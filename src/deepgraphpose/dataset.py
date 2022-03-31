"""Dataset loader classes."""

import copy
import os
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import yaml
from moviepy.editor import VideoFileClip
from skimage.util import img_as_ubyte
from tensorflow.python.util import deprecation
from tqdm import tqdm
from os.path import isfile, join, split

deprecation._PRINT_DEPRECATION_WARNINGS = False

vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf


def calculate_motion_energy(video_path):
    clip = VideoFileClip(str(video_path))
    n_frames = np.ceil(clip.fps * clip.duration).astype('int')

    # calculate the me per frame
    motion_energy = np.zeros((n_frames))
    previous_frame = None
    frame_idx = 0
    for frame_idx, frame in enumerate(clip.iter_frames()):
        if previous_frame is not None:
            motion_energy[frame_idx] = np.mean(np.abs(frame - previous_frame))
        previous_frame = frame
    motion_energy = motion_energy[:frame_idx + 1]
    clip.close()
    return motion_energy


def select_hidden_frames(ns, pv_all, pvh_sorted, n_frames, n_max_frames, ns_jump=None):
    """Get index of selected hidden frames"""

    # ns_jump is window to skip over values if too close
    # ns_jump=0 : skip all p^h indices in neighborhood of p^v
    # ns_jump=ns include indices even if in neighborhood of p^v
    if ns_jump is None:
        ns_jump = ns
    ns_small = max(ns - ns_jump, 1)

    # do not consider ph in pv_windowed
    # use real ns for pv_windowed not small chunk
    pv_windowed = get_neighboring_window(pv_all, ns, n_frames)

    ph_all = np.empty(0, dtype='int')

    if len(pv_windowed) >= n_max_frames:
        print(
            'Visible frames + window exceed n_max_frames; skipping selection of hidden frames'
        )
        return ph_all

    # add frames for training with highest me that are outside visible window
    ph_valid = pvh_sorted[~np.in1d(pvh_sorted, pv_windowed)]

    # select top random_perm frames
    pvh = pv_all.copy()

    selected = 0
    skipped = 0
    ph_skipped = []
    for ph_current in ph_valid:
        # print('ph_current',ph_current)
        # print('pvh',pvh)
        # print('ns_small',ns_small)
        # skip if we are too close to previous training frame
        if len(pvh) > 0:
            if np.min(np.abs(ph_current - pvh)) < ns_small:
                skipped += 1
                ph_skipped.append(ph_current)
                continue
        # do not exceed max frames
        frames_to_extract = get_neighboring_window(np.append(pvh, ph_current),
                                                   ns, n_frames)
        if len(frames_to_extract) > n_max_frames:
            break
        else:
            ph_all = np.append(ph_all, ph_current)
            pvh = np.append(pvh, ph_current)
            selected += 1

    print('Selected additional {} hidden frames'.format(selected))
    print(
        'Skipped {} high motion energy (me) frames since in visible window or close to higher me hidden frame'
            .format(skipped))
    return ph_all


def make_neighboring_window(window_size=5):
    """
    Make window of size -n:n
    """
    window = np.arange(window_size + 1)
    window = np.unique(np.concatenate((-1 * window[::-1], window)))
    return window


def get_neighboring_window(pv_all, ns, nt_max, nt_min=0):
    """get neighboring window"""
    window_hidden = make_neighboring_window(ns)
    pv_windowed = np.unique((pv_all[:, None] + window_hidden[None, :]))
    pv_windowed = pv_windowed[((pv_windowed >= nt_min) &
                               (pv_windowed < nt_max))]
    return pv_windowed


def initialize_resnet(dlc_cfg, nx_in, ny_in, allow_growth=True):
    from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

    TF.reset_default_graph()
    inputs = TF.placeholder(TF.float32, shape=[1, nx_in, ny_in, 3])
    pn = pose_net(dlc_cfg)

    # extract resnet outputs
    net, end_points = pn.extract_features(inputs)

    # restore from snapshot
    if 'snapshot' in dlc_cfg.init_weights:
        print('restoring from snapshot')
        variables_to_restore = slim.get_variables_to_restore()
    else:
        variables_to_restore = slim.get_variables_to_restore(
            include=["resnet_v1"])

    restorer = TF.train.Saver(variables_to_restore)

    # initialize tf session
    config_TF = TF.ConfigProto()
    config_TF.gpu_options.allow_growth = allow_growth
    sess = TF.Session(config=config_TF)

    # initialize weights
    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # restore the weights from disk
    restorer.restore(sess, dlc_cfg.init_weights)

    return sess, net, inputs


def find_marker_index(pv, ph, joint_loc):
    target_ind = pv
    nj = joint_loc.shape[1]
    # not really but none are nan
    nan_ind = []
    for jj in range(nj):
        joint_loc_tongue = joint_loc[:, jj, 0]  # whathever
        nan_ind += [list(nj * target_ind[np.isnan(joint_loc_tongue)] + jj)]

    from itertools import chain
    nan_ind = list(chain(*nan_ind))
    nan_ind = list(np.sort(nan_ind))

    ph_ts = ph * nj
    ph_ts_list = []
    for i in range(nj):
        ph_ts_list += [ph_ts + i]
    ph_ts = np.sort(np.array(ph_ts_list).flatten())
    ph_ts = np.sort(list(ph_ts) + list(nan_ind))

    pv_ts = pv * nj
    pv_ts_list = []
    for i in range(nj):
        pv_ts_list += [pv_ts + i]
    pv_ts0 = np.sort(np.array(pv_ts_list).flatten())
    pv_ts = np.sort(np.setdiff1d(pv_ts0, nan_ind))

    return pv_ts, ph_ts


def gen_idx_chunk(visible_frame_indices, hidden_frame_indices, joint_loc):
    """
    Parameters
    ----------
    visible_frame_indices : indices for visible frames, int array
    hidden_frame_indices : indices for hidden frames, int array
    joint_loc : joints for visible frames,
        hidden markers in visible frames have nan values,
        the number of indices in visible_frame_indices matches the number of rows in joint_loc

    Returns
    -------
    visible_marker : indices for visible markers, int array
    hidden_marker : indices for hidden markers, int array
    visible_marker_in_targets : indices for visible markers within visible frames, int array

    """
    nj = joint_loc.shape[1]  # number of joints

    # look for nan in joint_loc, these are hidden markers in visible frames
    nan_ind = []
    for jj in range(nj):
        joint_loc_jj = joint_loc[:, jj, 0]
        nan_ind += [list(nj * visible_frame_indices[np.isnan(joint_loc_jj)] + jj)]
    nan_ind = list(chain(*nan_ind))
    nan_ind = list(np.sort(nan_ind))

    # generate hidden marker indices from hidden frame indices, then combine them with nan_ind
    hidden_marker = hidden_frame_indices * nj
    hidden_marker_list = []
    for i in range(nj):
        hidden_marker_list += [hidden_marker + i]
    hidden_marker = np.sort(np.array(hidden_marker_list).flatten())
    hidden_marker = np.sort(list(hidden_marker) + list(nan_ind))

    # generate visible marker indices from visible frame indices
    visible_marker = visible_frame_indices * nj
    visible_marker_list = []
    for i in range(nj):
        visible_marker_list += [visible_marker + i]
    visible_marker0 = np.sort(np.array(visible_marker_list).flatten())
    visible_marker = np.sort(np.setdiff1d(visible_marker0, nan_ind))

    if len(visible_marker) == 0:  # if there is no visible marker
        visible_marker = np.empty(0, dtype='int')
        visible_marker_in_targets = np.empty(0, dtype='int')
    else:
        visible_marker_in_targets = np.nonzero(np.in1d(visible_marker0, np.setdiff1d(visible_marker0, nan_ind)))[0]

    if len(hidden_marker) == 0:  # if there is no hidden marker
        hidden_marker = np.empty(0, dtype='int')

    return visible_marker, hidden_marker, visible_marker_in_targets


def build_batch_key(ns_jump, step, ns, nc, n_max_frames, **kwargs):
    return 'nsjump=%s_step=%i_ns=%i_nc=%i_max=%i' % (ns_jump, step, ns, nc, n_max_frames)


def coord2map(pdata, joint_loc, nx_out, ny_out, nj):
    sm_size = np.array([nx_out, ny_out])
    locref_targets_all = []
    locref_mask_all = []

    for ii in range(joint_loc.shape[0]):
        joint_ii = joint_loc[ii, :, :].squeeze() * 8 + 4
        joint_ii = np.flip(joint_ii, 1)
        joint_ii = [joint_ii]
        joint_id = [np.array(range(nj))]
        jj = np.where(np.nan_to_num(joint_ii).sum(0).sum(1) != 0)[0]
        joint_id = [joint_id[0][jj]]
        joint_ii = [joint_ii[0][jj, :]]
        _, _, locref_targets, locref_mask = pdata.compute_target_part_scoremap(
            joint_id, joint_ii, 0, sm_size, 1)
        locref_targets_all += [locref_targets]
        locref_mask_all += [locref_mask]

    locref_targets_all = np.array(locref_targets_all).squeeze()
    locref_mask_all = np.array(locref_mask_all).squeeze()

    if len(locref_targets_all.shape) == 3:
        locref_targets_all = locref_targets_all[None, :, :, :]
        locref_mask_all = locref_mask_all[None, :, :, :]

    return locref_targets_all, locref_mask_all


def get_frame_idxs_from_train_mat(data_array, video):
    idxs = []
    for dat in data_array:
        idx = int(split(dat[0][0])[-1][3:].split('.')[0])

        dat00 = os.path.normpath(dat[0][0])
        dat00 = dat00.split(os.sep)

        if video in dat00:
            idxs.append(idx)
    return np.sort(idxs)


def get_frame_idxs_val(video, train_idxs):
    # TODO
    return np.array([])


def calculate_num_frames(clip):
    # np.ceil(self.video_clip.fps * self.video_clip.duration *1.0).astype('int')
    nframes = clip.duration * clip.fps
    nframes_fsec = nframes - int(nframes)
    # %%
    if (nframes_fsec < 1 / clip.fps):
        nframes = np.floor(nframes).astype('int')
    else:
        nframes = np.ceil(nframes).astype('int')
        print('Warning. Check the number of frames')
    return int(nframes)


class Dataset:

    def __init__(self, video_path, dlc_config, paths):

        self.video_path = video_path
        self.video_name = os.path.basename(str(
            self.video_path)).rpartition('.')[0]
        self.video_clip = VideoFileClip(str(self.video_path))
        self.video_n_frames = calculate_num_frames(self.video_clip)

        self.dlc_config = dlc_config
        self.paths = copy.deepcopy(paths)

        # record data dims
        #self.n_frames = int(self.video_clip.duration * self.video_clip.fps *
        #                    1.0)
        # TO DO: del
        self.n_frames = calculate_num_frames(self.video_clip)
        self.nj = self.dlc_config.num_joints
        # to fill upon creating batches
        self.ny_in, self.nx_in = self.video_clip.size
        self.nx_out, self.ny_out = self._compute_pred_dims()  # x, y dims of model output

        # load manual labels
        filename = join(self.dlc_config['project_path'],
                                self.dlc_config['dataset'])
        data = sio.loadmat(filename)['dataset'][0]
        idxs_train = get_frame_idxs_from_train_mat(data, self.video_name)
        print('### video name: ', self.video_name)
        print('### idxs_train: ', idxs_train)
        idxs_val = get_frame_idxs_val(dlc_config, idxs_train)
        print('### idxs_val: ', idxs_val)
        #assert len(idxs_train) > 0
        self.idxs = {'vis': {'train': idxs_train, 'val': idxs_val}}

        self.curr_batch = 0
        self.batch_data = None  # to speed up schedule 0

    def __str__(self):
        """Pretty printing of dataset info"""
        format_str = str('%s\n' % self.video_path)
        return format_str

    def _compute_pred_dims(self):
        """Compute output dims of dgp prediction layer by pushing fake data through network."""
        from deepgraphpose.models.fitdgp_util import dgp_prediction_layer
        from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

        TF.reset_default_graph()

        nc = 3
        inputs = TF.placeholder(TF.float32, shape=[None, self.nx_in, self.ny_in, nc])

        pn = pose_net(self.dlc_config)
        conv_inputs, end_points = pn.extract_features(inputs)

        x = dgp_prediction_layer(
            None, None, self.dlc_config, conv_inputs, 'confidencemap', self.nj, 0,
            nc, 1)

        sess = TF.Session(config=TF.ConfigProto())
        sess.run(TF.global_variables_initializer())
        sess.run(TF.local_variables_initializer())
        feed_dict = {inputs: np.zeros([1, self.nx_in, self.ny_in, nc])}
        x_np = sess.run(x, feed_dict)

        return x_np.shape[1], x_np.shape[2]

    def create_batches_from_resnet_output(self, batch_info, batches_path):
        """

        Parameters
        ----------
        batch_info
        batches_path

        Returns
        -------

        """

        from deepgraphpose.models.fitdgp_util import find_hidden_markers, find_visible_markers, \
            find_nan_ind
        self.paths['batched_data'] = batches_path
        video_name = split(str(self.video_path))[-1]
        self.batch_key = build_batch_key(**batch_info)
        self.paths['batched_data_hdf5'] = batches_path / str(
            '%s__%s.hdf5' % (video_name.split('.')[0], self.batch_key))

        pv_idxs, ph_idxs = self._process_video(batch_info)

        # store indices
        self.idxs['pv'] = pv_idxs
        self.idxs['ph'] = ph_idxs

        chunk_id = np.concatenate([pv_idxs, ph_idxs])
        chunk_len = len(chunk_id)
        ns_new = np.ceil(batch_info['n_max_frames'] / chunk_len / 2)
        if ns_new > batch_info['ns']:
            ns_new = batch_info['ns']
        self.idxs['chunk'] = get_neighboring_window( # all indices for visible, hidden, and window
            np.concatenate([pv_idxs, ph_idxs]), ns_new, self.video_n_frames)
        self.idxs['pv_chunk'] = np.where(np.in1d(self.idxs['chunk'], pv_idxs))[0]
        self.idxs['ph_chunk'] = np.where(np.in1d(self.idxs['chunk'], ph_idxs))[0]
        self.idxs['ph_all_chunk'] = np.where(~np.in1d(self.idxs['chunk'], pv_idxs))[0]

        # store labels
        if len(pv_idxs) > 0:
            target_2d, target_idxs, _, _ = self._compute_targets()
        else:
            target_2d, target_idxs = np.empty((0, self.nj, 2)), np.array([])
        self.labels = target_2d
        self.labels_idxs = target_idxs

        # find labels with nan values
        nan_idxs = find_nan_ind(self.idxs['pv_chunk'], self.labels)
        self.ph_all_ts = find_hidden_markers(self.idxs['ph_all_chunk'], self.nj,
                                    nan_idxs)
        _, self.pv_all_ts = find_visible_markers(self.idxs['pv_chunk'], self.nj,
                                       nan_idxs)

    def _process_video(self, batch_info):
        # use labeled frames as visible frames
        pv_idxs = self.idxs['vis']['train']
        # randomly select "good" hidden frames (with high motion energy)
        ph_idxs = self._find_good_hidden_frames(pv_idxs, batch_info)
        return pv_idxs, ph_idxs

    def _create_batches_from_resnet_output(self, pv_idxs, ph_idxs, batch_info):
        """

        Parameters
        ----------
        pv_idxs
        ph_idxs
        batch_info

        Returns
        -------

        """

        hdf5_file = self.paths['batched_data_hdf5']
        if not os.path.exists(os.path.dirname(hdf5_file)):
            os.makedirs(os.path.dirname(hdf5_file))

        # get indices of all visible/hidden frames and their windows
        data_idxs = get_neighboring_window(np.concatenate([pv_idxs, ph_idxs]),
                                           batch_info['ns'],
                                           self.video_n_frames)
        n_frames_to_extract = len(data_idxs)

        # -----------------
        # initialize resnet
        # -----------------
        ny_in, nx_in = self.video_clip.size
        # set number of final resnet channels to output
        nc = batch_info['nc']
        nc = 2048 if nc is None else nc
        sess, net, inputs = initialize_resnet(self.dlc_config, nx_in, ny_in)

        # --------------------------
        # push frames through resent
        # --------------------------
        print('Processing %i video frames\nOutput saved to %s' %
              (n_frames_to_extract, hdf5_file))
        with h5py.File(hdf5_file, 'w', libver='latest', swmr=True) as f:

            f.swmr_mode = True  # single write multi-read

            # save out frames and processed frames
            g_out = f.create_group('resnet_out')
            g_in = f.create_group('frames')
            g_idxs = f.create_group('idxs')
            g_dims = f.create_group('dims')

            # save out indices
            g_idxs.create_dataset('visible', data=pv_idxs, dtype='int')
            ph_idxs_all = data_idxs[~np.in1d(data_idxs, pv_idxs)]
            g_idxs.create_dataset('hidden', data=ph_idxs_all, dtype='int')
            g_idxs.create_dataset('hidden_selected', data=ph_idxs, dtype='int')

            pbar = tqdm(total=n_frames_to_extract,
                        desc='Processing video frames')
            for idx in data_idxs:

                # get resnet output
                ff = img_as_ubyte(
                    self.video_clip.get_frame(idx * 1. / self.video_clip.fps))
                net_output = sess.run(net,
                                      feed_dict={inputs: ff[None, :, :, :]})
                if isinstance(nc, int):
                    ff_out = net_output[:, :, :, :nc]
                elif isinstance(nc, np.ndarray):
                    ff_out = net_output[:, :, :, nc]
                else:
                    raise Exception('Not proper resnet channel selection')

                # store resnet output of each frame in hdf5 file
                g_in.create_dataset('%i' % idx, data=ff, dtype='uint8')
                g_out.create_dataset('%i' % idx, data=ff_out, dtype='float32')

                pbar.update(1)

            # save resnet output dims for easy loading later
            g_dims.create_dataset('resnet_out',
                                  data=ff_out.shape[1:3],
                                  dtype='int')

        pbar.close()
        sess.close()

    def _find_good_hidden_frames(self, pv_idxs, batch_info):

        # see if frames have been previously stored
        video_name = split(str(self.video_path))[-1]
        idxs_file = self.paths['batched_data'] / str(
            '%s__%s_idxs.npy' % (video_name.split('.')[0], self.batch_key))

        if os.path.exists(idxs_file):
            idxs = np.load(idxs_file, allow_pickle=True).item()
            if np.all(np.sort(pv_idxs) == np.sort(idxs['pv'])):
                print('loading hidden indices from %s' % idxs_file)
                ph_idxs_sel = idxs['ph']
                return ph_idxs_sel

        # compute me of video to find good hidden frames
        me = calculate_motion_energy(self.video_path)
        idxs_me_sort = np.argsort(me).flatten()[::-1]

        # select "good" hidden frames (with highest motion energy, not within visible windows)
        ph_idxs_sel_all = select_hidden_frames(
            ns=batch_info['ns'],
            pv_all=pv_idxs,
            pvh_sorted=idxs_me_sort,
            n_frames=self.video_n_frames,
            n_max_frames=batch_info['n_max_frames'],
            ns_jump=batch_info['ns_jump'])
        ph_idxs_sel_all = np.sort(ph_idxs_sel_all)

        # select a subset of these "good" frames by skipping over neighbors
        # TODO: this ends up with many fewer than n_max_frames, need to update
        idxs_subset = np.arange(0, len(ph_idxs_sel_all),
                                batch_info['step']).astype(np.int)
        ph_idxs_sel = ph_idxs_sel_all[idxs_subset]

        # store indices to speed up future calls
        if not os.path.exists(os.path.dirname(idxs_file)):
            os.makedirs(os.path.dirname(idxs_file))
        np.save(idxs_file, {'pv': pv_idxs, 'ph': ph_idxs_sel})

        return ph_idxs_sel

    def _add_labels_to_batches(self):

        # load labels
        target_2d, target_idxs, nx_out, ny_out = self._compute_targets()

        # find matrices corresponding to train/val labels
        idxs = self.idxs['vis']['train']
        assert len(idxs) == target_2d.shape[0]
        assert len(idxs) == len(target_idxs)

        # save 2d labels to hdf5
        with h5py.File(self.paths['batched_data_hdf5'],
                       'a',
                       libver='latest',
                       swmr=True) as f:
            f.swmr_mode = True  # single write multi-read

            g_dims = f['dims']
            g_dims.create_dataset('dgp_out',
                                  data=np.array([nx_out, ny_out]),
                                  dtype='int')

            g_labels = f.create_group('labels')

            for i, idx in enumerate(target_idxs):
                g_labels.create_dataset('%i' % idx,
                                        data=target_2d[i],
                                        dtype='float32')

    def _compute_targets(self):

        from deeplabcut.pose_estimation_tensorflow.dataset.factory import \
            create as create_dataset

        dlc_config = copy.deepcopy(self.dlc_config)
        dlc_config['deterministic'] = True
        # switch to default dataset_type to produce expected batch output
        dlc_config['dataset_type'] = 'default'
        dataset = create_dataset(dlc_config)
        nt = len(self.idxs['vis']['train'])  # number of training frames
        # assert nt >= 1
        nj = self.nj #Taiga edit 8/31/21: untested.  max([dat_.joints[0].shape[0] for dat_ in dataset.data])
        stride = dlc_config['stride']

        def extract_frame_num(img_path):
            return int(split(img_path)[-1][3:].split('.')[0])

        frame_idxs = []
        joinss = []
        counter = 0
        while counter < nt:

            data = dataset.next_batch()
            data_keys = list(data.keys())
            # inputs = 0
            # part_score_targets = 1
            # part_score_weights = 2
            # locref_targets = 3
            # locref_mask = 4
            # pairwise_targets = 5
            # pairwise_mask = 6
            # data_item = 7

            im_path = data[data_keys[5]].im_path
            # skip if frame belongs to another video
            im_path_split = os.path.normpath(im_path).split(os.sep)
            if self.video_name not in im_path_split:
            #if im_path.find('/' + self.video_name + '/') == -1:
                continue
            # skip if image has already been processed
            frame_idx = extract_frame_num(im_path)
            if frame_idx in frame_idxs:
                continue

            # inputs = data[data_keys[0]].squeeze()
            # part_score_targets = data[data_keys[1]].squeeze()  # multi class labels
            # part_score_weights = data[data_keys[2]].squeeze()
            # locref_targets = data[data_keys[3]].squeeze()
            # locref_mask  = data[data_keys[4]].squeeze()
            data_item = data[data_keys[-1]]
            joinss += [np.copy(data_item.joints[0])]

            frame_idxs.append(frame_idx)
            counter += 1

        # to find 2D coordinates, we must update
        targets_2d = np.zeros((nt, nj, 2)) * np.nan  # nt x nj x 2
        for ntt in range(nt):
            cjoin = joinss[ntt]  # D x nj
            njtt = cjoin.shape[0]
            for njj_id in range(njtt):
                njj = cjoin[njj_id][0]
                joinss_ntt_njj = cjoin[njj_id][1:]
                targets_2d[ntt, njj] = np.flip(
                    (joinss_ntt_njj - stride / 2) / stride)

        nx_out = int(data[data_keys[1]].squeeze().shape[0])
        ny_out = int(data[data_keys[1]].squeeze().shape[1])

        return targets_2d, frame_idxs, nx_out, ny_out

    def load_frame_idxs_from_hdf5(self):
        with h5py.File(self.paths['batched_data_hdf5'], 'r') as f:
            pv_idxs = f['idxs']['visible'][()]
            ph_idxs = f['idxs']['hidden_selected'][()]
        return pv_idxs, ph_idxs

    def reset(self):
        """randomly shuffle visible and hidden frame indices."""
        # np.random.seed()?
        np.random.shuffle(self.idxs['pv'])
        np.random.shuffle(self.idxs['ph'])
        self.curr_batch = 0

    def next_batch(self, schedule, batch_dict, pv_idxs=None, ph_idxs=None):
        """Load image data and labels for a given batch.

        If `pv_idxs` and `ph_idxs` are None, these indices are randomly chosen depending on the
        schedule.

        Parameters
        ----------
        schedule : int
        batch_dict : dict
            - ns (int): window size (one-sided)
            - nc (int): number of channels when loading resnet outputs
        pv_idxs : array-like, optional
            visible frame indices for batch
        ph_idxs : array-like, optional
            hidden frame indices for batch, does not include temporal windows

        Returns
        -------
        :obj:`tuple`
            - pv_idxs (list): visible indices (full video reference)
            - ph_idxs (list): hidden indices (full video reference)
            - pv_idxs_b (list): visible indices (batch reference)
            - network_input (np.ndarray): input to model for batch
            - labels (np.ndarray): labels (network target) for batch
            - batch_mask (np.ndarray)
            - batch_ts (np.ndarray)
            - addn_batch_info (tuple)

        """

        if pv_idxs is None and ph_idxs is None:
            # get pv/ph idxs depending on iter/schedule/batch_num
            pv_idxs, ph_idxs = self.get_visible_hidden_idxs(
                schedule, batch_dict['ns'])
            # pv_idxs = np.random.choice(np.copy(pv_idxs), size=10, replace=False)

        # check if we can use previous batch (only relevant for single dataset in schedule 0)
        if self.batch_data is not None:
            pv_idxs_old = self.batch_data[0]
            ph_idxs_old = self.batch_data[1]
        else:
            pv_idxs_old = np.array([])
            ph_idxs_old = np.array([])

        pv_overlap = pv_idxs_old == pv_idxs
        ph_overlap = ph_idxs_old == ph_idxs
        pv_size_match = pv_idxs_old.size == pv_idxs.size
        ph_size_match = ph_idxs_old.size == ph_idxs.size
        if np.all(pv_overlap) and np.all(ph_overlap) and pv_size_match and ph_size_match:
            print('using previous batch')
        else:
            idxs_video = np.sort(np.concatenate([pv_idxs, ph_idxs]))

            # load batch from hdf5/video
            network_input, labels = self.load_data(idxs_video, pv_idxs)

            # get batch-based idxs of visible/hidden frames
            pv_idxs_b = np.where(np.in1d(idxs_video, pv_idxs))[0]
            ph_idxs_b = np.where(np.in1d(idxs_video, ph_idxs))[0]

            # compute batch mask
            batch_mask = np.zeros(len(idxs_video) - 1, dtype=np.int)
            batch_mask[np.where(np.diff(idxs_video) == 1)[0]] = 1

            # compute batch_ts
            pv_chunk = np.where(np.in1d(self.idxs['chunk'], pv_idxs))[0]
            ph_chunk = np.where(np.in1d(self.idxs['chunk'], ph_idxs))[0]
            pv_full_ts, ph_full_ts = find_marker_index(pv_chunk, ph_chunk, labels)
            batch_ts0 = np.unique(list(pv_full_ts) + list(ph_full_ts))
            batch_ts = self.global_offset * self.nj + batch_ts0

            # compute additional batch info
            addn_batch_info = gen_idx_chunk(pv_idxs_b, ph_idxs_b, labels)

            self.batch_data = (pv_idxs, ph_idxs, pv_idxs_b, network_input,
                               labels, batch_mask, batch_ts, addn_batch_info)

        return self.batch_data

    def get_visible_hidden_idxs(self, schedule, ns):
        pv_idxs = self.idxs['pv']
        ph_idxs = self.idxs['ph']
        if schedule == 0:
            # schedule 0: load all visible frames with no hidden frames or windows
            if self.curr_batch == 1:
                raise StopIteration
            pv = pv_idxs
            ph = np.asarray([])
        elif schedule == 1:
            # schedule 1: load selected visible frame plus temporal window, no hidden frames
            if self.curr_batch == len(pv_idxs):
                print('iterator for dataset empty; skipping')
                raise StopIteration
            pb = [pv_idxs[self.curr_batch]]
            print('pb', pb)
            pb1 = []
            for vv in pb:
                pb1 += list(range(vv - ns, vv + ns + 1))
            pb1 = np.unique(np.array(pb1))
            pv = np.array([
                i for i in pb1
                if (i in pv_idxs) and (i < self.n_frames) and (i >= 0)
            ])
            ph = np.array([
                i for i in pb1
                if (i not in pv_idxs) and (i < self.n_frames) and (i >= 0)
            ])
        elif schedule == 2:
            # schedule 2: load selected hidden frame and visible frame, plus temporal windows
            idx_h = self.curr_batch  # np.mod(iter, len(ph_idxs))
            idx_v = np.mod(self.curr_batch, len(pv_idxs))
            if idx_v == 0:
                # reshuffle visible indices
                np.random.shuffle(self.idxs['pv'])
                pv_idxs = self.idxs['pv']
                idx_v = np.mod(self.curr_batch, len(pv_idxs))
            if idx_h == len(ph_idxs):
                print('iterator for dataset empty; skipping')
                raise StopIteration
            pb = [pv_idxs[idx_v]] + [ph_idxs[idx_h]]
            print('pb', pb)
            pb1 = []
            for vv in pb:
                pb1 += list(range(vv - ns, vv + ns + 1))
            pb1 = np.unique(np.array(pb1))
            pv = np.array([
                i for i in pb1
                if (i in pv_idxs) and (i < self.n_frames) and (i >= 0)
            ])
            ph = np.array([
                i for i in pb1
                if (i not in pv_idxs) and (i < self.n_frames) and (i >= 0)
            ])
        else:
            raise ValueError('%i is not a valid schedule' % schedule)
        self.curr_batch += 1
        return np.sort(pv), np.sort(ph)

    def load_data(self, idxs_video, pv_idxs):
        images = np.zeros((len(idxs_video), self.nx_in, self.ny_in, 3))
        for i, idx in enumerate(idxs_video):
            images[i] = img_as_ubyte(
                self.video_clip.get_frame(idx * 1. / self.video_clip.fps))

        # get idxs for labels (wrt all visible frames)
        idxs_labels = [(np.where(self.labels_idxs == i)[0]).item() for i in pv_idxs]
        labels = self.labels[idxs_labels]

        return images, labels


class MultiDataset:

    def __init__(self, config_yaml, video_sets=None, shuffle=None, S0=None):

        self.datasets = []
        self.paths = {}

        # load dataset config
        from deepgraphpose.utils_model import get_train_config

        # load config.yaml file for project
        with open(config_yaml, 'r') as stream:
            self.proj_config = yaml.safe_load(stream)
        if video_sets is None:
            self.proj_config['video_path'] = self.proj_config['video_sets']  # backwards compat
        else:
            video_set_keys = self.proj_config['video_sets'].keys()
            video_set_keys = [split(v)[-1] for v in video_set_keys]
            #print('video_set_keys: ', video_set_keys)
            video_set_input = [split(v)[-1] for v in video_sets]
            #print('video_set_input: ', video_set_input)
            if set(video_set_keys)==set(video_set_input):
                self.proj_config['video_path'] = self.proj_config['video_sets']
            else:
                self.proj_config['video_path'] = {v: {} for v in video_sets}  # backwards compat
                self.proj_config['video_sets'] = {v: {} for v in video_sets}

        # update video paths from relative to absolute
        self.proj_config['video_sets'] = {
            join(self.proj_config['project_path'], key): val
            for key, val in self.proj_config['video_sets'].items()
        }

        self.dlc_config = get_train_config(self.proj_config, shuffle)

        # save path info
        self.paths['project'] = Path(self.dlc_config.project_path)
        self.paths['dlc_model'] = Path(self.dlc_config.snapshot_prefix).parent
        self.paths['batched_data'] = ''

        # create a dataset for each video
        self.video_files = self.proj_config['video_sets'].keys()
        assert len(self.video_files) > 0
        self.batch_ratios = []
        for video_file in self.video_files:
            self.datasets.append(Dataset(video_file, self.dlc_config, self.paths))
            self.batch_ratios.append(len(self.datasets[-1].idxs['vis']['train']))
        self.batch_ratios = np.array(self.batch_ratios) / np.sum(self.batch_ratios)
        # collect info about datasets
        self.n_datasets = len(self.datasets)
        self.nj = self.datasets[0].nj
        self.nx_in = None
        self.ny_in = None
        self.nx_out = None
        self.ny_out = None
        self.S0 = S0

        self.n_visible_frames_total = 0  # labeled frames
        self.n_hidden_frames_total = 0  # selected hidden (unlabeled) frames
        self.n_frames_total = 0  # labeled + selected hidden + temporal windows

        self.curr_batch = 0  # keep track of batches served per schedule

    def __str__(self):
        """Pretty printing of dataset info"""
        format_str = str('MultiDataset contains %i videos:\n' % self.n_datasets)
        for dataset in self.datasets:
            format_str += dataset.__str__()
        return format_str

    def __len__(self):
        return self.n_datasets

    def create_batches_from_resnet_output(self,
                                          snapshot,
                                          ns_jump=None,
                                          ns=10,
                                          nc=200,
                                          step=2,
                                          n_max_frames=1000):
        """Push train/test data through resnet and store outputs in hdf5.

        Parameters
        ----------
        snapshot : int
            dlc training snapshot index
        ns_jump : int
        ns : int
            one-sided window length around visible/hidden frames
        nc : int
            number of resnet output channels to keep
        step : int
        n_max_frames : int
            total number of frames (visible + hidden + windows) to store for later training

        """

        self.snapshot = snapshot
        self.batch_info = {
            'ns_jump': ns_jump,
            'ns': ns,
            'nc': nc,
            'step': step,
            'n_max_frames': n_max_frames
        }
        self.paths['batched_data'] = \
            self.paths['dlc_model'] / 'batched_data' / 'snapshot-{}'.format(snapshot)
        self.batch_key = build_batch_key(ns_jump, step, ns, nc, n_max_frames)

        print('\n\n')
        print('Creating training datasets')
        print('--------------------------')
        for dataset in self.datasets:
            dataset.create_batches_from_resnet_output(self.batch_info, self.paths['batched_data'])
        print('\n\n')

        # update data dims
        self.nx_in = self.datasets[0].nx_in
        self.ny_in = self.datasets[0].ny_in
        self.nx_out, self.ny_out = self.datasets[0].nx_out, self.datasets[0].ny_out

        # update info on full dataset
        self.n_visible_frames_total = 0  # labeled frames
        self.n_hidden_frames_total = 0  # selected hidden (unlabeled) frames
        self.n_frames_total = 0  # labeled + selected hidden + temporal windows
        for i, dataset in enumerate(self.datasets):
            print('Video: ', dataset.video_name, ' has ', len(dataset.idxs['pv']), ' visible frames selected; ', len(dataset.idxs['ph']), ' hidden frames selected.')
            self.n_visible_frames_total += len(dataset.idxs['pv'])
            self.n_hidden_frames_total += len(dataset.idxs['ph'])

            dataset.global_offset = self.n_frames_total
            self.n_frames_total += len(dataset.idxs['chunk'])

    def reset(self):
        """Reset iterators so that all data is available."""
        for dataset in self.datasets:
            dataset.reset()
        self.curr_batch = 0

    def next_batch(self, schedule, dataset=None, pv_idxs=None, ph_idxs=None):
        """Return next batch of data.

        The data generator iterates randomly through datasets and frames. Once a dataset runs out
        of frames it is skipped.

        If all of `dataset`, `pv_idxs`, and `ph_idxs` are not None, these indices from the
        specified dataset are loaded.

        Parameters
        ----------
        schedule : int
        dataset : int, optional
            index into dataset list
        pv_idxs : array-like, optional
            visible frame indices for batch
        ph_idxs : array-like, optional
            hidden frame indices for batch, does not include temporal windows

        Returns
        -------
        :obj:`tuple`
            - data (:obj:`tuple`): data batch
            - dataset (:obj:`int`): dataset from which data batch is drawn

        """
        if dataset is None or pv_idxs is None or ph_idxs is None:
            # pick random dataset/indices
            while True:

                if schedule == 0:
                    if self.curr_batch % self.n_datasets == 0:
                        if self.curr_batch != 0 and self.n_datasets != 1:
                            print(
                                'processed all visible frames; resetting batcher'
                            )
                        self.reset()
                elif schedule == 1:
                    if self.curr_batch % self.n_visible_frames_total == 0:
                        if self.curr_batch != 0:
                            print(
                                'processed all visible frames; resetting batcher'
                            )
                        self.reset()
                elif schedule == 2:
                    if self.curr_batch % self.n_hidden_frames_total == 0:
                        if self.curr_batch != 0:
                            print(
                                'processed all hidden frames; resetting batcher'
                            )
                        self.reset()
                else:
                    raise Exception('invalid schedule number %i' % schedule)

                # get next dataset - ratios are based on number of visible frames
                dataset = np.random.choice(np.arange(self.n_datasets),
                                           p=self.batch_ratios)

                # get this session data
                try:
                    data = self.datasets[dataset].next_batch(
                        schedule, self.batch_info)
                    break
                except StopIteration:
                    continue

            self.curr_batch += 1
        else:
            data = self.datasets[dataset].next_batch(schedule,
                                                     self.batch_info,
                                                     pv_idxs=pv_idxs,
                                                     ph_idxs=ph_idxs)

        return data, dataset


