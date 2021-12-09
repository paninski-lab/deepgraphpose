import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_slim as slim
import yaml
from moviepy.editor import VideoFileClip
from skimage.draw import circle
from skimage.util import img_as_ubyte
from tqdm import tqdm
from pathlib import Path
from os.path import isfile, join, split

vers = tf.__version__.split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


def get_clip_frames(clip, frames_idxs, num_channels=3):
    numframes = len(frames_idxs)
    xlim, ylim = clip.size
    fps = clip.fps

    frames_arr = np.zeros((numframes, ylim, xlim, num_channels))

    for idx, frame_idx in enumerate(frames_idxs):
        # print(frame_idx)
        frame_idx_sec = frame_idx / fps
        frames_arr[idx] = clip.get_frame(frame_idx_sec)
    return frames_arr


def make_cmap(number_colors, cmap="cool"):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


def create_annotated_movie(clip, df_x, df_y, mask_array=None, dotsize=5, colormap="cool",
        filename="movie.mp4"):
    """Helper function for creating annotated videos.

    Parameters
    ----------
    clip : moviepy.editor.VideoFileClip
    df_x : np.ndarray
        shape T x n_joints
    df_y : np.ndarray
        shape T x n_joints
    mask_array : np.ndarray, boolean, optional
        shape T x n_joints, same as df_x and df_y; any timepoints/joints with a False entry will
        not be plotted
    dotsize : int
        size of marker dot on labeled video
    colormap : str
        matplotlib color map for markers
    filename : str, optional
        video file name

    """

    if mask_array is None:
        mask_array = ~np.isnan(df_x)
    # ------------------------------
    # Get number of colorvars to plot

    number_body_parts, T = df_x.shape

    # Set colormap for each color
    colors = make_cmap(number_body_parts, cmap=colormap)

    nx, ny = clip.size
    duration = int(clip.duration - clip.start)
    fps = clip.fps
    int(duration * fps)

    print("Duration of video [s]: ", round(duration, 2), ", recorded with ",
        round(fps, 2), "fps!", )

    # print("Overall # of frames: ", nframes, "with cropped frame dimensions: ", nx, ny)
    # print("Generating frames and creating video.")

    # add marker to each frame t, where t is in sec
    def add_marker(get_frame, t):
        image = get_frame(t * 1.0)

        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * 1.0 * fps))

        if index % 1000 == 0:
            print("\nTime frame @ {} [sec] is {}".format(t, index))

        for bpindex in range(number_body_parts):
            if index >= T:
                print('SKipped frame {}, marker {}'.format(index, bpindex))
                continue
            if mask_array[bpindex, index]:
                xc = min(int(df_x[bpindex, index]), nx - 1)
                yc = min(int(df_y[bpindex, index]), ny - 1)
                # rr, cc = circle_perimeter(yc, xc, dotsize, shape=(ny, nx))
                rr, cc = circle(yc, xc, dotsize, shape=(ny, nx))
                frame[rr, cc, :] = colors[bpindex]

        return frame

    clip_marked = clip.transform(add_marker)

    clip_marked.write_videofile(str(filename), codec="mpeg4", fps=fps, bitrate="1000k")
    clip_marked.close()
    return


def create_movie_comparison(dgp_movie, dlc_movie, fnameout='dgp_dlc_compare.mp4'):
    from moviepy.editor import VideoFileClip, clips_array

    # add concatenated video
    outputmovie0 = VideoFileClip(str(dlc_movie))
    outputmovie1 = VideoFileClip(str(dgp_movie))
    # dgp_name = TextClip('DGP', #font='Amiri-Bold',
    #                    color="white",fontsize=20).set_position(('left', 'top')).set_duration(outputmovie1.duration)
    # dlc_name = TextClip('DLC', #font='Amiri-Bold',
    #                    color="white",fontsize=20).set_position(('left', 'top')).set_duration(outputmovie1.duration)

    # final_clip0 = CompositeVideoClip([outputmovie0, dlc_name])
    # final_clip1 = CompositeVideoClip([outputmovie1, dgp_name])

    # concat_final_clip = clips_array([[final_clip1, final_clip0]])
    concat_final_clip = clips_array([[outputmovie0, outputmovie1]])
    concat_final_clip.write_videofile(str(fnameout))

    concat_final_clip.close()
    outputmovie0.close()
    outputmovie1.close()
    print('\n Movie in {}'.format(fnameout))
    return


def setup_dgp_eval_graph(dlc_cfg, dgp_model_file, loc_ref=False, gauss_len=1, gamma=1):
    """Helper function to set up dgp graph and return input/output tensors.

    Parameters
    ----------
    dlc_cfg : str
        dlc model config file (.yaml)
    dgp_model_file : str
        dgp model weights; .ckpt if fitting full resnet, .npy if only fitting final conv layer
    loc_ref : bool
        True to use on location refinement (only when fitting full resnet)
    gauss_len : float
    gamma : float

    Returns
    -------
    tuple
        tf session (tf.Session object)
        mu_n (tf.Tensor)
        softmax_tensor (tf.Tensor)
        scmap (tf.Tensor)
        locref (tf.Tensor)
        inputs (tf.Tensor)

    """
    from deeplabcut.pose_estimation_tensorflow.nnets import PoseNetFactory
    from deepgraphpose.models.fitdgp_util import argmax_2d_from_cm, dgp_prediction_layer

    # -------------------
    # define model
    # -------------------
    TF.compat.v1.reset_default_graph()
    inputs = TF.compat.v1.placeholder(tf.float32, shape=[1, None, None, 3])
    pn = PoseNetFactory.create(dlc_cfg)
    # extract resnet outputs
    net, end_points = pn.extract_features(inputs)

    with tf.compat.v1.variable_scope('pose', reuse=None):
        scmap = dgp_prediction_layer(None, None, dlc_cfg, net, name='part_pred',
            num_outputs=dlc_cfg['num_joints'], init_flag=False, nc=None, train_flag=True)
        if loc_ref:
            locref = dgp_prediction_layer(None, None, dlc_cfg, net, name='locref_pred',
                num_outputs=dlc_cfg['num_joints'] * 2, init_flag=False, nc=None,
                train_flag=True)
        else:
            locref = None
    variables_to_restore = slim.get_variables_to_restore()
    restorer = TF.compat.v1.train.Saver(variables_to_restore)
    weights_location = str(dgp_model_file)

    mu_n, softmax_tensor = argmax_2d_from_cm(scmap, dlc_cfg['num_joints'], gamma, gauss_len)

    # initialize tf session
    config_TF = TF.compat.v1.ConfigProto()
    config_TF.gpu_options.allow_growth = True
    sess = TF.compat.v1.Session(config=config_TF)

    # initialize weights
    sess.run(TF.compat.v1.global_variables_initializer())
    sess.run(TF.compat.v1.local_variables_initializer())

    # restore resnet from dlc trained weights
    print('loading resnet model weights from %s...' % weights_location, end='')
    restorer.restore(sess, weights_location)
    print('done')

    return sess, mu_n, softmax_tensor, scmap, locref, inputs


def estimate_pose(proj_cfg_file, dgp_model_file, video_file, output_dir, shuffle=1,
                  save_pose=True, save_str='', new_size=None, crop_size=None):
    """Estimate pose on an arbitrary video.

    Parameters
    ----------
    proj_cfg_file : str, optional
        dlc project config file (.yaml) (if `label_dir` is None)
    dgp_model_file : str, optional
        dgp model weights; .ckpt if fitting full resnet, .npy if only fitting final conv layer
    video_file : str
        video to label
    output_dir : str
        output directory to store labeled video
    shuffle : int, optional
        dlc shuffle number
    save_pose : bool, optional
        True to save out pose in csv/hdf5 file
    save_str : str, optional
        additional string to append to labeled video file name

    Returns
    -------
    dict

    """
    from deepgraphpose.utils_model import get_train_config

    f = os.path.basename(video_file).rsplit('.', 1)
    save_file = join(output_dir, f[0] + '_labeled%s' % save_str)
    if os.path.exists(save_file + '.csv'):
        print('labels already exist! video at %s will not be processed' % video_file)
        return save_file + '.csv'

    # -------------------
    # loading
    # -------------------
    # load video
    print('initializing video clip...', end='')
    video_clip = VideoFileClip(str(video_file))
    n_frames = np.ceil(video_clip.fps * video_clip.duration).astype('int')
    print('done')

    # load dlc project config file
    print('loading dlc project config...', end='')
    with open(proj_cfg_file, 'r') as stream:
        proj_config = yaml.safe_load(stream)
    proj_config['video_path'] = None
    dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
    print('done')

    # -------------------
    # extract pose
    # -------------------
    try:
        dlc_cfg['net_type'] = 'resnet_50'
        sess, mu_n, _, scmap, _, inputs = setup_dgp_eval_graph(dlc_cfg, dgp_model_file)
    except:
        dlc_cfg['net_type'] = 'resnet_101'
        sess, mu_n, _, scmap, _, inputs = setup_dgp_eval_graph(dlc_cfg, dgp_model_file)

    print('\n')
    """
    pbar = tqdm(total=n_frames, desc='processing video frames')
    markers = np.zeros((n_frames, dlc_cfg.num_joints, 2))
    likelihoods = np.zeros((n_frames, dlc_cfg.num_joints))
    for i, frame in enumerate(video_clip.iter_frames()):
        # get resnet output
        ff = img_as_ubyte(frame)
        mu_n_batch = sess.run(mu_n, feed_dict={inputs: ff[None, :, :, :]})
        markers[i] = mu_n_batch * dlc_cfg.stride + 0.5 * dlc_cfg.stride
        likelihoods[i] = 0.5

        pbar.update(1)
    """
    # %%
    nj = dlc_cfg['num_joints']
    nx, ny = video_clip.size
    nx_out, ny_out = int((nx - dlc_cfg['stride'] / 2) / dlc_cfg['stride'] + 1) + 5, int(
        (ny - dlc_cfg['stride'] / 2) / dlc_cfg['stride'] + 1) + 5

    # %%
    markers = np.zeros((n_frames, dlc_cfg['num_joints'], 2))

    mu_likelihoods = np.zeros((n_frames, nj, 2)).astype('int')
    likelihoods = np.zeros((n_frames, nj))
    offset_mu_jj = 0

    pbar = tqdm(total=n_frames, desc='processing video frames')
    for ii, frame in enumerate(video_clip.iter_frames()):
        frame = Image.fromarray(frame)
        if new_size is not None:
            scale_x = frame.width / new_size[1]
            scale_y = frame.height / new_size[0]
            # width, height
            frame = frame.resize(size=(new_size[1], new_size[0]))
        else:
            scale_x = 1
            scale_y = 1
        if crop_size is not None:
            # crop
            frame = frame.crop(crop_size)
            xmin = crop_size[0]
            ymin = crop_size[1]
        else:
            xmin = 0
            ymin = 0

        frame = np.asarray(frame)
        ff = img_as_ubyte(frame)

        mu_n_batch, scmap_np = sess.run([mu_n, scmap], feed_dict={inputs: ff[None, :, :, :]})
        markers[ii] = mu_n_batch[0]
        softmaxtensor = scmap_np[0]
        for jj_idx in range(nj):
            mu_jj = markers[ii, jj_idx]
            ends_floor = np.floor(mu_jj).astype('int') - offset_mu_jj
            ends_ceil = np.ceil(mu_jj).astype('int') + 1 + offset_mu_jj
            sigmoid_pred_np_jj = np.exp(softmaxtensor[:, :, jj_idx]) / \
                                 (np.exp(softmaxtensor[:, :, jj_idx]) + 1)
            spred_centered = sigmoid_pred_np_jj[
                             ends_floor[0]:ends_ceil[0], ends_floor[1]:ends_ceil[1]]
            mu_likelihoods[ii, jj_idx] = np.unravel_index(np.argmax(spred_centered),
                                                          spred_centered.shape)
            mu_likelihoods[ii, jj_idx] += [ends_floor[0], ends_floor[1]]
            likelihoods[ii, jj_idx] = sigmoid_pred_np_jj[
                int(mu_likelihoods[ii, jj_idx][0]), int(mu_likelihoods[ii, jj_idx][1])]

        pbar.update(1)

    pbar.close()
    sess.close()
    video_clip.close()

    # %%
    xr = markers[:, :, 1] * dlc_cfg['stride'] + 0.5 * dlc_cfg['stride']  # T x nj
    yr = markers[:, :, 0] * dlc_cfg['stride'] + 0.5 * dlc_cfg['stride']
    # %%
    # true xr
    xr *= scale_x
    yr *= scale_y
    print('Finished collecting markers')
    print('Storing data')

    # -------------------
    # save labels
    # -------------------
    labels = {'x': xr, 'y': yr, 'likelihoods': likelihoods}

    # convert to DLC-like csv/hdf5
    if save_pose:
        if not Path(save_file).parent.exists():
            os.makedirs(os.path.dirname(save_file))
        export_pose_like_dlc(labels, os.path.basename(dgp_model_file),
                             dlc_cfg['all_joints_names'], save_file)
    return labels


def estimate_pose_obsolete(proj_cfg_file, dgp_model_file, video_file, output_dir, shuffle=1,
                  save_pose=True, save_str='', new_size=None):
    """Estimate pose on an arbitrary video.
    Parameters
    ----------
    proj_cfg_file : str, optional
        dlc project config file (.yaml) (if `label_dir` is None)
    dgp_model_file : str, optional
        dgp model weights; .ckpt if fitting full resnet, .npy if only fitting final conv layer
    video_file : str
        video to label
    output_dir : str
        output directory to store labeled video
    shuffle : int, optional
        dlc shuffle number
    save_pose : bool, optional
        True to save out pose in csv/hdf5 file
    save_str : str, optional
        additional string to append to labeled video file name
    Returns
    -------
    dict
    """
    from deepgraphpose.utils_model import get_train_config

    f = os.path.basename(video_file).rsplit('.', 1)
    save_file = os.path.join(output_dir, f[0] + '_labeled%s' % save_str)
    if os.path.exists(save_file + '.csv'):
        print('labels already exist! video at %s will not be processed' % video_file)
        return save_file + '.csv'

    # -------------------
    # loading
    # -------------------
    # load video
    print('initializing video clip...', end='')
    video_clip = VideoFileClip(str(video_file))
    n_frames = np.ceil(video_clip.fps * video_clip.duration).astype('int')
    print('done')

    # load dlc project config file
    print('loading dlc project config...', end='')
    with open(proj_cfg_file, 'r') as stream:
        proj_config = yaml.safe_load(stream)
    proj_config['video_path'] = None
    dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
    print('done')

    # -------------------
    # extract pose
    # -------------------
    try:
        dlc_cfg['net_type'] = 'resnet_50'
        sess, mu_n, _, scmap, _, inputs = setup_dgp_eval_graph(dlc_cfg, dgp_model_file)
    except:
        dlc_cfg['net_type'] = 'resnet_101'
        sess, mu_n, _, scmap, _, inputs = setup_dgp_eval_graph(dlc_cfg, dgp_model_file)

    print('\n')
    """
    pbar = tqdm(total=n_frames, desc='processing video frames')
    markers = np.zeros((n_frames, dlc_cfg.num_joints, 2))
    likelihoods = np.zeros((n_frames, dlc_cfg.num_joints))
    for i, frame in enumerate(video_clip.iter_frames()):
        # get resnet output
        ff = img_as_ubyte(frame)
        mu_n_batch = sess.run(mu_n, feed_dict={inputs: ff[None, :, :, :]})
        markers[i] = mu_n_batch * dlc_cfg.stride + 0.5 * dlc_cfg.stride
        likelihoods[i] = 0.5
        pbar.update(1)
    """
    # %%
    nj = dlc_cfg.num_joints
    nx, ny = video_clip.size
    nx_out, ny_out = int((nx - dlc_cfg.stride / 2) / dlc_cfg.stride + 1) + 5, int(
        (ny - dlc_cfg.stride / 2) / dlc_cfg.stride + 1) + 5
    # %%
    markers = np.zeros((n_frames + 1, dlc_cfg.num_joints, 2))
    softmaxtensors = np.zeros((n_frames + 1, ny_out, nx_out, nj))
    pbar = tqdm(total=n_frames, desc='processing video frames')
    for ii, frame in enumerate(video_clip.iter_frames()):
        if new_size is not None:
            frame = Image.fromarray(frame)
            scale_x = frame.width / new_size[1]
            scale_y = frame.height / new_size[0]
            # width, height
            frame = frame.resize(size=(new_size[1], new_size[0]))
            frame = np.asarray(frame)
        else:
            scale_x = 1
            scale_y = 1
        ff = img_as_ubyte(frame)

        mu_n_batch, scmap_np = sess.run([mu_n, scmap],
                                        feed_dict={inputs: ff[None, :, :, :]})
        markers[ii] = mu_n_batch[0]
        softmaxtensors[ii, :int(scmap_np[0].shape[0]), :int(scmap_np[0].shape[1])] = \
        scmap_np[0]
        pbar.update(1)

    n_frames = ii + 1
    pbar.close()
    sess.close()
    video_clip.close()
    markers = markers[:n_frames]
    softmaxtensors = softmaxtensors[:n_frames]

    softmaxtensors = softmaxtensors[:, :int(scmap_np[0].shape[0]),
                     :int(scmap_np[0].shape[1])]
    # %%
    xr = markers[:, :, 1] * dlc_cfg.stride + 0.5 * dlc_cfg.stride  # T x nj
    yr = markers[:, :, 0] * dlc_cfg.stride + 0.5 * dlc_cfg.stride
    # %%
    # true xr
    xr *= scale_x
    yr *= scale_y
    print('Finished collecting markers')
    # %%
    print('Calculate likelihoods')
    sigmoid_pred_np = np.exp(softmaxtensors) / (np.exp(softmaxtensors) + 1)
    mu_likelihoods = np.zeros((n_frames, nj, 2)).astype('int')
    likelihoods = np.zeros((n_frames, nj))
    offset_mu_jj = 0
    for ff_idx in range(n_frames):
        for jj_idx in range(nj):
            mu_jj = markers[ff_idx, jj_idx]
            ends_floor = np.floor(mu_jj).astype('int') - offset_mu_jj
            ends_ceil = np.ceil(mu_jj).astype('int') + 1 + offset_mu_jj
            sigmoid_pred_np_jj = sigmoid_pred_np[ff_idx, :, :, jj_idx]
            spred_centered = sigmoid_pred_np_jj[ends_floor[0]:ends_ceil[0],
                             ends_floor[1]:ends_ceil[1]]
            mu_likelihoods[ff_idx, jj_idx] = np.unravel_index(np.argmax(spred_centered),
                spred_centered.shape)
            mu_likelihoods[ff_idx, jj_idx] += [ends_floor[0], ends_floor[1]]
            likelihoods[ff_idx, jj_idx] = sigmoid_pred_np_jj[
                int(mu_likelihoods[ff_idx, jj_idx][0]), int(
                    mu_likelihoods[ff_idx, jj_idx][1])]
    print('Storing data')

    # -------------------
    # save labels
    # -------------------
    labels = {'x': xr, 'y': yr, 'likelihoods': likelihoods}

    # convert to DLC-like csv/hdf5
    if save_pose:
        if not Path(save_file).parent.exists():
            os.makedirs(os.path.dirname(save_file))
        export_pose_like_dlc(labels, os.path.basename(dgp_model_file),
                             dlc_cfg.all_joints_names, save_file)
    return labels


def estimate_pose0(proj_cfg_file, dgp_model_file, video_file, output_dir, shuffle=1,
        save_pose=True, save_str=''):
    """Estimate pose on an arbitrary video.

    Parameters
    ----------
    proj_cfg_file : str, optional
        dlc project config file (.yaml) (if `label_dir` is None)
    dgp_model_file : str, optional
        dgp model weights; .ckpt if fitting full resnet, .npy if only fitting final conv layer
    video_file : str
        video to label
    output_dir : str
        output directory to store labeled video
    shuffle : int, optional
        dlc shuffle number
    save_pose : bool, optional
        True to save out pose in csv/hdf5 file
    save_str : str, optional
        additional string to append to labeled video file name

    Returns
    -------
    dict

    """
    from deepgraphpose.utils_model import get_train_config

    f = os.path.basename(video_file).rsplit('.', 1)
    save_file = join(output_dir, f[0] + '_labeled%s' % save_str)
    if os.path.exists(save_file + '.csv'):
        print('labels already exist! video at %s will not be processed' % video_file)
        return save_file + '.csv'

    # -------------------
    # loading
    # -------------------
    # load video
    print('initializing video clip...', end='')
    video_clip = VideoFileClip(str(video_file))
    n_frames = np.ceil(video_clip.fps * video_clip.duration).astype('int')
    print('done')

    # load dlc project config file
    print('loading dlc project config...', end='')
    with open(proj_cfg_file, 'r') as stream:
        proj_config = yaml.safe_load(stream)
    proj_config['video_path'] = None
    dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
    print('done')

    # -------------------
    # extract pose
    # -------------------
    try:
        dlc_cfg.net_type = 'resnet_50'
        sess, mu_n, _, _, _, inputs = setup_dgp_eval_graph(dlc_cfg, dgp_model_file)
    except:
        dlc_cfg.net_type = 'resnet_101'
        sess, mu_n, _, _, _, inputs = setup_dgp_eval_graph(dlc_cfg, dgp_model_file)

    print('\n')
    pbar = tqdm(total=n_frames, desc='processing video frames')
    markers = np.zeros((n_frames, dlc_cfg.num_joints, 2))
    likelihoods = np.zeros((n_frames, dlc_cfg.num_joints))
    for i, frame in enumerate(video_clip.iter_frames()):
        # get resnet output
        ff = img_as_ubyte(frame)
        mu_n_batch = sess.run(mu_n, feed_dict={inputs: ff[None, :, :, :]})
        markers[i] = mu_n_batch * dlc_cfg.stride + 0.5 * dlc_cfg.stride
        likelihoods[i] = 0.5

        pbar.update(1)

    pbar.close()
    sess.close()
    video_clip.close()

    # -------------------
    # save labels
    # -------------------
    labels = {'x': markers[:, :, 1], 'y': markers[:, :, 0], 'likelihoods': likelihoods}

    # convert to DLC-like csv/hdf5
    if save_pose:
        if not Path(save_file).parent.exists():
            os.makedirs(os.path.dirname(save_file))
        export_pose_like_dlc(labels, os.path.basename(dgp_model_file),
            dlc_cfg.all_joints_names, save_file)

    return labels


def export_pose_like_dlc(labels, scorer, joints_names, save_file):
    """Adapted from deeplabcut.pose_estimation_tensorflow.predict_videos.analyze_videos."""

    import pandas as pd

    n_frames, n_labels = labels['x'].shape

    PredictedData = np.empty((n_frames, 3 * n_labels), dtype=labels['x'].dtype)
    PredictedData[:, 0::3] = labels['x']
    PredictedData[:, 1::3] = labels['y']
    PredictedData[:, 2::3] = labels['likelihoods']

    pdindex = pd.MultiIndex.from_product(
        [[scorer], joints_names, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    imagenames = np.arange(n_frames)

    DataMachine = pd.DataFrame(PredictedData, columns=pdindex, index=imagenames)

    # save as both hdf5 and csv
    DataMachine.to_hdf(save_file + '.h5', 'df_with_missing', format='table', mode='w')
    DataMachine.to_csv(save_file + '.csv')

    return


def load_pose_from_dlc_to_dict(filename):
    from numpy import genfromtxt
    dlc = genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
    dlc = dlc[3:, 1:].astype('float')  # get rid of headers, etc.
    labels = {'x': dlc[:, 0::3], 'y': dlc[:, 1::3], 'likelihoods': dlc[:, 2::3]}
    return labels


def evaluate_dgp(proj_cfg_file, dgp_model_file, shuffle=1, loc_ref=None,
                 loc_ref_calc='dlc'):
    """Evaluate model by computing RMSE on train/test data across all joints.

    Parameters
    ----------
    proj_cfg_file : str
        dlc project config file (.yaml)
    dgp_model_file : str
        dgp model weights; .ckpt if fitting full resnet, .npy if only fitting final conv layer
    shuffle : int
        dlc shuffle number
    loc_ref : bool
        evaluate with or without location refinement
    loc_ref_calc : str
        method used to combine location refinement with part score map; 'dlc' to use deeplabcut
        functions, 'dgp' to use custom dgp function

    Returns
    -------
    pandas dataframe
        RMSE for each frame/joint, including all train/test data

    """

    import pandas as pd
    from deeplabcut.utils.auxfun_videos import imread
    from deeplabcut.pose_estimation_tensorflow.evaluate import pairwisedistances
    from deeplabcut.pose_estimation_tensorflow.nnet import predict
    from deeplabcut.utils import auxiliaryfunctions
    from deepgraphpose.utils_model import get_train_config

    # -------------------
    # loading
    # -------------------

    # load dlc project config file
    print('loading dlc project config...', end='')
    with open(proj_cfg_file, 'r') as stream:
        proj_config = yaml.safe_load(stream)
    proj_config['video_path'] = None
    dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
    print('done')

    # specify location refinement
    # default to dlc config file
    loc_ref = dlc_cfg.location_refinement if loc_ref is None else loc_ref
    # no location refinement if only training final layer
    if not loc_ref:
        dlc_cfg.location_refinement = False

    # -------------------
    # extract pose
    # -------------------

    try:
        dlc_cfg.net_type = 'resnet_50'
        sess, mu_n, softmax_tensor, scmap_tf, locref_tf, inputs = setup_dgp_eval_graph(
            dlc_cfg, dgp_model_file, loc_ref=loc_ref)
    except:
        dlc_cfg.net_type = 'resnet_101'
        sess, mu_n, softmax_tensor, scmap_tf, locref_tf, inputs = setup_dgp_eval_graph(
            dlc_cfg, dgp_model_file, loc_ref=loc_ref)

    # load human annotated data, taken from
    # deeplabcut.pose_estimation_tensorflow.evaluate.evaluate_network
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(proj_config)
    DLCscorer = 'DGP'
    Data = pd.read_hdf(join(proj_config["project_path"], str(trainingsetfolder),
        'CollectedData_' + proj_config["scorer"] + '.h5'), 'df_with_missing')
    # get list of body parts to evaluate network for
    comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        proj_config, 'all')

    datafn, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,
        proj_config['TrainingFraction'][0], shuffle, proj_config)

    # Load meta data
    data, trainIndices, testIndices, trainFraction = auxiliaryfunctions.LoadMetadata(
        join(proj_config['project_path'], metadatafn))

    nj = len(dlc_cfg['all_joints_names'])
    Numimages = len(Data.index)
    PredicteData = np.ones((Numimages, 3 * nj))
    print("Analyzing data...")
    for imageindex, imagename in tqdm(enumerate(Data.index)):
        image = imread(join(proj_config['project_path'], imagename), mode='RGB')

        if loc_ref:
            if loc_ref_calc.lower() == 'dlc':
                outputs_np = sess.run([scmap_tf, locref_tf],
                    feed_dict={inputs: image[None, :, :, :]})
                scmap, locref = predict.extract_cnn_output(outputs_np, dlc_cfg)
                pose = predict.argmax_pose_predict(scmap, locref, dlc_cfg.stride)

            else:
                [lr, st] = sess.run([locref_tf, softmax_tensor],
                    feed_dict={inputs: image[None, :, :, :]})
                locref = np.squeeze(lr)
                shape = locref.shape
                locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
                locref *= dlc_cfg.locref_stdev

                nx = shape[0]
                ny = shape[1]
                xg, yg = np.meshgrid(np.linspace(0, nx - 1, nx),
                                     np.linspace(0, ny - 1, ny))
                alpha = np.array([xg, yg]).swapaxes(1, 2)  # 2 x nx_out x ny_out

                pose_hard_st = []
                pose_hard_st1 = []
                for joint_idx in range(dlc_cfg.num_joints):
                    st_j = np.expand_dims(st[0, :, :, joint_idx], 0)
                    lr_j = np.transpose(locref[:, :, joint_idx, :], [2, 0, 1])
                    pose_f8 = alpha.astype(
                        'float') * dlc_cfg.stride + 0.5 * dlc_cfg.stride + lr_j
                    spatial_soft_argmax = np.sum(np.sum(st_j * alpha, 1),
                        1) * dlc_cfg.stride + 0.5 * dlc_cfg.stride
                    spatial_soft_argmax_offset = np.sum(np.sum(st_j * pose_f8, 1), 1)

                    offset = np.sum(np.sum(st_j * lr_j, 1), 1)
                    spatial_soft_argmax_offset1 = spatial_soft_argmax + offset

                    pose_hard_st.append(np.hstack((spatial_soft_argmax_offset[::-1])))
                    pose_hard_st1.append(np.hstack((spatial_soft_argmax_offset1[::-1])))

                pose_hard_st = np.hstack((np.array(pose_hard_st), np.ones((nj, 1))))
                pose_hard_st1 = np.hstack((np.array(pose_hard_st1), np.ones((nj, 1))))

                pose = np.array(pose_hard_st1)

        else:
            mu_n_batch = sess.run(mu_n, feed_dict={inputs: image[None, :, :, :]})
            pose = mu_n_batch * dlc_cfg.stride + 0.5 * dlc_cfg.stride
            pose = np.hstack([pose[0, :, ::-1], np.ones((nj, 1))])

        PredicteData[imageindex, :] = pose.flatten()

    sess.close()  # closes the current tf session

    index = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    DataMachine = pd.DataFrame(PredicteData, columns=index, index=Data.index.values)
    # DataMachine.to_hdf(resultsfilename, 'df_with_missing', format='table', mode='w')

    DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T

    RMSE, _ = pairwisedistances(DataCombined, proj_config["scorer"], DLCscorer,
        proj_config["pcutoff"], comparisonbodyparts)
    testerror = np.nanmean(RMSE.iloc[testIndices].values.flatten())
    trainerror = np.nanmean(RMSE.iloc[trainIndices].values.flatten())

    print("Train error:", np.round(trainerror, 2), " pixels")
    print("Test error:", np.round(testerror, 2), " pixels")

    return RMSE


def plot_dgp(video_file, output_dir='', label_dir=None, proj_cfg_file=None,
        dgp_model_file=None, shuffle=1, dotsize=3, colormap='jet', save_str='',
        mask_threshold=0.1, new_size=None):
    """Produce video overlaid with predicted markers.

    Parameters
    ----------
    video_file : str
        video to label
    output_dir : str
        output directory to store labeled video
    label_dir : str, optional
        location of labels for each frame of video
    proj_cfg_file : str, optional
        dlc project config file (.yaml) (if `label_dir` is None)
    dgp_model_file : str, optional
        dgp model weights; .ckpt if fitting full resnet, .npy if only fitting final conv layer
        (if `label_dir` is None)
    shuffle : int
        dlc shuffle number
    dotsize : int
        size of marker dot on labeled video
    colormap : str
        matplotlib color map for markers
    save_str : str, optional
        additional string to append to video file name
    mask_threshold : float
        when plotting videos, masks all markers with likelihoods lower than specified threshold.
    Returns
    -------
    str
        location of saved labeled video

    """
    f = os.path.basename(video_file).rsplit('.', 1)
    save_file = join(output_dir, f[0] + '_labeled%s.mp4' % save_str)

    if label_dir is None:
        label_dir = output_dir
    label_file = join(label_dir, f[0] + '_labeled%s.csv' % save_str)

    # export labels if necessary
    if not os.path.exists(label_file):
        estimate_pose(proj_cfg_file, dgp_model_file, video_file, label_dir,
            shuffle=shuffle, save_str=save_str, new_size=new_size)

    # load labels
    labels = load_pose_from_dlc_to_dict(label_file)

    video_clip = VideoFileClip(str(video_file))

    mask_array = labels['likelihoods'].T > mask_threshold
    # make movie
    create_annotated_movie(video_clip, labels['x'].T, labels['y'].T,
        mask_array=mask_array, filename=save_file, dotsize=dotsize, colormap=colormap)

    video_clip.close()

    return save_file
