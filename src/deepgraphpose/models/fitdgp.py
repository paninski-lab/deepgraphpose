"""Main fitting function for DGP.

NOTES:
    - schedule 0 [# visible frames]: use all visible, no hidden on each iteration
    - schedule 1 [# visible frames]: use batch of size 2*ns + 1, centered around a visible frame
      (may include multiple visible frames)
    - schedule 2 [# visible frames]: use batch of size 2 * (2*ns + 1), centered around two visible frames
      (concatenate windows)
    - repeat once (total of two sweeps through schedules)

"""

import importlib
import logging
import time
import os
from os import listdir
from os.path import isfile, join, split
from pathlib import Path
from random import randint

import numpy as np
import tensorflow as tf
import tf_slim as slim

import deeplabcut
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.datasets import (
    PoseDatasetFactory,
    ImgaugPoseDataset )
from deeplabcut.pose_estimation_tensorflow.nnets import (
    PoseNetFactory,
    PoseResnet
)
from deeplabcut.pose_estimation_tensorflow.core.train import LearningRate, get_batch_spec, \
    setup_preloading, start_preloading, get_optimizer
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.nnets.layers import prediction_layer

from deepgraphpose.dataset import MultiDataset, coord2map
from deepgraphpose.models.fitdgp_util import gen_batch, argmax_2d_from_cm, combine_all_marker, build_aug, data_aug, learn_wt


importlib.reload(logging)
logging.shutdown()

vers = tf.__version__.split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf


# %%
def fit_dlc(
        snapshot, dlcpath, shuffle=1, step=0, saveiters=1000, displayiters=100, maxiters=50000,
trainingsetindex=0):
    """Run the original DLC code.
    Parameters
    ----------
    snapshot : the snapshot to start from, str
        'resnet_v1_50/101.ckpt' or the saved snapshot from a previous run DLC
    dlcpath : the path for the DLC project, str
    shuffle : shuffle index, int, optional
        default value is 1
    step : the index of training step, int, optional
        default value is 0
    saveiters : number of iterations to save snapshot, int, optional
        default value is 1000
    displayiters : number of iterations to display, int, optional
        default value is 100
    maxiters : number of total iterations, int, optional
        default value is 200000

    Returns
    -------
    None

    """

    # Load config.yaml and pose_cfg.yaml from dlcpath.
    dlc_base_path = Path(dlcpath)
    config_path = dlc_base_path / 'config.yaml'
    print('config_path', config_path)
    cfg = auxiliaryfunctions.read_config(config_path)
    modelfoldername = auxiliaryfunctions.GetModelFolder(
        cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)
    pose_config_yaml = Path(
        os.path.join(cfg["project_path"], str(modelfoldername), "train",
                     "pose_cfg.yaml"))

    # Change dlc_cfg as we want, here we set the default values
    # TODO: it would be better to set the default values when making config.yaml and pose_cfg.yaml, double check the default setting for these two yamls.
    dlc_cfg = load_config(pose_config_yaml)
    print(pose_config_yaml)
    print(dlc_cfg)
    dlc_cfg['crop'] = False
    dlc_cfg['cropratio'] = 0.4
    dlc_cfg['global_scale'] = 0.8
    dlc_cfg['multi_step'] = [[0.001, 10000], [0.005, 430000], [0.002, 730000],
                          [0.001, 1030000]]
    if "snapshot" in snapshot:
        train_path = dlc_base_path / modelfoldername / 'train'
        init_weights = str(train_path / snapshot)
    else:
        parent_path = Path(os.path.dirname(deeplabcut.__file__))
        snapshot = dlc_cfg['net_type'].split('_')[0] + '_v1_' + dlc_cfg['net_type'].split('_')[1] + '.ckpt'
        init_weights = str(
            parent_path /
            join('pose_estimation_tensorflow', 'models', 'pretrained', snapshot))

    dlc_cfg['init_weights'] = init_weights
    dlc_cfg['pos_dist_thresh'] = 8
    dlc_cfg['output_stride'] = 16

    # skip this DLC step if it's already done.
    model_name = dlc_cfg['snapshot_prefix'] + '-step0-final--0.index'
    if os.path.isfile(model_name):
        print(model_name, '  exists! The original DLC has already been run.', flush=True)
        return None

    # Build loss function
    TF.compat.v1.reset_default_graph()
    dlc_cfg['dataset_type'] = 'deterministic'
    dataset = PoseDatasetFactory.create(dlc_cfg)
    batch_spec = get_batch_spec(dlc_cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = PoseNetFactory.create(dlc_cfg).train(batch)
    total_loss = losses["total_loss"]
    print(losses.items())
    for k, t in losses.items():
        TF.compat.v1.summary.scalar(k, t)
    merged_summaries = TF.compat.v1.summary.merge_all()

    if "snapshot" in Path(dlc_cfg['init_weights']).stem:
        print("Loading already trained DLC with backbone:",
              dlc_cfg['net_type'],
              flush=True)
        variables_to_restore = slim.get_variables_to_restore()
    else:
        print("Loading ImageNet-pretrained", dlc_cfg['net_type'], flush=True)
        # loading backbone from ResNet, MobileNet etc.
        if "resnet" in dlc_cfg['net_type']:
            variables_to_restore = slim.get_variables_to_restore(
                include=["resnet_v1"])
        elif "mobilenet" in dlc_cfg['net_type']:
            variables_to_restore = slim.get_variables_to_restore(
                include=["MobilenetV2"])
        else:
            print("Wait for DLC 2.3.")

    restorer = TF.compat.v1.train.Saver(variables_to_restore)
    saver = TF.compat.v1.train.Saver(
        max_to_keep=5
    )  # selects how many snapshots are stored,
    # see https://github.com/AlexEMG/DeepLabCut/issues/8#issuecomment-387404835

    allow_growth = True
    if allow_growth:
        config = TF.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TF.compat.v1.Session(config=config)
    else:
        sess = TF.compate.v1.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = TF.compat.v1.summary.FileWriter(dlc_cfg['log_dir'], sess.graph)
    learning_rate, train_op, tstep = get_optimizer(total_loss, dlc_cfg)

    sess.run(TF.compat.v1.global_variables_initializer())
    sess.run(TF.compat.v1.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, dlc_cfg['init_weights'])

    # Run iterations
    if displayiters is None:
        display_iters = max(1, int(dlc_cfg['display_iters']))
    else:
        display_iters = max(1, int(displayiters))
        print("Display_iters overwritten as", display_iters, flush=True)

    if saveiters is None:
        save_iters = max(1, int(dlc_cfg['save_iters']))
    else:
        save_iters = max(1, int(saveiters))
        print("Save_iters overwritten as", save_iters, flush=True)

    if maxiters is None:
        max_iter = int(dlc_cfg['multi_step'][-1][1])
    else:
        max_iter = min(int(dlc_cfg['multi_step'][-1][1]), int(maxiters))
        print("Max_iters overwritten as", max_iter, flush=True)

    lr_gen = LearningRate(dlc_cfg)  # learning rate
    stats_path = Path(pose_config_yaml).with_name("learning_stats.csv")
    lrf = open(str(stats_path), "w")
    cumloss, partloss, locrefloss = 0.0, 0.0, 0.0

    print("Training parameters:", flush=True)
    print(dlc_cfg, flush=True)
    print("Starting training....", flush=True)
    for it in range(max_iter + 1):
        current_lr = lr_gen.get_lr(it)
        [_, alllosses, loss_val, summary] = sess.run(
            [train_op, losses, total_loss, merged_summaries],
            feed_dict={learning_rate: current_lr},
        )

        # collect loss
        partloss += alllosses["part_loss"]  # scoremap loss
        if dlc_cfg['location_refinement']:
            locrefloss += alllosses["locref_loss"]
        cumloss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0 and it > 0:
            logging.info(
                "iteration: {} loss: {} scmap loss: {} locref loss: {} lr: {}".
                    format(
                    it,
                    "total loss {0:.4f}".format(cumloss / display_iters),
                    "scoremap loss {0:.4f}".format(partloss / display_iters),
                    "learning rate {0:.4f}".format(locrefloss / display_iters),
                    current_lr,
                ))

            lrf.write(
                "iteration: {}, loss: {}, scmap loss: {}, locref loss: {}, lr: {}\n"
                    .format(
                    it,
                    "total loss {0:.4f}".format(cumloss / display_iters),
                    "scoremap loss {0:.4f}".format(partloss / display_iters),
                    "learning rate {0:.4f}".format(locrefloss / display_iters),
                    current_lr,
                ))

            lrf.flush()

        # Save snapshot
        if (it % save_iters == 0 and it != 0) or it == max_iter:
            model_name = dlc_cfg['snapshot_prefix'] + '-step' + str(
                step)
            saver.save(sess, model_name, global_step=it)
            if it == max_iter:
                model_name = dlc_cfg['snapshot_prefix'] + '-step' + str(
                    step) + '-final-'
                saver.save(sess, model_name, global_step=0)

    print('Finish training {} iterations\n'.format(it), flush=True)
    # Close everything
    lrf.close()
    sess.close()
    coord.request_stop()
    coord.join([thread])

    return None


def fit_dgp_labeledonly(
        snapshot, dlcpath, shuffle=1, step=1, saveiters=1000, displayiters=5, maxiters=50000,
        ns=10, nc=2048, n_max_frames=2000, aug=True,trainingsetindex=0):
    """Run the DGP with labeled frames only.
    Parameters
    ----------
    snapshot : the saved snapshot, str
        snapshot = 'snapshot-step0-final--0' or the specified DLC snapshot
    dlcpath : the path for the DLC project, str
    shuffle : shuffle index, int, optional
        default value is 1
    step : the index of training step, int, optional
        default value is 1
    saveiters : number of iterations to save snapshot, int, optional
        default value is 1000
    displayiters : number of iterations to display, int, optional
        default value is 5
    maxiters : number of total iterations, int, optional
        default value is 50000
    ns : frames on either side of visible frames, int, optional
        default value is 10
    nc : resnet channels, int, optional
        default value is 2048
    n_max_frames : total number of frames (visible + hidden + windows) to train on, int, optional
        default value is 2000

    Returns
    -------
    None

    """

    # Load config.yaml and set up paths.
    dlc_base_path = Path(dlcpath)
    config_path = dlc_base_path / 'config.yaml'
    print('config_path', config_path)
    cfg = auxiliaryfunctions.read_config(config_path)
    modelfoldername = auxiliaryfunctions.GetModelFolder(
        cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)

    train_path = dlc_base_path / modelfoldername / 'train'
    init_weights = str(train_path / snapshot)
    video_path = str(dlc_base_path / 'videos_dgp')
    if not (os.path.exists(video_path)):
        print(video_path + " does not exist!")
        video_sets = list(cfg['video_sets'])
    else:
        video_sets = [
            join(video_path, f) for f in listdir(video_path)
            if isfile(join(video_path, f)) and (
                    f.find('avi') > 0 or f.find('mp4') > 0 or f.find('mov') > 0 or f.find('mkv') > 0)
        ]
    print('video_sets: ', video_sets)

    # structure info
    bodyparts = cfg['bodyparts']
    skeleton = cfg['skeleton']
    if skeleton is None:
        skeleton = []
    S0 = np.zeros((len(skeleton), len(bodyparts)))
    for s in range(len(skeleton)):
        sk = skeleton[s]
        ski = bodyparts.index(sk[0])
        skj = bodyparts.index(sk[1])
        S0[s, ski] = 1
        S0[s, skj] = -1

    # batching info
    batch_dict = dict(
        ns_jump=None,  # obsolete
        step=1,  # obsolete
        ns=ns,  # frames on either side of visible frames
        nc=nc,  # resnet channels
        n_max_frames=n_max_frames  # total number of frames (visible + hidden + windows) to train on
    )

    # %%
    # ------------------------------------------------------------------------------------
    # initialize data
    # ------------------------------------------------------------------------------------
    # initialize data batcher
    data_batcher = MultiDataset(config_yaml=config_path,
                                video_sets=video_sets,
                                shuffle=shuffle,
                                S0=S0)
    dgp_cfg = data_batcher.dlc_config
    dgp_cfg['ws'] = 0  # the spatial clique parameter
    dgp_cfg['ws_max'] = 1.2  # the multiplier for the upper bound of spatial distance
    dgp_cfg['wt'] = 0  # the temporal clique parameter
    dgp_cfg['wt_max'] = 0  # the upper bound of temporal distance
    dgp_cfg['wn_visible'] = 1  # the network clique parameter for visible frames
    dgp_cfg['wn_hidden'] = 0  # the network clique parameter for hidden frames
    dgp_cfg['gamma'] = 1  # the multiplier for the softmax confidence map
    dgp_cfg['gauss_len'] = 1  # the length scale for the Gaussian kernel convolving the softmax confidence map
    dgp_cfg['lengthscale'] = 1  # the length scale for the Gaussian target map
    dgp_cfg['max_to_keep'] = 5  # max number of snapshots to keep
    dgp_cfg['batch_size'] = 1  # batch size
    dgp_cfg['n_times_all_frames'] = 100  # the number of times each selected frames is iterated over
    dgp_cfg['lr'] = 0.005  # learning rate
    # dgp_cfg['net_type'] = 'resnet_50'
    dgp_cfg['gm2'] = 0  # scale target by confidence level
    dgp_cfg['gm3'] = 0  # scale hidden loss by confidence level
    dgp_cfg['aug'] = aug  # data augmentation

    # skip this DGP with labeled frames only step if it's already done.
    model_name = dgp_cfg['snapshot_prefix'] + '-step1-final-0.index'
    if os.path.isfile(model_name):
        print(model_name, '  exists! DGP with labeled frames has already been run.', flush=True)
        return None

    # create training data - find good hidden frames
    snapshot = 0
    data_batcher.create_batches_from_resnet_output(snapshot, **batch_dict)

    # unpack from data batcher
    nj = data_batcher.nj
    n_frames_total = data_batcher.n_frames_total
    n_visible_frames_total = data_batcher.n_visible_frames_total
    n_hidden_frames_total = n_frames_total - n_visible_frames_total
    print('n_hidden_frames_total', n_hidden_frames_total, flush=True)
    print('n_visible_frames_total', n_visible_frames_total, flush=True)
    print('n_frames_total', n_frames_total, flush=True)

    # get all selected frame indices
    visible_frame_total = [d.idxs['pv'] for d in data_batcher.datasets]
    # hidden_frame_total = [d.idxs['ph'] for d in data_batcher.datasets]
    # all_frame_total = [d.idxs['chunk'] for d in data_batcher.datasets]

    # %%
    # ------------------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------------------
    TF.compat.v1.reset_default_graph()
    loss, total_loss, total_loss_visible, placeholders = dgp_loss(data_batcher, dgp_cfg)
    learning_rate = TF.compat.v1.compat.v1.placeholder(tf.float32, shape=[])

    # Restore network parameters for RESNET and COVNET
    variables_to_restore0 = slim.get_variables_to_restore(
        include=['pose/part_pred'])
    variables_to_restore1 = slim.get_variables_to_restore(
        include=['pose/locref_pred'])
    variables_to_restore2 = slim.get_variables_to_restore(include=['resnet'])
    restorer = TF.compat.v1.train.Saver(variables_to_restore0 + variables_to_restore1 +
                              variables_to_restore2)
    saver = TF.compat.v1.train.Saver(max_to_keep=dgp_cfg['max_to_keep'])

    # Set up session
    allow_growth = True
    if allow_growth:
        config = TF.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TF.compat.v1.Session(config=config)
    else:
        sess = TF.compat.v1.Session()

    # Set up optimizer
    all_train_vars = TF.compat.v1.trainable_variables()
    optimizer = TF.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients, variables = zip(
        *optimizer.compute_gradients(total_loss_visible, var_list=all_train_vars))
    gradients, _ = TF.clip_by_global_norm(gradients, 10.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    sess.run(TF.compat.v1.global_variables_initializer())
    sess.run(TF.compat.v1.local_variables_initializer())

    # Restore RESNET var
    print('restoring resnet weights from %s' % init_weights)
    restorer.restore(sess, init_weights)

    # %%
    # ------------------------------------------------------------------------------------
    # Begin training
    # ------------------------------------------------------------------------------------
    nepoch = np.min([int(n_visible_frames_total * dgp_cfg['n_times_all_frames']), maxiters])
    visible_frame_total_dict = []
    for i, v in enumerate(visible_frame_total):
        for vv in v:
            visible_frame_total_dict.extend((i, vv))
    visible_frame_total_dict = np.array(visible_frame_total_dict).reshape(-1, 2)
    batch_ind_all = np.random.randint(0, visible_frame_total_dict.shape[0], size=nepoch)
    save_iters = saveiters
    maxiters = batch_ind_all.shape[0]

    dgp_cfg['dataset_type'] = 'deterministic'
    pdata = PoseDatasetFactory.create(dgp_cfg)
    data_batcher.reset()

    # %%
    print('Begin Training for {} iterations'.format(maxiters))
    if dgp_cfg['aug']:
        pipeline = build_aug(apply_prob=0.8)
    time_start = time.time()

    for it in range(maxiters):

        current_lr = dgp_cfg['lr']

        # get batch index
        visible_batch_ind = visible_frame_total_dict[batch_ind_all[it]]
        dataset_i = visible_batch_ind[0]
        nx_out = data_batcher.datasets[
            dataset_i].nx_out  # x dimension of the output confidence map from the neural network
        ny_out = data_batcher.datasets[
            dataset_i].ny_out  # y dimension of the output confidence map from the neural network

        visible_frame_batch_i = np.array([visible_batch_ind[1]])
        hidden_frame_batch_i = np.array([])

        (visible_frame, hidden_frame, _, all_data_batch, joint_loc, wt_batch_mask,
         all_marker_batch, addn_batch_info), d = \
            data_batcher.next_batch(0, dataset_i, visible_frame_batch_i, hidden_frame_batch_i)
        nt_batch = len(visible_frame) + len(hidden_frame)
        visible_marker, hidden_marker, visible_marker_in_targets = addn_batch_info
        all_frame = np.sort(list(visible_frame) + list(hidden_frame))
        visible_frame_within_batch = [np.where(all_frame == i)[0][0] for i in visible_frame]

        # batch data for placeholders
        if dgp_cfg['wt'] > 0:
            vector_field = learn_wt(all_data_batch)  # vector field from optical flow
        else:
            vector_field = np.zeros((1,1,1))
        wt_batch = np.ones(nt_batch - 1, ) * dgp_cfg['wt']

        # data augmentation for visible frames
        if dgp_cfg['aug'] and dgp_cfg['wt'] == 0 and len(visible_frame_within_batch) > 0:
            all_data_batch, joint_loc = data_aug(all_data_batch, visible_frame_within_batch, joint_loc, pipeline, dgp_cfg)

        locref_targets_batch, locref_mask_batch = coord2map(pdata, joint_loc, nx_out, ny_out, nj)
        if locref_mask_batch.shape[0] != 0:
            locref_targets_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))
            locref_targets_all_batch[
            visible_frame_within_batch, :, :, :] = locref_targets_batch
            locref_mask_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))
            locref_mask_all_batch[visible_frame_within_batch, :, :, :] = locref_mask_batch
        else:
            locref_targets_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))
            locref_mask_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))

        # generate 2d grids
        xg, yg = np.meshgrid(np.linspace(0, nx_out - 1, nx_out),
                             np.linspace(0, ny_out - 1, ny_out))
        alpha = np.array([xg, yg]).swapaxes(1, 2)  # 2 x nx_out x ny_out

        feed_dict = {
            placeholders['inputs']: all_data_batch,
            placeholders['targets']: joint_loc,
            placeholders['locref_map']: locref_targets_all_batch,
            placeholders['locref_mask']: locref_mask_all_batch,
            placeholders['visible_marker_pl']: visible_marker,
            placeholders['hidden_marker_pl']: hidden_marker,
            placeholders['visible_marker_in_targets_pl']: visible_marker_in_targets,
            placeholders['wt_batch_mask_pl']: wt_batch_mask,
            placeholders['vector_field_tf']: vector_field,
            placeholders['nt_batch_pl']: nt_batch,
            placeholders['wt_batch_pl']: wt_batch,
            placeholders['alpha_tf']: alpha,
            learning_rate: current_lr,
        }

        start_time00 = time.time()
        [loss_eval, _] = sess.run([loss, train_op], feed_dict)
        end_time00 = time.time()
        if it % displayiters == 0 and it > 0:
            print('\nIteration {}/{}'.format(it, maxiters))

            print('dataset_i: ', dataset_i, flush=True)
            print('visible_frame_batch_i: ', visible_frame_batch_i, flush=True)
            print('hidden_frame_batch_i: ', hidden_frame_batch_i, flush=True)

            print('\n running time: ', end_time00 - start_time00, flush=True)
            print('\n loss: ', loss_eval, flush=True)

        # Save snapshot
        if (it % save_iters == 0) or (it + 1) == maxiters:
            model_name = dgp_cfg['snapshot_prefix'] + '-step' + str(step)
            saver.save(sess, model_name, global_step=it)
            saver.save(sess, model_name, global_step=0)
            if (it + 1) == maxiters:
                model_name = dgp_cfg['snapshot_prefix'] + '-step' + str(step) + '-final'
                saver.save(sess, model_name, global_step=0)

    time_end = time.time()
    print('Finished training {} iterations\n'.format(it), flush=True)
    print('\n\n TOTAL TIME ELAPSED: ', time_end - time_start)

    return None


def fit_dgp(
        snapshot, dlcpath, batch_size=1, shuffle=1, step=2, saveiters=500, displayiters=10,
        maxiters=30000, ns=10, nc=2048, n_max_frames=2000, gm2=0, gm3=0, nepoch=100, wt=0, aug=True,
        debug='', trainingsetindex=0):
    """Run DGP.
    Parameters
    ----------
    snapshot : the saved snapshot, str
        snapshot = 'snapshot-step0-final--0' or the specified DLC snapshot
    dlcpath : the path for the DLC project, str
    batch_size : batch size, int, optional
        default value is 10
    shuffle : shuffle index, int, optional
        default value is 1
    step : the index of training step, int, optional
        default value is 2
    saveiters : number of iterations to save snapshot, int, optional
        default value is 1000
    displayiters : number of iterations to display, int, optional
        default value is 5
    maxiters : number of total iterations, int, optional
        default value is 50000
    ns : frames on either side of visible frames, int, optional
        default value is 10
    nc : resnet channels, int, optional
        default value is 2048
    n_max_frames : total number of frames (visible + hidden + windows) to train on, int, optional
        default value is 2000

    Returns
    -------
    None

    """

    # Load config.yaml and set up paths.
    dlc_base_path = Path(dlcpath)
    config_path = dlc_base_path / 'config.yaml'
    print('config_path', config_path)
    cfg = auxiliaryfunctions.read_config(config_path)
    modelfoldername = auxiliaryfunctions.GetModelFolder(
        cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)

    train_path = dlc_base_path / modelfoldername / 'train'
    init_weights = str(train_path / snapshot)
    video_path = str(dlc_base_path / 'videos_dgp')
    if not (os.path.exists(video_path)):
        print(video_path + " does not exist!")
        video_sets = list(cfg['video_sets'])
    else:
        video_sets = [
            join(video_path, f) for f in listdir(video_path)
            if isfile(join(video_path, f)) and (
                    f.find('avi') > 0 or f.find('mp4') > 0 or f.find('mov') > 0 or f.find('mkv') > 0)
        ]
    print('video_sets: ', video_sets)

    # structure info
    bodyparts = cfg['bodyparts']
    skeleton = cfg['skeleton']
    if skeleton is None:
        skeleton = []
    S0 = np.zeros((len(skeleton), len(bodyparts)))
    for s in range(len(skeleton)):
        sk = skeleton[s]
        ski = bodyparts.index(sk[0])
        skj = bodyparts.index(sk[1])
        S0[s, ski] = 1
        S0[s, skj] = -1

    # batching info
    batch_dict = dict(
        ns_jump=None,  # obsolete
        step=1,  # obsolete
        ns=ns,  # frames on either side of visible frames
        nc=nc,  # resnet channels
        n_max_frames=n_max_frames  # total number of frames (visible + hidden + windows) to train on
    )

    # %%
    # ------------------------------------------------------------------------------------
    # initialize data
    # ------------------------------------------------------------------------------------
    # initialize data batcher
    data_batcher = MultiDataset(config_yaml=config_path,
                                video_sets=video_sets,
                                shuffle=shuffle,
                                S0=S0)
    dgp_cfg = data_batcher.dlc_config
    dgp_cfg['ws'] = 1000  # the spatial clique parameter
    dgp_cfg['ws_max'] = 1.2  # the multiplier for the upper bound of spatial distance
    dgp_cfg['wt'] = wt  # the temporal clique parameter
    dgp_cfg['wt_max'] = 0  # the upper bound of temporal distance
    dgp_cfg['wn_visible'] = 5  # the network clique parameter for visible frames
    dgp_cfg['wn_hidden'] = 3  # the network clique parameter for hidden frames
    dgp_cfg['gamma'] = 1  # the multiplier for the softmax confidence map
    dgp_cfg['gauss_len'] = 1  # the length scale for the Gaussian kernel convolving the softmax confidence map
    dgp_cfg['lengthscale'] = 1  # the length scale for the Gaussian target map
    dgp_cfg['max_to_keep'] = 5  # max number of snapshots to keep
    dgp_cfg['batch_size'] = batch_size  # batch size
    dgp_cfg['n_times_all_frames'] = nepoch  # the number of times each selected frames is iterated over
    dgp_cfg['lr'] = 0.005  # learning rate
    # dgp_cfg['net_type'] = 'resnet_50'
    dgp_cfg['gm2'] = gm2  # scale target by confidence level
    dgp_cfg['gm3'] = gm3  # scale hidden loss by confidence level
    dgp_cfg['aug'] = aug  # data augmentation

    # skip this DGP step if it's already done.
    model_name = dgp_cfg['snapshot_prefix'] + '-step{}-final-0.index'.format(step)
    if os.path.isfile(model_name):
        print(model_name, '  exists! DGP has already been run.', flush=True)
        return None

    # create training data - find good hidden frames
    snapshot = 0
    data_batcher.create_batches_from_resnet_output(snapshot, **batch_dict)

    # unpack from data batcher
    nj = data_batcher.nj
    n_frames_total = data_batcher.n_frames_total
    n_visible_frames_total = data_batcher.n_visible_frames_total
    n_hidden_frames_total = n_frames_total - n_visible_frames_total
    print('n_hidden_frames_total', n_hidden_frames_total, flush=True)
    print('n_visible_frames_total', n_visible_frames_total, flush=True)
    print('n_frames_total', n_frames_total, flush=True)

    # get all selected frame indices
    visible_frame_total = [d.idxs['pv'] for d in data_batcher.datasets]
    hidden_frame_total = [d.idxs['ph'] for d in data_batcher.datasets]
    all_frame_total = [d.idxs['chunk'] for d in data_batcher.datasets]

    # %%
    # ------------------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------------------
    TF.compat.v1.reset_default_graph()
    loss, total_loss, total_loss_visible, placeholders = dgp_loss(data_batcher, dgp_cfg)
    learning_rate = TF.compat.v1.placeholder(tf.float32, shape=[])

    # Restore network parameters for RESNET and COVNET
    variables_to_restore0 = slim.get_variables_to_restore(
        include=['pose/part_pred'])
    variables_to_restore1 = slim.get_variables_to_restore(
        include=['pose/locref_pred'])
    variables_to_restore2 = slim.get_variables_to_restore(include=['resnet'])
    restorer = TF.compat.v1.train.Saver(variables_to_restore0 + variables_to_restore1 +
                              variables_to_restore2)
    saver = TF.compat.v1.train.Saver(max_to_keep=dgp_cfg['max_to_keep'])

    # Set up session
    allow_growth = True
    if allow_growth:
        config = TF.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TF.compat.v1.Session(config=config)
    else:
        sess = TF.compat.v1.Session()

    # Set up optimizer
    all_train_vars = TF.compat.v1.trainable_variables()
    optimizer = TF.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients, variables = zip(
        *optimizer.compute_gradients(total_loss, var_list=all_train_vars))
    gradients, _ = TF.clip_by_global_norm(gradients, 10.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    sess.run(TF.compat.v1.global_variables_initializer())
    sess.run(TF.compat.v1.local_variables_initializer())

    # Restore RESNET var
    print('restoring resnet weights from %s' % init_weights)
    restorer.restore(sess, init_weights)

    # %%
    # ------------------------------------------------------------------------------------
    # Begin training
    # ------------------------------------------------------------------------------------
    batch_ind_all = gen_batch(visible_frame_total, hidden_frame_total, all_frame_total, dgp_cfg, maxiters)
    save_iters = np.int(saveiters / dgp_cfg['batch_size'])
    maxiters = len(batch_ind_all)


    dgp_cfg['dataset_type'] = 'deterministic'
    pdata = PoseDatasetFactory.create(dgp_cfg)
    data_batcher.reset()

    # %%
    print('Begin Training for {} iterations'.format(maxiters))
    if dgp_cfg['aug']:
        pipeline = build_aug(apply_prob=0.8)

    time_start = time.time()
    for it in range(maxiters):

        current_lr = dgp_cfg['lr']

        # get batch index
        batch_ind = batch_ind_all[it]
        dataset_i = batch_ind[-1]
        nx_out = data_batcher.datasets[
            dataset_i].nx_out  # x dimension of the output confidence map from the neural network
        ny_out = data_batcher.datasets[
            dataset_i].ny_out  # y dimension of the output confidence map from the neural network

        all_frame_batch = batch_ind[:-1]
        visible_frame_i = visible_frame_total[dataset_i]
        all_frame_i = list(all_frame_total[dataset_i]) + list(hidden_frame_total[dataset_i])

        visible_frame_batch_i = np.sort(np.array([i for i in all_frame_batch if i in visible_frame_i]))
        if len(visible_frame_batch_i) == 0 and len(visible_frame_i) > 0:
            random_int = randint(0, len(visible_frame_total[dataset_i]) - 1)
            visible_frame_batch_i = np.array([visible_frame_total[dataset_i][random_int]])
        hidden_frame_batch_i = np.sort(
            np.array([i for i in all_frame_batch if (i in all_frame_i) and (i not in visible_frame_i)]))

        (visible_frame, hidden_frame, _, all_data_batch, joint_loc, wt_batch_mask,
         all_marker_batch, addn_batch_info), d = \
            data_batcher.next_batch(0, dataset_i, visible_frame_batch_i, hidden_frame_batch_i)
        nt_batch = len(visible_frame) + len(hidden_frame)
        visible_marker, hidden_marker, visible_marker_in_targets = addn_batch_info
        all_frame = np.sort(list(visible_frame) + list(hidden_frame))
        visible_frame_within_batch = [np.where(all_frame == i)[0][0] for i in visible_frame]

        # batch data for placeholders
        if dgp_cfg['wt'] > 0:
            vector_field = learn_wt(all_data_batch)  # vector field from optical flow
        else:
            vector_field = np.zeros((1,1,1))
        wt_batch = np.ones(nt_batch - 1, ) * dgp_cfg['wt']

        # data augmentation for visible frames
        if dgp_cfg['aug'] and dgp_cfg['wt'] == 0 and len(visible_frame_within_batch) > 0:
            all_data_batch, joint_loc = data_aug(all_data_batch, visible_frame_within_batch, joint_loc, pipeline, dgp_cfg)

        locref_targets_batch, locref_mask_batch = coord2map(pdata, joint_loc, nx_out, ny_out, nj)
        if locref_mask_batch.shape[0] != 0:
            locref_targets_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))
            locref_targets_all_batch[
            visible_frame_within_batch, :, :, :] = locref_targets_batch
            locref_mask_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))
            locref_mask_all_batch[visible_frame_within_batch, :, :, :] = locref_mask_batch
        else:
            locref_targets_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))
            locref_mask_all_batch = np.zeros(
                (len(all_frame), nx_out, ny_out, nj * 2))

        # generate 2d grids
        xg, yg = np.meshgrid(np.linspace(0, nx_out - 1, nx_out),
                             np.linspace(0, ny_out - 1, ny_out))
        alpha = np.array([xg, yg]).swapaxes(1, 2)  # 2 x nx_out x ny_out

        feed_dict = {
            placeholders['inputs']: all_data_batch,
            placeholders['targets']: joint_loc,
            placeholders['locref_map']: locref_targets_all_batch,
            placeholders['locref_mask']: locref_mask_all_batch,
            placeholders['visible_marker_pl']: visible_marker,
            placeholders['hidden_marker_pl']: hidden_marker,
            placeholders['visible_marker_in_targets_pl']: visible_marker_in_targets,
            placeholders['wt_batch_mask_pl']: wt_batch_mask,
            placeholders['vector_field_tf']: vector_field,
            placeholders['nt_batch_pl']: nt_batch,
            placeholders['wt_batch_pl']: wt_batch,
            placeholders['alpha_tf']: alpha,
            learning_rate: current_lr,
        }

        start_time00 = time.time()
        [loss_eval, _] = sess.run([loss, train_op], feed_dict)
        end_time00 = time.time()
        if it % displayiters == 0 and it > 0:
            print('\nIteration {}/{}'.format(it, maxiters))

            print('dataset_i: ', dataset_i, flush=True)
            print('visible_frame_batch_i: ', visible_frame_batch_i, flush=True)
            print('hidden_frame_batch_i: ', hidden_frame_batch_i, flush=True)

            print('\n running time: ', end_time00 - start_time00, flush=True)
            print('\n loss: ', loss_eval, flush=True)

        # Save snapshot
        if (it % save_iters == 0) or (it + 1) == maxiters:
            model_name = dgp_cfg['snapshot_prefix'] + '-step' + str(step)
            #print('Storing model {}'.format(model_name))
            saver.save(sess, model_name, global_step=it)
            saver.save(sess, model_name, global_step=0)
            if (it + 1) == maxiters:
                model_name = dgp_cfg['snapshot_prefix'] + '-step' + str(step) +'-final'
                #print('Storing model {}'.format(model_name))
                saver.save(sess, model_name, global_step=0)

    time_end = time.time()
    print('Finished {} iterations\n'.format(it), flush=True)
    print('\n\n TOTAL TIME ELAPSED: ', time_end - time_start)

    return None


def dgp_loss(data_batcher, dgp_cfg):
    """Construct the loss for DGP.
    Parameters
    ----------
    data_batcher : dict for data info
    dgp_cfg : dict for configuration info

    Returns
    -------
    loss : dict for losses
    total_loss : total loss for DGP
    total_loss_visible : total loss for DGP with labeled frames only
    placeholders : dict for placeholders

    """

    # Unpack parameters from data batcher
    S0 = data_batcher.S0  # structural matrix for spatial clique
    nj = data_batcher.nj  # number of joints
    nl = S0.shape[0]  # number of limbs
    n_frames_total = data_batcher.n_frames_total  # total number of frames across video sets
    # number of labeled frames across video sets, we refer to them as visible frames
    n_visible_frames_total = data_batcher.n_visible_frames_total
    # number of unlabeled labeled frames across video sets, we refer to them as hidden frames
    n_hidden_frames_total = n_frames_total - n_visible_frames_total

    # Calculate the upper bounds for spatial distances
    joint_locs = [d.labels for d in data_batcher.datasets]
    joint_loc_full = np.empty((0, nj, 2))
    for j in joint_locs:
        if len(j) > 0:
            joint_loc_full = np.vstack((j, joint_loc_full))

    joint_loc_full1 = np.copy(joint_loc_full).swapaxes(1, 2).reshape(-1, nj)
    joint_loc_full1[np.isnan(joint_loc_full1)] = 1e10
    limb_full = np.matmul(joint_loc_full1, S0.T)
    limb_full[np.abs(limb_full) > 1e5] = 0
    limb_full = np.reshape(limb_full, [joint_loc_full.shape[0], 2, -1])
    limb_full = np.sqrt(np.sum(np.square(limb_full), 1))
    limb_full = limb_full.T * dgp_cfg['stride'] + dgp_cfg['stride'] / 2
    ws_max = np.max(np.nan_to_num(limb_full), 1) * dgp_cfg['ws_max']

    limb_full = np.true_divide(limb_full.sum(1), (limb_full != 0).sum(1))
    ws = 1 / (np.nan_to_num(
        limb_full) + 1e-20) * dgp_cfg['ws']  # spatial clique parameter based on the limb length and dlc_cfg.ws

    # Define placeholders
    # input and output
    inputs = TF.compat.v1.placeholder(TF.float32, shape=[None, None, None, 3])
    targets = TF.compat.v1.placeholder(TF.float32, shape=[None, nj, 2])
    targets_nonan = TF.where(TF.compat.v1.is_nan(targets), TF.ones_like(targets) * 0, targets)  # set nan to be 0 in targets

    # local refinement
    locref_map = TF.compat.v1.placeholder(TF.float32, shape=[None, None, None, nj * 2])
    locref_mask = TF.compat.v1.placeholder(TF.float32, shape=[None, None, None, nj * 2])

    # placeholders for parameters
    visible_marker_pl = TF.compat.v1.placeholder(TF.int32, shape=[
        None,
    ])  # placeholder for visible marker index in the batch
    hidden_marker_pl = TF.compat.v1.placeholder(TF.int32, shape=[
        None,
    ])  # placeholder for hidden marker index in the batch
    visible_marker_in_targets_pl = TF.compat.v1.placeholder(TF.int32, shape=[
        None,
    ])  # placeholder for visible marker index in targets/visible frames

    nt_batch_pl = TF.compat.v1.placeholder(TF.int32, shape=[])  # placeholder for the total number of frames in the batch

    wt_batch_pl = TF.compat.v1.placeholder(TF.float32, shape=[
        None, ])  # placeholder for the temporal clique wt; it's a vector which can contain different clique values for different frames
    wt_batch_mask_pl = TF.compat.v1.placeholder(TF.float32, shape=[
        None,
    ])  # placeholder for the batch mask for wt, 1 means wt is in the batch; 0 means wt is not in the batch
    wt_batch_tf = TF.multiply(wt_batch_pl, wt_batch_mask_pl)  # wt vector for the batch
    wt_max_tf = TF.constant(dgp_cfg['wt_max'], TF.float32)  # placeholder for the upper bounds for the temporal clique wt

    wn_visible_tf = TF.constant(dgp_cfg['wn_visible'],
                                TF.float32)  # placeholder for the upper bounds for the spatial clique ws; it varies across joints
    wn_hidden_tf = TF.constant(dgp_cfg['wn_hidden'],
                               TF.float32)  # placeholder for the upper bounds for the spatial clique ws; it varies across joints

    ws_tf = TF.constant(ws, TF.float32)  # placeholder for the spatial clique ws; it varies across joints
    ws_max_tf = TF.constant(ws_max,
                            TF.float32)  # placeholder for the upper bounds for the spatial clique ws; it varies across joints
    vector_field_tf = TF.compat.v1.placeholder(TF.float32, shape=[None, None, None])  # placeholder for the vector fields

    # Build the network
    pn = PoseResnet(dgp_cfg)
    net, end_points = pn.extract_features(inputs)
    scope = "pose"
    reuse = None
    heads = {}
    # two convnets, one is the prediction network, the other is the local refinement network.
    with TF.compat.v1.variable_scope(scope, reuse=reuse):
        heads["part_pred"] = prediction_layer(dgp_cfg, net, "part_pred", nj)
        heads["locref"] = prediction_layer(dgp_cfg, net, "locref_pred", nj * 2)

    # Read the 2D targets from pred
    pred = heads['part_pred']
    nx_out, ny_out = tf.shape(pred)[1], tf.shape(pred)[2]
    targets_pred, confidencemap_softmax = argmax_2d_from_cm(pred, nj, dgp_cfg['gamma'], dgp_cfg['gauss_len'])
    targets_pred_marker = TF.reshape(targets_pred, [-1, 2])  # 2d locations for all markers
    targets_pred_hidden_marker = TF.gather(targets_pred_marker,
                                           hidden_marker_pl)  # 2d locations for hidden markers, predicted targets from the network

    # Targets for visible markers only
    targets_visible_marker = TF.reshape(targets_nonan, [-1, 2])
    targets_visible_marker = TF.gather(targets_visible_marker,
                                       visible_marker_in_targets_pl)  # 2d locations for visible markers, observed targets

    # Combine visible markers and hidden markers to construct a full vector for all frames and all markers
    targets_all_marker = combine_all_marker(targets_pred_hidden_marker, targets_visible_marker, hidden_marker_pl,
                                            visible_marker_pl, nj, nt_batch_pl)

    # Construct Gaussian targets for all markers
    target_expand = TF.expand_dims(TF.expand_dims(targets_all_marker, 2), 3)  # (nt*nj) x 2 x 1 x 1

    # 2d grid of the output
    alpha_tf = TF.compat.v1.placeholder(tf.float32, shape=[2, None, None], name="2dgrid")
    alpha_expand = TF.expand_dims(alpha_tf, 0)  # 1 x 2 x nx_out x ny_out

    # normalize the Gaussian bump for the target so that the peak is 1, nt * nx_out * ny_out * nj
    targets_gauss = TF.exp(-TF.reduce_sum(TF.square(alpha_expand - target_expand), axis=1) /
                           (2 * (dgp_cfg['lengthscale'] ** 2)))
    gauss_max = TF.reduce_max(TF.reduce_max(targets_gauss, [1]), [1]) + TF.constant(1e-5, TF.float32)
    gauss_max = TF.expand_dims(TF.expand_dims(gauss_max, [1]), [2])
    targets_gauss = targets_gauss / gauss_max
    targets_gauss = TF.transpose(TF.reshape(targets_gauss, [-1, nj, nx_out, ny_out]), [0, 2, 3, 1])

    # Now calculate the cross entropy given the network outputs and the Gaussian targets
    n_hidden_frames_batch = TF.cast(TF.shape(hidden_marker_pl)[0], tf.float32)  # number of hidden markers in the batch
    n_visible_frames_batch = TF.cast(TF.shape(visible_marker_pl)[0],
                                     tf.float32)  # number of hidden markers in the batch
    # if n_visible_frames_batch is 0, set n_visible_frames_batch=n_hidden_frames_batch
    n_visible_frames_batch = TF.sign(n_visible_frames_batch) * n_visible_frames_batch + (
            1 - TF.sign(n_visible_frames_batch)) * n_hidden_frames_batch

    # Separate gauss targets and output pred for visible and hidden markers
    targets_gauss = TF.reshape(TF.transpose(targets_gauss, [0, 3, 1, 2]), [-1, nx_out, ny_out])
    targets_gauss_v = TF.gather(targets_gauss, visible_marker_pl)  # gauss targets for visible markers
    targets_gauss_h = TF.gather(targets_gauss, hidden_marker_pl)  # gauss targets for hidden markers
    pred = TF.reshape(TF.transpose(pred, [0, 3, 1, 2]), [-1, nx_out, ny_out])
    pred_v = TF.gather(pred, visible_marker_pl)  # output pred for visible markers
    pred_h = TF.gather(pred, hidden_marker_pl)  # output pred for hidden markers

    if dgp_cfg['gm2'] == 1:
        # scale crossentropy loss terms by confidence
        pred_h_sigmoid = tf.sigmoid(pred_h)
        pgm_h1 = tf.reduce_max(tf.reduce_max(pred_h_sigmoid, [1]), [1])  # + EPSILON_tf
        pgm_h2 = tf.expand_dims(tf.expand_dims(pgm_h1, [1]), [2])
        # scale target
        targets_gauss_h = targets_gauss_h * pgm_h2
        # scale network
        pred_h_scaled = pred_h_sigmoid * pgm_h2
        pred_h_scaled1 = -tf.compat.v1.log(1 - pred_h_scaled + 1e-20) + tf.compat.v1.log(
            pred_h_scaled + 1e-20)
    elif dgp_cfg['gm2'] == 2:
        # scale crossentropy loss input by confidence
        pred_h_sigmoid = tf.sigmoid(pred_h)
        pgm_h1 = tf.reduce_max(tf.reduce_max(pred_h_sigmoid, [1]), [1])  # + EPSILON_tf
        pgm_h2 = tf.expand_dims(tf.expand_dims(pgm_h1, [1]), [2])
        # pred_h_sigmoid = TF.gather(pred_sigmoid, hidden_marker_pl)
        # scmap_h = scmap_h#*pmg_h1#tf.math.maximum(pmg_h, gmnormal_pgm_h)
        # scale target
        targets_gauss_h = targets_gauss_h  # *pgm_h2
        # scaled the network output
        pred_h_scaled = pred_h_sigmoid * pgm_h2
        pred_h_scaled1 = -tf.compat.v1.log(1 - pred_h_scaled + 1e-20) + tf.compat.v1.log(
            pred_h_scaled + 1e-20)
    elif dgp_cfg['gm2'] == 0:
        pass
    else:
        raise Exception('Not implemented')

    # %%
    loss = {}
    loss["visible_loss_pred"] = TF.compat.v1.losses.sigmoid_cross_entropy(targets_gauss_v, pred_v, 1.0)
    if dgp_cfg['gm3'] == 3:
        loss["hidden_loss_pred"] = TF.compat.v1.losses.sigmoid_cross_entropy(targets_gauss_h, pred_h_scaled1,
                                                                   weights=(1 - pgm_h2)) * \
                                   n_visible_frames_total / n_hidden_frames_total * \
                                   n_hidden_frames_batch / n_visible_frames_batch * wn_hidden_tf / wn_visible_tf

    elif dgp_cfg['gm3'] == 0:
        loss["hidden_loss_pred"] = TF.compat.v1.losses.sigmoid_cross_entropy(targets_gauss_h, pred_h, 1.0) * \
                                   n_visible_frames_total / n_hidden_frames_total * \
                                   n_hidden_frames_batch / n_visible_frames_batch * wn_hidden_tf / wn_visible_tf
    else:
        raise Exception('Not implemented')

    total_loss = loss["visible_loss_pred"] + loss["hidden_loss_pred"]

    # Calculate the loss for local refinement
    # network output
    locref_pred = heads['locref']
    locref_pred_reshape = TF.reshape(TF.transpose(locref_pred, [0, 3, 1, 2]), [-1, 2, nx_out, ny_out])
    locref_pred_v = TF.gather(locref_pred_reshape, visible_marker_pl)

    # observed locref target and mask
    locref_map_reshape = TF.reshape(TF.transpose(locref_map, [0, 3, 1, 2]), [-1, 2, nx_out, ny_out])
    locref_map_v = TF.gather(locref_map_reshape, visible_marker_pl)
    locref_mask_reshape = TF.reshape(TF.transpose(locref_mask, [0, 3, 1, 2]), [-1, 2, nx_out, ny_out])
    locref_mask_v = TF.gather(locref_mask_reshape, visible_marker_pl)
    
    loss_func = tf.compat.v1.losses.huber_loss if dgp_cfg['locref_huber_loss'] else TF.compat.v1.losses.mean_squared_error
    loss['visible_loss_locref'] = dgp_cfg['locref_loss_weight'] * loss_func(locref_map_v, locref_pred_v, locref_mask_v)
    total_loss = total_loss + loss['visible_loss_locref']

    # ------------------------------------------------------------------------------------
    # Cliques
    # ------------------------------------------------------------------------------------
    targets_all_marker_3c = TF.reshape(targets_all_marker, [nt_batch_pl, nj, -1])  # targets_all_marker with 3 columns

    # Spatial clique
    if nl > 0:
        S = TF.constant(S0, dtype=TF.float32)
        targets_all_marker_spatial = TF.reshape(TF.transpose(targets_all_marker_3c, [1, 2, 0]),
                                                [nj, -1]) * dgp_cfg['stride'] + 0.5 * dgp_cfg['stride']
        dist_targets = TF.sqrt(
            TF.reduce_sum(
                TF.square(TF.reshape(TF.matmul(S, targets_all_marker_spatial), [nl, 2, -1])), [1]))
        ws_max_tf_expand = TF.expand_dims(ws_max_tf, [1])
        dist_targets_th = TF.nn.relu(dist_targets - ws_max_tf_expand) + ws_max_tf_expand

        loss['ws_loss'] = TF.reduce_sum(dist_targets_th * TF.reshape(ws_tf, [-1, 1])) / tf.cast(nx_out, tf.float32) / tf.cast(ny_out, tf.float32)
        loss['ws_loss'] = loss['ws_loss'] * n_visible_frames_total / n_visible_frames_batch / (
                n_visible_frames_total + n_hidden_frames_total) / wn_visible_tf
        total_loss += loss['ws_loss']

    # Temporal clique
    if dgp_cfg['wt'] > 0:
        targets_all_marker_temporal = targets_all_marker_3c * dgp_cfg.stride + 0.5 * dgp_cfg.stride
        targets_all_marker_temporal0 = targets_all_marker_temporal[:-1, :, :]
        targets_all_marker_temporal1 = targets_all_marker_temporal[1:, :, :]
        time_dif0 = TF.sqrt(TF.reduce_sum(TF.square(targets_all_marker_temporal0 - targets_all_marker_temporal1), 2))

        nx_in, ny_in = tf.cast(tf.shape(vector_field_tf)[1], tf.float32), tf.cast(tf.shape(vector_field_tf)[2], tf.float32)

        targets_box_row0 = tf.reshape(targets_all_marker_temporal0[:, :, 0], [-1, ])
        targets_box_col0 = tf.reshape(targets_all_marker_temporal0[:, :, 1], [-1, ])
        targets_box_row1 = tf.reshape(targets_all_marker_temporal1[:, :, 0], [-1, ])
        targets_box_col1 = tf.reshape(targets_all_marker_temporal1[:, :, 1], [-1, ])

        targets_box_row_min = tf.reduce_min(tf.stack((targets_box_row0, targets_box_row1), axis=1), 1)
        targets_box_row_max = tf.reduce_max(tf.stack((targets_box_row0, targets_box_row1), axis=1), 1)
        targets_box_col_min = tf.reduce_min(tf.stack((targets_box_col0, targets_box_col1), axis=1), 1)
        targets_box_col_max = tf.reduce_max(tf.stack((targets_box_col0, targets_box_col1), axis=1), 1)

        window = 10
        targets_box_row_min = tf.math.maximum(tf.constant(0, tf.float32), targets_box_row_min - window)
        targets_box_row_max = tf.math.minimum(nx_in, targets_box_row_max + window)
        targets_box_col_min = tf.math.maximum(tf.constant(0, tf.float32), targets_box_col_min - window)
        targets_box_col_max = tf.math.minimum(ny_in, targets_box_col_max + window)

        boxes = tf.transpose(tf.stack((
                                      targets_box_row_min / nx_in, targets_box_col_min / ny_in, targets_box_row_max / nx_in,
                                      targets_box_col_max / ny_in)))
        box_indices = tf.reshape(tf.transpose(tf.reshape(tf.tile(tf.range(0, nt_batch_pl - 1), [nj]), [nj, -1])), [-1, ])

        vector_field_tf3 = tf.expand_dims(vector_field_tf, 3)
        vector_field_tf_crop = tf.image.crop_and_resize(vector_field_tf3, boxes, box_indices, [nx_in, ny_in])
        vector_field_tf_crop = tf.reshape(vector_field_tf_crop, [-1, nj, nx_in, ny_in])
        vector_field_tf_meanflow = tf.reduce_mean(vector_field_tf_crop, [2, 3])

        vector_field_tf_meanflow_inv = 1 / (vector_field_tf_meanflow + tf.constant(1e-10))
        vector_field_tf_meanflow_inv = tf.math.minimum(vector_field_tf_meanflow_inv, 1)
        vector_field_tf_meanflow_inv = tf.exp(tf.log(vector_field_tf_meanflow_inv) * 3)
        vector_field_tf_meanflow_inv = tf.math.minimum(vector_field_tf_meanflow_inv, 1)
        vector_field_tf_meanflow_inv = vector_field_tf_meanflow_inv * TF.reshape(wt_batch_tf, [-1, 1]) / tf.cast(nx_out, tf.float32) / tf.cast(ny_out, tf.float32)

        dist_targets_th_wt = (TF.nn.relu(time_dif0 - wt_max_tf) + wt_max_tf) * vector_field_tf_meanflow_inv

        loss['wt_loss'] = TF.norm(dist_targets_th_wt, 2)
        loss['wt_loss'] = loss['wt_loss'] * n_visible_frames_total / n_visible_frames_batch / (
                n_visible_frames_total + n_hidden_frames_total) / wn_visible_tf
        total_loss += loss['wt_loss']

    loss['total_loss'] = total_loss

    total_loss_visible = loss['visible_loss_pred'] + loss['visible_loss_locref']

    placeholders = {'inputs': inputs,
                    'targets': targets,
                    'locref_map': locref_map,
                    'locref_mask': locref_mask,
                    'visible_marker_pl': visible_marker_pl,
                    'hidden_marker_pl': hidden_marker_pl,
                    'visible_marker_in_targets_pl': visible_marker_in_targets_pl,
                    'wt_batch_mask_pl': wt_batch_mask_pl,
                    'vector_field_tf': vector_field_tf,
                    'nt_batch_pl': nt_batch_pl,
                    'wt_batch_pl': wt_batch_pl,
                    'alpha_tf': alpha_tf,
                    }

    return loss, total_loss, total_loss_visible, placeholders
