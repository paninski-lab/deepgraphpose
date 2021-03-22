import os

import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from deepgraphpose.models.fitdgp import fit_dgp
from deepgraphpose.models.fitdgp import compute_epipolar_loss
from deepgraphpose.models.fitdgp import dgp_loss
from deepgraphpose.models.fitdgp import define_placeholders

import importlib
import logging
import time
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

import deeplabcut
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.factory import (
    create as create_dataset, )
from deeplabcut.pose_estimation_tensorflow.dataset.pose_defaultdataset import PoseDataset
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from deeplabcut.pose_estimation_tensorflow.train import LearningRate, get_batch_spec, \
    setup_preloading, start_preloading, get_optimizer
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import PoseNet, losses, \
    prediction_layer

from deepgraphpose.dataset import MultiDataset, coord2map
from deepgraphpose.models.fitdgp_util import gen_batch, argmax_2d_from_cm, combine_all_marker, build_aug, data_aug, learn_wt


vers = tf.__version__.split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf
#
#
# def get_img_points(df, img_name):
#     img_lbls = df.loc[df['scorer'] == img_name]
#     # drop first column
#     img_lbls = img_lbls.drop(columns=['scorer'])
#     # convert to list
#     img_lbls = img_lbls.to_numpy(dtype=float)
#     # get points
#     num_points = int(img_lbls.shape[1] / 2)
#     points = np.zeros(shape=(num_points, 2))
#     for i in range(num_points):
#         x = img_lbls[0][i * 2]
#         y = img_lbls[0][i * 2 + 1]
#         points[i] = np.array([x, y])
#
#     return points
#
# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape[0:2]
#     # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
#     # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1.astype(int), pts2.astype(int)):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv.circle(img2,tuple(pt2),5,color,-1)
#     return img1, img2
#
# def run_test_numpy(dlcpath, shuffle, batch_size, snapshot):
#     img1_name = "labeled-data/lBack_bodyCrop/img019942.png"
#     img1 = cv.imread("/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img1_name)
#     img2_name = "labeled-data/lTop_bodyCrop/img019942.png"
#     img2 = cv.imread("/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img2_name)
#     # plt.imshow(img1)
#     # plt.show()
#     # plt.imshow(img2)
#     # plt.show()
#     # read the dataframe
#     df = pd.read_csv("/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01/training-datasets/iteration-0/UnaugmentedDataSet_bird1Jan1/CollectedData_selmaan.csv")
#     print(df.describe())
#     col_names = df.loc[df['scorer'] == 'bodyparts']
#
#     # get points
#     im1_pts = get_img_points(df, img1_name)
#     im2_pts = get_img_points(df, img2_name)
#
#     # Now you can knock yourself out writing the loss function
#     # compute fundamental matrix
#     # todo: note a minimum of 8 corresponding points are needed
#     F, mask = cv.findFundamentalMat(im1_pts, im2_pts)
#     print(F)
#     # this selects *only* the inlier points. I think this is unnecessary, because all points are guaranteed to be in
#     # the visible space of the image plane
#     im1_pts = im1_pts[mask.ravel() == 1]
#     im2_pts = im2_pts[mask.ravel() == 1]
#
#     # convert to homogeneous
#     ones = np.ones(shape=(im1_pts.shape[0], 1))
#     im1_pts_hom = np.hstack((im1_pts, ones))
#     im2_pts_hom = np.hstack((im2_pts, ones))
#
#     # compute to x^Fx
#     z = np.sum(np.dot(im2_pts_hom, F) * im1_pts_hom, axis=1)
#     # compute loss as magnitude of x^Fx
#     loss = np.linalg.norm(z, 2)
#     print(z)
#     print(loss)
#
#     # naive x^Fx
#     for i, im1_pt in enumerate(im1_pts_hom):
#         im2_pt = im2_pts_hom[i]
#         z = np.dot(im2_pt.T, np.dot(F, im1_pt))
#         print(z)
#         # z = np.dot(np.dot(im1_pt, F.T), im2_pt.T)
#         # print(z)
#
#     # x(F.T)x^
#     lines1_z = np.zeros(shape=(18,3))
#     for i, x in enumerate(im2_pts_hom):
#         lines1_z[i] = np.dot(F.T, x)
#     a = np.sum(im1_pts_hom * lines1_z, axis=1)
#     print(a)
#
#     lines2_z = np.zeros(shape=(18, 3))
#     for i, x in enumerate(im1_pts_hom):
#         lines2_z[i] = np.dot(F, x)
#
#
#     print("here")
#
#     # Find epilines corresponding to points in right image (second image) and
#     # drawing its lines on left image
#     lines1 = cv.computeCorrespondEpilines(im2_pts.reshape(-1, 1, 2), 2, F)
#     lines1 = lines1.reshape(-1, 3)
#     img5, img6 = drawlines(np.copy(img1), np.copy(img2), lines1, im1_pts, im2_pts)
#     # Find epilines corresponding to points in left image (first image) and
#     # drawing its lines on right image
#     lines2 = cv.computeCorrespondEpilines(im1_pts.reshape(-1, 1, 2), 1, F)
#     lines2 = lines2.reshape(-1, 3)
#     img3, img4 = drawlines(np.copy(img2), np.copy(img1), lines2, im2_pts, im1_pts)
#     plt.subplot(121), plt.imshow(img4)
#     plt.subplot(122), plt.imshow(img3)
#     plt.show()
#     plt.subplot(121), plt.imshow(img6)
#     plt.subplot(122), plt.imshow(img5)
#     plt.show()
#
#
#     # todo: convert labels to heatmaps (how?)
#     # Construct Gaussian targets for all markers
#     # target_expand = TF.expand_dims(TF.expand_dims(targets_all_marker, 2), 3)  # (nt*nj) x 2 x 1 x 1
#     #
#     # # 2d grid of the output
#     # alpha_tf = TF.placeholder(tf.float32, shape=[2, None, None], name="2dgrid")
#     # alpha_expand = TF.expand_dims(alpha_tf, 0)  # 1 x 2 x nx_out x ny_out
#     #
#     # # normalize the Gaussian bump for the target so that the peak is 1, nt * nx_out * ny_out * nj
#     # targets_gauss = TF.exp(-TF.reduce_sum(TF.square(alpha_expand - target_expand), axis=1) /
#     #                        (2 * (dgp_cfg.lengthscale ** 2)))
#     # gauss_max = TF.reduce_max(TF.reduce_max(targets_gauss, [1]), [1]) + TF.constant(1e-5, TF.float32)
#     # gauss_max = TF.expand_dims(TF.expand_dims(gauss_max, [1]), [2])
#     # targets_gauss = targets_gauss / gauss_max
#     # targets_gauss = TF.transpose(TF.reshape(targets_gauss, [-1, nj, nx_out, ny_out]), [0, 2, 3, 1])
#     #
#     # # Separate gauss targets and output pred for visible and hidden markers
#     # targets_gauss = TF.reshape(TF.transpose(targets_gauss, [0, 3, 1, 2]), [-1, nx_out, ny_out])
#     # targets_gauss_v = TF.gather(targets_gauss
#     #                             # , visible_marker_pl
#     #                             )  # gauss targets for visible markers
#     # pred = TF.reshape(TF.transpose(pred, [0, 3, 1, 2]), [-1, nx_out, ny_out])
#     # pred_v = TF.gather(pred,
#     #                    # visible_marker_pl
#     #                    )  # output pred for visible markers
#
#     print("here")
#
#     # step = 2
#     # gm2, gm3 = 1, 3
#     # fit_dgp_labeledonly(snapshot,
#     #                     dlcpath,
#     #                     shuffle=shuffle,
#     #                     step=step,
#     #                     maxiters=5,
#     #                     displayiters=1)
#
#     # fit_dgp(snapshot,
#     #         dlcpath,
#     #         batch_size=batch_size,
#     #         shuffle=shuffle,
#     #         step=step,
#     #         maxiters=5,
#     #         displayiters=1,
#     #         gm2=gm2,
#     #         gm3=gm3)
#
#
# def run_test_tf(dlcpath, shuffle, batch_size, snapshot):
#     img1_name = "labeled-data/lBack_bodyCrop/img019942.png"
#     img1 = cv.imread("/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img1_name)
#     img2_name = "labeled-data/lTop_bodyCrop/img019942.png"
#     img2 = cv.imread("/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img2_name)
#     # plt.imshow(img1)
#     # plt.show()
#     # plt.imshow(img2)
#     # plt.show()
#     # read the dataframe
#     df = pd.read_csv(
#         "/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01/training-datasets/iteration-0/UnaugmentedDataSet_bird1Jan1/CollectedData_selmaan.csv")
#     print(df.describe())
#     col_names = df.loc[df['scorer'] == 'bodyparts']
#
#     # get points (just a helper for this test)
#     im1_pts = get_img_points(df, img1_name)
#     im2_pts = get_img_points(df, img2_name)
#
#
#     # Now you can knock yourself out writing the loss function
#     # compute fundamental matrix
#     # todo: note a minimum of 8 corresponding points are needed
#     F, mask = cv.findFundamentalMat(im1_pts, im2_pts)
#     print(F)
#     # this selects *only* the inlier points. I think this is unnecessary, because all points are guaranteed to be in
#     # the visible space of the image plane
#     im1_pts = im1_pts[mask.ravel() == 1]
#     im2_pts = im2_pts[mask.ravel() == 1]
#
#     # convert to homogeneous
#     ones = np.ones(shape=(im1_pts.shape[0], 1))
#     im1_pts_hom = np.hstack((im1_pts, ones))
#     im2_pts_hom = np.hstack((im2_pts, ones))
#     tf.enable_eager_execution()
#     ###### TF ######
#     im1_pts_hom = tf.convert_to_tensor(im1_pts_hom)
#     im2_pts_hom = tf.convert_to_tensor(im2_pts_hom)
#     F = tf.convert_to_tensor(F)
#
#     # compute to x^Fx
#     z = tf.math.reduce_sum(tf.math.multiply(tf.tensordot(im2_pts_hom, F, axes=1), im1_pts_hom), axis=1)
#     # compute loss as magnitude of x^Fx
#     loss = tf.norm(z, ord=2)
#     print(z)
#     print(loss)
#
def fit_dgp_eager(
        snapshot, dlcpath, batch_size=10, shuffle=1, step=2, saveiters=1000, displayiters=5,
        maxiters=200000, ns=10, nc=2048, n_max_frames=2000, gm2=0, gm3=0, nepoch=100, wt=0, aug=True,
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

    # enable eager execution
    tf.enable_eager_execution()

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
            video_path + '/' + f for f in listdir(video_path)
            if isfile(join(video_path, f)) and (
                    f.find('avi') > 0 or f.find('mp4') > 0 or f.find('mov') > 0 or f.find('mkv') > 0)
        ]

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
    # todo: self.nx_out, self.ny_out for MultiDataset class need to be specified explicitly when running in eager ((40, 40) for selmaan)
    data_batcher = MultiDataset(config_yaml=config_path,
                                video_sets=video_sets,
                                shuffle=shuffle,
                                S0=S0)
    dgp_cfg = data_batcher.dlc_config
    dgp_cfg.ws = 1000  # the spatial clique parameter
    dgp_cfg.ws_max = 1.2  # the multiplier for the upper bound of spatial distance
    dgp_cfg.wt = wt  # the temporal clique parameter
    dgp_cfg.wt_max = 0  # the upper bound of temporal distance
    dgp_cfg.wn_visible = 5  # the network clique parameter for visible frames
    dgp_cfg.wn_hidden = 3  # the network clique parameter for hidden frames
    dgp_cfg.gamma = 1  # the multiplier for the softmax confidence map
    dgp_cfg.gauss_len = 1  # the length scale for the Gaussian kernel convolving the softmax confidence map
    dgp_cfg.lengthscale = 1  # the length scale for the Gaussian target map
    dgp_cfg.max_to_keep = 5  # max number of snapshots to keep
    dgp_cfg.batch_size = batch_size  # batch size
    dgp_cfg.n_times_all_frames = nepoch  # the number of times each selected frames is iterated over
    dgp_cfg.lr = 0.005  # learning rate
    # dgp_cfg.net_type = 'resnet_50'
    dgp_cfg.gm2 = gm2  # scale target by confidence level
    dgp_cfg.gm3 = gm3  # scale hidden loss by confidence level
    dgp_cfg.aug = aug  # data augmentation

    # skip this DGP step if it's already done.
    model_name = dgp_cfg.snapshot_prefix + '-step{}{}-final--0.index'.format(step, debug)
    if os.path.isfile(model_name):
        print(model_name, '  exists! DGP has already been run.', flush=True)
        # return None

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
    TF.reset_default_graph()

    # Restore network parameters for RESNET and COVNET
    # variables_to_restore0 = slim.get_variables_to_restore(
    #     include=['pose/part_pred'])
    # variables_to_restore1 = slim.get_variables_to_restore(
    #     include=['pose/locref_pred'])
    # variables_to_restore2 = slim.get_variables_to_restore(include=['resnet'])
    # restorer = TF.train.Saver(variables_to_restore0 + variables_to_restore1 +
    #                           variables_to_restore2)
    # saver = TF.train.Saver(max_to_keep=dgp_cfg.max_to_keep)

    # Set up session
    # allow_growth = True
    # if allow_growth:
    #     config = TF.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = TF.Session(config=config)
    # else:
    #     sess = TF.Session()

    # %%
    # ------------------------------------------------------------------------------------
    # Begin training
    # ------------------------------------------------------------------------------------
    batch_ind_all = gen_batch(visible_frame_total, hidden_frame_total, all_frame_total, dgp_cfg, maxiters)
    save_iters = np.int(saveiters / dgp_cfg.batch_size)
    maxiters = batch_ind_all.shape[0]

    pdata = PoseDataset(dgp_cfg)
    data_batcher.reset()

    # %%
    print('Begin Training for {} iterations'.format(maxiters))
    if dgp_cfg.aug:
        pipeline = build_aug(apply_prob=0.8)

    time_start = time.time()
    for it in range(maxiters):

        current_lr = dgp_cfg.lr

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

        # todo: this loop should be conditional based on how many videos are in the data_batcher?
        # todo: rewrite this to BE a part of the data_batcher?
        all_data_batch_ids = []
        video_names = []
        for dataset_id in range(len(data_batcher.datasets)):
            (visible_frame, hidden_frame, _, all_data_batch, joint_loc, wt_batch_mask,
             all_marker_batch, addn_batch_info), d = \
                data_batcher.next_batch(0, dataset_id, visible_frame_batch_i, hidden_frame_batch_i)
            # add data from a single view to the batch
            all_data_batch_ids.append(all_data_batch)
            # add the corresponding name of the view to a list of video_names (important that these added at the same time to their respective lists to preserve ordering)
            video_names.append(data_batcher.datasets[dataset_id].video_name)

        # (visible_frame, hidden_frame, _, all_data_batch, joint_loc, wt_batch_mask,
        #  all_marker_batch, addn_batch_info), d = \
        #     data_batcher.next_batch(0, dataset_i, visible_frame_batch_i, hidden_frame_batch_i)
        nt_batch = len(visible_frame) + len(hidden_frame)
        visible_marker, hidden_marker, visible_marker_in_targets = addn_batch_info
        all_frame = np.sort(list(visible_frame) + list(hidden_frame))
        visible_frame_within_batch = [np.where(all_frame == i)[0][0] for i in visible_frame]

        # batch data for placeholders
        if dgp_cfg.wt > 0:
            vector_field = learn_wt(all_data_batch)  # vector field from optical flow # todo: optical flow?
        else:
            vector_field = np.zeros((1, 1, 1))
        wt_batch = np.ones(nt_batch - 1, ) * dgp_cfg.wt

        # data augmentation for visible frames
        if dgp_cfg.aug and dgp_cfg.wt == 0:
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

        all_data_batch_ids = np.concatenate(all_data_batch_ids)
        video_names = tf.convert_to_tensor(video_names)
        feed_dict = {
            'inputs': all_data_batch_ids,
            'targets': joint_loc,
            'locref_map': locref_targets_all_batch,
            'locref_mask': locref_mask_all_batch,
            'visible_marker_pl': visible_marker,
            'hidden_marker_pl': hidden_marker,
            'visible_marker_in_targets_pl': visible_marker_in_targets,
            'wt_batch_mask_pl': wt_batch_mask,
            'vector_field_tf': vector_field,
            'nt_batch_pl': nt_batch,
            'wt_batch_pl': wt_batch,
            'alpha_tf': alpha,
            'learning_rate': current_lr,
            'video_names': video_names
        }

        # loss = dgp_loss(data_batcher, dgp_cfg, placeholders=feed_dict)










        # Epipolar clique
        # todo: make this conditional based on whether or not training is "multiview"
        # todo: as it stands right now, I am not incorporating any hard labels, strictly constraining the predictions -> may be helpful to incorporate hard labels
        # todo: consider scaling loss by confidence of prediction?
        # todo: need a weight for the clique?
        # Build the network
        pn = PoseNet(dgp_cfg)
        net, end_points = pn.extract_features(all_data_batch_ids)
        scope = "pose"
        reuse = None
        heads = {}
        # two convnets, one is the prediction network, the other is the local refinement network.
        with TF.variable_scope(scope, reuse=reuse):
            heads["part_pred"] = prediction_layer(dgp_cfg, net, "part_pred", nj)
            heads["locref"] = prediction_layer(dgp_cfg, net, "locref_pred", nj * 2)

        pred = heads['part_pred']
        targets_pred, confidencemap_softmax = argmax_2d_from_cm(pred, nj, dgp_cfg.gamma, dgp_cfg.gauss_len)

        # Read the 2D targets from pred
        nx_out, ny_out = tf.shape(pred)[1], tf.shape(pred)[2]
        targets_pred, confidencemap_softmax = argmax_2d_from_cm(pred, nj, dgp_cfg.gamma, dgp_cfg.gauss_len)
        targets_pred_marker = TF.reshape(targets_pred, [-1, 2])  # 2d locations for all markers
        targets_pred_hidden_marker = TF.gather(targets_pred_marker,
                                               hidden_marker)  # 2d locations for hidden markers, predicted targets from the network

        F_dict = data_batcher.fundamental_mat_dict
        num_pts_per_frame = targets_pred.shape[1]
        num_pts_per_view = tf.dtypes.cast(num_pts_per_frame * nt_batch,
                                          tf.int64)  # need to cast this as an int64 for some reason or it breaks
        # loss['epipolar_loss'] = 0
        for key, F in F_dict.items():
            v1_name, v2_name = key.split(data_batcher.F_dict_key_delim)
            # get coordinates of predictions for video 1
            name1_idx = tf.where(tf.equal(video_names, v1_name))[0][0]
            v1_pts = targets_pred_marker[name1_idx * num_pts_per_view:name1_idx * num_pts_per_view + num_pts_per_view]
            # get coordinates of predictions for video 2
            name2_idx = tf.where(tf.equal(video_names, v2_name))[0][0]
            v2_pts = targets_pred_marker[name2_idx * num_pts_per_view:name2_idx * num_pts_per_view + num_pts_per_view]
            # compute epipolar loss. (every point in v1_pts should correspond to the same point in space as the point at
            # the same index in v2_pts. I.e. v1_pts[n] and v2_pts[n] correspond to the same point in space)
            epipolar_loss = 0.1 * compute_epipolar_loss(v1_pts, v2_pts, F)
            # loss['epipolar_loss'] += epipolar_loss
        print(epipolar_loss)
        # total_loss += loss['epipolar_loss']


        start_time00 = time.time()
#
#
# def compute_epipolar_loss(v1_pts, v2_pts, F):
#     # convert to homogeneous coordinates
#     ones = tf.ones_like(v1_pts)[:,0]
#     ones = tf.expand_dims(ones, axis=1)
#     im1_pts_hom = tf.concat([v1_pts, ones], axis=1)
#     im2_pts_hom = tf.concat([v2_pts, ones], axis=1)
#
#     F = tf.convert_to_tensor(F, dtype=tf.float32)
#     # compute x`Fx
#     z = tf.math.reduce_sum(tf.math.multiply(tf.tensordot(im2_pts_hom, F, axes=1), im1_pts_hom), axis=1)
#     # compute loss as magnitude of x`Fx
#     epipolar_loss = tf.norm(z, ord=2)
#     return epipolar_loss
#
#
def dgp_loss_eager(data_batcher, dgp_cfg, feed_dict):
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
    limb_full = limb_full.T * dgp_cfg.stride + dgp_cfg.stride / 2
    ws_max = np.max(np.nan_to_num(limb_full), 1) * dgp_cfg.ws_max

    limb_full = np.true_divide(limb_full.sum(1), (limb_full != 0).sum(1))
    ws = 1 / (np.nan_to_num(
        limb_full) + 1e-20) * dgp_cfg.ws  # spatial clique parameter based on the limb length and dlc_cfg.ws

    # feed_dict
    inputs = feed_dict['inputs']
    targets = feed_dict['targets']
    targets_nonan = TF.where(TF.is_nan(targets), TF.ones_like(targets) * 0, targets)  # set nan to be 0 in targets
    locref_map = feed_dict['locref_map']
    locref_mask = feed_dict['locref_mask']
    visible_marker_pl = feed_dict['visible_marker_pl']
    hidden_marker_pl = feed_dict['hidden_marker_pl']
    visible_marker_in_targets_pl = feed_dict['visible_marker_in_targets_pl']
    nt_batch_pl = feed_dict['nt_batch_pl']
    wt_batch_pl = feed_dict['wt_batch_pl']
    wt_batch_mask_pl = feed_dict['wt_batch_mask_pl']
    video_names = feed_dict['video_names'] # used in getting the appropiate views for computing epipolar loss
    alpha_tf = feed_dict['alpha_tf']


    wt_batch_tf = TF.multiply(wt_batch_pl, wt_batch_mask_pl)  # wt vector for the batch
    wt_max_tf = TF.constant(dgp_cfg.wt_max, TF.float32)  # placeholder for the upper bounds for the temporal clique wt

    wn_visible_tf = TF.constant(dgp_cfg.wn_visible,
                                TF.float32)  # placeholder for the upper bounds for the spatial clique ws; it varies across joints
    wn_hidden_tf = TF.constant(dgp_cfg.wn_hidden,
                               TF.float32)  # placeholder for the upper bounds for the spatial clique ws; it varies across joints

    ws_tf = TF.constant(ws, TF.float32)  # placeholder for the spatial clique ws; it varies across joints
    ws_max_tf = TF.constant(ws_max,
                            TF.float32)  # placeholder for the upper bounds for the spatial clique ws; it varies across joints
    vector_field_tf = TF.placeholder(TF.float32, shape=[None, None, None])  # placeholder for the vector fields

    # Build the network
    pn = PoseNet(dgp_cfg) # todo: why is this here? (as opposed to outside of the loss function)
    net, end_points = pn.extract_features(inputs)
    scope = "pose"
    reuse = None
    heads = {}
    # two convnets, one is the prediction network, the other is the local refinement network.
    with TF.variable_scope(scope, reuse=reuse):
        heads["part_pred"] = prediction_layer(dgp_cfg, net, "part_pred", nj)
        heads["locref"] = prediction_layer(dgp_cfg, net, "locref_pred", nj * 2)

    # Read the 2D targets from pred
    pred = heads['part_pred']
    nx_out, ny_out = tf.shape(pred)[1], tf.shape(pred)[2]
    targets_pred, confidencemap_softmax = argmax_2d_from_cm(pred, nj, dgp_cfg.gamma, dgp_cfg.gauss_len)
    targets_pred_marker = TF.reshape(targets_pred, [-1, 2])  # 2d locations for all markers
    targets_pred_hidden_marker = TF.gather(targets_pred_marker,
                                           hidden_marker_pl)  # 2d locations for hidden markers, predicted targets from the network

    # Targets for visible markers only
    targets_visible_marker = TF.reshape(targets_nonan, [-1, 2])
    targets_visible_marker = TF.gather(targets_visible_marker,
                                       visible_marker_in_targets_pl)  # 2d locations for visible markers, observed targets

    # Combine visible markers and hidden markers to construct a full vector for all frames and all markers
    # set hidden_marker_pl and visible_marker_pl to int32 (instead of int64) for some reason
    hidden_marker_pl = hidden_marker_pl.astype('int32')
    visible_marker_pl = visible_marker_pl.astype('int32')
    targets_all_marker = combine_all_marker(targets_pred_hidden_marker, targets_visible_marker, hidden_marker_pl,
                                            visible_marker_pl, nj, nt_batch_pl)

    # Construct Gaussian targets for all markers
    target_expand = TF.expand_dims(TF.expand_dims(targets_all_marker, 2), 3)  # (nt*nj) x 2 x 1 x 1

    # 2d grid of the output
    alpha_expand = TF.expand_dims(alpha_tf, 0)  # 1 x 2 x nx_out x ny_out

    # normalize the Gaussian bump for the target so that the peak is 1, nt * nx_out * ny_out * nj
    target_expand = tf.dtypes.cast(target_expand, tf.float64)
    targets_gauss = TF.exp(-TF.reduce_sum(TF.square(alpha_expand - target_expand), axis=1) /
                           (2 * (dgp_cfg.lengthscale ** 2)))
    gauss_max = TF.reduce_max(TF.reduce_max(targets_gauss, [1]), [1]) + TF.constant(1e-5, TF.float64)  # todo: look here
    gauss_max = TF.expand_dims(TF.expand_dims(gauss_max, [1]), [2])
    targets_gauss = targets_gauss / gauss_max
    targets_gauss = TF.transpose(TF.reshape(targets_gauss, [-1, nj, nx_out, ny_out]), [0, 2, 3, 1])

    # Now calculate the cross entropy given the network outputs and the Gaussian targets # todo: cross entropy computed here
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

    if dgp_cfg.gm2 == 1:
        # scale crossentropy loss terms by confidence
        pred_h_sigmoid = tf.sigmoid(pred_h)
        pgm_h1 = tf.reduce_max(tf.reduce_max(pred_h_sigmoid, [1]), [1])  # + EPSILON_tf
        pgm_h2 = tf.expand_dims(tf.expand_dims(pgm_h1, [1]), [2])
        # scale target
        targets_gauss_h = targets_gauss_h * pgm_h2
        # scale network
        pred_h_scaled = pred_h_sigmoid * pgm_h2
        pred_h_scaled1 = -tf.log(1 - pred_h_scaled + 1e-20) + tf.log(
            pred_h_scaled + 1e-20)
    elif dgp_cfg.gm2 == 2:
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
        pred_h_scaled1 = -tf.log(1 - pred_h_scaled + 1e-20) + tf.log(
            pred_h_scaled + 1e-20)
    elif dgp_cfg.gm2 == 0:
        pass
    else:
        raise Exception('Not implemented')

    # %%
    loss = {}
    loss["visible_loss_pred"] = TF.losses.sigmoid_cross_entropy(targets_gauss_v, pred_v, 1.0)
    if dgp_cfg.gm3 == 3:
        loss["hidden_loss_pred"] = TF.losses.sigmoid_cross_entropy(targets_gauss_h, pred_h_scaled1,
                                                                   weights=(1 - pgm_h2)) * \
                                   n_visible_frames_total / n_hidden_frames_total * \
                                   n_hidden_frames_batch / n_visible_frames_batch * wn_hidden_tf / wn_visible_tf

    elif dgp_cfg.gm3 == 0:
        loss["hidden_loss_pred"] = TF.losses.sigmoid_cross_entropy(targets_gauss_h, pred_h, 1.0) * \
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

    loss_func = losses.huber_loss if dgp_cfg.locref_huber_loss else TF.losses.mean_squared_error
    loss['visible_loss_locref'] = dgp_cfg.locref_loss_weight * loss_func(locref_map_v, locref_pred_v, locref_mask_v)
    total_loss = total_loss + loss['visible_loss_locref']

    # ------------------------------------------------------------------------------------
    # Cliques
    # ------------------------------------------------------------------------------------
    targets_all_marker_3c = TF.reshape(targets_all_marker, [nt_batch_pl, nj, -1])  # targets_all_marker with 3 columns

    F_dict = data_batcher.fundamental_mat_dict
    num_pts_per_frame = targets_pred.shape[1]
    num_pts_per_view = num_pts_per_frame * nt_batch_pl
    loss['epipolar_loss'] = 0
    for key, F in F_dict.items():
        v1_name, v2_name = key.split(data_batcher.F_dict_key_delim)
        # get coordinates of predictions for video 1
        name1_idx = tf.where(tf.equal(video_names, v1_name))[0][0]
        v1_pts = targets_pred_marker[name1_idx * num_pts_per_view:name1_idx * num_pts_per_view + num_pts_per_view]
        # get coordinates of predictions for video 2
        name2_idx = tf.where(tf.equal(video_names, v2_name))[0][0]
        v2_pts = targets_pred_marker[name2_idx * num_pts_per_view:name2_idx * num_pts_per_view + num_pts_per_view]
        # compute epipolar loss. (every point in v1_pts should correspond to the same point in space as the point at
        # the same index in v2_pts. I.e. v1_pts[n] and v2_pts[n] correspond to the same point in space)
        epipolar_loss = compute_epipolar_loss(v1_pts, v2_pts, F)
        loss['epipolar_loss'] += epipolar_loss

    total_loss += loss['epipolar_loss']

    # Spatial clique
    if nl > 0:
        S = TF.constant(S0, dtype=TF.float32)
        targets_all_marker_spatial = TF.reshape(TF.transpose(targets_all_marker_3c, [1, 2, 0]),
                                                [nj, -1]) * dgp_cfg.stride + 0.5 * dgp_cfg.stride
        dist_targets = TF.sqrt(
            TF.reduce_sum(
                TF.square(TF.reshape(TF.matmul(S, targets_all_marker_spatial), [nl, 2, -1])), [1]))
        ws_max_tf_expand = TF.expand_dims(ws_max_tf, [1])
        dist_targets_th = TF.nn.relu(dist_targets - ws_max_tf_expand) + ws_max_tf_expand

        loss['ws_loss'] = TF.reduce_sum(dist_targets_th * TF.reshape(ws_tf, [-1, 1])) / tf.cast(nx_out,
                                                                                                tf.float32) / tf.cast(
            ny_out, tf.float32)
        loss['ws_loss'] = loss['ws_loss'] * n_visible_frames_total / n_visible_frames_batch / (
                n_visible_frames_total + n_hidden_frames_total) / wn_visible_tf
        total_loss += loss['ws_loss']

    # Temporal clique
    if dgp_cfg.wt > 0:
        targets_all_marker_temporal = targets_all_marker_3c * dgp_cfg.stride + 0.5 * dgp_cfg.stride
        targets_all_marker_temporal0 = targets_all_marker_temporal[:-1, :, :]
        targets_all_marker_temporal1 = targets_all_marker_temporal[1:, :, :]
        time_dif0 = TF.sqrt(TF.reduce_sum(TF.square(targets_all_marker_temporal0 - targets_all_marker_temporal1), 2))

        nx_in, ny_in = tf.cast(tf.shape(vector_field_tf)[1], tf.float32), tf.cast(tf.shape(vector_field_tf)[2],
                                                                                  tf.float32)

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
        box_indices = tf.reshape(tf.transpose(tf.reshape(tf.tile(tf.range(0, nt_batch_pl - 1), [nj]), [nj, -1])),
                                 [-1, ])

        vector_field_tf3 = tf.expand_dims(vector_field_tf, 3)
        vector_field_tf_crop = tf.image.crop_and_resize(vector_field_tf3, boxes, box_indices, [nx_in, ny_in])
        vector_field_tf_crop = tf.reshape(vector_field_tf_crop, [-1, nj, nx_in, ny_in])
        vector_field_tf_meanflow = tf.reduce_mean(vector_field_tf_crop, [2, 3])

        vector_field_tf_meanflow_inv = 1 / (vector_field_tf_meanflow + tf.constant(1e-10))
        vector_field_tf_meanflow_inv = tf.math.minimum(vector_field_tf_meanflow_inv, 1)
        vector_field_tf_meanflow_inv = tf.exp(tf.log(vector_field_tf_meanflow_inv) * 3)
        vector_field_tf_meanflow_inv = tf.math.minimum(vector_field_tf_meanflow_inv, 1)
        vector_field_tf_meanflow_inv = vector_field_tf_meanflow_inv * TF.reshape(wt_batch_tf, [-1, 1]) / tf.cast(nx_out,
                                                                                                                 tf.float32) / tf.cast(
            ny_out, tf.float32)

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
                    'video_names': video_names
                    }

    return loss, total_loss, total_loss_visible, placeholders

if __name__ == '__main__':
    # dlcpath = "/Users/sethdonaldson/sourceCode/neuro/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01" # personal machine path
    base_path = os.getcwd()[:os.getcwd().find("deepgraphpose")]
    dlcpath = base_path + "/deepgraphpose/data/track_graph3d/bird1-selmaan-2030-01-01" # axon path
    shuffle = 1
    batch_size = 10
    snapshot = 'snapshot-step0-final--0'
    step = 2
    # gm2, gm3 = 1, 3
    #
    # # tf.enable_eager_execution()
    print(dlcpath)
    fit_dgp_eager(snapshot, dlcpath, shuffle=shuffle, step=step, batch_size=batch_size)

    # run_test_numpy(dlcpath, shuffle, batch_size, snapshot)
