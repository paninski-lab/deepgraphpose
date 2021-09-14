# If you have collected labels using DLC's GUI you can run DGP with the following
"""Main fitting function for DGP.
   step 0: run DLC
   step 1: run DGP with labeled frames only
   step 2: run DGP with spatial clique
   step 3: do prediction on all videos
"""
import argparse
import os
from os import listdir
from os.path import isfile, join, split
from pathlib import Path
import sys
import yaml

import pandas as pd
from deeplabcut.utils.video_processor import (
    VideoProcessorCV as vp,
)  # used to CreateVideo
from deeplabcut.utils import auxiliaryfunctions, CreateVideo, visualization


if sys.platform == 'darwin':
    import wx
    if int(wx.__version__[0]) > 3:
        wx.Thread_IsMain = wx.IsMainThread

os.environ["DLClight"] = "True"
os.environ["Colab"] = "True"
from deeplabcut.utils import auxiliaryfunctions

from deepgraphpose.models.fitdgp import fit_dlc, fit_dgp, fit_dgp_labeledonly
from deepgraphpose.models.fitdgp_util import get_snapshot_path
from deepgraphpose.models.eval import plot_dgp


def update_config_files(dlcpath):
    base_path = os.getcwd()

    # project config
    proj_cfg_path = join(base_path, dlcpath, 'config.yaml')
    with open(proj_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['project_path'] = join(base_path, dlcpath)
        video_loc = join(base_path, dlcpath, 'videos', 'reachingvideo1.avi')
        try:
            yaml_cfg['video_sets'][video_loc] = yaml_cfg['video_sets'].pop(join('videos','reachingvideo1.avi'))
        except:
            yaml_cfg['video_sets'][video_loc] = yaml_cfg['video_sets'].pop(video_loc)
    with open(proj_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # train model config
    model_cfg_path = get_model_cfg_path(base_path, 'train')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = get_init_weights_path(base_path)
        yaml_cfg['project_path'] = join(base_path, dlcpath)
    with open(model_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # download resnet weights if necessary
    if not os.path.exists(yaml_cfg['init_weights']):
        raise FileNotFoundError('Must download resnet-50 weights; see README for instructions')

    # test model config
    model_cfg_path = get_model_cfg_path(base_path, 'test')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = get_init_weights_path(base_path)
    with open(model_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    return join(base_path, dlcpath)


def return_configs():
    base_path = os.getcwd()
    dlcpath = join('data','Reaching-Mackenzie-2018-08-30')

    # project config
    proj_cfg_path = join(base_path, dlcpath, 'config.yaml')
    with open(proj_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['project_path'] = dlcpath
        video_loc = join(base_path, dlcpath, 'videos', 'reachingvideo1.avi')
        yaml_cfg['video_sets'][join('videos','reachingvideo1.avi')] = yaml_cfg['video_sets'].pop(video_loc)
    with open(proj_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # train model config
    model_cfg_path = get_model_cfg_path(base_path, 'train')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = 'resnet_v1_50.ckpt'
        yaml_cfg['project_path'] = dlcpath
    with open(model_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # test model config
    model_cfg_path = get_model_cfg_path(base_path, 'test')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = 'resnet_v1_50.ckpt'
    with open(model_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)


def get_model_cfg_path(base_path, dtype):
    return join(
        base_path, dlcpath, 'dlc-models', 'iteration-0', 'ReachingAug30-trainset95shuffle1',
        dtype, 'pose_cfg.yaml')


def get_init_weights_path(base_path):
    return join(
        base_path, 'resnet_v1_50.ckpt')


if __name__ == '__main__':

    # %% set up dlcpath for DLC project and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dlcpath",
        type=str,
        default=None,
        help="the absolute path of the DLC project",
    )

    parser.add_argument(
        "--dlcsnapshot",
        type=str,
        default=None,
        help="use the DLC snapshot to initialize DGP",
    )

    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="use the DGP snapshot",
    )

    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="path to video",
    )

    parser.add_argument(
        "--video-path-out",
        type=str,
        default=None,
        help="path to output video",
    )


    parser.add_argument("--shuffle", type=int, default=1, help="Project shuffle")

    input_params = parser.parse_known_args()[0]
    print(input_params)

    dlcpath = input_params.dlcpath
    shuffle = input_params.shuffle
    snapshot = input_params.snapshot
    video_path = input_params.video_path
    video_path_out = input_params.video_path_out


    print(dlcpath)


    cfg_yaml =  dlcpath + '/config.yaml'


    print(cfg_yaml)



    # ------------------------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------------------------
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    
    # --------------------------------------------------------------------------------
    # Test DGP model
    # --------------------------------------------------------------------------------
    # %% step 3 predict on all videos in videos_dgp folder
    print(
        '''
        ==========================
        |                        |
        |                        |
        |    Predict with DGP    |
        |                        |
        |                        |
        ==========================
        '''
        , flush=True)
    cfg = auxiliaryfunctions.read_config(cfg_yaml)
    bodyparts2connect = cfg["skeleton"]
    skeleton_color = cfg["skeleton_color"]
    draw_skeleton = True
    color_by = 'bodypart'
    displaycropped = False
    bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, "all"
    )
    cropping = False
    x1, x2, y1, y2 = 0,0,0,0
    trailpoints = 0
    if not (os.path.exists(video_path)):
        print(video_path + " does not exist!")
        video_sets = list(cfg['video_sets'])
    else:
        video_sets = [
            join(video_path, f) for f in listdir(video_path)
            if isfile(join(video_path, f)) and (
                    f.find('avi') > 0 or f.find('mp4') > 0 or f.find('mov') > 0 or f.find(
                'mkv') > 0)
        ]
    video_pred_path = video_path_out
    if not os.path.exists(video_pred_path):
        os.makedirs(video_pred_path)
    print('video_sets', video_sets, flush=True)
    for video_file in video_sets:
        plot_dgp(str(video_file),
                 str(video_pred_path),
                 proj_cfg_file=str(cfg_yaml),
                 dgp_model_file=str(snapshot),
                 shuffle=shuffle)

                 
        filename = video_pred_path + "/" + os.path.basename(video_file)
        videooutname = filename.split(".")[0] + "_dgp_labeled.mp4"
        print("VIDEO OUT NAME")
        print(videooutname)
        clip = vp(fname=video_file,sname=videooutname)
        filepath = filename.split(".")[0] + "_labeled.h5"
        df = pd.read_hdf(filepath)

        labeled_bpts = [
            bp
            for bp in df.columns.get_level_values("bodyparts").unique()
            if bp in bodyparts
        ]

        CreateVideo(
            clip,
            df,
            cfg["pcutoff"],
            cfg["dotsize"],
            cfg["colormap"],
            labeled_bpts,
            trailpoints,
            cropping,
            x1,
            x2,
            y1,
            y2,
            bodyparts2connect,
            skeleton_color,
            draw_skeleton,
            displaycropped,
            color_by,
        )