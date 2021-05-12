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
from os.path import isfile, join
from pathlib import Path
import sys
import yaml

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
    proj_cfg_path = os.path.join(base_path, dlcpath, 'config.yaml')
    with open(proj_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['project_path'] = os.path.join(base_path, dlcpath)
        video_loc = os.path.join(base_path, dlcpath, 'videos', 'reachingvideo1.avi')
        try:
            yaml_cfg['video_sets'][video_loc] = yaml_cfg['video_sets'].pop('videos/reachingvideo1.avi')
        except KeyError:    
            ## check if update has already been done. 
            assert yaml_cfg["video_sets"].get(video_loc,False), "Can't find original or updated video path in config file."
    with open(proj_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # train model config
    model_cfg_path = get_model_cfg_path(base_path, 'train')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = get_init_weights_path(base_path)
        yaml_cfg['project_path'] = os.path.join(base_path, dlcpath)
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

    return os.path.join(base_path, dlcpath)


def return_configs():
    base_path = os.getcwd()
    dlcpath = 'data/Reaching-Mackenzie-2018-08-30'

    # project config
    proj_cfg_path = os.path.join(base_path, dlcpath, 'config.yaml')
    with open(proj_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['project_path'] = dlcpath
        video_loc = os.path.join(base_path, dlcpath, 'videos', 'reachingvideo1.avi')
        yaml_cfg['video_sets']['videos/reachingvideo1.avi'] = yaml_cfg['video_sets'].pop(video_loc)
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
    return os.path.join(
        base_path, dlcpath, 'dlc-models', 'iteration-0', 'ReachingAug30-trainset95shuffle1',
        dtype, 'pose_cfg.yaml')


def get_init_weights_path(base_path):
    return os.path.join(
        base_path, 'src', 'DeepLabCut', 'deeplabcut', 'pose_estimation_tensorflow',
        'models', 'pretrained', 'resnet_v1_50.ckpt')


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

    parser.add_argument("--shuffle", type=int, default=1, help="Project shuffle")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="size of the batch, if there are memory issues, decrease it value")
    parser.add_argument("--test", action='store_true', default=False)

    input_params = parser.parse_known_args()[0]
    print(input_params)

    dlcpath = input_params.dlcpath
    shuffle = input_params.shuffle
    dlcsnapshot = input_params.dlcsnapshot
    batch_size = input_params.batch_size
    test = input_params.test

    # update config files
    dlcpath = update_config_files(dlcpath)
    update_configs = True

    ## Specifying snapshot manually at the moment assuming training. 
    step = 2
    snapshot = 'snapshot-step{}-final--0'.format(step)

    try:

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

        snapshot_path, cfg_yaml = get_snapshot_path(snapshot, dlcpath, shuffle=shuffle)
        cfg = auxiliaryfunctions.read_config(cfg_yaml)

        video_path = str(Path(dlcpath) / 'videos_dgp')
        if not (os.path.exists(video_path)):
            print(video_path + " does not exist!")
            video_sets = list(cfg['video_sets'])
        else:
            video_sets = [
                video_path + '/' + f for f in listdir(video_path)
                if isfile(join(video_path, f)) and (
                        f.find('avi') > 0 or f.find('mp4') > 0 or f.find('mov') > 0 or f.find(
                    'mkv') > 0)
            ]

        video_pred_path = str(Path(dlcpath) / 'videos_pred')
        if not os.path.exists(video_pred_path):
            os.makedirs(video_pred_path)

        print('video_sets', video_sets, flush=True)

        if test:
            for video_file in [video_sets[0]]:
                from moviepy.editor import VideoFileClip
                clip =VideoFileClip(str(video_file))
                if clip.duration > 10:
                    clip = clip.subclip(10)
                video_file_name = video_file.rsplit('/', 1)[-1].rsplit('.',1)[0] + '.mp4'
                print('\nwriting {}'.format(video_file_name))
                clip.write_videofile(video_file_name)
                output_dir = os.getcwd() + '/'
                plot_dgp(video_file=str(video_file_name),
                         output_dir=output_dir,
                         proj_cfg_file=str(cfg_yaml),
                         dgp_model_file=str(snapshot_path),
                         shuffle=shuffle)
        else:
            for video_file in video_sets:
                plot_dgp(str(video_file),
                         str(video_pred_path),
                         proj_cfg_file=str(cfg_yaml),
                         dgp_model_file=str(snapshot_path),
                         shuffle=shuffle)
    finally:

        #if update_configs:
        #    return_configs()
