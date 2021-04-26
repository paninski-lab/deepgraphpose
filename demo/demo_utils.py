import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import sys
import yaml

def get_model_cfg_path(base_path, dlcpath, dtype):
    return os.path.join(
        base_path, dlcpath, 'dlc-models', 'iteration-0', 'ReachingAug30-trainset95shuffle1',
        dtype, 'pose_cfg.yaml')


def get_init_weights_path(base_path):
    return os.path.join(
        base_path, 'src', 'DeepLabCut', 'deeplabcut', 'pose_estimation_tensorflow',
        'models', 'pretrained', 'resnet_v1_50.ckpt')

def update_config_files(dlcpath):
    base_path = os.getcwd()

    # project config
    proj_cfg_path = os.path.join(base_path, dlcpath, 'config.yaml')
    with open(proj_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['project_path'] = os.path.join(base_path, dlcpath)
        video_loc = os.path.join(base_path, dlcpath, 'videos', 'reachingvideo1.avi')
        yaml_cfg['video_sets'][video_loc] = yaml_cfg['video_sets'].pop('videos/reachingvideo1.avi')
    with open(proj_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # train model config
    model_cfg_path = get_model_cfg_path(base_path, dlcpath, 'train')
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
    model_cfg_path = get_model_cfg_path(base_path, dlcpath, 'test')
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
    model_cfg_path = get_model_cfg_path(base_path, dlcpath, 'train')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = 'resnet_v1_50.ckpt'
        yaml_cfg['project_path'] = dlcpath
    with open(model_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

    # test model config
    model_cfg_path = get_model_cfg_path(base_path, dlcpath, 'test')
    with open(model_cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        yaml_cfg['init_weights'] = 'resnet_v1_50.ckpt'
    with open(model_cfg_path, 'w') as f:
        yaml.dump(yaml_cfg, f)

def print_steps(step=None):
    if step == 0:
        print(
            '''
            =====================
            |                   |
            |                   |
            |    Running DLC    |
            |                   |
            |                   |
            =====================
            '''
            , flush=True)
    elif step == 1:
        print(
            '''
            ===============================================
            |                                             |
            |                                             |
            |    Running DGP with labeled frames only     |
            |                                             |
            |                                             |
            ===============================================
            '''
            , flush=True)
    elif step == 2:
        print(
            '''
            =====================
            |                   |
            |                   |
            |    Running DGP    |
            |                   |
            |                   |
            =====================
            '''
            , flush=True)

    elif step == "prediction":
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

def clip_video(video_file, crop_duration=10):
    "cropping a video for 10 seconds"
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(str(video_file))
    if clip.duration > crop_duration:
        clip = clip.subclip(t_start=0, t_end=crop_duration)
    video_file_name = video_file.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '_short' + '.mp4'
    print('\nwriting {}'.format(video_file_name))
    clip.write_videofile(video_file_name)
    output_dir = os.getcwd() + '/'
    return video_file_name, output_dir

def get_video_sets(dlcpath, cfg):
    """we want to get the list of videos for prediction"""
    video_path = str(Path(dlcpath) / 'videos_dgp')
    if not (os.path.exists(video_path)):  # if there's no folder called "videos_dgp" with videos
        print(video_path + " does not exist!")
        video_sets = list(cfg['video_sets'])  # take videos that were used for training
    else:  # pick formats ".avi", ".mp4", ".mov", ".mkv"
        video_sets = [
            video_path + '/' + f for f in listdir(video_path)
            if isfile(join(video_path, f)) and (
                    f.find('avi') > 0 or f.find('mp4') > 0 or f.find('mov') > 0 or f.find(
                'mkv') > 0)
        ]
    return video_sets

def set_or_open_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("Opened a new folder at: {}".format(folder_path))
    else:
        print("The folder already exists at: {}".format(folder_path))
    return Path(folder_path) # a PosixPath object