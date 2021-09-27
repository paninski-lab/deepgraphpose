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
from deepgraphpose.contrib.segment_videos import split_video
from run_dgp_demo import update_config_files_general

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
        "--snapshot",
        type=str,
        default=None,
        help="use snapshot for prediction. If not given, assumes `snapshot-step2-final--0`",
    )

    parser.add_argument("--shuffle", type=int, default=1, help="Project shuffle")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="size of the batch, if there are memory issues, decrease it value")
    parser.add_argument("--test", action='store_true', default=False)

    parser.add_argument("--split", action = "store_true",help = "whether or not we should run inference on chopped up videos")
    parser.add_argument("--splitlength", default = 6000, help= "number of frames in block if splitting videos. ")

    input_params = parser.parse_known_args()[0]
    print(input_params)

    dlcpath = input_params.dlcpath
    shuffle = input_params.shuffle
    if input_params.snapshot is not None:
        snapshot = input_params.snapshot
    else:
        ## Specifying snapshot as the end product of training.  
        step = 2
        snapshot = 'snapshot-step{}-final--0'.format(step)

    batch_size = input_params.batch_size
    test = input_params.test
    splitflag,splitlength = input_params.split,input_params.splitlength 


    # update config files
    dlcpath = update_config_files_general(dlcpath,shuffle)
    update_configs = True


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

        if splitflag: 
            video_cut_path = str(Path(dlcpath) / 'videos_cut')
            if not os.path.exists(video_cut_path):
                os.makedirs(video_cut_path)
                clip_sets = []
                for v in video_sets: 
                   clip_sets.extend(split_video(v,int(splitlength),suffix = "demo",outputloc = video_cut_path))
            else:    
                clip_sets = [os.path.join(video_cut_path,v) for v in os.listdir(video_cut_path)]
            video_sets = clip_sets ## replace video_sets with clipped versions.   
        print('video_sets', video_sets, flush=True)

        if test:
            for video_file in [video_sets[0]]:
                from moviepy.editor import VideoFileClip
                clip =VideoFileClip(str(video_file))
                if clip.duration > 10:
                    clip = clip.subclip(10)
                video_file_name = os.path.splitext(video_file)[0] +"test"+ ".mp4" 
                print('\nwriting {}'.format(video_file_name))
                clip.write_videofile(video_file_name)
                plot_dgp(str(video_file_name),
                         str(video_pred_path),
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
        pass

        #if update_configs:
        #    return_configs()
