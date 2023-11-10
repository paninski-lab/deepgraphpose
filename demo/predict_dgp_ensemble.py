# If you have collected labels using DLC's GUI you can run DGP with the following
"""Ensemble fitting function for DGP.
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
from joblib import Memory
import sys
import yaml
import pandas as pd
import numpy as np

if sys.platform == 'darwin':
    import wx
    if int(wx.__version__[0]) > 3:
        wx.Thread_IsMain = wx.IsMainThread

os.environ["DLClight"] = "True"
os.environ["Colab"] = "True"

from moviepy.editor import VideoFileClip
                
from deeplabcut.utils import auxiliaryfunctions

from deepgraphpose.models.fitdgp import fit_dlc, fit_dgp, fit_dgp_labeledonly
from deepgraphpose.models.fitdgp_util import get_snapshot_path
from deepgraphpose.models.eval import plot_dgp
from run_dgp_demo import update_config_files_general

from dgp_ensembletools.models import Ensemble 

if __name__ == '__main__':

    # %% set up dlcpath for DLC project and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelpaths",
        nargs="+",
        type=str,
        default=None,
        help="the absolute path of the DLC projects you want to ensemble",
    )

    parser.add_argument(
        "--dlcsnapshot",
        type=str,
        default=None,
        help="use the DLC snapshot to initialize DGP",
    )

    parser.add_argument(
        "--videopath",
        type=str,
        default=None,
        help="path to video",
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

    modelpaths = input_params.modelpaths
    shuffle = input_params.shuffle
    videopath = input_params.videopath
    dlcsnapshot = input_params.dlcsnapshot
    batch_size = input_params.batch_size
    test = input_params.test

    # update config files
    for modelpath in modelpaths:
        dlcpath = update_config_files_general(modelpath,shuffle)
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
            ==================================
            |                                |
            |                                |
            |    Predict with DGP Ensemble   |
            |                                |
            |                                |
            ==================================
            '''
            , flush=True)

        ## get ensembleparameters
        topdir = os.path.dirname(modelpaths[0]) ## they're all loaded into the same anyway. 
        modeldirs = [os.path.basename(m) for m in modelpaths]
        videoext = os.path.splitext(videopath)[-1].split(".",1)[-1] ## remove the dot as well. 
        video = VideoFileClip(videopath)
        ## Write results to:
        resultpath = os.path.join(os.path.dirname(os.path.dirname(videopath)),"results")



        framelength = int(video.duration*video.fps)
        ## can do some processing based on length here. 
        framerange = range(0,framelength)

        remoteensemble = Ensemble(topdir,modeldirs,videoext,memory = Memory(os.path.dirname(videopath)))
        [model.predict(videopath) for model in remoteensemble.models.values()]
        predict_videoname = "_labeled".join(os.path.splitext(os.path.basename(videopath)))
        predict_h5name = "_labeled".join([os.path.splitext(os.path.basename(videopath))[0],".h5"])
        consensus_videoname = "_labeled_consensus".join(os.path.splitext(os.path.basename(videopath)))
        consensus_csvname = "_labeled_consensus".join([os.path.splitext(os.path.basename(videopath))[0],".csv"])
        consensus_h5name = "_labeled_consensus".join([os.path.splitext(os.path.basename(videopath))[0],".h5"])
        ## outputs pose of shape xy, time, body part 
        meanx,meany = remoteensemble.get_mean_pose(predict_videoname,framerange,snapshot = snapshot, shuffle = shuffle) 

        ## reshape and save in shape of existing: 
        likearray = np.empty(meanx.shape) ## don't get likelihoods right now. 
        likearray[:] = np.NaN
        stacked = np.stack((meanx,meany,likearray),axis = -1)
        dfshaped = stacked.reshape(stacked.shape[0],stacked.shape[1]*stacked.shape[2])
        ## get sample dataframe:
        sampledf = pd.read_hdf(os.path.join(modelpaths[0],"videos_pred",predict_h5name))
        sampledf.iloc[:len(dfshaped),:] = dfshaped
        sampledf.drop([i for i in range(len(dfshaped),len(sampledf))],inplace = True)
        
        if not os.path.isdir(resultpath):
            os.makedirs(resultpath)

        sampledf.to_csv(os.path.join(resultpath,consensus_csvname))
        sampledf.to_hdf(os.path.join(resultpath,consensus_h5name),key="consensus")

        ### Not writing video of consensus for now: 
        #if test:
        #    for video_file in [video_sets[0]]:
        #        clip =VideoFileClip(str(video_file))
        #        if clip.duration > 10:
        #            clip = clip.subclip(10)
        #        video_file_name = os.path.splitext(video_file)[0] +"test"+ ".mp4" 
        #        print('\nwriting {}'.format(video_file_name))
        #        clip.write_videofile(video_file_name)
        #        plot_dgp(str(video_file_name),
        #                 str(video_pred_path),
        #                 proj_cfg_file=str(cfg_yaml),
        #                 dgp_model_file=str(snapshot_path),
        #                 shuffle=shuffle)
        #else:
        #    for video_file in video_sets:
        #        plot_dgp(str(video_file),
        #                 str(video_pred_path),
        #                 proj_cfg_file=str(cfg_yaml),
        #                 dgp_model_file=str(snapshot_path),
        #                 shuffle=shuffle)
    finally:
        pass

        #if update_configs:
        #    return_configs()
