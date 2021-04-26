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
from demo_utils import get_video_sets, print_steps, get_model_cfg_path, get_init_weights_path, \
    update_config_files, return_configs, clip_video, set_or_open_folder


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
    parser.add_argument("--multiview", action='store_true', default=False)
    # consider changing
    parser.add_argument("--start_step", type=int, default=0,
                        help="0:dlc, 1:dgp_labeledonly, 2:dgp, 3:multiview labeled only, 4:dgp multiview ")
    input_params = parser.parse_known_args()[0]
    print(input_params)

    dlcpath = input_params.dlcpath
    shuffle = input_params.shuffle
    dlcsnapshot = input_params.dlcsnapshot
    batch_size = input_params.batch_size
    test = input_params.test
    multiview = input_params.multiview
    start_step = input_params.start_step

    if test:  # we run for a short period
        maxiters_dgp = 10
        maxiters_dlc = 20
        displayiters = 2
        saveiters = 5
    else:
        maxiters_dgp = 50000
        maxiters_dlc = 200000
        displayiters = 100
        saveiters = 1000
        # TODO: check if these numbers should be different for DLC

    update_configs = False
    if dlcpath == 'data/Reaching-Mackenzie-2018-08-30':
        # update config files
        dlcpath = update_config_files(dlcpath)
        update_configs = True

    # ------------------------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------------------------

    try:

        # %% step 0 DLC
        if dlcsnapshot is None:  # run DLC from scratch
            step=0
            print_steps(step=step)
            snapshot = 'resnet_v1_50.ckpt'
            fit_dlc(snapshot=snapshot, dlcpath=dlcpath,
                    shuffle=shuffle, step=0, saveiters=saveiters,
                    displayiters=displayiters, maxiters=maxiters_dlc)
            snapshot = 'snapshot-step{}-final--0'.format(step)  # snapshot for initializing next step

        else:  # use the specified DLC snapshot to initialize DGP, and skip step 0
            snapshot = dlcsnapshot  # snapshot for step 1

        # %% step 1 DGP labeled frames only
        step = 1
        print_steps(step)

        fit_dgp_labeledonly(snapshot=snapshot,
                            dlcpath=dlcpath,
                            shuffle=shuffle,
                            step=step,
                            saveiters=saveiters,
                            displayiters=displayiters,
                            maxiters=maxiters_dgp,
                            multiview=multiview)
        snapshot = 'snapshot-step{}-final--0'.format(step)

        # %% step 2 DGP
        step = 2
        print_steps(step)
        gm2, gm3 = 1, 3 # regularization constants
        fit_dgp(snapshot=snapshot,
                dlcpath=dlcpath,
                batch_size=batch_size,
                shuffle=shuffle,
                step=step,
                saveiters=saveiters,
                maxiters=maxiters_dgp,
                displayiters=displayiters,
                gm2=gm2,
                gm3=gm3,
                multiview=multiview)

        snapshot = 'snapshot-step{}-final--0'.format(step)

        # --------------------------------------------------------------------------------
        # Test DGP model
        # --------------------------------------------------------------------------------

        # %% predict on all videos in videos_dgp folder
        print_steps('prediction')
        snapshot_path, cfg_yaml = get_snapshot_path(snapshot, dlcpath, shuffle=shuffle)
        cfg = auxiliaryfunctions.read_config(cfg_yaml)

        video_sets = get_video_sets(dlcpath, cfg)
        print('video_sets', video_sets, flush=True)

        video_pred_path = set_or_open_folder(os.path.join(dlcpath, 'videos_pred'))

        if test: # analyze a short clip from one video
            print('clipping one video for test.')
            short_video, _ = clip_video(video_sets[0], 10)
            video_sets = [short_video]

        for video_file in video_sets:
            plot_dgp(video_file=str(video_file),
                     output_dir=str(video_pred_path),
                     proj_cfg_file=str(cfg_yaml),
                     dgp_model_file=str(snapshot_path),
                     shuffle=shuffle)

    finally:

        if update_configs:
            return_configs()
