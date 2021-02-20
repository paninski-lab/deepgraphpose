#!/usr/bin/env python

"""
STEP 0:
Create labeled dataset compatible with dlc library
from preprocessed labels
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime as dt
from pathlib import Path

import argparse
# %%
vers = tf.__version__.split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf


#%%
def create_labels(task, date, overwrite_flag=False, check_labels=False, verbose=False, training_fraction=0.95):
    from deepgraphpose.utils_data import local_extract_frames
    from deepgraphpose.PoseDataLoader import DataLoader
    import deeplabcut

    #%%
    #task = 'reach'
    #date ='2030-12-12'
    #%%
    data_info = DataLoader(task)
    # %%
    videoname_ = data_info.vname + data_info.videotype
    scorer = data_info.scorer

    # %%
    assert data_info.raw_data_dir.exists()
    if not data_info.model_data_dir.exists():
        os.makedirs(data_info.model_data_dir)
    # %%
    # The data should be in raw dats dir
    videofile_path = data_info.raw_data_dir / videoname_
    assert videofile_path.exists()
    videos = [videofile_path]

    # %%
    wd = Path(data_info.model_data_dir).resolve()
    project_name = "{pn}-{exp}-{date}".format(pn=task, exp=scorer, date=date)
    project_path = wd / project_name

    print("Project_path: {}\n".format(project_path))
    # %%
    if not project_path.exists():
        path_config_file = deeplabcut.create_new_project(
            project=task,
            experimenter=scorer,
            videos=videos,  # list of video
            working_directory=str(data_info.model_data_dir),
            copy_videos=True,
            videotype=data_info.videotype,
            date=date,
        )
        print("\n Created  new project")

    else:
        if not overwrite_flag:
            raise Exception(" Your config file exists, Turn on overwriting flag Goodbye!")
        else:
            path_config_file = project_path / "config.yaml"
            print("\nYou are overwritting your local file!!")
    # %% set config path
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(path_config_file))

    #print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in cfg.items()) + "}")
    #%%
    cfg["move2corner"] = data_info.move2corner
    cfg["numframes2pick"] = data_info.numframes2pick
    cfg["TrainingFraction"] = [training_fraction] #data_info.TrainingFraction
    cfg["cropping"] = data_info.cropping
    cfg["dotsize"] = data_info.dotsize
    cfg["batch_size"] = data_info.batch_size
    # cfg['pos_dist_thresh'] # should be 4 or os for single pixel but need to be large enough to recover all pixels
    cfg["skeleton"] = data_info.skeleton
    cfg["bodyparts"] = data_info.bodyparts
    # %%
    # iteration
    # snapshot index -1
    # rewrite config file
    deeplabcut.auxiliaryfunctions.write_config(path_config_file, cfg)

    # %%
    # Load curated datasets
    # #### pick curated datasets instead of automatic
    #print("Loading labeled frames and indices")
    frames_index_keep = data_info.get_labeled_frames()

    # %%
    #print("CREATING-SOME LABELS FOR THE FRAMES")
    # directory called on frame extractor
    frames_dir = project_path / "labeled-data" / data_info.vname
    frames = os.listdir(frames_dir)

    #print("Number of training frames {}".format(len(frames)))
    #%%
    if len(frames) == 0:
        assert len(np.unique(frames_index_keep)) >= cfg["numframes2pick"]
        frames2pick = np.sort(
            np.random.choice(frames_index_keep, cfg["numframes2pick"], replace=False)
        )
        local_extract_frames(path_config_file, frames2pick)
        frames = os.listdir(frames_dir)

    #%%
    csv_file = (
            project_path
            / "labeled-data"
            / data_info.vname
            / Path("CollectedData_" + scorer + ".csv")
    )
    hdf_file = (
            project_path
            / "labeled-data"
            / data_info.vname
            / Path("CollectedData_" + scorer + ".h5")
    )

    # %%
    if not (csv_file.exists() and hdf_file.exists()):
        # num_frames = len(frames)
        # num_bodyparts = len(cfg["bodyparts"])

        # clip = VideoFileClip(str(videofile_path))
        data = np.load(str(videofile_path).split(".")[0] + ".npy", allow_pickle=True)[
            ()
        ]
        xr = data["xr"]
        yr = data["yr"]

        # Extracts the frame number
        extract_frame_num = lambda x: int(x.split("/")[-1].split(".")[0][3:])

        # As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
        for index, bodypart in enumerate(cfg["bodyparts"]):
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
            #print(columnindex)
            frame_index_name = [
                os.path.join("labeled-data", data_info.vname, fn) for fn in frames
            ]
            frame_indices = np.asarray(
                [extract_frame_num(frame_) for frame_ in frame_index_name]
            )

            frame = pd.DataFrame(
                np.vstack((xr[index, frame_indices], yr[index, frame_indices])).T,
                columns=columnindex,
                index=frame_index_name,
            )
            if index == 0:
                dataFrame = frame
            else:
                dataFrame = pd.concat([dataFrame, frame], axis=1)

        dataFrame.to_csv(str(csv_file))
        dataFrame.to_hdf(str(hdf_file), "df_with_missing", format="table", mode="w")

        #print("Plot labels...")
        if check_labels:
            deeplabcut.check_labels(path_config_file)
    else:
        print('FIle exists')
    return path_config_file


#%%
def create_labels_md(config_path, video_path, scorer, overwrite_flag=False, check_labels=False, verbose=False,seed= None):
    from deepgraphpose.utils_data import local_extract_frames_md
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(config_path)

    project_path = Path(str(Path(config_path).parent))
    numframes2pick = []
    for video in video_path:
        print(video)
        try:
           # For windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path = os.path.realpath(video)]
           rel_video_path = str(Path.resolve(Path(video)))
        except:
           rel_video_path = os.readlink(str(video))
    
        rel_video_path_parent = Path(rel_video_path).parent / 'dlc_labels_check.npy'
        videoname = Path(rel_video_path).name
        frames_index_keep = np.load(str(rel_video_path_parent), allow_pickle=True)[()]['keep']
    
        print("Number of training frames {} for video {}".format(len(frames_index_keep), videoname))
        
        #%
        if cfg["numframes2pick"] is not None:
            np.random.seed(seed) ## option to set seed. 
            assert len(np.unique(frames_index_keep)) >= cfg["numframes2pick"]
            frames2pick = np.sort(
                np.random.choice(frames_index_keep, cfg["numframes2pick"], replace=False)
            )
            np.random.seed() ## make sure we return to randomness 
        else:
            frames2pick = frames_index_keep
        numframes2pick += [len(frames2pick)]
        
        # extract frames for one video
        local_extract_frames_md(config_path, frames2pick, video)
    
        #%
        vname = videoname.split(".")[0]
        csv_file = (
                project_path
                / "labeled-data"
                / vname
                / Path("CollectedData_" + scorer + ".csv")
        )
        hdf_file = (
                project_path
                / "labeled-data"
                / vname
                / Path("CollectedData_" + scorer + ".h5")
        )
        
        frames_dir = project_path / "labeled-data" / vname
        frames = os.listdir(frames_dir)
        
        # %
        if not (csv_file.exists() and hdf_file.exists()):
            # num_frames = len(frames)
            # num_bodyparts = len(cfg["bodyparts"])
        
            # clip = VideoFileClip(str(videofile_path))
            data = np.load(str(rel_video_path).split(".")[0] + ".npy", allow_pickle=True)[()]
            xr = data["xr"]
            yr = data["yr"]
        
            # Extracts the frame number
            extract_frame_num = lambda x: int(x.split("/")[-1].split(".")[0][3:])
        
            # As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
            for index, bodypart in enumerate(cfg["bodyparts"]):
                columnindex = pd.MultiIndex.from_product(
                    [[scorer], [bodypart], ["x", "y"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                #print(columnindex)
                frame_index_name = [
                    os.path.join("labeled-data", vname, fn) for fn in frames
                ]
                frame_indices = np.asarray(
                    [extract_frame_num(frame_) for frame_ in frame_index_name]
                )
        
                frame = pd.DataFrame(
                    np.vstack((xr[index, frame_indices], yr[index, frame_indices])).T,
                    columns=columnindex,
                    index=frame_index_name,
                )
                if index == 0:
                    dataFrame = frame
                else:
                    dataFrame = pd.concat([dataFrame, frame], axis=1)
        
            dataFrame.to_csv(str(csv_file))
            dataFrame.to_hdf(str(hdf_file), "df_with_missing", format="table", mode="w")
        
            #print("Plot labels...")
            if check_labels:
                deeplabcut.check_labels(config_path)
        else:
            print('FIle exists')
    #    return path_config_file
    
    return numframes2pick

#%%

if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="reach", help="task for run")
    parser.add_argument(
        "--date",
        type=str,
        default=dt.today().strftime("%Y-%m-%d"),
        help="Project run date",
    )

    parser.add_argument(
        "--overwrite_flag",
        type=bool,
        default=False,
        help="Flag to overwrite current project",
    )

    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=create_labels, argv=[sys.argv[0]] + unparsed)
    input_params = parser.parse_args()
    create_labels(input_params.task, input_params.date, input_params.overwrite_flag)
    #%%
