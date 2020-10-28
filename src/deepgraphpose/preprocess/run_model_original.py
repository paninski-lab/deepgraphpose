"""
step 3: run model original
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import tensorflow as tf
from pathlib import Path
import deeplabcut
from moviepy.editor import VideoFileClip
from PoseDataLoader import DataLoader
from datetime import datetime as dt
from src.deepgraphpose import get_model_config
import argparse

vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf

#%%


def run_original_dlc_shuffle(task, date, shuffle, allow_growth=True, max_snapshots_to_keep=None,
                             keepdeconvweights=True, maxiters=None):
    data_info = DataLoader(task)
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    path_config_file = Path(cfg["project_path"]) / "config.yaml"
    # dlc_cfg = get_train_config(cfg, shuffle=shuffle)
    # %%
    print("\nTraining network")
    deeplabcut.train_network(
        str(path_config_file),
        shuffle=shuffle,
        trainingsetindex=cfg["iteration"],
        allow_growth=allow_growth,
        max_snapshots_to_keep=max_snapshots_to_keep,
        keepdeconvweights=keepdeconvweights,
        maxiters=maxiters,  # inherits from dlc_cfg file being read
    )
    #%%
    # Evaluate run
    print("\nEVALUATE")
    deeplabcut.evaluate_network(
        str(path_config_file),
        plotting=False,
        Shuffles=[shuffle],
        trainingsetindex=cfg["iteration"],
    )
    # %%
    print("Make copy of video for evaluation")
    # Read video
    video_file = str(
        Path(cfg["project_path"]) / "videos" / (data_info.vname + data_info.videotype)
    )

    # %%
    dfolder = Path(cfg["project_path"]) / "videos" / "moriginal_iter_{}_shuffle_{}".format(cfg['iteration'], shuffle)
    if not os.path.isdir(dfolder):
        os.makedirs(dfolder)

    # Make video copy for evaluation
    newvideo = str(dfolder / (data_info.vname + '.mp4'))

    newclip = VideoFileClip(video_file)
    newclip.write_videofile(newvideo)

    #%%
    # deeplabcut.analyze_videos(path_config_file,
    #                          [newvideo],
    #                          save_as_csv=True,
    #                          shuffle=shuffle,
    #                          trainingsetindex=trainingsetindex,
    #                          destfolder=dfolder,
    #                          dynamic=(True, .1, 5))

    # %%
    # cannot use dynamic cropping wo locref
    print("\nanalyze again...")
    deeplabcut.analyze_videos(
        str(path_config_file),
        [newvideo],
        save_as_csv=True,
        destfolder=dfolder,
        shuffle=shuffle,
        trainingsetindex=cfg["iteration"],
    )
    # %%
    print("\nCreate Labeled video")
    deeplabcut.create_labeled_video(
        str(path_config_file),
        [newvideo],
        destfolder=dfolder,
        shuffle=shuffle,
        trainingsetindex=cfg["iteration"],
        save_frames=False,
    )

    # %%
    # print("\nMaking plots")
    # deeplabcut.plot_trajectories(str(path_config_file),
    #                             [newvideo],
    #                             destfolder=dfolder,
    #                             shuffle=shuffle,
    #                             trainingsetindex=trainingsetindex)
    # % note when training gails we are not updating the model
    #%%
    return


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

    parser.add_argument("--shuffle", type=int, default=1, help="Project shuffle")

    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=create_labels, argv=[sys.argv[0]] + unparsed)
    input_params = parser.parse_args()
    run_original_dlc_shuffle(input_params.task, input_params.date, input_params.shuffle)
    #%%