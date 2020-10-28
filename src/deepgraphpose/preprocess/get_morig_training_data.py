#!/usr/bin/env python

"""
Step 1
Create training dataset compatible with dlc library
from labels
"""

import tensorflow as tf
import numpy as np

from datetime import datetime as dt
from pathlib import Path

from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    mergeandsplit,
    get_largestshuffle_index,
    create_training_dataset,
)
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deepgraphpose.PoseDataLoader import DataLoader
from deepgraphpose.utils_model import get_model_config
from deepgraphpose.helpers.scheduling import default_scheduling
import argparse
import logging

# %%
vers = tf.__version__.split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf

def load_train_test_indices(path_config_file, trainindex=0, uniform=True):

    #%%
    project_path = Path(path_config_file).parent
    #%%
    # store train and test indices for schedule
    filename = project_path / "dlc_project_trainindex_{}_uniform_{}.npy".format(
        trainindex, uniform
    )
    #print(filename)
    if not filename.exists():
        print("Creating train and test indices for current project")

        # deeplabcut.create_training_model_comparison(str(path_config_file),trainindex=True)
        trainIndexes, testIndexes = mergeandsplit(
            str(path_config_file), trainindex=trainindex, uniform=uniform
        )
        np.save(
            str(filename),
            {
                "trainIndexes": trainIndexes,
                "testIndexes": testIndexes,
                "config": str(path_config_file),
            },
        )
        old_iteration_shuffle = 0
    else:
        print("Loading train and test indices")
        trainIndexes = np.load(str(filename), allow_pickle=True)[()]["trainIndexes"]
        testIndexes = np.load(str(filename), allow_pickle=True)[()]["testIndexes"]
        old_iteration_shuffle = 1
    return trainIndexes, testIndexes, old_iteration_shuffle



def add_train_shuffle(get_max_shuffle_idx, cfg, trainIndexes, testIndexes, schedule_config={}):
    project_path = Path(cfg["project_path"])
    path_config_file = project_path / "config.yaml"
    TrainingFraction = cfg["TrainingFraction"]
    iteration = cfg['iteration']

    modelfoldername = auxiliaryfunctions.GetModelFolder(
        TrainingFraction[iteration], get_max_shuffle_idx, cfg
    )
    path_train_folder = project_path / Path(modelfoldername)

    if path_train_folder.exists():
        print('Iteration {} shuffle {} already exists'.format(iteration, get_max_shuffle_idx))
        return
    else:
        print("\nAdding shuffle {}".format(get_max_shuffle_idx))

    #get_max_shuffle_idx    #schedule_config['project_path'] = str(str(project_path))
    #get_max_shuffle_idx = largestshuffleindex + schedule_config_idx + 1
    # Create dataset for that shuffle with deterministic data
    print("\nCreating shuffle {}".format(get_max_shuffle_idx))
    create_training_dataset(
        str(path_config_file),
        Shuffles=[get_max_shuffle_idx],
        trainIndexes=trainIndexes,
        testIndexes=testIndexes,
        items2change_pose=schedule_config,
    )
    return


def add_train_shuffle_from_schedule(task, date, schedules=None, trainindex=0, uniform=True, schedules_default=2):
    #%%
    data_info = DataLoader(task)
    # %%
    # get project configuration file
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    project_path = Path(cfg["project_path"])
    path_config_file = project_path / "config.yaml"

    trainIndexes, testIndexes, old_iteration_shuffle = load_train_test_indices(path_config_file, trainindex=trainindex, uniform=uniform)

    # get the largest index
    largestshuffleindex = get_largestshuffle_index(str(path_config_file))

    shuffle_indices = []

    if schedules is None:
        schedules = default_scheduling(schedules_default, str(project_path))

    #%%
    for schedule_config_idx, schedule_config in enumerate(schedules):
        get_max_shuffle_idx = largestshuffleindex + schedule_config_idx + old_iteration_shuffle
        # Create dataset for that shuffle with deterministic data
        add_train_shuffle(get_max_shuffle_idx,cfg,trainIndexes,testIndexes,schedule_config=schedule_config)
        shuffle_indices.append(get_max_shuffle_idx)
    return shuffle_indices


def create_train_sets(
        task, date, schedule_id=2, trainindex=0, uniform=True, create_default_run=False
):
    #%%
    # Deprecated
    #%%
    shuffle_indices = []
    #%%
    data_info = DataLoader(task)

    # get project configuration file
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    project_path = Path(cfg["project_path"])
    path_config_file = project_path / "config.yaml"
    #%%
    trainIndexes, testIndexes, _ = load_train_test_indices(path_config_file, trainindex=trainindex, uniform=uniform)
    # store train and test indices for schedule
    #%%
    # get the largest index
    largestshuffleindex = get_largestshuffle_index(str(path_config_file))
    #%%
    TrainingFraction = cfg["TrainingFraction"]

    #%%
    # create log file
    logger_name = "training_model_comparison_trainindex_{}_uniform_{}".format(
        trainindex, uniform
    )
    log_file_name = str(project_path / (logger_name + ".log"))
    logger = logging.getLogger(logger_name)
    # print(logger.handlers)
    #%%
    if not logger.handlers:
        # create logger comparison data
        new_iteration_shuffle = True
        logger = logging.getLogger(logger_name)
        hdlr = logging.FileHandler(log_file_name)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
    else:
        # logger exists! shuffle should be set to > max
        new_iteration_shuffle = False

    #%%
    if create_default_run:
        # Create shuffle 0 for default params
        if new_iteration_shuffle & largestshuffleindex == 0:
            create_training_dataset(
                str(path_config_file),
                Shuffles=[largestshuffleindex],
                trainIndexes=trainIndexes,
                testIndexes=testIndexes,
            )

        # reading file struct for training and test files as pose_yaml files
        if new_iteration_shuffle & largestshuffleindex == 0:
            trainFraction = TrainingFraction[0]
            modelfoldername = auxiliaryfunctions.GetModelFolder(
                trainFraction, largestshuffleindex, cfg
            )
            path_train_config = str(
                project_path / Path(modelfoldername) / "train" / "pose_cfg.yaml"
            )
            # path_test_config = str(project_path / Path(modelfoldername) /'test' /'pose_cfg.yaml')

            # cfg_train = auxiliaryfunctions.read_config(path_train_config)
            cfg_train = load_config(filename=path_train_config)

            # load_config gets the training datas
            log_info = str(
                "Shuffle index: {},".format(largestshuffleindex)
                + "".join("{!r}: {!r},".format(k, v) for k, v in cfg_train.items())
            )
            logger.info(log_info)

            shuffle_indices.append(largestshuffleindex)
    #%% Run schedule
    schedule = default_scheduling(schedule_id, str(project_path))
    print("Setting up {} experiments".format(len(schedule)))
    print("Last scheduled experiment was shuffle {}".format(largestshuffleindex))
    #%%
    # TO DO: add read log_info function to avoid modified params
    # TO DO: skip creating new data
    for schedule_config_idx, schedule_config in enumerate(schedule):
        print(schedule_config)
        #schedule_config['project_path'] = str(str(project_path))
        get_max_shuffle_idx = largestshuffleindex + schedule_config_idx + 1
        # Create dataset for that shuffle with deterministic data
        add_train_shuffle(get_max_shuffle_idx,cfg,trainIndexes,testIndexes,schedule_config=schedule_config)

        # replace by all config files
        # log_info = str("Shuffle index: {},".format(get_max_shuffle_idx) + "".join("{!r}: {!r},".format(k, v)
        #                  for k, v in schedule_config.items()))

        trainFraction = TrainingFraction[cfg["iteration"]]
        modelfoldername = auxiliaryfunctions.GetModelFolder(
            trainFraction, get_max_shuffle_idx, cfg
        )
        path_train_config = str(
            project_path / Path(modelfoldername) / "train" / "pose_cfg.yaml"
        )
        # path_test_config = str(project_path / Path(modelfoldername) /'test' /'pose_cfg.yaml')

        # cfg_train = auxiliaryfunctions.read_config(path_train_config)
        cfg_train = load_config(filename=path_train_config)
        # load_config gets the training datas
        log_info = str(
            "Shuffle index: {},".format(get_max_shuffle_idx)
            + "".join("{!r}: {!r},".format(k, v) for k, v in cfg_train.items())
        )
        logger.info(log_info)

        print(log_info)

        shuffle_indices.append(get_max_shuffle_idx)
    logging.shutdown()
    return shuffle_indices


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
        "--schedule_id",
        type=int,
        default=None,
        help="input an id default in schedules_default",
    )

    parser.add_argument("--trainindex", type=int, default=0, help="Train index")

    parser.add_argument("--uniform", type=bool, default=True, help="Split of labels")

    parser.add_argument(
        "--schedules_default", type=int, default=2, help="Create default run"
    )
    input_params = parser.parse_known_args()[0]

    #%%

    shuffle_indices = add_train_shuffle_from_schedule(
        task=input_params.task,
        date=input_params.date,
        schedules=input_params.schedule_id,
        trainindex=input_params.trainindex,
        uniform=input_params.uniform,
        schedules_default=input_params.schedules_default,
    )

    print("For this schedule, evaluate shuffles {}".format(shuffle_indices))
    print(shuffle_indices)
    #%%


#%%