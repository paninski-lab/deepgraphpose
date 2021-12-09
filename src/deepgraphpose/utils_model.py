import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
from easydict import EasyDict as edict
vers = (tf.__version__).split('.')

if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf


# %%
def get_snapshot_path(snapshot, dlcpath, shuffle=1,trainingsetindex = 0):
    dlc_base_path = Path(dlcpath)
    config_path = dlc_base_path / 'config.yaml'
    cfg = auxiliaryfunctions.read_config(config_path)
    modelfoldername = auxiliaryfunctions.GetModelFolder(
        cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)

    train_path = dlc_base_path / modelfoldername / 'train'
    snapshot_path = str(train_path / snapshot)
    return snapshot_path, config_path


def read_dlc_csv(cfg, dlc_cfg, data_info, shuffle, snapshot=4999):
    iteration = cfg['iteration']
    dlc_csv_fname = dlc_cfg.project_path / "videos" / (
        "moriginal_iter_{}_shuffle_{}".format(iteration, shuffle)) / (
            data_info.vname + "DLC_{}_{}{}shuffle{}_{}.csv".format(
                cfg['default_net_type'].replace('_', ''), cfg['Task'],
                cfg['date'], shuffle, snapshot))
    assert dlc_csv_fname.exists()

    df = pd.read_csv(dlc_csv_fname)
    df = df.values
    dfxy = df[2:, 1:].reshape(df.shape[0] - 2, -1, 3)
    dfxy = dfxy.astype(np.float)
    dfxy = dfxy[:, :, :-1]

    xr = dfxy[:, :, 0]
    yr = dfxy[:, :, 1]

    xr = np.transpose(xr, [1, 0])
    yr = np.transpose(yr, [1, 0])

    return xr, yr


def get_model_plot_outdir(cfg, snapshot=4999, mdir='va_semi', shuffle=1):
    mfulldir = Path(
        cfg['project_path']) / 'videos' / (mdir + '_iter_{}_shuffle_{}'.format(
            cfg['iteration'], shuffle)) / 'snapshot-{}'.format(snapshot)
    if not mfulldir.exists():
        os.makedirs(mfulldir)
    return mfulldir


def get_model_dir(dataset):
    semi_dlc_outdir = dataset.resnet_out_dir / 'va_semi' / 'snapshot-{}'.format(
        dataset.snapshot)
    #mfulldir = Path(dlc_cfg.snapshot_prefix).parent / mdir / 'snapshot-{}'.format(snapshot)
    #assert mfulldir.exists()
    return semi_dlc_outdir


def get_model_config(task, wd, scorer='kelly', date='2019-12-13'):
    # construct proj name
    from deeplabcut.utils import auxiliaryfunctions
    project_name = '{pn}-{exp}-{date}'.format(pn=task, exp=scorer, date=date)
    project_path = wd / project_name
    path_config_file = project_path / 'config.yaml'
    assert path_config_file.exists()
    cfg = auxiliaryfunctions.read_config(str(path_config_file))
    cfg['project_path'] = project_path

    video_sets = list(cfg['video_sets'].keys())
    # forcing to be 1 video -- update when handling multiple videos
    assert len(video_sets) == 1
    # to make this work in old config update video_sets to be a relative path
    video_name = video_sets[0]
    cfg['video_path'] = cfg['project_path'] / video_name
    return edict(cfg)


def get_train_config(cfg, shuffle=0):
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    project_path = cfg['project_path']
    iteration = cfg['iteration']
    TrainingFraction = cfg['TrainingFraction'][iteration]
    modelfolder = os.path.join(
        project_path,
        str(auxiliaryfunctions.GetModelFolder(TrainingFraction, shuffle, cfg)))

    path_test_config = Path(modelfolder) / 'train' / 'pose_cfg.yaml'
    print(path_test_config)

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, TrainingFraction))
    # from get_model_config
    dlc_cfg['video_path'] = cfg['video_path']
    dlc_cfg['project_path'] = cfg['project_path']
    return dlc_cfg


def load_dlc_snapshot(dlc_cfg,
                      snapshotindex=-1,
                      overwrite_snapshot=None,
                      verbose=False):
    """
    Load snapshot index : uses max by default (snapshot index), otherwise uses specific overwrite_snapshot
    :param dlc_cfg:
    :param snapshotindex:
    :param overwrite_snapshot:
    :param verbose:
    :return:
    """
    Snapshots = []
    try:
        for fn in os.listdir(Path(dlc_cfg.snapshot_prefix).parent):
            if 'index' in fn:
                # Snapshots = np.array([fn.split('.')[0]])
                Snapshots.append(fn.split('.')[0])
            # Filter into
    except FileNotFoundError:
        raise FileNotFoundError('Did not find snapshot')
        #raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))
    Snapshots = np.asarray(Snapshots)
    if verbose:
        print('available snapshots to load are {}'.format(Snapshots))

    if len(Snapshots) > 0:
        increasing_indices = np.argsort(
            [int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        if verbose:
            print(Snapshots)
        if not (overwrite_snapshot is None):
            trainingsnapshot_name = 'snapshot-{}'.format(
                int(overwrite_snapshot))
        else:
            # Check if data already was generated and set init weight to specific snapshot:
            trainingsnapshot_name = Snapshots[snapshotindex]
        if verbose:
            print("Using %s" % trainingsnapshot_name, "for model")

        trainingsnapshot = int(trainingsnapshot_name.split('-')[-1])

        # if 0 use the default
        if trainingsnapshot > 0:
            dlc_cfg['init_weights'] = dlc_cfg.snapshot_prefix + "-{}".format(
                trainingsnapshot)
            if not Path(dlc_cfg['init_weights'] + '.index').exists():
                raise Exception(
                    'Init weight file doesn'
                    't exist: \n {}'.format(dlc_cfg['init_weights'] +
                                            '.index'))
            if verbose:
                print('Set the init weights as {}'.format(trainingsnapshot))
    else:
        # if you are running this in order you shouldn't get here
        if verbose:
            print('using resnet')
        trainingsnapshot_name = 'snapshot-{}'.format(0)
        trainingsnapshot = 0

        #assert Path(dlc_cfg['init_weights'] + '.index').exists()
    print('snapshot {}'.format(trainingsnapshot))
    return trainingsnapshot_name, trainingsnapshot, dlc_cfg
