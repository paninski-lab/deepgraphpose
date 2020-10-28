"""
function version of wt+ws model
"""
# %load_ext autoreload
# %autoreload 2
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from PoseDataLoader import DataLoader
from PoseDataLoader import TrainDataLoader as train_loader
from src.deepgraphpose import load_dlc_snapshot, get_train_config, get_model_config
# from graphpose.plotting.plot_model import get_traces_fname, make_annotated_movie
from preprocess.get_morig_resnet_outputs import extract_resnet_output

# %%
task = 'ibl1'
date = '2031-05-01'
shuffle = 1
snapshot = 5000
# %%
data_info = DataLoader(task)
# construct proj name
cfg = get_model_config(task, data_info.model_data_dir, scorer=data_info.scorer, date=date)
dlc_cfg = get_train_config(cfg, shuffle)
# load training file
trainingsnapshot_name, snapshot, dlc_cfg = load_dlc_snapshot(dlc_cfg,
                                                             overwrite_snapshot=snapshot)
# %%
resout_from_file = extract_resnet_output(
    task=task, date=date, shuffle=shuffle, overwrite_snapshot=snapshot)

# %%
debug_key = ''
# Load training data
dataset = train_loader(dlc_cfg, debug_key=debug_key)

# %%
images = dataset.load_train_data()[0]

# %%
train_indices = dataset.train_indices
# %%

np.allclose(resout_from_file[train_indices], images[:, :, :, :200], rtol=1e-04, atol=1e-4)
# %%
# pass_video_through_resnet
#
