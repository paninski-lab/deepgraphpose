#%%
# if we read the targets we extracted what do we get?
#%load_ext autoreload
#%autoreload 2

from pathlib import Path
from os.path import isfile, join, split

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

from PoseDataLoader import DataLoader
from src.deepgraphpose import get_targets  # , get_targets_v1
from src.deepgraphpose import load_dlc_snapshot, get_train_config, get_model_config

#%%
tindex = 5
tseed =1
date = "2031-{:02d}-{:02d}".format(tindex, tseed)
print(date)
#date_names.append(date_name)
#%%
task = 'ibl1'
shuffle = 1

#%%
datasv0 = get_targets(task,date, shuffle)
#datasv1 = get_targets_v1(task,date, shuffle)

#%%
frame_names = datasv0[0]
frame_coord0 = datasv0[1]*8+4 # T x d1 x d2
#frame_coord1 = datasv1[1]*8+4 # T x d1 x d2
#%%
data_info = DataLoader(task)
myclip = VideoFileClip(str(data_info.videofile_path))

#%%
#filename = mydir / ("train_idx_{}_frame_{}.png".format(tf_idx, frame_idx))
mydir = Path(join('data','libraries','deepgraphpose','tests'))

#%%
# read dlc targets
# Load labeled data
task = "ibl1"  # or ibl1 or reach
data_info = DataLoader(task)
shuffle = 1
cfg = get_model_config(task, data_info.model_data_dir, scorer=data_info.scorer, date=date)
# Load pretrained weights
dlc_cfg = get_train_config(cfg, shuffle)
# load training file
trainingsnapshot_name, trainingsnapshot, dlc_cfg = load_dlc_snapshot(dlc_cfg,
                                                                     overwrite_snapshot=49999)

# Load metadata
from deeplabcut.utils import auxiliaryfunctions
import os
trainingsetindex = 0
trainFraction=cfg["TrainingFraction"][trainingsetindex]
trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))

#%%
scale=dlc_cfg['global_scale']
Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')*scale
#%%
num_labels, _ = Data.values.shape
labeled_frames = np.empty(num_labels).astype('int')
for frame_idx in range(num_labels):
    idx_name = int(Path(Data.iloc[frame_idx].name).stem[3:])
    labeled_frames[frame_idx] = idx_name

xys = Data.values.reshape((-1, data_info.nj, 2))
xrlabeled = xys[:,:,0]
yrlabeled = xys[:,:,1]
#yr_labeled =

labeled_frames_train = labeled_frames[trainIndices]
xr_train = xrlabeled[trainIndices,:]
yr_train = yrlabeled[trainIndices,:]
#%%
def marker_distance(xr, yr, xr1, yr1):
    # euclidean distance
    # each T x D
    #T = xr.shape[0]
    dx = (xr- xr1)**2 # T x D
    dy = (yr - yr1)**2 # T x D
    dist = np.sqrt((dx.sum() + dy.sum()))  # T
    return dist # T
#%%
mse_v0 = np.zeros(len(frame_names))
mse_v1 = np.zeros(len(frame_names))
#%%
for idx in range(len(frame_names)):
    #idx = 0
    frame_idx = frame_names[idx]
    print(frame_idx)
    # Load DLC out
    dx_idx = np.argwhere(labeled_frames_train == frame_idx).flatten()[0]
    dxs = xr_train[dx_idx]
    dys = yr_train[dx_idx]

    dxs1 = frame_coord0[idx][:,1]
    dys1 = frame_coord0[idx][:,0]

    #dxs2 = frame_coord1[idx][:,1]
    #dys2 = frame_coord1[idx][:,0]
    #filename = mydir / "compare_frame_{}.png".format(frame_idx)

    mse_v0[idx] = marker_distance(dxs, dys, dxs1, dys1)
    #mse_v1[idx] = marker_distance(dxs, dys, dxs2, dys2)

    #compare_frame(myclip, dxs, dys, dxs1, dys1, dxs2, dys2, frame_idx, store=True, filename=filename, names=['Ref','V0','V1'])

#visualize_frame(myclip, dxs, dys, frame_idx, store=True, filename=filename)

#%%

#%%