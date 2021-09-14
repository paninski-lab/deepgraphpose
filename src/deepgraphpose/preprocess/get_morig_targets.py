"""
step 5: get targets
hardcoded version - try to see if we can get the reconstructions to match
# get 2d coordinates for target location for training data of the same size as outputs of convnet
"""

import numpy as np
import os
from os.path import isfile, join, split
from pathlib import Path
import scipy.io as sio
from PoseDataLoader import DataLoader
from src.deepgraphpose import load_dlc_snapshot, get_train_config, get_model_config
from datetime import datetime as dt
import argparse
#%%

def extract_frame_num(x):
    return int(split(x)[-1].split(".")[0][3:])


#%%
def get_targets(task, date, shuffle):
    from deeplabcut.pose_estimation_tensorflow.dataset.factory import (
        create as create_dataset,
    )
    #%%
    #task = 'reach'
    #date = '2020-02-19'
    #shuffle = 0
    #%%
    data_info = DataLoader(task)
    # Load project configuration
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    # Load training configuration for cfg file
    dlc_cfg = get_train_config(cfg, shuffle=shuffle)

    # update
    _,_, dlc_cfg = load_dlc_snapshot(dlc_cfg)
    # %%
    dlc_cfg["deterministic"] = True

    #%% Create dataset
    dataset = create_dataset(dlc_cfg)
    nt = len(dataset.data)  # number of training frames
    assert nt >= 1
    nj = max([dat_.joints[0].shape[0] for dat_ in dataset.data])
    ncolors, nxraw, nyraw = dataset.data[0].im_size

    #%%
    stride = dlc_cfg['stride']

    #%%
    #clip = VideoFileClip(str(dlc_cfg.video_path))

    #%%
    # TO DO: make exact should be around 1/8 for resnet 50
    nx = nxraw//4
    ny = nyraw//4
    #%%
    extract_frame_num = lambda x: int(split(x)[-1].split(".")[0][3:])
    frame_ids = []
    # frame_imgs = []
    counter = 0
    #datas = []
    datas = np.zeros((nt, nx, ny, nj))
    joinss = []
    #%%
    while counter < (nt):
        # for ii in range(nt):
        data = dataset.next_batch()
        data_keys = list(data.keys())

        # inputs = 0
        # part_score_targets = 1
        # part_score_weights = 2
        # locref_targets = 3
        # locref_mask = 4
        # pairwise_targets = 5
        # pairwise_mask = 6
        # data_item = 7

        data_item = data[data_keys[-1]]
        joinss += [np.copy(data_item.joints[0])]

        inputs = data[data_keys[0]].squeeze()
        print(inputs.shape)
        part_score_targets = data[data_keys[1]].squeeze()  # multi class labels
        nx, ny = part_score_targets.shape[0], part_score_targets.shape[1]

        # part_score_weights = data[data_keys[2]].squeeze()
        # locref_targets = data[data_keys[3]].squeeze()
        # locref_mask  = data[data_keys[4]].squeeze()

        frame_id = extract_frame_num(data[data_keys[5]].im_path)
        # print(part_score_targets.max((0, 1)))
        print(frame_id)

        if frame_id in frame_ids:
            continue
        else:
            print("Adding frame {}".format(frame_id))
            frame_ids.append(frame_id)
            datas[counter,:nx,:ny,:] = part_score_targets
            #datas.append(part_score_targets)
            counter += 1

    #%%
    #datas = np.stack(datas, 0)
    datas = datas[:, :nx, :ny, :]
    #nx, ny = datas.shape[1:3]
    frame_ids = np.asarray(frame_ids)
    #%%
    # ignore when tongue is not present?
    # assert datas.sum((1, 2)).min() >= 1
    print(datas.sum((1, 2)).min())

    # %%
    # To find 2D coordinates, we must update
    target2d_train = np.zeros((nt, nj, 2))*np.nan # nt x nj x 2
    for ntt in range(nt):
        cjoin = joinss[ntt] # D x nj
        njtt = cjoin.shape[0]
        for njj_id in range(njtt):
            njj = cjoin[njj_id][0]
            joinss_ntt_njj =  cjoin[njj_id][1:]
            target2d_train[ntt, njj] = np.flip((joinss_ntt_njj-stride/2)/stride)

    # %%
    # visualize some randomly
    # ntt = 23
    # njj = 0
    # center_loc = target2d_train[ntt, njj]

    # plt.imshow(datas[ntt, :, :, njj])
    # plt.plot(center_loc[1], center_loc[0], 'ro')
    # plt.show(block=True)
    # %%
    # target is shared for all snapshots regargless of whathever else

    nx_out = int(part_score_targets.shape[0])
    ny_out = int(part_score_targets.shape[1])
    return frame_ids, target2d_train, nx_out, ny_out, datas
    #%%

    #%%
def store_original_targets(task, date, shuffle, debug_key=''):

    #%%
    #task = 'ibl1'
    #date = '2031-05-01'
    #%%
    data_info = DataLoader(task)
    # Load project configuration
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    # Load training configuration for cfg file
    dlc_cfg = get_train_config(cfg, shuffle=shuffle)

    # update
    _,_, dlc_cfg = load_dlc_snapshot(dlc_cfg)

    #%%
    if debug_key == '':
        frame_ids, target2d_train, nx_out, ny_out, datas = get_targets(task, date, shuffle)

        #elif debug_key == '_v1':
        #frame_ids, target2d_train, nx_out, ny_out, datas = get_targets_v1(task, date, shuffle)
    else:
        raise ('Not implemeneted')
     # %%
    # target is shared for all snapshots regargless of whathever else
    target_outdir = Path(dlc_cfg.snapshot_prefix).parent / "target_mat"
    if not target_outdir.exists():
        os.makedirs(target_outdir)
    print(target_outdir)
    # %%
    ss = os.path.join(str(target_outdir), "targets{}.mat".format(debug_key))
    sio.savemat(
        ss,
        {
            "target_ind": frame_ids,  # ntrain,
            "target2d": target2d_train,  # ntrain x nj x 2
            "nx_out": nx_out,  # int
            "ny_out": ny_out,  # int
            "dlc_outs": datas,
        },
    )
    #%%
    print("Stored targets in:\n{}".format(ss))
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

    parser.add_argument("--shuffle", type=int, default=0, help="Project shuffle")

    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=create_labels, argv=[sys.argv[0]] + unparsed)
    input_params = parser.parse_args()
    store_original_targets(input_params.task, input_params.date, input_params.shuffle)
    #%%


# %%
