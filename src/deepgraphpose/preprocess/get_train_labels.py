"""
STEP 2:
Get training label ids for a specific shuffle
"""
import scipy.io as sio
import numpy as np
from pathlib import Path
from datetime import datetime as dt
from os.path import isfile, join, split

from PoseDataLoader import DataLoader
from src.deepgraphpose import get_model_config, get_train_config

import argparse

#%%
def store_train_labels(task, date, shuffle):
    data_info = DataLoader(task)
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    cfg_train = get_train_config(cfg, shuffle=shuffle)
    #%%
    filename = Path(cfg_train.project_path) / cfg_train.dataset
    assert filename.exists()
    #%%
    # Look up manually labeled datasets
    data = sio.loadmat(filename)
    data = data["dataset"][0]
    # n_frames = len(data)
    train_data = [int(split(dat_[0][0].split(".")[0])[-1][3:]) for dat_ in data]
    train_data = np.sort(train_data)

    assert len(train_data) > 0
    #%%
    val_frame = np.setdiff1d(data_info.get_labeled_frames(), train_data)

    #%%
    frame_index_dir = filename.parent / (filename.stem + "_vframe_indices.npy")
    print(frame_index_dir)
    #%%
    np.save(
        str(frame_index_dir),
        {"train_data": train_data, "val_data": val_frame, "fname": cfg_train.dataset},
    )
    print("\nStored train and validation values in \n{}\n".format(frame_index_dir))
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
    store_train_labels(input_params.task, input_params.date, input_params.shuffle)
    #%%


#%%