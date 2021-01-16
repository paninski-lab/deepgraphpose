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

if sys.platform == "darwin":
    import wx

    if int(wx.__version__[0]) > 3:
        wx.Thread_IsMain = wx.IsMainThread

os.environ["DLClight"] = "True"
os.environ["Colab"] = "True"
from deeplabcut.utils import auxiliaryfunctions
from deepgraphpose.models.fitdgp import fit_dlc, fit_dgp, fit_dgp_labeledonly
from deepgraphpose.models.fitdgp_util import get_snapshot_path
from deepgraphpose.models.eval import plot_dgp, estimate_pose
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="causal-gen")

    # dataset
    parser.add_argument("--video_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True
    )
    parser.add_argument("--dot_sizes", type=int, nargs="+", default=3)
    parser.add_argument("--new_hw", type=int, nargs="+")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    video_file_list = args.video_paths
    config = Path(args.config).resolve()
    snapshot = Path(args.snapshot).resolve()
    output_dir = Path(args.output_dir).resolve()
    dot_sizes = args.dot_sizes
    output_dir.mkdir(exist_ok=True, parents=True)
    if len(dot_sizes) == 1:
        dot_sizes *= len(video_file_list)

    for i, video_file in enumerate(video_file_list):
        video_file = Path(video_file)

        estimate_pose(
            proj_cfg_file=config,
            dgp_model_file=snapshot,
            video_file=video_file,
            output_dir=output_dir,
            save_str=video_file.stem,
            new_size=args.new_hw
        )
        plot_dgp(
            video_file=video_file,
            output_dir=output_dir,
            label_dir=output_dir,
            proj_cfg_file=config,
            dgp_model_file=snapshot,
            save_str=video_file.stem,
            dotsize=dot_sizes[i]
        )

