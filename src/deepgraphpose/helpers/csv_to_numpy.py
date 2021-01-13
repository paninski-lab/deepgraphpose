from pathlib import Path
import argparse
import pandas as pd
import numpy as np

"""Convert CSV representation to numpy array
"""


def get_args():
    parser = argparse.ArgumentParser(description="dgp-csv-to-numpy")

    # dataset
    parser.add_argument("--csv_paths", type=str, nargs="+", required=True, help='Order by views since they will be stacked')
    parser.add_argument("--output_path", type=str, default="./output.npy")
    parser.add_argument("--thresh", type=float, default=0.0)

    return parser.parse_args()


def count_bodyparts(df):
    count = 0
    for col in df.columns:
        if 'x' in col:
            count += 1
    
    return count


if __name__ == "__main__":
    args = get_args()
    csv_file_list = args.csv_paths
    thresh = args.thresh

    multi_x_coords = []
    multi_y_coords = []

    for i, csv_file in enumerate(csv_file_list):
        csv_file = Path(csv_file).resolve()
        df = pd.read_csv(csv_file, skiprows=[0, 1])
        # x, y then x.1, y.1, then x.2, y.2, etc.
        num_bp = count_bodyparts(df)
        x_coords = []
        y_coords = []
        likelihoods = []

        for j in range(num_bp):
            if i == 1:
                j = num_bp - j - 1

            if j == 0:
                index = ''
            else:
                index = '.' + str(j)
        
            x = df['x' + index].values
            y = df['y' + index].values
            lh = df['likelihood' + index].values
            x_coords.append(x)
            y_coords.append(y)
            likelihoods.append(lh)

        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)
        likelihoods = np.asarray(likelihoods)

        nan_indices = likelihoods <= thresh
        x_coords[nan_indices] = np.nan
        y_coords[nan_indices] = np.nan

        multi_x_coords.append(x_coords)
        multi_y_coords.append(y_coords)

    multi_x_coords = np.asarray(multi_x_coords).transpose(0, 2, 1)[:, :, :, np.newaxis]
    multi_y_coords = np.asarray(multi_y_coords).transpose(0, 2, 1)[:, :, :, np.newaxis]

    multiview_arr = np.concatenate((multi_x_coords, multi_y_coords), axis=-1)

    np.save(args.output_path, multiview_arr)
