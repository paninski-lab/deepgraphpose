import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

vers = tf.__version__.split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf


def get_img_points(df, img_name):
    img_lbls = df.loc[df['scorer'] == img_name]
    # drop first column
    img_lbls = img_lbls.drop(columns=['scorer'])
    # convert to list
    img_lbls = img_lbls.to_numpy(dtype=float)
    # get points
    points = []
    for i in range(img_lbls.shape[1] / 2):
        x = img_lbls[0][i * 2]
        y = img_lbls[0][i * 2 + 1]
        points.append((y, x))

    return points

def run_test(dlcpath, shuffle, batch_size, snapshot):
    # get labeled frame from view 1

    # get labeled frame from view 2

    img1_name = "labeled-data/lBack_bodyCrop/img019942.png"
    img1 = cv.imread("/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img1_name)
    img2_name = "labeled-data/lTop_bodyCrop/img019942.png"
    img2 = cv.imread("/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img2_name)
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()

    df = pd.read_csv("/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01/training-datasets/iteration-0/UnaugmentedDataSet_bird1Jan1/CollectedData_selmaan.csv")
    print(df.describe())
    col_names = df.loc[df['scorer'] == 'bodyparts']

    # get points
    im1_pts = get_img_points(df, img1_name)
    im2_pts = get_img_points(df, img2_name)

    # Now you can knock yourself out writing the loss function


    print("here")

    # todo: convert labels to heatmaps (how?)
    # Construct Gaussian targets for all markers
    # target_expand = TF.expand_dims(TF.expand_dims(targets_all_marker, 2), 3)  # (nt*nj) x 2 x 1 x 1
    #
    # # 2d grid of the output
    # alpha_tf = TF.placeholder(tf.float32, shape=[2, None, None], name="2dgrid")
    # alpha_expand = TF.expand_dims(alpha_tf, 0)  # 1 x 2 x nx_out x ny_out
    #
    # # normalize the Gaussian bump for the target so that the peak is 1, nt * nx_out * ny_out * nj
    # targets_gauss = TF.exp(-TF.reduce_sum(TF.square(alpha_expand - target_expand), axis=1) /
    #                        (2 * (dgp_cfg.lengthscale ** 2)))
    # gauss_max = TF.reduce_max(TF.reduce_max(targets_gauss, [1]), [1]) + TF.constant(1e-5, TF.float32)
    # gauss_max = TF.expand_dims(TF.expand_dims(gauss_max, [1]), [2])
    # targets_gauss = targets_gauss / gauss_max
    # targets_gauss = TF.transpose(TF.reshape(targets_gauss, [-1, nj, nx_out, ny_out]), [0, 2, 3, 1])
    #
    # # Separate gauss targets and output pred for visible and hidden markers
    # targets_gauss = TF.reshape(TF.transpose(targets_gauss, [0, 3, 1, 2]), [-1, nx_out, ny_out])
    # targets_gauss_v = TF.gather(targets_gauss
    #                             # , visible_marker_pl
    #                             )  # gauss targets for visible markers
    # pred = TF.reshape(TF.transpose(pred, [0, 3, 1, 2]), [-1, nx_out, ny_out])
    # pred_v = TF.gather(pred,
    #                    # visible_marker_pl
    #                    )  # output pred for visible markers

    print("here")

    # step = 2
    # gm2, gm3 = 1, 3
    # fit_dgp_labeledonly(snapshot,
    #                     dlcpath,
    #                     shuffle=shuffle,
    #                     step=step,
    #                     maxiters=5,
    #                     displayiters=1)

    # fit_dgp(snapshot,
    #         dlcpath,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         step=step,
    #         maxiters=5,
    #         displayiters=1,
    #         gm2=gm2,
    #         gm3=gm3)





if __name__ == '__main__':
    # print(tf.__version__)
    dlcpath = "/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01"
    shuffle = 1
    batch_size = 1
    snapshot = 'snapshot-step0-final--0'
    run_test(dlcpath, shuffle, batch_size, snapshot)
