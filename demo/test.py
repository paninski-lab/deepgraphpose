import tensorflow as tf
import cv2 as cv
import numpy as np
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
    num_points = int(img_lbls.shape[1] / 2)
    points = np.zeros(shape=(num_points, 2))
    for i in range(num_points):
        x = img_lbls[0][i * 2]
        y = img_lbls[0][i * 2 + 1]
        points[i] = np.array([x, y])

    return points

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[0:2]
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1.astype(int), pts2.astype(int)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2

def run_test(dlcpath, shuffle, batch_size, snapshot):
    img1_name = "labeled-data/lBack_bodyCrop/img019942.png"
    img1 = cv.imread("/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img1_name)
    img2_name = "labeled-data/lTop_bodyCrop/img019942.png"
    img2 = cv.imread("/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01/%s" % img2_name)
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    # read the dataframe
    df = pd.read_csv("/Users/sethdonaldson/data/track_graph3d/bird1-selmaan-2030-01-01/training-datasets/iteration-0/UnaugmentedDataSet_bird1Jan1/CollectedData_selmaan.csv")
    print(df.describe())
    col_names = df.loc[df['scorer'] == 'bodyparts']

    # get points
    im1_pts = get_img_points(df, img1_name)
    im2_pts = get_img_points(df, img2_name)

    # Now you can knock yourself out writing the loss function
    # compute fundamental matrix
    # todo: note a minimum of 8 corresponding points are needed
    F, mask = cv.findFundamentalMat(im1_pts, im2_pts)
    print(F)
    # todo: what's going on here?
    # this selects *only* the inlier points. I think this is unnecessary, because all points are guaranteed to be in
    # the visible space of the image plane
    im1_pts = im1_pts[mask.ravel() == 1]
    im2_pts = im2_pts[mask.ravel() == 1]

    # todo: get lines?
    # todo: convert to homogeneous
    ones = np.ones(shape=(im1_pts.shape[0], 1))
    im1_pts_hom = np.hstack((im1_pts, ones))
    im2_pts_hom = np.hstack((im2_pts, ones))

    # equivalent to x^Fx
    z = np.sum(np.dot(im2_pts_hom, F) * im1_pts_hom, axis=1)
    loss = np.linalg.norm(z, 2)
    print(z)
    print(loss)

    # naive x^Fx
    for i, im1_pt in enumerate(im1_pts_hom):
        im2_pt = im2_pts_hom[i]
        z = np.dot(im2_pt.T, np.dot(F, im1_pt))
        print(z)
        # z = np.dot(np.dot(im1_pt, F.T), im2_pt.T)
        # print(z)

    # x(F.T)x^
    lines1_z = np.zeros(shape=(18,3))
    for i, x in enumerate(im2_pts_hom):
        lines1_z[i] = np.dot(F.T, x)
    a = np.sum(im1_pts_hom * lines1_z, axis=1)
    print(a)

    lines2_z = np.zeros(shape=(18, 3))
    for i, x in enumerate(im1_pts_hom):
        lines2_z[i] = np.dot(F, x)


    print("here")

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(im2_pts.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(np.copy(img1), np.copy(img2), lines1, im1_pts, im2_pts)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(im1_pts.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(np.copy(img2), np.copy(img1), lines2, im2_pts, im1_pts)
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
    plt.subplot(121), plt.imshow(img6)
    plt.subplot(122), plt.imshow(img5)
    plt.show()


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
