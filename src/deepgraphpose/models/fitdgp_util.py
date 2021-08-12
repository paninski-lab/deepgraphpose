import tensorflow as tf
from pathlib import Path
import numpy as np
import tf_slim as slim
from deeplabcut.utils import auxiliaryfunctions
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2
import random

vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf


def dgp_prediction_layer(weight_dlc,
                     bias_dlc,
                     dlc_cfg,
                     inputs,
                     name,
                     num_outputs,
                     init_flag,
                     nc,
                     train_flag,
                     stride=2,
                     kernel_size=[3, 3],
                     scope='block4'):
    """
    nj  : number of joints
    T   : batch size
    :param weight_dlc   : np.ndarray (n x n x nj x nc)
    :param bias_dlc     : np.ndarray  (1 x nj)
    :param dlc_cfg      : easydict.EasyDict
            used to extract variables: weight_decay (float)
    :param inputs       : tf Tensor (T x nx_in x ny_in x nch)
    :param name: str
    :param num_outputs  : nj
    :param init_flag    : bool
            whether to initialize network from weight_dlc and bias_dlc
    :param nc: int
            number of channels
    :param train_flag   : bool
            whether to train the network
    :return:
            pred  : tf.Tensor (T x nx_out, ny_out x nj)
            output of conv netwoek
    """
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(
                            dlc_cfg['weight_decay'])):
        with TF.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            if init_flag:
                pred = slim.conv2d_transpose(
                    inputs,
                    num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    scope=scope,
                    weights_initializer=tf.constant_initializer(weight_dlc[:, :, :, :nc]),
                    biases_initializer=tf.constant_initializer(bias_dlc),
                    trainable=train_flag)
            else:
                pred = slim.conv2d_transpose(inputs,
                                             num_outputs,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             scope=scope,
                                             trainable=train_flag)
            return pred


def find_nan_ind(target_ind, joint_loc):
    """
    :param target_ind: list of #nvisible integers  (nvisible, )
        containts indices of visible markers
    :param joint_loc: nump.ndarray (nvisible x nj x 2)
        2d coordinates of visible integers
    :return:
    nan_id list nnan
        list of nan indices in nvisible integers
    """
    if len(target_ind) == 0:
        return np.empty(0, dtype='int')
    from itertools import chain
    nvisible, nj, _ = joint_loc.shape
    #
    assert len(target_ind) == nvisible
    # not really but none are nan
    nan_ind = []
    for jj in range(nj):
        joint_loc_j = joint_loc[:, jj, 0]  # joint loc is nv x nj x 2
        nan_ind += [list(nj * target_ind[np.isnan(joint_loc_j)] + jj)]

    nan_ind = list(chain(*nan_ind))
    nan_ind = list(np.sort(nan_ind))
    return nan_ind


def find_hidden_markers(hidden_frame, nj, nan_ind):
    """
    Find array with indices of hidden arrays
    include those in nan_ind
    :param hidden_frame:
    :param nj: int
    :param nan_ind: list
    :return:
    """
    if len(hidden_frame) == 0:
        return np.empty(0, dtype='int')
    hidden_marker = hidden_frame * nj
    hidden_marker_list = []
    for i in range(nj):
        hidden_marker_list += [hidden_marker + i]
    hidden_marker = np.sort(np.array(hidden_marker_list).flatten())
    hidden_marker = np.sort(list(hidden_marker) + list(nan_ind))
    return hidden_marker


def find_visible_markers(visible_frame, nj, nan_ind):
    """
    Find array with indices of visible arrays
    excluding those in nan_ind
    :param visible_frame:
    :param nj:
    :param nan_ind:
    :return:
    """
    if len(visible_frame) == 0:
        assert len(nan_ind) == 0
        return np.empty(0, dtype='int'), np.empty(0, dtype='int')

    visible_marker = visible_frame * nj
    visible_frame_list = []
    for i in range(nj):
        visible_frame_list += [visible_marker + i]
    visible_frame0 = np.sort(np.array(visible_frame_list).flatten())
    visible_marker = np.sort(np.setdiff1d(visible_frame0, nan_ind))
    return visible_frame0, visible_marker


def gen_batch(visible_frame_total, hidden_frame_total, all_frame_total, dgp_cfg, maxiters):
    """Generate batch for DGP.
    Parameters
    ----------
    visible_frame_total : list of visible frames in all datasets
    hidden_frame_total :  list of hidden frames in all datasets
    all_frame_total :  list of all frames in all datasets
    dgp_cfg : dict for configuration info
    maxiters : max number of iterations

    Returns
    -------
    batch_ind_all : pre-calculated batch list, each element corresponds to the frames run in one iteration
        the last value in each element (a list) corresponds to the index of the dataset.

    """

    batch_size = dgp_cfg['batch_size']
    n_frames_total = np.sum([len(v) for v in all_frame_total])
    n_datasets = len(all_frame_total)
    nepoch = np.min([int(n_frames_total * dgp_cfg['n_times_all_frames'] /
                                batch_size), maxiters])

    print('nepoch: ', nepoch)
    print('n_datasets: ', n_datasets)

    batch_ind_all = []#np.empty((0, batch_size + 1))
    for i in range(n_datasets):
        index_v_i = visible_frame_total[i]
        index_vh_i = list(all_frame_total[i]) + list(hidden_frame_total[i])
        index_all_i = np.unique(list(index_v_i) + list(index_vh_i))

        batch_size = dgp_cfg['batch_size']
        batchsize_i = max([1, int(nepoch / n_frames_total * len(index_all_i))])

        if len(index_all_i) < batch_size:
            batch_ind = np.random.randint(0, len(index_all_i),
                                          size=batchsize_i)
            batch_size = 1
        else:
            batch_ind = np.random.randint(0,
                                          len(index_all_i) - batch_size,
                                          size=batchsize_i)

        adds = np.linspace(0, batch_size - 1, batch_size)
        batch_ind = batch_ind.reshape(-1, 1) + adds.reshape(1, -1)

        batch_reshape = batch_ind.reshape(-1, )
        batch_ind = index_all_i[batch_reshape.astype(np.int)].reshape(-1, batch_size)
        batch_ind = np.hstack((batch_ind, i * np.ones(
            (batch_ind.shape[0], 1))))

        batch_ind_all += [b.astype(np.int32) for b in batch_ind]

    batch_ind_all = random.sample(batch_ind_all, len(batch_ind_all))

    return batch_ind_all


def get_snapshot_path(snapshot, dlcpath, shuffle=1,trainingsetindex=0):
    """Get the full path for the snapshot.
    Parameters
    ----------
    snapshot : snapshot name, str
    dlcpath : the path for the DLC project, str
    shuffle : shuffle index, int, optional
        default value is 1

    Returns
    -------
    snapshot_path : full path of the snapshot
    config_path : full path of the configuration file

    """

    dlc_base_path = Path(dlcpath)
    config_path = dlc_base_path / 'config.yaml'
    cfg = auxiliaryfunctions.read_config(config_path)
    modelfoldername = auxiliaryfunctions.GetModelFolder(
        cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)

    train_path = dlc_base_path / modelfoldername / 'train'
    snapshot_path = str(train_path / snapshot)
    return snapshot_path, config_path


def combine_all_marker(targets_pred_hidden_marker, targets_visible_marker, hidden_marker_pl, visible_marker_pl, nj, nt_batch_pl):
    """
    Stacks hidden and visible coordinates in tf
    array (nt*nj x 2)

    nt = number of image inputs in batch
    nj = number of targets per image
    nb = nt*nj = number of targets per batch
    nb = nbh + nbv
    nbh = number of unlabeled targets in batch
    nbv = number of labeled targets in batch
    :param targets_pred_hidden_marker: tf (nbh, 2)
    :param targets_visible_marker: tf (nbv, 2)
    :param hidden_marker_pl: tf (nbh, )
    :param visible_marker_pl: tf (nbv, )
    :param nj: in32
    :param nt_batch_pl: int32
    :return:
    targets_all_marker : # (nt*nj x 2)

    """
    indices = TF.reshape(hidden_marker_pl, [-1, 1])
    shape = TF.reshape(nt_batch_pl * nj, [
        1,
    ])

    scatter0 = TF.scatter_nd(indices, targets_pred_hidden_marker[:, 0], shape)
    scatter1 = TF.scatter_nd(indices, targets_pred_hidden_marker[:, 1], shape)
    scatter_mu = TF.transpose(TF.stack([scatter0, scatter1]))

    indices = TF.reshape(visible_marker_pl, [-1, 1])
    shape = TF.reshape(nt_batch_pl * nj, [
        1,
    ])
    scatter0 = TF.scatter_nd(indices, targets_visible_marker[:, 0], shape)
    scatter1 = TF.scatter_nd(indices, targets_visible_marker[:, 1], shape)
    scatter_yv = TF.transpose(TF.stack([scatter0, scatter1]))

    targets_all_marker = scatter_mu + scatter_yv

    return targets_all_marker


def rank(tensor):
    # return the rank of a Tensor
    return len(tensor.get_shape())


# Make Gaussian kernel following SciPy logic
def make_gaussian_2d_kernel(sigma, truncate=1.0, dtype=TF.float32):
    radius = TF.compat.v1.to_int32(sigma * truncate)
    x = TF.cast(TF.range(-radius, radius + 1), dtype=dtype)
    k = TF.exp(-0.5 * TF.square(x / sigma))
    k = k / TF.reduce_sum(k)
    return TF.expand_dims(k, 1) * k


def apply_gaussian_2d_kernel(image, gauss_len, nj):
    # image tensor N x H x W x C
    # gauss_len int

    # N = TF.shape(image)[0]
    # H = TF.shape(image)[1]
    # W = TF.shape(image)[2]
    # C = TF.shape(image)[3]

    #    nj = image.shape.as_list()[-1]

    # make kernel
    kernel = make_gaussian_2d_kernel(gauss_len)
    kernel = TF.tile(kernel[:, :, TF.newaxis, TF.newaxis], [1, 1, nj, 1])

    # apply border replication to handle edge artifact
    # make pad array for image
    padd_gl = TF.cast(TF.reshape([gauss_len, gauss_len], [1, 2]), dtype=TF.int32)
    padd_0 = TF.zeros((1, 2), dtype=TF.int32)
    padd_image = TF.concat((padd_0, padd_gl, padd_gl, padd_0), axis=0)

    image_padded = TF.pad(image, padd_image, "CONSTANT")
    # convolve padded image with kernel
    tensor2 = TF.nn.separable_conv2d(image_padded, kernel, TF.eye(nj, batch_shape=[1, 1]),
                                     strides=[1, 1, 1, 1], padding='VALID')

    return tensor2


def make_2Dgrids(H, W):
    """
    Make 2D grid for soft arg max
    :param H:
    :param W:
    :return:
    """
    xh = TF.cast(TF.range(0, H, dtype=TF.int32), TF.float32)
    xh = TF.reshape(xh, [1, -1])
    xh = TF.transpose(TF.math.multiply(TF.ones([W, 1], dtype=TF.float32), xh))
    xh = TF.expand_dims(xh, 2)

    xw = TF.cast(TF.range(0, W, dtype=TF.int32), TF.float32)
    xw = TF.reshape(xw, [1, -1])
    xw = TF.math.multiply(TF.ones([H, 1], dtype=TF.float32), xw)
    xw = TF.expand_dims(xw, 2)

    image_coords = TF.concat([xh, xw], 2)
    image_coords = TF.expand_dims(image_coords, 2)
    image_coords = TF.cast(image_coords, dtype=TF.float32)

    return image_coords


def argmax_2d_from_cm(tensor, nj, gamma=1, gauss_len=2, th=None):
    """
    Given a tensor  (T x nx_out x ny_out x nj), where apply a
    spatial gaussian smoother along dimension -1, and then
    apply softargmax function
    :param tensor: (T x nx_out x ny_out x nj)
    :param nj: int
    :param gamma: float
    :param gauss_len: float
    :return:
    spatial_soft_argmax: TF.Tensor (T x nj x 2)
    softmax_tensor0: TF.Tensor (T x nx_out x ny_out x nj)
    """
    # input format: BxHxWxD
    assert rank(tensor) == 4

    N = TF.shape(tensor)[0]
    H = TF.shape(tensor)[1]
    W = TF.shape(tensor)[2]
    C = TF.shape(tensor)[3]
    # nj = tensor.shape.as_list()[-1]

    tensor2 = tensor
    # % flatten the Tensor along the height and width axes
    features = TF.transpose(tensor2, [0, 3, 1, 2])
    flat_tensor = TF.reshape(features, (N * C, -1))
    softmax_tensor = TF.nn.softmax(flat_tensor * gamma)

    # Reshape and transpose back to original format.
    softmax_tensor = TF.transpose(TF.reshape(softmax_tensor, [N, C, H, W]), [0, 2, 3, 1])

    # gaussian smoothing
    softmax_tensor = apply_gaussian_2d_kernel(softmax_tensor, gauss_len, nj)
    softmax_tensor_sum = TF.reduce_sum(softmax_tensor, [1, 2])
    softmax_tensor_sum = TF.expand_dims(TF.expand_dims(softmax_tensor_sum, 1), 1)
    softmax_tensor = softmax_tensor / (softmax_tensor_sum + 1e-100)

    if th is not None:
        st = TF.reshape(TF.transpose(softmax_tensor, [0, 3, 1, 2]),
                        [-1, H, W])
        mst = TF.expand_dims(TF.expand_dims(TF.reduce_max(TF.reduce_max(st, 1), 1), 1), 1)

        softmax_tensor_th = TF.where(st < mst * th, TF.zeros_like(st) * 0, st)
        softmax_tensor_th = TF.reshape(softmax_tensor_th, [-1, nj, H, W])
        softmax_tensor = TF.transpose(softmax_tensor_th, [0, 2, 3, 1])
        softmax_tensor_sum = TF.reduce_sum(softmax_tensor, [1, 2])
        softmax_tensor_sum = TF.expand_dims(TF.expand_dims(softmax_tensor_sum, 1), 1)
        softmax_tensor = softmax_tensor / (softmax_tensor_sum + 1e-100)

    softmax_tensor0 = softmax_tensor

    softmax_tensor = TF.expand_dims(softmax_tensor, -1)
    # Convert image coords to shape [H, W, 1, 2]
    image_coords = make_2Dgrids(H, W)

    # Multiply (with broadcasting) and reduce over image dimensions to get the result
    # of shape [N, C, 2]
    spatial_soft_argmax = TF.compat.v1.reduce_sum(softmax_tensor * image_coords, reduction_indices=[1, 2])

    # stack and return 2D coordinates
    return spatial_soft_argmax, softmax_tensor0


def array2list(joint_loc, direct=1):
    if direct == 1:
        joint_loc_list = [list([tuple(f) for f in v.tolist()]) for v in joint_loc]
    else:
        joint_loc_list = np.array(joint_loc)
    return joint_loc_list

def build_aug(apply_prob=0.5):
    sometimes = lambda aug: iaa.Sometimes(apply_prob, aug)
    pipeline = iaa.Sequential(random_order=False)

    pipeline.add(sometimes(iaa.Fliplr(0.5)))
    pipeline.add(sometimes(iaa.Affine(rotate=(-10, 10))))
    # pipeline.add(sometimes(iaa.AllChannelsHistogramEqualization()))
    pipeline.add(sometimes(iaa.MotionBlur(k=3, angle=(-90, 90))))
    pipeline.add(
        sometimes(iaa.CoarseDropout((0, 0.02), size_percent=(0.01, 0.05)))
    )  # , per_channel=0.5)))
    pipeline.add(sometimes(iaa.ElasticTransformation(sigma=5, alpha=(0, 10))))
    pipeline.add(
        sometimes(
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5
            )
        )
    )
    pipeline.add(
        iaa.Sometimes(
            0.4, iaa.CropAndPad(percent=(-0.3, 0.1), keep_size=True)
        )
    )
    return pipeline


def data_aug(all_data_batch, visible_frame_within_batch, joint_loc, pipeline, dgp_cfg):
    visible_data = all_data_batch[visible_frame_within_batch, :, :, :].astype(np.uint8)

    # array to list
    joint_loc_list = array2list(np.flip(joint_loc, 2) * dgp_cfg['stride'] + dgp_cfg['stride'] / 2, direct=1)
    batch_images, batch_joints = pipeline(images=visible_data, keypoints=joint_loc_list)
    # list to array
    joint_loc_aug = np.flip(array2list(batch_joints, direct=-1) / dgp_cfg['stride'] - 0.5, 2)

    all_data_batch_aug = np.copy(all_data_batch)
    all_data_batch_aug[visible_frame_within_batch, :, :, :] = batch_images

    return all_data_batch_aug, joint_loc_aug


def learn_wt(all_data_batch):
    vector_fields = []
    for ff in range(all_data_batch.shape[0] - 1):
        prvs = all_data_batch[ff].astype(np.uint8)
        next1 = all_data_batch[ff + 1].astype(np.uint8)

        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        next1 = cv2.cvtColor(next1, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vector_fields += [np.abs(flow).sum(2)]

    vector_fields = np.array(vector_fields)
    return vector_fields
