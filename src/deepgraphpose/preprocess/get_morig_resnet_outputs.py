"""
step 4 get_resnet outputs
"""
import numpy as np
import os
from pathlib import Path
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from skimage.util import img_as_ubyte
import scipy.io as sio
from PoseDataLoader import DataLoader
from src.deepgraphpose import load_dlc_snapshot, get_train_config, get_model_config
import tensorflow.contrib.slim as slim
import tensorflow as tf
from datetime import datetime as dt
import argparse
# TODO: error in nframes max for ibl3, check 147th chunk for 1000-long chunks
#%%
vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf

#%%


def load_resnet_files(resnet_out_path, indices,nc=200):
    images = []
    for im_id in indices:
        image_path = resnet_out_path / 'resnet_output_{:03d}.mat'.format(
            im_id)
        image = sio.loadmat(str(image_path))['net_output'].squeeze()
        if nc is None:
            # load all channels
            images.append(image)
        elif isinstance(nc, int):
            # load top nc channels
            assert np.ndim(image) == 3
            images.append(image[:, :, :nc])
        elif isinstance(nc, np.ndarray):
            # load nc specified channels
            assert np.ndim(nc) == 1
            images.append(image[:, :, nc])

    images = np.array(images).squeeze()

    return images


def get_resnet_outsize(videofile_path, dlc_cfg):
    from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
    #%%
    clip = VideoFileClip(str(videofile_path))
    #%%
    ny_in, nx_in = clip.size

    # %%
    TF.reset_default_graph()
    inputs = TF.placeholder(tf.float32, shape=[1, nx_in, ny_in, 3])
    pn = pose_net(dlc_cfg)
    # extract resnet outputs
    net, end_points = pn.extract_features(inputs)
    nx_out, ny_out = net.shape.as_list()[1:3]
    return nx_out, ny_out


def pass_video_through_resnet(videofile_path, dlc_cfg, allow_growth=True, indices=None, nc=None,
                              step_pbar=500):
    """
    Pass frames through nextwork
    can pass selected frames
    :param videofile_path:
    :param dlc_cfg:
    :param allow_growth:
    :param indices:
    :param nc:
    :return:
    """
    #%%
    from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

    clip = VideoFileClip(str(videofile_path))
    #%%
    ny_in, nx_in = clip.size
    if indices is None:
        #%%
        nframes = clip.duration*clip.fps
        nframes_fsec = nframes - int(nframes)
        if (nframes_fsec < 1/clip.fps):
            nframes = np.floor(nframes).astype('int')
        else:
            nframes = np.ceil(nframes).astype('int')
            print('Warning. Check the number of frames')
            # raise Exception('You shouldn''t be here. Check video reader')
        indices = np.arange(nframes)
    else:
        nframes = len(indices)

    # %%
    TF.reset_default_graph()
    inputs = TF.placeholder(tf.float32, shape=[1, nx_in, ny_in, 3])
    pn = pose_net(dlc_cfg)
    # extract resnet outputs
    net, end_points = pn.extract_features(inputs)
    # heads = pn.prediction_layers(net, end_points)
    # %%
    # restore from snapshot
    if 'snapshot' in dlc_cfg.init_weights:
        variables_to_restore = slim.get_variables_to_restore()
    else:
        variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])

    restorer = TF.train.Saver(variables_to_restore)

    # Init session
    config_TF = TF.ConfigProto()
    config_TF.gpu_options.allow_growth = allow_growth
    sess = TF.Session(config=config_TF)

    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # %%
    # Restore the one variable from disk
    restorer.restore(sess, dlc_cfg.init_weights)

    #%%
    nx_out, ny_out = net.shape.as_list()[1:3]
    #%%
    if nc is None:
        # load all channels
        nchannels = 2048
        nc = 2048
    elif isinstance(nc, int):
        nchannels = nc
    elif isinstance(nc, np.ndarray):
        nchannels = len(nc)
    else:
        raise Exception('Check nc')

    # Here we fix a large #
    resnet_outputs = np.zeros(
        (nframes, nx_out, ny_out, nchannels), dtype="float32"
    )

    #%%
    pbar = tqdm(total=nframes, desc='Read video frames')
    step = int(max(step_pbar, nframes // 3))

    for counter, index in enumerate(indices):
        ff = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
        [net_output] = sess.run([net], feed_dict={inputs: ff[None, :, :, :]}) # 1 x nx_out x ny_out x nchannels
        if isinstance(nc, int):
            resnet_outputs[counter, :, :] = net_output[0, :, :, :nc]
        elif isinstance(nc, np.ndarray):
            resnet_outputs[counter, :, :] = net_output[0, :, :, nc]
        else:
            raise Exception('Not proper resnet channel selection')

        if (counter % step == 0) or (counter == nframes -1):
            pbar.update(min(counter,  step))

    pbar.close()
    #%%
    assert counter == (nframes-1)
    clip.close()
    sess.close()
    #%%
    return resnet_outputs


def extract_resnet_output(
            task, date, shuffle, overwrite_snapshot=None, allow_growth=True,
            videofile_path=None, indices=None, nc=200):
    #%%
    if isinstance(videofile_path, str):
        videofile_path = Path(videofile_path)
    else:
        pass
        #assert isinstance(videofile_path, Path)
    #%%
    data_info = DataLoader(task)
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    #%%
    dlc_cfg = get_train_config(cfg, shuffle)
    # dlc_cfg_init = edict(dlc_cfg)
    #%%
    trainingsnapshot_name, trainingsnapshot, dlc_cfg = load_dlc_snapshot(
        dlc_cfg, overwrite_snapshot=overwrite_snapshot
    )
    if not (overwrite_snapshot is None):
        assert trainingsnapshot == overwrite_snapshot
    # dlc_cfg_init = edict(dlc_cfg)
    #%%# Update dlc_cfg files just to init network
    dlc_cfg["batch_size"] = 1
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)
    dlc_cfg["deterministic"] = True

    # %%
    # Load data
    if videofile_path is None:
        videofile_path = Path(data_info.videofile_path)

    resnet_outputs = pass_video_through_resnet(videofile_path, dlc_cfg, allow_growth=allow_growth, indices=indices, nc=nc)

    return resnet_outputs
    #%%


def store_resnet_output(
        task, date, shuffle, overwrite_snapshot=None, allow_growth=True,
        videofile_path=None, resnet_output_dir=None):
    from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

    """
    task = 'ibl1'
    date = '2020-01-25'
    shuffle = 1
    overwrite_snapshot = 5000
    ibl_chunk_path = '/data/libraries/deepgraphpose/etc/lab_ekb/debug_va_semi_pipeline/run_long_video_aqweight/movies'
    videofile_path = Path(ibl_chunk_path) / 'movie_chunk_00.mp4'
    resnet_output_dir = videofile_path.parent / videofile_path.stem
    allow_growth = True
    """
    #%%
    if isinstance(videofile_path, str):
        videofile_path = Path(videofile_path)
    else:
        pass
        # assert isinstance(videofile_path, Path)
    if isinstance(resnet_output_dir, str):
        resnet_output_dir = Path(resnet_output_dir)
    else:
        pass
        #assert isinstance(resnet_output_dir, Path)

    #%%
    data_info = DataLoader(task)
    cfg = get_model_config(
        task, data_info.model_data_dir, scorer=data_info.scorer, date=date
    )
    #%%
    dlc_cfg = get_train_config(cfg, shuffle)
    # dlc_cfg_init = edict(dlc_cfg)
    #%%
    trainingsnapshot_name, trainingsnapshot, dlc_cfg = load_dlc_snapshot(
        dlc_cfg, overwrite_snapshot=overwrite_snapshot
    )
    if not overwrite_snapshot == None:
        assert trainingsnapshot == overwrite_snapshot
    # dlc_cfg_init = edict(dlc_cfg)
    #%% Update dlc_cfg files just to init network
    dlc_cfg["batch_size"] = 1
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)
    dlc_cfg["deterministic"] = True
    # %%
    # Load data
    if videofile_path is None:
        videofile_path = Path(data_info.videofile_path)

    #%%

    cap = cv2.VideoCapture(str(videofile_path))
    nframes = int(cap.get(7))
    #%%
    # We want to pass all frames through network
    nx_in, ny_in = int(cap.get(4)), int(cap.get(3))

    frames = np.zeros(
        (nframes, nx_in, ny_in, 3), dtype="ubyte"
    )  # this keeps all frames in a batch
    pbar = tqdm(total=nframes)
    counter = 0
    step = nframes // 3 #max(10, int(nframes / 100))

    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[counter] = img_as_ubyte(frame)
            counter = counter + 1
        else:
            print("the end")
            break
    pbar.close()
    # read all frames
    assert counter == nframes

    # %%
    TF.reset_default_graph()
    inputs = TF.placeholder(tf.float32, shape=[1, nx_in, ny_in, 3])
    pn = pose_net(dlc_cfg)
    # extract freatures using resnet
    net, end_points = pn.extract_features(inputs)
    # heads = pn.prediction_layers(net, end_points)
    # %%
    # always restore from snapshot do not restore from IBL
    if trainingsnapshot == 0:
        variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    else:
        variables_to_restore = slim.get_variables_to_restore()
    restorer = TF.train.Saver(variables_to_restore)

    # Init session
    config_TF = TF.ConfigProto()
    config_TF.gpu_options.allow_growth = allow_growth
    sess = TF.Session(config=config_TF)

    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # %%
    # Restore the one variable from disk
    restorer.restore(sess, dlc_cfg.init_weights)

    #%%
    if resnet_output_dir is None:
        resnet_output_dir = Path(dlc_cfg.init_weights).parent
        print(resnet_output_dir)
    #%%
    resnet_outdir = resnet_output_dir / "resnet_output_mat" / ("{}".format(trainingsnapshot_name))
    if not resnet_outdir.exists():
        os.makedirs(resnet_outdir)
    #%%
    for ii in range(nframes):
        if ii % 10 == 0:
            print("iter {}/{}".format(ii, nframes))
        ff = frames[ii, :, :, :]
        ff = np.expand_dims(ff, axis=0)
        [net_output] = sess.run([net], feed_dict={inputs: ff})
        # net_heads = sess.run(heads, feed_dict={inputs: ff})
        ss = resnet_outdir / ("resnet_output_{:03d}.mat".format(ii))
        sio.savemat(str(ss), {"net_output": net_output})
    print("Stored resnet outputs in:\n{}".format(resnet_outdir))
    #%%
    sess.close()
    return


if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ibl1", help="task for run")
    parser.add_argument(
        "--date",
        type=str,
        default=dt.today().strftime("%Y-%m-%d"),
        help="Project run date",
    )

    parser.add_argument("--shuffle", type=int, default=0, help="Project shuffle")

    parser.add_argument(
        "--overwrite_snapshot",
        type=int,
        default=None,
        help=" set to snapshot # to overwrite snapshot read",
    )
    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=create_labels, argv=[sys.argv[0]] + unparsed)
    input_params = parser.parse_args()
    store_resnet_output(
        input_params.task,
        input_params.date,
        input_params.shuffle,
        input_params.overwrite_snapshot,
    )
    #%%


# %%
