# even as we run some test code we need to pass all frames through the resnet
# we will store the video in chunks and load resnet
#%load_ext autoreload
#%autoreload 2
#%%
import os
from datetime import datetime as dt
import argparse

import numpy as np
import tensorflow as tf
from PoseDataLoader import DataLoader, TrainDataLoader as train_loader
from src.deepgraphpose import load_dlc_snapshot, get_train_config, get_model_config
from moviepy.editor import VideoFileClip
import  h5py
# %%
vers = (tf.__version__).split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# %%
def get_size_training_chunk(task,date, shuffle, snapshot=4999,chunk = 0, nc=200, ns=10
):
    #task = 'ibl3'
    #date = '2020-03-16'
    # ns = 10
    # nc = 200
    # %%
    #shuffle = 0
    data_info = DataLoader(task)
    #snapshot = 4999  # set to snapshot # to overwrite snapshot read
    #chunk_size = 1000  # distribute among gpus
    #num_chunks = 1
    cfg = get_model_config(task, data_info.model_data_dir, scorer=data_info.scorer, date=date)
    dlc_cfg = get_train_config(cfg, shuffle)
    trainingsnapshot_name, snapshot, dlc_cfg = load_dlc_snapshot(dlc_cfg,
                                                                 overwrite_snapshot=snapshot)
    # %% Load training data

    #%%
    dataset = train_loader(dlc_cfg)
    # %%
    outputdir = dataset.resnet_out_chunk_path / ('ns{}nc{}'.format(ns, nc))
    # %%
    if not outputdir.exists() or (len(os.listdir(str(outputdir))) == 0):
        raise FileExistsError('\n\nRun training process\n')
        #store_training_features_v1(dlc_cfg, ns=ns, num_chunks=num_chunks,chunk_size=chunk_size, nc=nc)
    # %%
    print('\n Reading stored resnet outputs in {}'.format(outputdir))
    # load a single chunk
    #num_chunks = len(os.listdir(outputdir))
    resnet_out_chunk = dataset.load_resnet_outs_chunk(chunk=chunk, ns=ns, nc=nc)[0]
    #%%
    # test frame is 0
    nt_chunk = resnet_out_chunk.shape[0]
    return nt_chunk


def store_test_resnet_output_chunks(dlc_cfg, nc=200, chunk_size=1000,allow_growth= True,debug_key=""):
    # debug_key = "nt_{}".format(nt_chunk)
    #
    #%%
    from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
    import tensorflow.contrib.slim as slim
    from tqdm import tqdm
    from skimage.util import img_as_ubyte
    from PoseDataLoader import TestDataLoader

    #%%
    clip = VideoFileClip(str(dlc_cfg.video_path))
    ny_raw, nx_raw = clip.size
    fps = clip.fps
    #%%
    nframes = clip.duration * clip.fps
    nframes_fsec = nframes - int(nframes)
    #%%
    if (nframes_fsec < 1 / clip.fps):
        nframes = np.floor(nframes).astype('int')
    else:
        nframes = np.ceil(nframes).astype('int')
        print('Warning. Check the number of frames')
    #%%
    # Build graph to pass frames through resnet
    TF.reset_default_graph()
    inputs = TF.placeholder(tf.float32, shape=[1, nx_raw, ny_raw, 3])
    pn = pose_net(dlc_cfg)
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
    # %%

    num_chunks_tvideo = int(np.ceil(nframes/chunk_size))
    print('Video is split in {} resnet_out files'.format(num_chunks_tvideo))

    #%% Make test dataset
    test_data = TestDataLoader(dlc_cfg, debug_key=debug_key)
    #%%
    if not test_data.video_data_chunks_dir.exists():
        os.makedirs(test_data.video_data_chunks_dir)

    if not test_data.resnet_output_chunks_dir.exists():
        os.makedirs(test_data.resnet_output_chunks_dir)

    #%%
    for chunk_id, video_start in enumerate(np.arange(0, nframes, chunk_size)):
        video_end = min(video_start + chunk_size, nframes)

        nvideoframes = video_end - video_start

        #%% Make movie file:
        start_sec = np.round(video_start/fps, 5)
        end_sec = np.round(video_end/fps, 5)
        bonus =(nvideoframes - (end_sec - start_sec)*fps)/fps
        if bonus<0:
            end_sec += 2*bonus

        mini_clip = clip.subclip(t_start=start_sec, t_end=end_sec)
        n_frames = sum(1 for x in mini_clip.iter_frames())
        if not(n_frames == nvideoframes):
            raise Exception('what for {}'.format(chunk_id))

        video_fname = test_data.get_video_data_chunks_fname(chunk_id)
        mini_clip.write_videofile(str(video_fname))
        #print('Wrote file:\n {}'.format(video_fname))
        #%%
        # Make resnet output file:
        indices = np.arange(video_start, video_end)
        resnet_outputs = np.zeros(
            (nvideoframes, nx_out, ny_out, nchannels), dtype="float32"
        )
        pbar = tqdm(total=nvideoframes, desc='Pass through resnet chunk {}'.format(chunk_id))
        step = nvideoframes // 3
        for counter, index in enumerate(indices):
            ff = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
            [net_output] = sess.run([net], feed_dict={inputs: ff[None, :, :, :]})
            if isinstance(nc, int):
                resnet_outputs[counter, :, :] = net_output[0, :, :, :nc]
            elif isinstance(nc, np.ndarray):
                resnet_outputs[counter, :, :] = net_output[0, :, :, nc]
            else:
                raise Exception('Not proper resnet channel selection')

            if (counter % step == 0) or (counter == nvideoframes):
                pbar.update(min(counter,step))
        pbar.close()

        resnet_fname = test_data.get_resnet_output_chunks_fname(chunk_id)
        with h5py.File(str(resnet_fname), 'w') as f:
            f.create_dataset("resnet_out", data=resnet_outputs)
            #f.create_dataset("resnet_idx", data=frames_in_chunk)
            #f.create_dataset("pv", data=pv_chunk)
            #f.create_dataset("ph", data=ph_chunk)
            f.create_dataset("start", data=video_start)
            f.create_dataset("stop", data=video_end)

    #print('Stored resnet output in:\n{}'.format(
    #    chunk_id, str(image_path)))

    return


def store_test_resnet_outputs(task,
                              date,
                              shuffle,
                              snapshot=None,
                              nc=200,
                              chunk_size=1000,
                              allow_growth=True,
                              debug_key=''):
    """
    Store resnet outputs for test data
    """
    #snapshot = 4999  # set to snapshot # to overwrite snapshot read
    #chunk_size = 1000  # distribute among gpus
    #%%
    from PoseDataLoader import DataLoader
    from src.deepgraphpose import load_dlc_snapshot, get_train_config, get_model_config

    # %%
    data_info = DataLoader(task)

    cfg = get_model_config(task,
                           data_info.model_data_dir,
                           scorer=data_info.scorer,
                           date=date)
    dlc_cfg = get_train_config(cfg, shuffle)
    trainingsnapshot_name, snapshot, dlc_cfg = load_dlc_snapshot(
        dlc_cfg, overwrite_snapshot=snapshot)
    # %%
    store_test_resnet_output_chunks(dlc_cfg,
                               nc=nc,
                               chunk_size=chunk_size,
                               allow_growth=allow_growth,
                               debug_key=debug_key)

    return



if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        type=str,
                        default="ibl1",
                        help="task for run")
    parser.add_argument(
        "--date",
        type=str,
        default=dt.today().strftime("%Y-%m-%d"),
        help="Project run date",
    )
    parser.add_argument(
        "--debug_key",
        type=str,
        default="",
        help="debug key",
    )
    parser.add_argument("--shuffle",
                        type=int,
                        default=0,
                        help="Project shuffle")

    parser.add_argument(
        "--snapshot",
        type=int,
        default=None,
        help=" set to snapshot # to overwrite snapshot read",
    )

    parser.add_argument("--nc",
                        type=int,
                        default=200,
                        help="channels to extract from resnet")

    parser.add_argument("--chunk_size",
                        type=int,
                        default=1000,
                        help="size of chunks")

    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=create_labels, argv=[sys.argv[0]] + unparsed)
    input_params = parser.parse_known_args()[0]
    store_test_resnet_outputs(
        task=input_params.task,
        date=input_params.date,
        shuffle=input_params.shuffle,
        snapshot=input_params.snapshot,
        nc=input_params.nc,
        chunk_size=input_params.chunk_size,
        debug_key=input_params.debug_key,
    )

    #%%