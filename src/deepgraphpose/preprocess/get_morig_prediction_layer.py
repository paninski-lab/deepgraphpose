"""
extract weights from pretrained network
"""
import os
# %%
from datetime import datetime as dt
from pathlib import Path

import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim

vers = (tf.__version__).split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf

import argparse

#%%
def store_prediction_layer(task, date, shuffle,overwrite_snapshot=None,allow_growth=True):
    #%%
    from deeplabcut.pose_estimation_tensorflow.dataset.factory import (
        create as create_dataset, )
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
    from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
    #
    from deeplabcut.pose_estimation_tensorflow.train import (
        get_batch_spec,
        setup_preloading,
        start_preloading,
        get_optimizer,
        LearningRate,
    )
    from deepgraphpose.PoseDataLoader import DataLoader
    from deepgraphpose.utils_model import load_dlc_snapshot, get_train_config, \
        get_model_config

    #%%
    data_info = DataLoader(task)

    #%%
    cfg = get_model_config(task, data_info.model_data_dir, scorer=data_info.scorer, date=date)

    #%%
    dlc_cfg = get_train_config(cfg, shuffle)
    trainingsnapshot_name, trainingsnapshot, dlc_cfg = load_dlc_snapshot(dlc_cfg,
                                                                         overwrite_snapshot=overwrite_snapshot)
    #%%
    # Batch is a class filled with indices
    TF.reset_default_graph()
    # create dataset
    dataset = create_dataset(dlc_cfg)
    #%%
    # train: inputs, part_score_targets, part_score_weights, locref_mask
    batch_spec = get_batch_spec(dlc_cfg)
    # queing
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)
    # init graph
    pn = pose_net(dlc_cfg)
    # extracts features, and runs it through a covnet,
    inputs = batch[Batch.inputs]
    net, end_points = pn.extract_features(inputs)
    # net is the input to the conv2d_transpose layer
    heads = pn.prediction_layers(net, end_points)

    #%%
    multi_class_labels = batch[Batch.part_score_targets]
    weigh_part_predictions = dlc_cfg.weigh_part_predictions
    part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0
    #%%
    from deeplabcut.pose_estimation_tensorflow.nnet import losses

    #%%
    def add_part_loss(multi_class_labels, logits, part_score_weights):
        return tf.losses.sigmoid_cross_entropy(multi_class_labels,
                                               logits,
                                               part_score_weights)

    loss = {}
    logits = heads['part_pred']
    loss['part_loss'] = add_part_loss(multi_class_labels,
                                      logits,
                                      part_score_weights)

    total_loss = loss['part_loss']
    if dlc_cfg.intermediate_supervision:
        logits_intermediate = heads['part_loss_interm']
        loss['part_loss_interm'] = add_part_loss(multi_class_labels,
                                                 logits_intermediate,
                                                 part_score_weights)
        total_loss = total_loss + loss['part_loss_interm']

    if dlc_cfg.location_refinement:
        locref_pred = heads['locref']
        locref_targets = batch[Batch.locref_targets]
        locref_weights = batch[Batch.locref_mask]

        loss_func = losses.huber_loss if dlc_cfg.locref_huber_loss else tf.losses.mean_squared_error
        loss['locref_loss'] = dlc_cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
        total_loss = total_loss + loss['locref_loss']

    # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
    loss['total_loss'] = total_loss

    #%%
    for k, t in loss.items():
        TF.summary.scalar(k, t)
    TF.summary.merge_all()

    #%%
    # restore from snapshot
    if trainingsnapshot == 0:
        variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    else:
        variables_to_restore = slim.get_variables_to_restore()
    restorer = TF.train.Saver(variables_to_restore)
    #%% Init session
    config_TF = TF.ConfigProto()
    config_TF.gpu_options.allow_growth = True
    sess = TF.Session(config=config_TF)

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    TF.summary.FileWriter(dlc_cfg.log_dir, sess.graph)
    learning_rate, train_op = get_optimizer(total_loss, dlc_cfg)
    #%%
    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())

    # Restore the one variable from disk
    restorer.restore(sess, dlc_cfg.init_weights)
    print('Restored variables from\n{}\n'.format(dlc_cfg.init_weights))

    #%%
    lr_gen = LearningRate(dlc_cfg)
    #%%
    dlc_params_outdir = Path(dlc_cfg.init_weights).parent / 'dlc_params_mat' / '{}'.format(trainingsnapshot_name)
    if not os.path.isdir(dlc_params_outdir):
        assert Path(dlc_cfg.init_weights).parent
        os.makedirs(dlc_params_outdir)
    print(dlc_params_outdir)

    #%%
    biases = [
        v for v in tf.global_variables()
        if v.name == "pose/part_pred/block4/biases:0"
    ][0]
    weights = [
        v for v in tf.global_variables()
        if v.name == "pose/part_pred/block4/weights:0"
    ][0]

    if dlc_cfg.location_refinement:
        biases_locref = [
            v for v in tf.global_variables()
            if v.name == "pose/locref_pred/block4/biases:0"
        ][0]
        weights_locref = [
            v for v in tf.global_variables()
            if v.name == "pose/locref_pred/block4/weights:0"
        ][0]

    # locref_pred
    #%%
    current_lr = lr_gen.get_lr(0)
    if dlc_cfg.location_refinement:
        [_, biases_out, weights_out, bias_locref_out, weight_locref_out] = sess.run([train_op, biases, weights,
                                                                                     biases_locref, weights_locref],
                                                                                    feed_dict={learning_rate: current_lr})

        ss = os.path.join(dlc_params_outdir, 'dlc_params.mat')
        sio.savemat(ss, {'weight': weights_out,
                         'bias': biases_out,
                         'weight_locref': weight_locref_out,
                         'bias_locref': bias_locref_out})
    else:
        [_, biases_out, weights_out] = sess.run([train_op, biases, weights],
                                                feed_dict={learning_rate: current_lr})

        ss = os.path.join(dlc_params_outdir, 'dlc_params.mat')
        sio.savemat(ss, {'weight': weights_out,
                         'bias': biases_out})
    print('\nStored output in\n{}\n'.format(str(ss)))
    sess.close()
    coord.request_stop()
    return

#%%
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
    store_prediction_layer(input_params.task, input_params.date, input_params.shuffle,
                           input_params.overwrite_snapshot)
    #%%
