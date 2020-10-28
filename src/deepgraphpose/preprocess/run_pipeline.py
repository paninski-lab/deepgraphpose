"""
Run all preprocessing steps
"""
import argparse
from datetime import datetime as dt

from preprocess.get_morig_labeled_data import create_labels
from src.deepgraphpose import create_train_sets, add_train_shuffle_from_schedule
from preprocess.get_morig_prediction_layer import store_prediction_layer
from src.deepgraphpose import store_original_targets
from preprocess.get_train_labels import store_train_labels
from src.deepgraphpose import run_original_dlc_shuffle


#%%
def run_model_shuffle(task, date, shuffle_index=1, snapshot_out=None):
    # step 2: get_train_labels call store_train_labels
    store_train_labels(task=task, date=date, shuffle=shuffle_index)
    # step 3: run_model_original call run_original_dlc_shuffle
    run_original_dlc_shuffle(task=task, date=date, shuffle=shuffle_index)
    # step 4: get_model_original_resnet_outputs call store_resnet_output
    # store_resnet_output(task=task, date=date, shuffle=shuffle_index, overwrite_snapshot=snapshot_out)
    # step 5: get_model_original_targets call store_original_targets
    store_original_targets(task=task, date=date, shuffle=shuffle_index)
    # step 6: get_morig_prediction_layer call store_prediction_layer
    store_prediction_layer(task=task, date=date, shuffle=shuffle_index, overwrite_snapshot=snapshot_out)
    return


def run_init_schedules(
        task, date, schedules, training_fraction=0.95, overwrite_flag=False, snapshot_out=None):
    # step 0: create_labeled_data call create_labels
    config_file = create_labels(
        task, date, training_fraction=training_fraction, overwrite_flag=overwrite_flag)
    # step 1: create_training_data call create_train_sets
    shuffle_indices = add_train_shuffle_from_schedule(task=task, date=date, schedules=schedules)
    # run only 1 schedules in config
    for ii, shuffle_index in enumerate(shuffle_indices):
        print('Running schedule {}/{}'.format(ii, len(shuffle_indices)))
        #run_model_shuffle(task, date, shuffle_index, snapshot_out=snapshot_out)
        #shuffle_index = shuffle_indices[0]
        #run_model_shuffle(task,date, shuffle_index, snapshot_out=snapshot_out)
        # step 2: get_train_labels call store_train_labels
        store_train_labels(task=task, date=date, shuffle=shuffle_index)
        # step 3: run_model_original call run_original_dlc_shuffle
        #run_original_dlc_shuffle(task=task, date=date, shuffle=shuffle_index)
        # step 4: get_model_original_resnet_outputs call store_resnet_output
        # store_resnet_output(task=task, date=date, shuffle=shuffle_index, overwrite_snapshot=snapshot_out)
        # step 5: get_model_original_targets call store_original_targets
        store_original_targets(task=task, date=date, shuffle=shuffle_index)
        # step 6: get_morig_prediction_layer call store_prediction_layer
        #store_prediction_layer(task=task, date=date, shuffle=shuffle_index, overwrite_snapshot=snapshot_out)

    print('\nFinished Running init schedules\n')
    return config_file


def run_preprocess(task, date, schedule_id=2, snapshot_out=None):
    # step 0: create_labeled_data call create_labels
    create_labels(task, date)
    # step 1: create_training_data call create_train_sets
    shuffle_indices = create_train_sets(task, date, schedule_id=schedule_id)
    # run only 1 schedules in config
    shuffle_index = shuffle_indices[0]
    run_model_shuffle(task,date, shuffle_index, snapshot_out=snapshot_out)

    print('\nFinished Running preprocess \n')
    return


def run_preprocess_schedules(task, date, schedules, training_fraction=0.95, snapshot_out=None):
    # step 0: create_labeled_data call create_labels
    create_labels(task, date, training_fraction=training_fraction)
    # step 1: create_training_data call create_train_sets
    shuffle_indices = add_train_shuffle_from_schedule(task=task, date=date, schedules=schedules)
    # run only 1 schedules in config
    for ii, shuffle_index in enumerate(shuffle_indices):
        print('Running schedule {}/{}'.format(ii, len(shuffle_indices)))
        run_model_shuffle(task, date, shuffle_index, snapshot_out=snapshot_out)
    print('\nFinished Running preprocess schedules\n')
    return shuffle_indices

#%%
if __name__ == '__main__':
    #%%

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='ibl1',
        help='task to run'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=dt.today().strftime('%Y-%m-%d'),
        help='Project run date'
    )

    parser.add_argument(
        '--schedule_id',
        type=int,
        default=2,
        help='Default scheduled run'
    )
    #%%
    input_params = parser.parse_known_args()[0]
    #%%
    run_preprocess(input_params.task, input_params.date,input_params.schedule_id)

    print('Finished running original model')
    #%%

#%%
