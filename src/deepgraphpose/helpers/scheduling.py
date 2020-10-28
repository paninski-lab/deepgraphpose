import collections
from itertools import product

def default_scheduling(schedule_id, project_path=None):
    """
    create default list of schedules schedule
    """
    if schedule_id == 0:
        schedule = create_schedule(
            {
                "crop": False,
                "crop_ratio": 1.0,
                "global_scale": 1.0,
                "location_refinement": [False],
                "locref_huber_loss": [False],
                "locref_loss_weight": 0.05,
                "multi_step": [[[0.001, 1000]]],
                "display_iters": 100,
                "save_iters": 200,
                "pos_dist_thresh": 8,
                "optimizer": ["sgd", "adam"],
                "scale_jitter_lo": 1,
                "scale_jitter_up": 1,
                "dataset_type": "deterministic",
            }
        )
    elif schedule_id == 1:
        schedule = create_schedule(
            {
                "crop": False,
                "crop_ratio": 1.0,
                "global_scale": 1.0,
                "location_refinement": [True],
                "locref_huber_loss": [True],
                "locref_loss_weight": [0.05, 1],
                "locref_stdev": 7.2801,
                "multi_step": [[[0.001, 1000]]],
                "display_iters": 100,
                "save_iters": 200,
                "pos_dist_thresh": 8,
                "optimizer": ["sgd", "adam"],
                "scale_jitter_lo": 1,
                "scale_jitter_up": 1,
                "dataset_type": "deterministic",
            }
        )
    elif schedule_id == 2:
        schedule = create_schedule(
            {
                "crop": False,
                "crop_ratio": 1.0,
                "global_scale": 1.0,
                "location_refinement": [True],
                "locref_huber_loss": [True],
                "locref_loss_weight": [0.05],
                "locref_stdev": 7.2801,
                "multi_step": [[[0.005, 5000]]],
                "display_iters": 100,
                "save_iters": 200,
                "pos_dist_thresh": 8,
                "optimizer": ["sgd"],
                "scale_jitter_lo": 1,
                "scale_jitter_up": 1,
                "dataset_type": "deterministic",
            }
        )
    elif schedule_id == 3:
        schedule = create_schedule(
            {
                "crop": False,
                "crop_ratio": 1.0,
                "global_scale": 1.0,
                "location_refinement": [True],
                "locref_huber_loss": [True],
                "locref_loss_weight": [0.05],
                "locref_stdev": 7.2801,
                "multi_step": [[[0.001, 50]]],
                "display_iters": 100,
                "save_iters": 200,
                "pos_dist_thresh": 8,
                "optimizer": ["sgd"],
                "scale_jitter_lo": 1,
                "scale_jitter_up": 1,
                "dataset_type": "deterministic",
            }
        )
    return schedule


def create_schedule(param_ranges, verbose=False):
    """
    Create schedule for experiment given dictionary of
    parameters. Each configuration in the schedule is
    a combination of the parameters (keys) and their values
    Inputs:
    _______
    :param param_ranges: dictionary of parameters
        {'param1': range(0, 10, 2), 'param2': 1, ...}
        The value of each key can be an int, float, list or array.
    :param verbose: bool
        Flag to print each configuration in schedule
    :return:
    schedule: list of configuration
        each configuration is an experiment to run
    """
    #Args:
    #    param_ranges: dict

    #Returns:
    #    Schedule containing all possible combinations of passed parameter values.

    param_lists = []

    # for each parameter-range pair ('p': range(x)),
    # create a list of the form [('p', 0), ('p', 1), ..., ('p', x)]
    for param, vals in param_ranges.items():
        if isinstance(vals, str):
            vals = [vals]
        # if a single value is passed for param...
        elif not isinstance(vals, collections.Iterable):
            vals = [vals]
        param_lists.append([(param, v) for v in vals])

    # permute the parameter lists
    schedule = [dict(config) for config in product(*param_lists)]

    #print('Created schedule containing {} configurations.'.format(len(schedule)))
    if verbose:
        for config in schedule:
            print(config)
        print('-----------------------------------------------')

    return schedule
