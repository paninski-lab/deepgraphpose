from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def generate_log_id(config, method_key='net_type'):
    """
    Produce a string from the experiment-configuration
    Args:
        config: dict containing parameters as keys pointing to their set values
        method_key: how to access the method run
    Returns:
    """
    # log_id += "--".join([f"{key}-{val_str}"] for key, val_str in config.items())
    method = config.get(method_key, 'unknownM')
    # log_id = '%s_%s' % (method, dataset)
    log_id = "{}-{}".format(method_key, method)
    for key, val in iter(sorted(config.items())):
        if key != method_key:
            if isinstance(val, list):
                print('ah')
                #for val_ in val:
                #    if isinstance(val_, list):
                #        for val__ in val_:
                #            val_str = "_".join([f"{v_}" for v_ in val__])
                #    else:
                #       val_str = "_".join([f"{v_}" for v_ in val_])
            elif isinstance(val, str):
                val_str = val
            elif isinstance(val, int):
                val_str = '%d' % val
            elif isinstance(val, float):
                if np.log10(np.abs(val)) >= -5:
                    val_str = '%.5f' % val
                else:
                    val_str = ('%.20f' % val).rstrip('0')
            #elif val is None:
            #    # inherits
            #    val_str = 'None'
            else:
                raise NotImplementedError

            log_id += "--{}-{}".format(key, val_str)
    return log_id
