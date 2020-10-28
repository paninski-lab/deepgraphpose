import tensorflow as tf

# [] Do we need to make a lock to not integrate across schedules?
def average_gradients(tower_grads):
    """
    Calculate average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks as follows:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_available_gpus():
    """
    Get information from available GPUS
    :return:
    """
    from tensorflow.python.client import device_lib
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU' ]
    if len(gpus) == 0: return ["/cpu:0"]
    tf.logging.info( 'Availble GPUs: {}'.format(', '.join(gpus)) )
    return gpus

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign


def create_parallel_optimization(model_fn, input_fn, optimizer, controller="/cpu:0"):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()

    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                # Compute loss and gradients, but don't apply them yet
                loss = model_fn(input_fn)

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)

    return apply_gradient_op, avg_loss

def create_parallel_optimization_v1(model_fn, input_fn,
                                    optimizer_nn,
                                    optimizer_wn, #optimizer,
                                    global_step_nn,
                                    global_step_wn,
                                    controller="/cpu:0"):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()

    # This list keeps track of the gradients per tower and the losses
    #tower_grads = []
    tower_grads_nn = []
    tower_grads_wn = []
    #losses = []
    losses_nn = []
    losses_wn = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                # Compute loss and gradients, but don't apply them yet
                # loss = model_fn(input_fn)
                # here maybe we need to read nn_var and wn_var from somwhere st they are shared?
                # they will be different for different inputs
                # but the updates should be the same
                total_loss_nn, total_loss_wn, nn_var, wn_var = model_fn(input_fn)

                #with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    #grads = optimizer.compute_gradients(loss)
                    #tower_grads.append(grads)
                # losses.append(loss)
                with tf.name_scope("compute_gradients_nn"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads_nn = optimizer_nn.compute_gradients(total_loss_nn, var_list=nn_var)
                    tower_grads_nn.append(grads_nn)

                with tf.name_scope("compute_gradients_wn"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads_wn = optimizer_wn.compute_gradients(total_loss_wn, var_list=wn_var)
                    tower_grads_wn.append(grads_wn)

                losses_nn.append(total_loss_nn)
                losses_wn.append(total_loss_wn)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients_nn"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.

        gradients_nn = average_gradients(tower_grads_nn)
        #global_step = tf.train.get_or_create_global_step()
        apply_gradient_op_nn = optimizer_nn.apply_gradients(gradients_nn, global_step_nn)
        avg_loss_nn = tf.reduce_mean(losses_nn)

    with tf.name_scope("apply_gradients_wn"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients_wn = average_gradients(tower_grads_wn)
        #global_step = tf.train.get_or_create_global_step()
        apply_gradient_op_wn = optimizer_wn.apply_gradients(gradients_wn, global_step_wn)
        avg_loss_wn = tf.reduce_mean(losses_wn)
    # apply_gradient_op, avg_loss
    return (apply_gradient_op_nn, apply_gradient_op_wn), (avg_loss_nn, avg_loss_wn)


def do_training(update_op, loss):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            step = 0
            while True:
                _, loss_value = sess.run((update_op, loss))
                if step % 100 == 0:
                    print('Step {} with loss {}'.format(step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final loss: {}'.format(loss_value))


def do_training_sch0(update_op, loss):
    # train here we assume only schedule 0
    # we are not doing the variable reloading
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            ii = 0
            while True:
                #
                (apply_gradient_op_nn, apply_gradient_op_wn) = update_op
                (avg_loss_nn, avg_loss_wn) = loss

                # we reload some variables here
                _, loss_value = sess.run((apply_gradient_op_nn, avg_loss_nn))
                if ii % 100 == 0:
                    print('Step {} with loss {}'.format(ii, loss_value))
                ii += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final loss: {}'.format(loss_value))


def parallel_training(model_fn, dataset):
    """
    tf.reset_default_graph()
    parallel_training(training_model, training_dataset(epochs=2))
    :param model_fn:
    :param dataset:
    :return:
    """
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    #optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)

    optimizer_nn = tf.train.AdamOptimizer(learning_rate=0.05)#, use_locking=True)
    optimizer_wn = tf.train.AdamOptimizer(learning_rate=0.05)#, use_locking=True)

    global_step_nn = tf.Variable(0, name='global_step_nn', trainable=False)
    global_step_wn = tf.Variable(0, name='global_step_wn', trainable=False)

    update_op, loss = create_parallel_optimization_v1(model_fn,
                                                   input_fn,
                                                   optimizer_nn,
                                                      optimizer_wn,
                                                      global_step_nn,
                                                      global_step_wn)

    do_training_sch0(update_op, loss)