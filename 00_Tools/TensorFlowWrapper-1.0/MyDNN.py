# -*- coding: utf-8 -*-
import tensorflow as tf

def feed_dict(dataset,label,x_pl,y_pl,ithSong=-1,numFrames=937,StartIdx=-1,EndIdx=-1):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if ithSong == -1:
        xs = dataset
        ys = label
    elif StartIdx == -1:
        StartIdx = ithSong*numFrames
        EndIdx = (ithSong+1)*numFrames
        xs = dataset[StartIdx:EndIdx,:]
        ys = label[StartIdx:EndIdx,:]
    else:
        xs = dataset[StartIdx:EndIdx,:]
        ys = label[StartIdx:EndIdx,:]

    return {x_pl: xs, y_pl: ys}

def weight_variable(shape,initializer=tf.contrib.layers.xavier_initializer()):
    """Create a weight variable with appropriate initialization."""
    # We can't initialize these variables to 0 - the network will get stuck.
    #initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape=shape))

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            #variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            #variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            #tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        #tf.summary.histogram('activations', activations)
        
        return activations

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_12x12(x):
    return tf.nn.max_pool(x, ksize=[1, 12, 12, 1], strides=[1, 12, 12, 1], padding='SAME')

def max_pool_5x12(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 12, 1], strides=[1, 5, 12, 1], padding='SAME')

def max_pool_3x12(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 12, 1], strides=[1, 3, 12, 1], padding='SAME')

def max_pool_1x2(x,layer_name):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME', name=layer_name)

def max_pool_1x3(x,layer_name):
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME', name=layer_name)

def max_pool_1x6(x,layer_name):
    return tf.nn.max_pool(x, ksize=[1, 1, 6, 1], strides=[1, 1, 6, 1], padding='SAME', name=layer_name)

def max_pool_1x8(x,layer_name):
    return tf.nn.max_pool(x, ksize=[1, 1, 8, 1], strides=[1, 1, 8, 1], padding='SAME', name=layer_name)

def max_pool_1x12(x,layer_name):
    return tf.nn.max_pool(x, ksize=[1, 1, 12, 1], strides=[1, 1, 12, 1], padding='SAME', name=layer_name)

def conv2d_layer(input_tensor, patch_width, patch_height, input_channel, output_channel, layer_name, act=tf.nn.relu):
    """
    Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([patch_width, patch_height, input_channel, output_channel])
        with tf.name_scope('biases'):
            biases = bias_variable([output_channel])
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            activations = act(preactivate, name='activation')
    return activations

def deconv2d(x, W, output_shape,strides):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding='SAME')

def deconv2d_layer(input_tensor, patch_width, patch_height, input_channel, output_channel, output_shape, strides, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([patch_width, patch_height, output_channel, input_channel])
        with tf.name_scope('biases'):
            biases = bias_variable([output_channel])
        with tf.name_scope('Wx_plus_b'):
            preactivate = deconv2d(input_tensor, weights, output_shape, strides) + biases
            activations = act(preactivate, name='activation')
    return activations

def batch_norm(x, scope, istrain):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 
                     if scope = fullconnect - a vector             
                     if scope = input or conv2d - 4D Batch-Height-Width-Depth input maps
        scope:       integer, 1 = input, 2 = conv2d, 3 = fullconnect
        istrain:     boolean tf.Varialbe, true indicates training phase
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[1]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[1]),name='gamma', trainable=True)
        if scope == 1:
            batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments_input')
        elif scope == 2:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments_conv2d')
        elif scope == 3:
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments_fullconnect')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(istrain,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def getTotalParameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
    return total_parameters