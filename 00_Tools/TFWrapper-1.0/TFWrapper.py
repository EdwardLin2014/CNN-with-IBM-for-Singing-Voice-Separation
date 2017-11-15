# -*- coding: utf-8 -*-
import tensorflow as tf

def weight_variable(shape,initializer=tf.contrib.layers.xavier_initializer()):
    """Create a weight variable with appropriate initialization."""
    # We can't initialize these variables to 0 - the network will get stuck.
    return tf.Variable(initializer(shape=shape))

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')
        
        return activations

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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
