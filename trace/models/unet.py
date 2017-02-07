import tensorflow as tf
from .common import *

class Conv2D

class UNet:
    def __init__(self):

        # Define the inputs
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])    # paper: 572
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 1])   # paper: 388

        prev_filter_size = 1

        # First Convolutional Grouping


        # Convolutional Layer 1
        n_f_channels = 64
        filter_size = 3
        filter_shape = [filter_size, filter_size, prev_filter_size, n_f_channels]

        weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[n_f_channels]))

        # Perform a dilated convolution on the previous layer, where the dilation rate is dependent on the
        # number of poolings so far.
        if self.is_valid:
            validity = 'VALID'
        else:
            validity = 'SAME'
        convolution = tf.nn.convolution(prev_layer, self.weights, strides=[1, 1], padding=validity,
                                        dilation_rate=[dilation_rate, dilation_rate])

        # Apply the activation function
        self.activations = self.activation_fn(convolution + self.biases)


