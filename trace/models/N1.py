import tensorflow as tf

from common import conv2d, bias_variable, weight_variable, max_pool

import numpy as np


def default_N1():
    params = {
        'fc': 500,
        'lr': 1e-3,
        'out': 1
    }
    return N1(params)


class N1:
    def __init__(self, params):

        # Hyperparameters
        fc = params['fc']
        learning_rate = params['lr']
        self.out = params['out']
        self.fov = 95
        self.inpt = 95

        # layer 0
        # Normally would have shape [1, inpt, inpt, 1], but None allows us to have a flexible validation set
        self.image = tf.placeholder(tf.float32, shape=[None, 95, 95, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, 2])
        # layer 1
        W_conv1 = weight_variable([4, 4, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(self.image, W_conv1) + b_conv1)
        # layer 2
        h_pool1 = max_pool(h_conv1, strides=[1, 1], dilation=1)
        # layer 3
        W_conv2 = weight_variable([5, 5, 48, 48])
        b_conv2 = bias_variable([48])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # layer 4
        h_pool2 = max_pool(h_conv2, strides=[1, 1], dilation=1)
        # layer 5
        W_conv3 = weight_variable([4, 4, 48, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        # layer 6
        h_pool3 = max_pool(h_conv3, strides=[1, 1], dilation=1)
        # layer 7
        W_conv4 = weight_variable([4, 4, 48, 48])
        b_conv4 = bias_variable([48])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        # layer 6
        h_pool4 = max_pool(h_conv4, strides=[1, 1], dilation=1)
        # layer 9
        W_fc1 = weight_variable([3 * 3 * 48, 200])
        b_fc1 = bias_variable([200])
        h_pool4_flat = tf.reshape(h_pool4, [-1, 3*3*48])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
        # layer 10
        W_fc2 = weight_variable([200, 2])
        b_fc2 = bias_variable([2])
        pred = tf.matmul(h_fc1, W_fc2) + b_fc2
        self.prediction = pred#tf.reshape(pred, [1,1,1,2])

        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.prediction, self.target))
        self.loss_summary = tf.scalar_summary('cross_entropy', self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        self.binary_prediction = tf.round(self.sigmoid_prediction)
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))
        self.pixel_error_summary = tf.summary.scalar('pixel_error', self.pixel_error)

        self.summary_op = tf.summary.merge([self.loss_summary,
                                       self.pixel_error_summary
                                       ])

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        self.model_name = 'single_layer'










