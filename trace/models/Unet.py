"""
U-Net implementation adapted from https://github.com/jakeret/tf_unet
"""

import tensorflow as tf
import numpy as np

from common import conv2d, conv2d_u, deconv2d_u, bias_variable, weight_variable, max_pool, max_pool_u, crop_and_concat

from collections import OrderedDict

def default_Unet():
    params = {
        #'m1': 48,
        #'m2': 48,
        #'m3': 48,
        #'m4': 48,
        #'fc': 200,
        'lr': 0.001,
        'out': 100
    }
    return Unet(params)


class Unet:
    def __init__(self, params):

        # Hyperparameters
        #map_1 = params['m1']
        #map_2 = params['m2']
        #map_3 = params['m3']
        #map_4 = params['m4']
        #fc = params['fc']
        learning_rate = params['lr']
        self.out = params['out']
        self.inpt = 140 # generalize later
        self.fov = self.inpt-self.out #?

        # Unet-specific params
        self.layers = 3
        self.features_root=64
        self.filter_size = 3
        self.channels = 1
        self.keep_prob = .8

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        in_node = self.image
        in_size = self.inpt
        size = in_size

        # DOWN CONV
        for layer in range(0,self.layers):
            features = 2**layer*self.features_root
            stddev = np.sqrt(2 / (self.filter_size**2 * features))
            if layer == 0:
                w1 = weight_variable([self.filter_size, self.filter_size, self.channels, features], stddev)
            else:
                w1 = weight_variable([self.filter_size, self.filter_size, features//2, features], stddev)
            w2 = weight_variable([self.filter_size, self.filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])

            conv1 = conv2d_u(in_node,w1,self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d_u(tmp_h_conv, w2, self.keep_prob)
            dw_h_convs[layer]=tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4

            if layer < self.layers-1:
                pools[layer]= max_pool_u(dw_h_convs[layer])
                in_node = pools[layer]
                size /= 2


        in_node = dw_h_convs[self.layers-1]
        pool_size=2

        # UP CONV
        for layer in range(self.layers-2, -1, -1):
            features = 2**(layer+1)*self.features_root
            stddev = np.sqrt(2 / (self.filter_size**2 * features))

            wd = weight_variable([pool_size, pool_size, features//2, features], stddev)
            bd = bias_variable([features//2])
            h_deconv = tf.nn.relu(deconv2d_u(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([self.filter_size, self.filter_size, features, features//2], stddev)
            w2 = weight_variable([self.filter_size, self.filter_size, features//2, features//2], stddev)
            b1 = bias_variable([features//2])
            b2 = bias_variable([features//2])

            conv1 = conv2d_u(h_deconv_concat, w1, self.keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d_u(h_conv, w2, self.keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

        #FINAL CONV
        weight = weight_variable([1, 1, self.features_root, 2], stddev)
        bias = bias_variable([2])
        conv = conv2d_u(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"]=output_map #changed

        self.prediction = output_map

        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(self.prediction,[-1,2]), tf.reshape(self.target,[-1,2])))
        #self.cross_entropy = -tf.reduce_mean(self.target*tf.log(tf.clip_by_value(pixel_wise_softmax_2(self.prediction),1e-10,1.0)))
        self.loss_summary = tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        self.binary_prediction = tf.round(self.sigmoid_prediction)
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))
        self.pixel_error_summary = tf.summary.scalar('pixel_error', self.pixel_error)
        self.validation_pixel_error_summary = tf.summary.scalar('validation pixel_error', self.pixel_error)

        self.rand_f_score = tf.placeholder(tf.float32)
        self.rand_f_score_merge = tf.placeholder(tf.float32)
        self.rand_f_score_split = tf.placeholder(tf.float32)
        self.vi_f_score = tf.placeholder(tf.float32)
        self.vi_f_score_merge = tf.placeholder(tf.float32)
        self.vi_f_score_split = tf.placeholder(tf.float32)

        self.rand_f_score_summary = tf.summary.scalar('rand f score', self.rand_f_score)
        self.rand_f_score_merge_summary = tf.summary.scalar('rand f merge score', self.rand_f_score_merge)
        self.rand_f_score_split_summary = tf.summary.scalar('rand f split score', self.rand_f_score_split)
        self.vi_f_score_summary = tf.summary.scalar('vi f score', self.vi_f_score)
        self.vi_f_score_merge_summary = tf.summary.scalar('vi f merge score', self.vi_f_score_merge)
        self.vi_f_score_split_summary = tf.summary.scalar('vi f split score', self.vi_f_score_split)

        self.score_summary_op = tf.summary.merge([self.rand_f_score_summary,
                                                 self.rand_f_score_merge_summary,
                                                 self.rand_f_score_split_summary,
                                                 self.vi_f_score_summary,
                                                 self.vi_f_score_merge_summary,
                                                 self.vi_f_score_split_summary
                                                 ])

        self.summary_op = tf.summary.merge([self.loss_summary,
                                       self.pixel_error_summary
                                       ])

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        self.model_name = str.format('out-{}_lr-{}_layers-{}_fr-{}_in-{}_keep-{}', self.out, learning_rate, self.layers,
                                     self.features_root, self.inpt,self.keep_prob)
