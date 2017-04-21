import tensorflow as tf
import numpy as np
import math
import models.common as com
import thirdparty.tf_models.spatial_transformer as spat
import utils as uti

import tensorflow.contrib.image as tfim

import scipy.stats as st

debug = False


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def get_weight_var_and_hist(name, shape, dtype=tf.float32, trainable=True):
    w = com.get_weight_variable(name, shape, dtype, trainable)
    hist = tf.summary.histogram(name, w)
    return w, hist


def get_bias_var_and_hist(name, shape, dtype=tf.float32, trainable=True):
    b = com.get_bias_variable(name, shape, dtype, trainable)
    hist = tf.summary.histogram(name, b)
    return b, hist


class Transform_2D(object):
    params_dim = None

    def __call__(self, off_slice, params):
        raise NotImplementedError('Abstract class, the inheritor must implement')


class TranslationTransform(Transform_2D):
    params_dim = 2

    def __init__(self, shift_scale_factor):
        self.__shift_scale_factor = shift_scale_factor

    def __call__(self, off_slice, params):
        print('SCALE')
        print(self.__shift_scale_factor)

        # Scale the shift by the scale factor, so that we limit the amount of translation
        params *= 1 - self.__shift_scale_factor

        print('PARAMS')
        print(params.get_shape())

        out_size = tf.shape(off_slice)[1]

        x_trans = tf.concat([tf.constant([1.0, 0.0]), params[0:1]], axis=0)
        y_trans = tf.concat([tf.constant([0.0, 1.0]), params[1:]], axis=0)

        trans = tf.concat([x_trans, y_trans], axis=0)

        transformed = spat.transformer(off_slice, trans, out_size=(out_size, out_size))

        # Squish down to 2d so that the scan transformation works in the batch aligner
        return tf.reshape(transformed, shape=[out_size, out_size]), trans


class RotationTransform(Transform_2D):
    params_dim = 1

    def __call__(self, off_slice, params):
        # Parameter should be [cos(th), -sin(th), 0, sin(th), cos(th), 0]
        pass


class RigidTransform(Transform_2D):
    params_dim = 3

    def __call__(self, off_slice, params):
        # Parameter should be [cos(th), -sin(th), dx, sin(th), cos(th), dy]
        pass


class AffineTransform(Transform_2D):
    params_dim = 6

    def __call__(self, off_slice, params):
        pass


class LocalizationNetwork(object):
    def __init__(self, in_dim, params_dim, trainable=True):
        self.params_dim = params_dim
        self.trainable = trainable
        self.in_dim = in_dim

    def __call__(self, ref_slice, off_slice):
        raise NotImplementedError('Abstract class, the inheritor must implement')


class Conv2DLocalizationNetwork(LocalizationNetwork):
    def __init__(self, in_dim, params_dim, l1_n, l2_n, l3_n, l4_n, fc1_n, trainable=True):
        super(Conv2DLocalizationNetwork, self).__init__(in_dim, params_dim, trainable)

        with tf.variable_scope('conv_localization'):
            with tf.variable_scope('layer1'):
                self.__w11, h_w11 = get_weight_var_and_hist('w11', [1, 3, 3, 1, l1_n], trainable=trainable)
                self.__w12, h_w12 = get_weight_var_and_hist('w12', [1, 3, 3, l1_n, l1_n], trainable=trainable)
                self.__b1, h_b1 = get_bias_var_and_hist('b1', [l1_n], dtype=tf.float32, trainable=trainable)

                self.__histogram1 = [h_w11, h_w12, h_b1]

                new_dim = self.in_dim / 2 + self.in_dim % 2
                print(new_dim)

            with tf.variable_scope('layer2'):
                self.__w21, h_w21 = get_weight_var_and_hist('w21', [1, 3, 3, l1_n, l2_n], trainable=trainable)
                self.__w22, h_w22 = get_weight_var_and_hist('w22', [1, 3, 3, l2_n, l2_n], trainable=trainable)
                self.__w23, h_w23 = get_weight_var_and_hist('w23', [1, 2, 2, l2_n, l2_n], trainable=trainable)
                self.__b2, h_b2 = get_bias_var_and_hist('b2', [l2_n], trainable=trainable)

                self.__histogram2 = [h_w21, h_w22, h_w23, h_b2]

                new_dim = (new_dim - 2 + 1) / 2 + (new_dim - 2 + 1) % 2
                print(new_dim)

            with tf.variable_scope('layer3'):
                self.__w31, h_w31 = get_weight_var_and_hist('w31', [1, 3, 3, l2_n, l3_n], trainable=trainable)
                self.__w32, h_w32 = get_weight_var_and_hist('w32', [1, 3, 3, l3_n, l3_n], trainable=trainable)
                self.__b3, h_b3 = get_bias_var_and_hist('b3', [l3_n], trainable=trainable)

                self.__histogram3 = [h_w31, h_w32, h_b3]

                new_dim = new_dim / 2 + new_dim % 2
                print(new_dim)

            with tf.variable_scope('layer4'):
                self.__w41, h_w41 = get_weight_var_and_hist('w41', [1, 3, 3, l3_n, l4_n], trainable=trainable)
                self.__w42, h_w42 = get_weight_var_and_hist('w42', [1, 3, 3, l4_n, l4_n], trainable=trainable)
                self.__w43, h_w43 = get_weight_var_and_hist('w43', [1, 2, 2, l4_n, l4_n], trainable=trainable)
                self.__b4, h_b4 = get_bias_var_and_hist('b4', [l4_n], trainable=trainable)

                self.__histogram4 = [h_w41, h_w42, h_w43, h_b4]

                new_dim = new_dim / 2 + new_dim % 2
                print(new_dim)

            with tf.variable_scope('layer_fc1'):
                # Fully connected 1
                flat_dim = new_dim ** 2
                self.__wfc1, h_wfc1 = get_weight_var_and_hist('wfc1', [2 * flat_dim * l4_n, fc1_n], trainable=trainable)
                self.__bfc1, h_bfc1 = get_bias_var_and_hist('bfc1', [fc1_n], trainable=trainable)

                self.__histogram5 = [h_wfc1, h_bfc1]

            with tf.variable_scope('layer_fc2'):
                # Fully connected 2
                self.__wfc2, h_wfc2 = get_weight_var_and_hist('wfc2', [fc1_n, params_dim], trainable=trainable)
                self.__bfc2, h_bfc2 = get_bias_var_and_hist('bfc2', [params_dim], trainable=trainable)

                self.__histogram6 = [h_wfc2, h_bfc2]

        self.saver = tf.train.Saver()

    def __call__(self, ref_slice, off_slice):
        # Stack so that they have the shape [z, y, x], then expand so they have [batch, z, y, x, chan]
        assert (len(ref_slice.get_shape()) == 4)
        assert (len(off_slice.get_shape()) == 4)

        # Stack on the z dim, then expand to batch=1
        stacked = tf.expand_dims(tf.concat([ref_slice, off_slice], axis=0), axis=0)

        # Layer 1
        l11 = tf.nn.convolution(stacked, self.__w11, padding='SAME')
        l12 = tf.nn.convolution(l11, self.__w12, padding='SAME')
        l1 = tf.nn.relu(l12 + self.__b1)

        self.__histogram1.append(tf.summary.histogram('l1', l1))
        histogram1 = tf.summary.merge(self.__histogram1)

        p1 = tf.nn.pool(l1, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p1.get_shape())

        # Layer 2
        l21 = tf.nn.convolution(p1, self.__w21, padding='SAME')
        l22 = tf.nn.convolution(l21, self.__w22, padding='SAME')
        l23 = tf.nn.convolution(l22, self.__w23, padding='VALID')
        l2 = tf.nn.relu(l23 + self.__b2)

        self.__histogram2.append(tf.summary.histogram('l2', l2))
        histogram2 = tf.summary.merge(self.__histogram2)

        p2 = tf.nn.pool(l2, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p2.get_shape())

        # Layer 3
        l31 = tf.nn.convolution(p2, self.__w31, padding='SAME')
        l32 = tf.nn.convolution(l31, self.__w32, padding='SAME')
        l3 = tf.nn.relu(l32 + self.__b3)

        self.__histogram3.append(tf.summary.histogram('l3', l3))
        histogram3 = tf.summary.merge(self.__histogram3)

        p3 = tf.nn.pool(l3, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p3.get_shape())

        # Layer 4
        l41 = tf.nn.convolution(p3, self.__w41, padding='SAME')
        l42 = tf.nn.convolution(l41, self.__w42, padding='SAME')
        l43 = tf.nn.convolution(l42, self.__w43, padding='SAME')
        l4 = tf.nn.relu(l43 + self.__b4)

        self.__histogram4.append(tf.summary.histogram('l4', l4))
        histogram4 = tf.summary.merge(self.__histogram4)

        p4 = tf.nn.pool(l4, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p4.get_shape())

        l4_flat = tf.reshape(p4, [1, -1])

        # Fully connected 1
        fc1 = tf.nn.tanh(tf.matmul(l4_flat, self.__wfc1) + self.__bfc1)

        self.__histogram5.append(tf.summary.histogram('l5', fc1))
        histogram5 = tf.summary.merge(self.__histogram5)

        # Fully connected 2
        theta = tf.nn.tanh(tf.matmul(fc1, self.__wfc2) + self.__bfc2)

        self.__histogram6.append(tf.summary.histogram('theta', theta))
        histogram6 = tf.summary.merge(self.__histogram6)

        histograms = tf.summary.merge([histogram1, histogram2, histogram3, histogram4, histogram5, histogram6])

        # Squish it down to 2d, so that our transformer layer can process it
        return tf.reshape(theta, [-1]), histograms


class FCLocalizationNetwork(LocalizationNetwork):
    def __init__(self, in_dim, params_dim, fc1_units, trainable=True):
        super(FCLocalizationNetwork, self).__init__(in_dim=in_dim, params_dim=params_dim, trainable=trainable)

        with tf.variable_scope('fc_localization'):
            with tf.variable_scope('layer_fc1'):
                # Fully connected 1
                self.__wfc1, h_wfc1 = get_weight_var_and_hist('wfc1', [2 * in_dim * in_dim, fc1_units],
                                                              dtype=tf.float32, trainable=trainable)
                self.__bfc1, b_wfc1 = get_bias_var_and_hist('bfc1', [fc1_units], dtype=tf.float32, trainable=trainable)

                self.__histogram1 = [h_wfc1, b_wfc1]

            with tf.variable_scope('layer_fc2'):
                # Fully connected 2. with bias set to 0
                self.__wfc2, h_wfc2 = get_weight_var_and_hist('wfc2', [fc1_units, params_dim], dtype=tf.float32,
                                                              trainable=trainable)
                self.__bfc2, b_wfc2 = get_bias_var_and_hist('bfc2', shape=[params_dim], dtype=tf.float32,
                                                            trainable=trainable)

                self.__histogram2 = [h_wfc2, b_wfc2]

    def __call__(self, ref_slice, off_slice):
        # make sure they're in the shape [z, y, x, chan]
        assert (len(ref_slice.get_shape()) == 4)
        assert (len(off_slice.get_shape()) == 4)

        # Stack on the z dim, then expand to batch=1
        stacked = tf.concat([ref_slice, off_slice], axis=0)

        # Flatten
        flat = tf.reshape(stacked, [1, -1])

        with tf.variable_scope('fc_localization'):
            with tf.variable_scope('layer_fc1'):
                # Fully connected 1
                fc1 = tf.nn.tanh(tf.matmul(flat, self.__wfc1) + self.__bfc1)

                self.__histogram1.append(tf.summary.histogram('fc1', fc1))
                histogram1 = tf.summary.merge(self.__histogram1)

            with tf.variable_scope('layer_fc2'):
                # Fully connected 2, w/ tanh
                theta = tf.nn.tanh(tf.matmul(fc1, self.__wfc2) + self.__bfc2)

                self.__histogram2.append(tf.summary.histogram('theta', theta))
                histogram2 = tf.summary.merge(self.__histogram2)

        histograms = tf.summary.merge([histogram1, histogram2])

        # Make sure theta is 1D
        return tf.reshape(theta, [-1]), histograms


class SpatialTransformer(object):
    def __call__(self, ref_slice, off_slice):
        raise NotImplementedError('Abstract class, the inheritor must implement.')


class TranslationSpatialTransformer(SpatialTransformer):
    def __init__(self, in_dim, max_shift):
        self.in_dim = in_dim
        self.max_shift = max_shift

        # Reduce the size of the inputs so that there is no 'black region' by 2 * max_shift
        self.reduced_dim = in_dim - 2 * max_shift

        # We introduce a scale factor so that our translation is within a known bound
        scale_factor = float(self.reduced_dim) / in_dim

        # Initialize a TranslationOperation, but not a localization_network
        self.transformer = TranslationTransform(scale_factor)
        self.localization_network = None

    def __call__(self, ref_slice, off_slice):
        # Crop the slices down to their reduced dim

        crop_start = self.max_shift
        crop_end = crop_start + self.reduced_dim

        cropped_ref = ref_slice[:, crop_start:crop_end, crop_start:crop_end, :]
        cropped_off = off_slice[:, crop_start:crop_end, crop_start:crop_end, :]

        # Compute the parameters
        params, self.histograms = self.localization_network(cropped_ref, cropped_off)
        realigned_off, theta = self.transformer(off_slice, params)

        return realigned_off, theta, self.reduced_dim


class FCTranslationSpatialTransformer(TranslationSpatialTransformer):
    def __init__(self, in_dim, fc1_units, max_shift, trainable):
        super(FCTranslationSpatialTransformer, self).__init__(in_dim, max_shift)

        self.localization_network = FCLocalizationNetwork(self.reduced_dim, self.transformer.params_dim, fc1_units,
                                                          trainable)
        self.saver = tf.train.Saver()

        assert (self.localization_network.params_dim == self.transformer.params_dim)


class ConvTranslationSpatialTransformer(TranslationSpatialTransformer):
    def __init__(self, in_dim, l1_n, l2_n, l3_n, l4_n, fc1_n, max_shift, trainable):
        super(ConvTranslationSpatialTransformer, self).__init__(in_dim, max_shift)

        self.localization_network = Conv2DLocalizationNetwork(self.reduced_dim, self.transformer.params_dim,
                                                              l1_n, l2_n, l3_n, l4_n, fc1_n, trainable)
        self.saver = tf.train.Saver()

        assert (self.localization_network.params_dim == self.transformer.params_dim)


class ConvAffineSpatialTransformer(SpatialTransformer):
    def __init__(self, in_dim, shift_bound, is_trainable=True):
        self.in_dim = in_dim

        # l/2 * math.sqrt(2) - 2*max_shift
        self.__crop_size = int(in_dim * (math.sqrt(2) / 2) - 2 * shift_bound)
        self.__crop_start = int((in_dim - self.__crop_size) / 2)

        # Declare all the variables for a simple convolution

        with tf.variable_scope('conv_aff_trans'):
            with tf.variable_scope('layer1'):
                self.__histogram1 = []

                # Layer 1 (724->360)
                self.__w11 = com.get_weight_variable('w11', [2, 3, 3, 1, 24], dtype=tf.float32, trainable=is_trainable)
                self.__w12 = com.get_weight_variable('w12', [2, 3, 3, 24, 24], dtype=tf.float32, trainable=is_trainable)
                self.__b1 = com.get_bias_variable('b1', [24], dtype=tf.float32, trainable=is_trainable)

                self.__histogram1.append(tf.summary.histogram('w11', self.__w11))
                self.__histogram1.append(tf.summary.histogram('w12', self.__w12))
                self.__histogram1.append(tf.summary.histogram('wb1', self.__b1))

                new_dim = self.__crop_size / 2 + self.__crop_size % 2
                print(new_dim)

            with tf.variable_scope('layer2'):
                self.__histogram2 = []

                # Layer 2 (360->178)
                self.__w21 = com.get_weight_variable('w21', [2, 3, 3, 24, 36], dtype=tf.float32, trainable=is_trainable)
                self.__w22 = com.get_weight_variable('w22', [2, 3, 3, 36, 36], dtype=tf.float32, trainable=is_trainable)
                self.__w23 = com.get_weight_variable('w23', [1, 2, 2, 36, 36], dtype=tf.float32, trainable=is_trainable)
                self.__b2 = com.get_bias_variable('b2', [36], dtype=tf.float32, trainable=is_trainable)

                self.__histogram2.append(tf.summary.histogram('w21', self.__w21))
                self.__histogram2.append(tf.summary.histogram('w22', self.__w22))
                self.__histogram2.append(tf.summary.histogram('w23', self.__w23))
                self.__histogram2.append(tf.summary.histogram('b2', self.__b2))

                new_dim = (new_dim - 2 + 1) / 2 + (new_dim - 2 + 1) % 2
                print(new_dim)

            with tf.variable_scope('layer3'):
                self.__histogram3 = []

                # Layer 3 (178->87)
                self.__w31 = com.get_weight_variable('w31', [2, 3, 3, 36, 48], dtype=tf.float32, trainable=is_trainable)
                self.__w32 = com.get_weight_variable('w32', [2, 3, 3, 48, 48], dtype=tf.float32, trainable=is_trainable)
                self.__b3 = com.get_bias_variable('b3', [48], dtype=tf.float32, trainable=is_trainable)

                self.__histogram3.append(tf.summary.histogram('w31', self.__w31))
                self.__histogram3.append(tf.summary.histogram('w32', self.__w32))
                self.__histogram3.append(tf.summary.histogram('b3', self.__b3))

                new_dim = new_dim / 2 + new_dim % 2
                print(new_dim)

            with tf.variable_scope('layer4'):
                self.__histogram4 = []

                # Layer 4 (87->42)
                self.__w41 = com.get_weight_variable('w41', [2, 3, 3, 48, 64], dtype=tf.float32, trainable=is_trainable)
                self.__w42 = com.get_weight_variable('w42', [2, 3, 3, 64, 64], dtype=tf.float32, trainable=is_trainable)
                self.__w43 = com.get_weight_variable('w43', [2, 2, 2, 64, 64], dtype=tf.float32, trainable=is_trainable)
                self.__b4 = com.get_bias_variable('b4', [64], dtype=tf.float32, trainable=is_trainable)

                new_dim = new_dim / 2 + new_dim % 2
                print(new_dim)

                self.__histogram4.append(tf.summary.histogram('w41', self.__w41))
                self.__histogram4.append(tf.summary.histogram('w42', self.__w42))
                self.__histogram4.append(tf.summary.histogram('w43', self.__w43))
                self.__histogram4.append(tf.summary.histogram('wb4', self.__b4))

            with tf.variable_scope('layer_fc1'):
                self.__histogram5 = []

                # Fully connected 1
                self.__wfc1 = com.get_weight_variable('wfc1', [2 * new_dim * new_dim * 64, 512], dtype=tf.float32,
                                                      trainable=is_trainable)
                self.__bfc1 = com.get_bias_variable('bfc1', [512], dtype=tf.float32, trainable=is_trainable)

                self.__histogram5.append(tf.summary.histogram('wfc1', self.__wfc1))
                self.__histogram5.append(tf.summary.histogram('bfc1', self.__bfc1))

            with tf.variable_scope('layer_fc2'):
                self.__histogram6 = []

                # Fully connected 2
                self.__wfc2 = com.get_weight_variable('wfc2', [512, 6], dtype=tf.float32, trainable=is_trainable)

                # Initialize the bias function to the identity
                initial = np.array([[1., 0, 0], [0, 1., 0]])
                initial = initial.astype('float32')
                initial = initial.flatten()
                self.__bfc2 = tf.get_variable('bfc2', initializer=initial, dtype=tf.float32, trainable=is_trainable)
                # self.__bfc2 = tf.Print(self.__bfc2, [self.__bfc2], message='bfc2', summarize=6)

                self.__histogram6.append(tf.summary.histogram('wfc2', self.__wfc2))
                self.__histogram6.append(tf.summary.histogram('bfc2', self.__bfc2))

        self.saver = tf.train.Saver()

    def __call__(self, ref_slice, off_slice):
        # Stack so that they have the shape [z, y, x], then expand so they have [batch, z, y, x, chan]
        assert (len(ref_slice.get_shape()) == 4)
        assert (len(off_slice.get_shape()) == 4)

        # Stack on the z dim, then expand to batch=1
        stacked = tf.expand_dims(tf.concat([ref_slice, off_slice], axis=0), axis=0)

        # Crop down to only the part that we definitely know will be visible

        start = self.__crop_start
        end = self.__crop_start + self.__crop_size
        cropped_for_viewing = stacked[:, :, start:end, start:end, :]

        print(stacked.get_shape())

        print(cropped_for_viewing.get_shape())

        # Layer 1
        l11 = tf.nn.convolution(cropped_for_viewing, self.__w11, padding='SAME')
        l12 = tf.nn.convolution(l11, self.__w12, padding='SAME')
        l1 = tf.nn.relu(l12 + self.__b1)

        self.__histogram1.append(tf.summary.histogram('l1', l1))
        histogram1 = tf.summary.merge(self.__histogram1)

        p1 = tf.nn.pool(l1, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p1.get_shape())

        # Layer 2
        l21 = tf.nn.convolution(p1, self.__w21, padding='SAME')
        l22 = tf.nn.convolution(l21, self.__w22, padding='SAME')
        l23 = tf.nn.convolution(l22, self.__w23, padding='VALID')
        l2 = tf.nn.relu(l23 + self.__b2)

        self.__histogram2.append(tf.summary.histogram('l2', l2))
        histogram2 = tf.summary.merge(self.__histogram2)

        p2 = tf.nn.pool(l2, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p2.get_shape())

        # Layer 3
        l31 = tf.nn.convolution(p2, self.__w31, padding='SAME')
        l32 = tf.nn.convolution(l31, self.__w32, padding='SAME')
        l3 = tf.nn.relu(l32 + self.__b3)

        self.__histogram3.append(tf.summary.histogram('l3', l3))
        histogram3 = tf.summary.merge(self.__histogram3)

        p3 = tf.nn.pool(l3, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p3.get_shape())

        # Layer 4
        l41 = tf.nn.convolution(p3, self.__w41, padding='SAME')
        l42 = tf.nn.convolution(l41, self.__w42, padding='SAME')
        l43 = tf.nn.convolution(l42, self.__w43, padding='SAME')
        l4 = tf.nn.relu(l43 + self.__b4)

        self.__histogram4.append(tf.summary.histogram('l4', l4))
        histogram4 = tf.summary.merge(self.__histogram4)

        p4 = tf.nn.pool(l4, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p4.get_shape())

        l4_flat = tf.reshape(p4, [1, -1])

        # Fully connected 1
        fc1 = tf.nn.sigmoid(tf.matmul(l4_flat, self.__wfc1) + self.__bfc1)

        self.__histogram5.append(tf.summary.histogram('l5', fc1))
        histogram5 = tf.summary.merge(self.__histogram5)

        # Fully connected 2, w/ sigmoid
        theta = tf.nn.sigmoid(tf.matmul(fc1, self.__wfc2) + self.__bfc2)

        self.__histogram6.append(tf.summary.histogram('theta', theta))
        histogram6 = tf.summary.merge(self.__histogram6)

        self.theta = tf.reshape(theta, [-1])

        # theta = tf.Print(theta, [theta], message='theta: ', summarize=6)

        transformed = spat.transformer(off_slice, theta, out_size=(self.in_dim, self.in_dim))

        self.histograms = tf.summary.merge([histogram1, histogram2, histogram3, histogram4, histogram5, histogram6])

        # Squish it down to 2d, so that our transformer layer can process it
        return tf.reshape(transformed, shape=[self.in_dim, self.in_dim])


def transformer_layer(stack, mode='AFFINE', custom_fn=None):
    if mode == 'AFFINE':
        transformer = ConvAffineSpatialTransformer()
        align_apply = transformer
    elif mode == 'CUSTOM':
        align_apply = custom_fn
    else:
        raise ValueError('Mode [%s] is not a valid mode' % mode)

    # We align to the top of the stack, for better or worse
    reference_image = stack[0]
    unaligned = stack[1:]

    # Iterate through the stack and align each image successively to the previous image
    # tf.scan produces
    realigned_stack = tf.scan(align_apply, unaligned, initializer=reference_image)

    complete_stack = tf.concat([tf.expand_dims(reference_image, axis=0), realigned_stack], axis=0)

    return complete_stack


class LocalizationSampler(object):
    def __init__(self, aligned_stack, rotation_aug=True, max_angle=math.pi / 8, translation_aug=True, max_shift=50):
        # 4d tensor [z_dim, y_dim, x_dim, n_chan]
        assert (len(aligned_stack.shape) == 4)
        assert (np.amax(aligned_stack) <= 1.0)

        # Put the stack into a constant
        tf_images = tf.constant(aligned_stack, dtype=tf.float32)

        # Sample 2 adjacent images from the aligned images
        aligned_sample = tf.random_crop(tf_images, size=(2,) + aligned_stack.shape[1:])

        # Pad the image on all 4 sides, so that when we rotate and shift we don't lose any information
        # pad = int((aligned_stack.shape[1] * (math.sqrt(2) - 1)) / 2 + max_shift)
        # padded_sample = tf.pad(aligned_sample, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

        # Break the sample into two slices
        reference_slice = aligned_sample[0:1]
        secondary_slice = aligned_sample[1:]

        # Function to apply a random rotation on inp with angle on interval [min_angle, max_angle]
        def tf_apply_random_rotation(inp, min_angle, max_angle):
            # Define a random angle
            angle = tf.random_uniform(shape=(), minval=min_angle, maxval=max_angle)

            # Rotate the image by that angle
            return tfim.rotate(inp, angle), angle

        # Function to apply a random translation on inp with dx on interval [-x_max_trans, x_max_trans] and dy
        # on interval [-y_max_trans, y_max_trans
        def tf_apply_random_translation(inp, x_max_trans, y_max_trans):
            # Define a random x and y translation
            x_trans = tf.round(tf.random_uniform(shape=(), minval=-x_max_trans, maxval=x_max_trans))
            y_trans = tf.round(tf.random_uniform(shape=(), minval=-y_max_trans, maxval=y_max_trans))

            # If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the output point (x, y)
            # to a transformed input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
            # where k = c0 x + c1 y + 1.
            #
            # From https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform

            # We only want to shift x by x_trans and y by y_trans
            transforms = [1, 0, x_trans, 0, 1, y_trans, 0, 0]

            # Perform the transforms
            return tfim.transform(inp, transforms), transforms

        ref_trans = dict()

        def save_trans(trans_dict, name, output, transform, ):
            trans_dict[name] = transform
            return output

        # APPLY RANDOM AUGMENTATIONS to the reference slice
        rotated_ref = uti.cond_apply(reference_slice,
                                     lambda inp: save_trans(ref_trans, 'rotation',
                                                            *tf_apply_random_rotation(inp, -max_angle, max_angle)),
                                     rotation_aug)

        translated_ref = uti.cond_apply(rotated_ref,
                                        lambda inp: save_trans(ref_trans, 'translation',
                                                               *tf_apply_random_translation(inp, max_shift, max_shift)),
                                        translation_aug)

        # APPLY DIFFERENT RANDOM AUGMENTATIONS to the secondary slice, only save the first output
        rotated_sec = uti.cond_apply(secondary_slice,
                                     lambda inp: tf_apply_random_rotation(inp, -max_angle, max_angle)[0],
                                     rotation_aug)

        translated_sec = uti.cond_apply(rotated_sec,
                                        lambda inp: tf_apply_random_translation(inp, max_shift, max_shift)[0],
                                        translation_aug)

        # APPLY THE SAME AUGMENTATION as the reference slice to the secondary slice, producing trans_lab
        rotated_lab = uti.cond_apply(secondary_slice,
                                     lambda inp: tfim.rotate(inp, ref_trans['rotation']), rotation_aug)

        trans_lab = uti.cond_apply(rotated_lab, lambda inp: tfim.transform(inp, ref_trans['translation']),
                                   translation_aug)

        # The reference slice, randomly transformed
        self.__transformed_reference = translated_ref
        # The secondary slice, randomly transformed
        self.__transformed_secondary = translated_sec
        # The secondary slice, transformed with the same transformations as the reference slice
        self.__secondary_label = trans_lab

    def get_sample_funcs(self):
        return self.__transformed_reference, self.__transformed_secondary, self.__secondary_label

    def sample(self):
        with tf.Session() as sess:
            t_ref, t_sec, sec_l = sess.run(
                [self.__transformed_reference, self.__transformed_secondary, self.__secondary_label])

        return t_ref, t_sec, sec_l


class LocalizationTrainer(object):
    def __init__(self, transformer, ckpt_folder):
        self.transformer = transformer
        self.ckpt_folder = ckpt_folder

    def train(self, sess, n_iter, sampler, lr):

        ref_op, sec_op, true_realign_op = sampler.get_sample_funcs()

        pred_realign, theta, valid_size = self.transformer(ref_op, sec_op)

        pred_realign = tf.expand_dims(tf.expand_dims(pred_realign, axis=0), axis=3)

        pixel_error = tf.reduce_mean(tf.abs(pred_realign - true_realign_op))

        size = tf.shape(pred_realign)[1]
        bound = (size - valid_size) / 2

        true_patch = true_realign_op[:, bound:(bound + valid_size), bound:(bound + valid_size), :]
        pred_patch = pred_realign[:, bound:(bound + valid_size), bound:(bound + valid_size), :]

        pat_as_filter = tf.expand_dims(tf.squeeze(pred_patch, axis=0), axis=3)

        # Smooth pat_as_filter so that the gradient is better... maybe
        # Idea is that xcorrelation is really not smoothly differentiable, so we smooth it out with a gaussian

        gaussian_kernel = np.asarray(np.expand_dims(np.expand_dims(gkern(10, 2), axis=2), axis=3), dtype=np.float32)
        normed_kernel = gaussian_kernel / np.sum(gaussian_kernel)

        smoothed_true = tf.nn.convolution(true_patch, normed_kernel, padding='SAME')

        # Calculate the cross-correlation
        patch_x_correlation = tf.reduce_mean(true_patch * pred_patch)

        patch_smoothed_x_correlation = tf.reduce_mean(smoothed_true * pred_patch)

        conv_x_corr = tf.squeeze(tf.nn.convolution(true_patch, pat_as_filter, padding='VALID')) / float(
            valid_size * valid_size)

        optimizer = tf.train.AdamOptimizer(lr).minimize(-patch_smoothed_x_correlation)

        summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        smoothed_x_cor_sum = tf.summary.scalar('smoothed_x_corr', patch_smoothed_x_correlation)
        x_cor_sum = tf.summary.scalar('cross_correlation', patch_x_correlation)
        pix_err_sum = tf.summary.scalar('pixel_error', pixel_error)
        conv_corr_sum = tf.summary.scalar('conv_cross_corr', conv_x_corr)

        training_summaries = tf.summary.merge([smoothed_x_cor_sum, x_cor_sum, pix_err_sum, conv_corr_sum])

        ref_summary = tf.summary.image('img_reference', ref_op)
        sec_summary = tf.summary.image('img_misaligned', sec_op)
        true_realign_summary = tf.summary.image('realigned_true', true_realign_op)
        pred_realign_summary = tf.summary.image('realigned_pred', pred_realign)

        true_patch_sum = tf.summary.image('patch_true', true_patch)
        pred_patch_sum = tf.summary.image('patch_pred', pred_patch)

        image_summaries = tf.summary.merge(
            [ref_summary, sec_summary, true_realign_summary, pred_realign_summary, true_patch_sum, pred_patch_sum])

        sess.run(tf.initialize_all_variables())

        for i in range(n_iter):
            # Run the optimizer
            sess.run(optimizer)

            print(i)

            # Output to tensorboard
            if i % 10 == 0:
                train_summary_rep, theta_vals = sess.run([training_summaries, theta])
                summary_writer.add_summary(train_summary_rep, i)
                print('theta: ' + ', '.join(["%.6f" % j for j in theta_vals.tolist()]))

            # Image summary
            if i % 100 == 0:
                loss_summary_rep, hist_summaries = sess.run([image_summaries, self.transformer.histograms])
                summary_writer.add_summary(hist_summaries, i)
                summary_writer.add_summary(loss_summary_rep, i)
                print('Computed images and histogram')

            # Model saver
            if i % 1000 == 0:
                self.transformer.saver.save(sess, self.ckpt_folder + 'model.ckpt')
                print('Model saved in %smodel.ckpt' % self.ckpt_folder)
