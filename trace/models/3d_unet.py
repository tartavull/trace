import tensorflow as tf
import numpy as np

dtype = tf.float32
shape_dict3d = {}
import pprint
import operator


def make_variable(shape, val=0.0):
    initial = tf.constant(val, dtype=dtype, shape=shape)
    var = tf.Variable(initial, dtype=dtype)
    return var


class ConvKernel():
    def transpose(self):
        return TransposeKernel(self)


class TransposeKernel(ConvKernel):
    def __init__(self, k):
        self.kernel = k

    def __call__(self, x):
        return self.kernel.transpose_call(x)

    def transpose(self):
        return self.kernel


class ConvKernel3d(ConvKernel):
    def __init__(self, size=(4, 4, 1), strides=(2, 2, 1), n_lower=1, n_upper=1, stddev=0.5, dtype=dtype):
        initial = tf.truncated_normal([size[0], size[1], size[2], n_lower, n_upper], stddev=stddev, dtype=dtype)
        self.weights = tf.Variable(initial, dtype=dtype)
        self.size = size
        self.strides = [1, strides[0], strides[1], strides[2], 1]
        self.n_lower = n_lower
        self.n_upper = n_upper
        # up_coeff and down_coeff are coefficients meant to keep the magnitude of the output independent of stride and size choices
        self.up_coeff = 1.0 / np.sqrt(reduce(operator.mul, size) * n_lower)
        self.down_coeff = 1.0 / np.sqrt(reduce(operator.mul, size) / (reduce(operator.mul, strides)) * n_upper)

    def transpose(self):
        return TransposeKernel(self)

    def __call__(self, x):
        with tf.name_scope('conv3d') as scope:
            self.in_shape = tf.shape(x)
            tmp = tf.nn.conv3d(x, self.up_coeff * self.weights, strides=self.strides, padding='VALID')
            shape_dict3d[(tuple(tmp._shape_as_list()[1:4]), self.size, tuple(self.strides))] = tuple(
                x._shape_as_list()[1:4])
        return tmp

    def transpose_call(self, x):
        with tf.name_scope('conv3d_t') as scope:
            if not hasattr(self, "in_shape"):
                self.in_shape = shape_dict3d[(tuple(x._shape_as_list()[1:4]), self.size, tuple(self.strides))] + (
                self.n_lower,)
            full_in_shape = (x._shape_as_list()[0],) + self.in_shape
            ret = tf.nn.conv3d_transpose(x, self.down_coeff * self.weights, output_shape=full_in_shape,
                                         strides=self.strides, padding='VALID')
        return tf.reshape(ret, full_in_shape)


sess = tf.Session()
# convolution variables
c1 = ConvKernel3d(size=(4, 4, 1), strides=(2, 2, 1), n_lower=1, n_upper=12)
c2 = ConvKernel3d(size=(4, 4, 1), strides=(2, 2, 1), n_lower=12, n_upper=24)
c3 = ConvKernel3d(size=(4, 4, 4), strides=(2, 2, 2), n_lower=24, n_upper=48)
c1t = ConvKernel3d(size=(4, 4, 1), strides=(2, 2, 1), n_lower=1, n_upper=12).transpose()
c2t = ConvKernel3d(size=(4, 4, 1), strides=(2, 2, 1), n_lower=12, n_upper=24).transpose()
c3t = ConvKernel3d(size=(4, 4, 4), strides=(2, 2, 2), n_lower=24, n_upper=48).transpose()
# bias variables
b0 = make_variable([1, 1, 1, 1, 1])
b1 = make_variable([1, 1, 1, 1, 12])
b2 = make_variable([1, 1, 1, 1, 24])
b3 = make_variable([1, 1, 1, 1, 48])
b0t = make_variable([1, 1, 1, 1, 1])
b1t = make_variable([1, 1, 1, 1, 12])
b2t = make_variable([1, 1, 1, 1, 24])
# Tensorflow convention is to represent volumes as (batch, x, y, z, channel)
inpt = tf.ones((1, 128, 128, 32, 1))
# Purely linear network, no skip connections
downsampled = c3(c2(c1(inpt)))
print
"Coarsest layer dimensions:", downsampled._shape_as_list()
upsampled = c1t(c2t(c3t(downsampled)))
print
"Output dimensions:", upsampled._shape_as_list()
otpt1 = upsampled
# Basic U-net with relu non-linearities and skip connections
l0 = inpt
l1 = tf.nn.relu(c1(l0) + b1)
l2 = tf.nn.relu(c2(l1) + b2)
l3 = tf.nn.relu(c3(l2) + b3)
l3t = l3
l2t = tf.nn.relu(c3t(l3t) + l2 + b2t)
l1t = tf.nn.relu(c2t(l2t) + l1 + b1t)
l0t = tf.nn.relu(c1t(l1t) + l0 + b0t)
otpt2 = l0t
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print
"Running linear network"
print
sess.run(otpt1).shape
print
"Running u-net"
print
sess.run(otpt2).shape
