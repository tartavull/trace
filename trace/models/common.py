import tensorflow as tf


def weight_variable(shape):
  """
  Xavier initialization
  """
  return tf.Variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate=[dilation, dilation])


def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')


class Layer(object):
    depth = 0
    def __init__(self, filter_size, n_feature_maps, activation_fn=lambda x: x):
        self.filter_size = filter_size
        self.n_feature_maps = n_feature_maps
        self.activation_fn = activation_fn

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        raise NotImplementedError("Abstract Class!")


class Conv2DLayer(Layer):
    layer_type = 'conv2d'

    def __init__(self, *args, **kwargs):
        self.is_valid = kwargs['is_valid']
        del kwargs['is_valid']
        super(self.__class__, self).__init__(*args, **kwargs)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        # Create the tensorflow variables
        filters_shape = [self.filter_size, self.filter_size, prev_n_feature_maps, self.n_feature_maps]
        self.weights = tf.Variable(tf.truncated_normal(filters_shape, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[self.n_feature_maps]))

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

        # Prepare the next values in the loop
        return self.activations, self.n_feature_maps


class PoolLayer(Layer):
    layer_type = 'pool'

    def __init__(self, filter_size):
        super(PoolLayer, self).__init__(filter_size, 0)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        # Max pool
        self.activations = tf.nn.pool(prev_layer, window_shape=[self.filter_size, self.filter_size],
                                  dilation_rate=[dilation_rate, dilation_rate], strides=[1, 1],
                                  padding='VALID',
                                  pooling_type='MAX')

        self.n_feature_maps = prev_n_feature_maps

        return self.activations, prev_n_feature_maps


class BNLayer(Layer):
    layer_type = 'bn_conv2d'

    def __init__(self, activation_fn=lambda x: x):
        super(BNLayer, self).__init__(1, 0, activation_fn)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):

        # Apply batch normalization
        bn_conv = tf.contrib.layers.batch_norm(prev_layer, center=True, scale=True, is_training=is_training, scope='bn')

        # Apply the activation function
        self.activations = self.activation_fn(bn_conv)

        self.n_feature_maps = prev_n_feature_maps

        # Prepare the next values in the loop
        return self.activations, prev_n_feature_maps
