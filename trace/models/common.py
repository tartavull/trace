import tensorflow as tf

import em_dataset as em


def weight_variable(shape):
  """
  Xavier initialization
  """
  return tf.Variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))


def get_weight_variable(name, shape):
  return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def get_bias_variable(name, shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name, initializer=initial)


def conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate=[dilation, dilation])


def same_conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='SAME', dilation_rate= [dilation, dilation])


def down_conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[2, 2], padding='SAME', dilation_rate= [dilation, dilation])


def conv2d_transpose(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')


def max_pool(x, dilation=1, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')


def dropout(x, keep_prob):
    mask = tf.ones(x.get_shape()[3])
    dropoutMask = tf.nn.dropout(mask, keep_prob)
    return x * dropoutMask


def crop(x1, x2, batch_size):
    offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
    x2_shape = tf.shape(x2)
    size = tf.pack([x2_shape[1], x2_shape[2]])
    return tf.image.extract_glimpse(x1, size=size, offsets=offsets, centered=True)


def crop_and_concat(x1, x2, batch_size):
    return tf.concat(3, [crop(x1, x2, batch_size), x2])


# Arguments:
#   - inputs: mini-batch of input images
#   - is_training: flag specifying whether to use mini-batch or population
#   statistics
#   - decay: the decay rate used to calculate exponential moving average
def batch_norm_layer(inputs, is_training, decay=0.9):
    epsilon = 1e-5
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    offset = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, offset, scale, epsilon)


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

class Architecture(object):
    def __init__(self, model_name, output_mode):
        self.output_mode = output_mode
        self.model_name = model_name

        if output_mode == em.BOUNDARIES_MODE:
            self.n_outputs = 1
        elif output_mode == em.AFFINITIES_2D_MODE:
            self.n_outputs = 2
        elif output_mode == em.AFFINITIES_3D_MODE:
            self.n_outputs = 3


class Model(object):
    def __init__(self, architecture):
        # Save the architecture
        self.architecture = architecture
        self.model_name = self.architecture.model_name
        self.fov = architecture.receptive_field

        # Create an input queue
        with tf.device('/cpu:0'):
            self.queue = tf.FIFOQueue(50, tf.float32)

        # Draw example from the queue and separate
        self.example = tf.placeholder_with_default(self.queue.dequeue(), shape=[None, None, None, architecture.n_outputs + 1])
        self.image = self.example[:, :, :, :1]
        # Crop the labels to the appropriate field of view
        self.target = self.example[:, self.fov // 2:-(self.fov // 2), self.fov // 2:-(self.fov // 2), 1:]
