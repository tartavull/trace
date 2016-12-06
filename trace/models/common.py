import tensorflow as tf


def weight_variable(shape):
  """
  One should generally initialize weights with a small amount of noise
  for symmetry breaking, and to prevent 0 gradients.
  Since we're using ReLU neurons, it is also good practice to initialize
  them with a slightly positive initial bias to avoid "dead neurons".
  """
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])


def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def max_unpool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

# convolution, batch normalization, and ReLU layer
def conv_norm_relu(inlayer, shape, dilation):
  # convolution
  W_conv  = weight_variable(shape)
  s_conv  = conv2d(inlayer, W_conv, dilation=dilation)

  # batch normalization
  mean_bn, var_bn = tf.nn.moments(s_conv, [0])
  offset_bn = tf.Variable(tf.zeros([shape[-1]]))
  scale_bn = tf.Variable(tf.ones([shape[-1]]))
  bn_conv = tf.nn.batch_normalization(s_conv, mean=mean_bn, variance=var_bn, 
    offset=offset_bn, scale=scale_bn, variance_epsilon=.0005)

  # ReLU
  return tf.nn.relu(bn_conv)