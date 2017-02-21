import tensorflow as tf


def weight_variable(shape):
  """
  Xavier initialization
  """
  return tf.Variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))


def bias_variable(shape):
  initial = tf.constant(val, dtype=dtype, shape=shape)
  var = tf.Variable(initial, dtype=dtype)
  return var

def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate=[dilation, dilation])

def conv3d(x, W, strides=[2,2,1], dilation=None):
  return tf.nn.convolution(x, W, stides=strides, padding='VALID', dilation_rate=[dilation, dilation])

def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')


class ConvKernel():
    def transpose(self):
        return TransposeKernel(self)

class TransposeKernel(ConvKernel):
    def __init__(self,k):
        self.kernel=k
    
    def __call__(self, x):
        return self.kernel.transpose_call(x)

    def transpose(self):
        return self.kernel

class ConvKernel3d(ConvKernel):
    def __init__(self, size=(4,4,1), strides=(2,2,1), n_lower=1, n_upper=1,stddev=0.5,dtype=tf.float32):
        initial = tf.truncated_normal([size[0],size[1],size[2],n_lower,n_upper], stddev=stddev, dtype=dtype)
        self.weights=tf.Variable(initial, dtype=dtype)
        self.bias=make_variable([1,1,1,1,n_lower])
        self.size=size
        self.strides=[1,strides[0],strides[1],strides[2],1]
        self.n_lower=n_lower
        self.n_upper=n_upper

        #up_coeff and down_coeff are coefficients meant to keep the magnitude of the output independent of stride and size choices
        self.up_coeff = 1.0/np.sqrt(reduce(operator.mul,size)*n_lower)
        self.down_coeff = 1.0/np.sqrt(reduce(operator.mul,size)/(reduce(operator.mul,strides))*n_upper)
    
    def transpose(self):
        return TransposeKernel(self)

    def __call__(self,x):

        with tf.name_scope('conv3d') as scope:
            self.in_shape = tf.shape(x)
            tmp=tf.nn.conv3d(x, self.up_coeff*self.weights, strides=self.strides, padding='VALID')
            shape_dict3d[(tuple(tmp._shape_as_list()[1:4]), self.size, tuple(self.strides))]=tuple(x._shape_as_list()[1:4])
        return tmp

    def transpose_call(self,x):
        with tf.name_scope('conv3d_t') as scope:
            if not hasattr(self,"in_shape"):
                self.in_shape=shape_dict3d[(tuple(x._shape_as_list()[1:4]),self.size,tuple(self.strides))]+(self.n_lower,)
            full_in_shape = (x._shape_as_list()[0],)+self.in_shape
            ret = tf.nn.conv3d_transpose(x, self.down_coeff*self.weights, output_shape=full_in_shape, strides=self.strides, padding='VALID')

        return tf.reshape(ret, full_in_shape)

class Layer(object):
    depth = 0
    def __init__(self, filter_size, n_feature_maps, depth=1, strides=[1,1], activation_fn=lambda x: x):
        self.filter_size = filter_size
        self.n_feature_maps = n_feature_maps
        self.activation_fn = activation_fn

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        raise NotImplementedError("Abstract Class!")

class Conv3DLayer(Layer):
    layer_type = 'conv3d'

    def __init__(self, *args, **kwargs):
        self.is_valid = kwargs['is_valid']
        del kwargs['is_valid']
        super(self.__class__, self).__init__(*args, **kwargs)

    def connect(self, prev_layer, n_in, dilation_rate, is_training):
        # Create the tensorflow variables
        filters_shape = [self.filter_size, self.filter_size, self.depth, prev_n_feature_maps, self.n_feature_maps]
        self.weights = tf.Variable(tf.truncated_normal(filters_shape, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[self.n_feature_maps]))
        self.strides = [1, self.strides[0], self.strides[1], self.strides[2], 1]

        # Perform a dilated convolution on the previous layer, where the dilation rate is dependent on the
        # number of poolings so far.
        if self.is_valid:
            validity = 'VALID'
        else:
            validity = 'SAME'
        convolution = tf.nn.convolution(prev_layer, self.weights, self.strides, padding=validity,
                                        dilation_rate=[dilation_rate, dilation_rate])

        # Apply the activation function
        self.activations = self.activation_fn(convolution + self.biases)

        # Prepare the next values in the loop
        return self.activations, self.n_feature_maps

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
        convolution = tf.nn.convolution(prev_layer, self.weights, self.strides, padding=validity,
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
        self.activations = tf.nn.pool(prev_layer, strides=self.strides,
                                  window_shape=[self.filter_size, self.filter_size],
                                  dilation_rate=[dilation_rate, dilation_rate], 
                                  padding='VALID',
                                  pooling_type='MAX')

        self.n_feature_maps = prev_n_feature_maps

        return self.activations, prev_n_feature_maps

class Pool3DLayer(Layer):
    layer_type = 'pool'

    def __init__(self, filter_size, depth):
        super(Pool3DLayer, self).__init__(filter_size, 0, depth)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        # Max pool
        self.activations = tf.nn.pool(prev_layer, strides= self.strides,
                                  window_shape=[self.filter_size, self.filter_size, self.depth],
                                  dilation_rate=[dilation_rate, dilation_rate],
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
