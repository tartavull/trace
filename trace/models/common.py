import tensorflow as tf
from utils import *
import operator
import numpy as np
import pprint
shape_dict3d={}

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


def conv3d(x, W, dilation=1, z_dilation=1):
    return tf.nn.convolution(x, W, strides=[1, 1, 1], padding='VALID', dilation_rate=[z_dilation, dilation, dilation])


def same_conv2d(x, W, dilation=1):
    return tf.nn.convolution(x, W, strides=[1, 1], padding='SAME', dilation_rate=[dilation, dilation])


def same_conv3d(x, W, dilation=1, z_dilation=1):
    return tf.nn.convolution(x, W, strides=[1, 1, 1], padding='SAME', dilation_rate=[z_dilation, dilation, dilation])


def down_conv2d(x, W, dilation=1):
    return tf.nn.convolution(x, W, strides=[2, 2], padding='SAME', dilation_rate=[dilation, dilation])


def down_conv3d(x, W, dilation=1, z_dilation=1):
    return tf.nn.convolution(x, W, strides=[1, 2, 2], padding='SAME', dilation_rate=[z_dilation, dilation, dilation])


def conv2d_transpose(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')


def conv3d_transpose(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 1, stride, stride, 1], padding='SAME')


def max_pool(x, dilation=1, strides=[2, 2], window_shape=[2, 2]):
    return tf.nn.pool(x, window_shape=window_shape, dilation_rate=[dilation, dilation],
                      strides=strides, padding='VALID', pooling_type='MAX')


def max_pool_3d(x, dilation=1, z_dilation=1, strides=[1, 2, 2], window_shape=[1, 2, 2]):
    return tf.nn.pool(x, window_shape=window_shape, dilation_rate=[z_dilation, dilation, dilation],
                      strides=strides, padding='VALID', pooling_type='MAX')

def dropout(x, keep_prob):
    mask = tf.ones(x.get_shape()[3])
    dropoutMask = tf.nn.dropout(mask, keep_prob)
    return x * dropoutMask


def crop(x1, x2, batch_size):
    offsets = tf.zeros(tf.stack([batch_size, 2]), dtype=tf.float32)
    x2_shape = tf.shape(x2)
    size = tf.stack([x2_shape[1], x2_shape[2]])
    return tf.image.extract_glimpse(x1, size=size, offsets=offsets, centered=True)


def crop_3d(x1, x2, batch_size):
    x2_shape = tf.shape(x2)
    x2_shape = tf.Print(x2_shape, [x2_shape, tf.shape(x1)])
    offsets = tf.zeros(tf.stack([x2_shape[1], 2]), dtype=tf.float32)
    size = tf.stack([x2_shape[2], x2_shape[3]])

    z_crop = tf.shape(x1)[1] - x2_shape[1]
    return tf.map_fn(lambda img: tf.image.extract_glimpse(img, size=size, offsets=offsets, centered=True), x1)[
           z_crop // 2: -(z_crop // 2)]


def crop_and_concat(x1, x2, batch_size):
    return tf.concat([crop(x1, x2, batch_size), x2], 3)


def crop_and_concat_3d(x1, x2, batch_size):
    return tf.concat([crop_3d(x1, x2, batch_size), x2], 4)

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
    def __init__(self, name, dict_key, size=(4,4,1), strides=(2,2,1), n_lower=1, n_upper=1):
        self.weights = get_weight_variable(name=name, shape=[size[0],size[1],size[2],n_lower,n_upper])
        self.size = size
        self.strides = [1,strides[0],strides[1],strides[2],1]
        self.n_lower = n_lower
        self.n_upper = n_upper
        self.dict_key = dict_key
        #up_coeff and down_coeff are coefficients meant to keep the magnitude of the output independent of stride and size choices
        self.up_coeff = 1.0/np.sqrt(reduce(operator.mul,size)*n_lower)
        self.down_coeff = 1.0/np.sqrt(reduce(operator.mul,size)/(reduce(operator.mul,strides))*n_upper)
    
    def transpose(self):
        return TransposeKernel(self)

    def __call__(self,x):
        with tf.name_scope('conv3d') as scope:
            self.in_shape = tf.Print(tf.shape(x), [tf.shape(x)])
            tmp=tf.nn.conv3d(x, self.up_coeff*self.weights, strides=self.strides, padding='VALID')
            shape_dict3d[self.dict_key] = self.in_shape[1:4]
        return tmp

    def transpose_call(self,x):
        with tf.name_scope('conv3d_t') as scope:
            if not hasattr(self,"in_shape"):
                self.in_shape = tf.concat([shape_dict3d[self.dict_key],tf.stack([self.n_lower])], 0)
            full_in_shape = tf.concat([tf.shape(x)[:1], self.in_shape], 0)
            ret = tf.nn.conv3d_transpose(x, self.down_coeff*self.weights, output_shape=full_in_shape, strides=self.strides, padding='VALID')
        return tf.reshape(ret, full_in_shape)

class Layer(object):
    depth = 0

    def __init__(self, dim, filter_size, n_feature_maps, activation_fn=lambda x: x, z_filter_size=1):
        self.dim = dim
        self.filter_size = filter_size
        self.z_filter_size = z_filter_size
        self.n_feature_maps = n_feature_maps
        self.activation_fn = activation_fn

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training, z_dilation_rate=1,
                skip_connect=None):
        raise NotImplementedError("Abstract Class!")

class UNet3DLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.layer_name = kwargs['layer_name']
        self.is_valid = kwargs['is_valid']
        self.is_residual = kwargs['is_residual']
        self.uses_max_pool = kwargs['uses_max_pool']
        self.num_convs = kwargs['num_convs']
        self.is_contracting = kwargs['is_contracting']
        self.is_expanding = kwargs['is_expanding']
        self.is_training = kwargs['is_training']
        del kwargs['layer_name']
        del kwargs['is_valid']
        del kwargs['is_residual']
        del kwargs['uses_max_pool']
        del kwargs['num_convs']
        del kwargs['is_contracting']
        del kwargs['is_expanding']
        del kwargs['is_training']

        kwargs['dim'] = 3

        super(UNet3DLayer, self).__init__(*args, **kwargs)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training, z_dilation_rate=1,
                skip_connect=None):
        weights = []
        biases = []
        convs = []

        cur_node = prev_layer
        batch_size = tf.shape(cur_node)[0]

        in_n_feature_maps = prev_n_feature_maps

        # Add skip-connection if expanding.
        if self.is_expanding and skip_connect != None:
            cur_node = tf.concat([skip_connect, cur_node], 4)
            convs.append(cur_node)
            in_n_feature_maps = prev_n_feature_maps * 2

        # Set up each convolution in the layer.
        for i in range(self.num_convs):
            # Create the weights and biases.
            if i == 0:
                w_i = get_weight_variable(self.layer_name + '_w0',
                                          [self.z_filter_size, self.filter_size, self.filter_size, in_n_feature_maps,
                                           self.n_feature_maps])
            else:
                w_i = get_weight_variable(self.layer_name + '_w' + str(i),
                                          [self.z_filter_size, self.filter_size, self.filter_size, self.n_feature_maps,
                                           self.n_feature_maps])
            b_i = get_bias_variable(self.layer_name + '_b' + str(i), [self.n_feature_maps])
            weights.append(w_i)
            biases.append(b_i)

            # Perform the convolution
            if self.is_valid:
                convFn = conv3d
            else:
                convFn = same_conv3d
            cur_node = tf.nn.elu(convFn(cur_node, w_i) + b_i)
            convs.append(cur_node)

        # If applicable, add the residual connection.
        if self.is_residual:
            if skip_connect == None:
                residual = prev_layer
            else:
                residual = skip_connect
            if self.is_valid:
                residual = crop(residual, cur_node, batch_size)
            if prev_n_feature_maps != self.n_feature_maps:
                residual = tf.tile(residual, (1, 1, 1, 1, self.n_feature_maps // prev_n_feature_maps))
            final_node = cur_node + residual

        # If on the contracting path, down sample using either max-pooling
        # or down-convolution.
        if self.is_contracting:
            if self.uses_max_pool:
                out_node = max_pool_3d(final_node)
                out_n_feature_maps = self.n_feature_maps
            else:
                w_d = get_weight_variable(self.layer_name + '_wd', [2, 2, 2, self.n_feature_maps, 2 * self.n_feature_maps])
                b_d = get_bias_variable(self.layer_name + '_bd', [2 * self.n_feature_maps])
                weights.append(w_d)
                biases.append(b_d)

                out_node = tf.nn.elu(down_conv3d(final_node, w_d) + b_d)
                out_n_feature_maps = self.n_feature_maps * 2
            convs.append(out_node)

            return out_node, final_node, out_n_feature_maps
        elif self.is_expanding:
            # If on the expanding path, upsample using transposed
            # convolution.
            w_u = get_weight_variable(self.layer_name + '_wu',
                                      [self.z_filter_size, self.filter_size, self.filter_size, self.n_feature_maps // 2,
                                       self.n_feature_maps])
            b_u = get_bias_variable(self.layer_name + '_bu', [self.n_feature_maps // 2])
            weights.append(w_u)
            biases.append(b_u)

            up_conv = tf.nn.elu(conv3d_transpose(final_node, w_u, stride=2) + b_u)
            convs.append(up_conv)
            out_n_feature_maps = self.n_feature_maps // 2

            return up_conv, out_n_feature_maps
        else:
            # Map to boundaries 
            w_o = get_weight_variable('w_o',
                                      [self.z_filter_size, self.filter_size, self.filter_size, self.n_feature_maps, 1])
            b_o = get_bias_variable('b_o', [1])
            out_node = same_conv3d(final_node, w_o) + b_o
            return out_node

class ConvLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_valid = kwargs['is_valid']
        del kwargs['is_valid']
        super(ConvLayer, self).__init__(*args, **kwargs)
        self.layer_type = 'conv' + str(self.dim) + 'd'

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training, z_dilation_rate=1):
        # Create the tensorflow variables
        filter_dims = [self.filter_size, self.filter_size]
        dilation_shape = [dilation_rate, dilation_rate]
        if self.dim == 3:
            filter_dims = [self.z_filter_size] + filter_dims
            dilation_shape = [z_dilation_rate] + dilation_shape
        filters_shape = filter_dims + [prev_n_feature_maps, self.n_feature_maps]
        self.weights = tf.Variable(tf.truncated_normal(filters_shape, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[self.n_feature_maps]))

        # Perform a dilated convolution on the previous layer, where the dilation rate is dependent on the
        # number of poolings so far.
        if self.is_valid:
            validity = 'VALID'
        else:
            validity = 'SAME'
        convolution = tf.nn.convolution(prev_layer, self.weights, strides=[1] * self.dim, padding=validity,
                                        dilation_rate=dilation_shape)

        # Apply the activation function
        self.activations = self.activation_fn(convolution + self.biases)

        # self.activations = tf.Print(self.activations, [tf.shape(self.activations)])

        # Prepare the next values in the loop
        return self.activations, self.n_feature_maps


class Conv2DLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        kwargs['dim'] = 2
        super(Conv2DLayer, self).__init__(*args, **kwargs)

    def connect(self, *args, **kwargs):
        return super(Conv2DLayer, self).connect(*args, **kwargs)


class Conv3DLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        kwargs['dim'] = 3
        super(Conv3DLayer, self).__init__(*args, **kwargs)

    def connect(self, *args, **kwargs):
        return super(Conv3DLayer, self).connect(*args, **kwargs)


class PoolLayer(Layer):
    layer_type = 'pool'

    def __init__(self, dim, filter_size):
        super(PoolLayer, self).__init__(dim, filter_size, 0)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training, z_dilation_rate=1):
        # Max pool
        filter_shape = [self.filter_size, self.filter_size]
        dilation_shape = [dilation_rate, dilation_rate]
        if self.dim == 3:
            filter_shape = [1] + filter_shape
            dilation_shape = [z_dilation_rate] + dilation_shape
        self.activations = tf.nn.pool(prev_layer, window_shape=filter_shape,
                                      dilation_rate=dilation_shape,
                                      strides=[1] * self.dim,
                                      padding='VALID',
                                      pooling_type='MAX')

        # self.activations = tf.Print(self.activations, [tf.shape(self.activations)])

        self.n_feature_maps = prev_n_feature_maps

        return self.activations, prev_n_feature_maps


class Pool2DLayer(PoolLayer):
    def __init__(self, filter_size):
        super(Pool2DLayer, self).__init__(2, filter_size)

    def connect(self, *args, **kwargs):
        return super(Pool2DLayer, self).connect(*args, **kwargs)


class Pool3DLayer(PoolLayer):
    def __init__(self, filter_size):
        self.z_filter_size = 1
        super(Pool3DLayer, self).__init__(3, filter_size)

    def connect(self, *args, **kwargs):
        return super(Pool3DLayer, self).connect(*args, **kwargs)


class BNLayer(Layer):
    layer_type = 'bn_conv2d'

    def __init__(self, activation_fn=lambda x: x):
        super(BNLayer, self).__init__(2, 1, 0, activation_fn)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        # Apply batch normalization
        bn_conv = tf.contrib.layers.batch_norm(prev_layer, center=True, scale=True, is_training=is_training, scope='bn')

        # Apply the activation function
        self.activations = self.activation_fn(bn_conv)

        self.n_feature_maps = prev_n_feature_maps

        # Prepare the next values in the loop
        return self.activations, prev_n_feature_maps


class Architecture(object):
    def __init__(self, model_name, output_mode, architecture_type):
        self.architecture_type = architecture_type
        self.output_mode = output_mode
        self.model_name = model_name

        if output_mode == BOUNDARIES:
            self.n_outputs = 1
        elif output_mode == AFFINITIES_2D:
            self.n_outputs = 2
        elif output_mode == AFFINITIES_3D:
            self.n_outputs = 3


class Model(object):
    def __init__(self, architecture, apply_mask=False):
        # Save the architecture
        self.architecture = architecture
        self.model_name = self.architecture.model_name
        self.fov = architecture.fov
        self.z_fov = architecture.z_fov
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.apply_mask = apply_mask
        self.num_output_channels = 1
        if self.architecture.output_mode == AFFINITIES_3D:
            self.num_output_channels = 3
        # Create an input queue
        with tf.device('/cpu:0'):
            self.queue = tf.FIFOQueue(50, tf.float32)

        # Inputs are tensor of shape [batch, z, y, x, n_chan]
        self.example = tf.placeholder_with_default(self.queue.dequeue(), shape=[None, None, None, None, None])

        self.raw_image = self.example[:, :, :, :, :1]

        # Standardize each input image, using map because per_image_standardization takes one image at a time
        self.image = tf.map_fn(lambda stack: tf.map_fn(lambda img: tf.image.per_image_standardization(img), stack), self.raw_image)

        # Comment out if we want to have averaged
        # mean, var = tf.nn.moments(self.raw_image, axes=[0, 1, 2, 3, 4], keep_dims=False)
        # self.image = (self.raw_image - mean) / tf.sqrt(var)

        # Crop the labels to the appropriate field of view
        if self.fov == 1 and self.z_fov == 1:
            if apply_mask:
                self.mask = self.example[:, :, :, :, (self.num_output_channels + 1):]
            self.target = self.example[:, :, :, :, 1:(self.num_output_channels + 1)]
        else:
            if apply_mask:
                self.mask = self.example[:, self.z_fov // 2:-(self.z_fov // 2), self.fov // 2:-(self.fov // 2),
                              self.fov // 2:-(self.fov // 2), (self.num_output_channels + 1):]
            self.target = self.example[:, self.z_fov // 2:-(self.z_fov // 2), self.fov // 2:-(self.fov // 2),
                            self.fov // 2:-(self.fov // 2), 1:(self.num_output_channels + 1)]

    def predict(self, session, inputs, pred_batch_shape, mirror_inputs=True):
        """Predict on a set of inputs, producing a tensor with the same shape.

        :param pred_batch_shape: When predicting, break into pieces of this shape for evaluation (3D tensor [z, y, x])
        :param session: Tensorflow session
        :param mirror_inputs: Decide to mirror the inputs every time
        :param inputs: A tensor of input stacks with dimension [batch, z, y, x, 1]
        """
        raise NotImplementedError('Abstract model does not implement prediction')

    # def predict_with_evaluation(self, session, inputs, metrics, labels, pred_batch_shape, mirror_inputs=True):
    #     """Predict on a set of inputs, and evaluate the model-specific metrics specified in metrics against labels
    #
    #     :param pred_batch_shape: When predicting, break into pieces of this shape for evaluation
    #     :param session: Tensorflow session
    #     :param mirror_inputs: Decide to mirror the inputs every time
    #     :param inputs: A tensor of input stacks with dimension [batch, z, y, x, 1]
    #     :param labels: Labels with dimension [batch, z, y, x, 1], against which we will evaluate our predictions
    #     :param metrics: A list of metrics on which we will evaluate
    #     """
    #     raise NotImplementedError('Abstract model does not implement prediction with evaluation')
    #
