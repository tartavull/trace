import tensorflow as tf
from .common import *
from collections import OrderedDict

from utils import *

FOV = 189
OUTPT = 192
INPT = 380


class UNetArchitecture(Architecture):
    def __init__(self, model_name, output_mode):
        super(UNetArchitecture, self).__init__(model_name, output_mode)
        self.receptive_field = FOV


RES_VNET = UNetArchitecture(
    model_name='res_vnet',
    output_mode=AFFINITIES_2D
)


class UNet(Model):
    '''
    Creates a new U-Net for the given parametrization.

    :param x: input tensor variable, shape [?, ny, nx, channels]
    :param keep_prob: dropout probability tensor
    :param layers: number of layers in the unet
    :param features_root: number of features in the first layer
    :param kernel_size: size of the convolutional kernel
    :param learning_rate: learning rate for the optimizer
    super(UNet, self).__init__(architecture)
    '''
    def __init__(self, architecture, is_training=False, num_layers=5, features_root=64, kernel_size=3):
        super(UNet, self).__init__(architecture)

        in_node = self.image
        batch_size = tf.shape(in_node)[0]
        in_size = tf.shape(in_node)[1]
        size = in_size

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        upconvs = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        histogram_dict = {}
        image_summaries = []

        # down layers
        for layer in range(num_layers):
            num_feature_maps = 2**layer * features_root
            layer_str = 'layer' + str(layer)

            # Input layer maps a 1-channel image to num_feature_maps channels
            if layer == 0:
                w1 = get_weight_variable(layer_str + '_w1', [kernel_size, kernel_size, 1, num_feature_maps])
            else:
                w1 = get_weight_variable(layer_str + '_w1', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            w2 = get_weight_variable(layer_str + '_w2', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            w3 = get_weight_variable(layer_str + '_w3', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            b1 = get_bias_variable(layer_str + '_b1', [num_feature_maps])
            b2 = get_bias_variable(layer_str + '_b2', [num_feature_maps])
            b3 = get_bias_variable(layer_str + '_b3', [num_feature_maps])

            h_conv1 = tf.nn.elu(same_conv2d(in_node, w1) + b1)
            h_conv2 = tf.nn.elu(conv2d(h_conv1, w2) + b2)
            h_conv3 = tf.nn.elu(conv2d(h_conv2, w2) + b3)

            in_node_cropped = crop(in_node, h_conv3, batch_size)
            if layer == 0:
                in_node_cropped = tf.tile(in_node_cropped, (1,1,1,num_feature_maps))
#            else:
#                in_node_cropped = tf.tile(in_node_cropped, (1,1,1,2))

            h_conv4 = h_conv3 + in_node_cropped
            dw_h_convs[layer] = h_conv4

            weights.append((w1, w2, w3))
            convs.append((h_conv1, h_conv2, dw_h_convs[layer]))
            histogram_dict[layer_str + '_in_node'] = in_node
            histogram_dict[layer_str + '_w1'] = w1
            histogram_dict[layer_str + '_w2'] = w2
            histogram_dict[layer_str + '_w3'] = w3
            histogram_dict[layer_str + '_b1'] = b1
            histogram_dict[layer_str + '_b2'] = b2
            histogram_dict[layer_str + '_b3'] = b3
            histogram_dict[layer_str + '_activations'] = dw_h_convs[layer]

            size -= 4


            # If not the bottom layer, do a max-pool (or down-convolution)
            if layer < num_layers - 1:
                w_d = get_weight_variable(layer_str + '_wd', [2, 2, num_feature_maps, 2 * num_feature_maps])
                b_d = get_bias_variable(layer_str + '_bd', [2 * num_feature_maps])
                pools[layer] = tf.nn.elu(down_conv2d(dw_h_convs[layer], w_d) + b_d)
                #pools[layer] = max_pool(dw_h_convs[layer])
                in_node = pools[layer]
                size //= 2


        in_node = dw_h_convs[num_layers-1]

        # Up layers
        # There is one less up layer than down layer.
        for layer in range(num_layers - 2, -1, -1):
            layer_str = 'layer_u' + str(layer)
            num_feature_maps = 2**layer * features_root

            wu = get_weight_variable(layer_str + '_wu', [kernel_size, kernel_size, num_feature_maps, num_feature_maps * 2])
            bu = get_bias_variable(layer_str + '_bu', [num_feature_maps])
            h_upconv = tf.nn.elu(conv2d_transpose(in_node, wu, stride=2) + bu)
            h_upconv_concat = crop_and_concat(dw_h_convs[layer], h_upconv, batch_size)
            upconvs[layer] = h_upconv_concat

            w1 = get_weight_variable(layer_str + '_w1', [kernel_size, kernel_size, num_feature_maps * 2, num_feature_maps])
            w2 = get_weight_variable(layer_str + '_w2', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            w3 = get_weight_variable(layer_str + '_w3', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            b1 = get_bias_variable(layer_str + '_b1', [num_feature_maps])
            b2 = get_bias_variable(layer_str + '_b2', [num_feature_maps])
            b3 = get_bias_variable(layer_str + '_b3', [num_feature_maps])

            h_conv1 = tf.nn.elu(same_conv2d(h_upconv_concat, w1) + b1)
            h_conv2 = tf.nn.elu(conv2d(h_conv1, w2) + b2)
            h_conv3 = tf.nn.elu(conv2d(h_conv2, w3) + b3)

            #h_upconv_cropped = crop(h_upconv, h_conv3, batch_size)
            skip_connect_cropped = crop(dw_h_convs[layer], h_conv3, batch_size)

            in_node = h_conv3 + skip_connect_cropped# + h_upconv_cropped
            up_h_convs[layer] = in_node

            weights.append((w1, w2, w3))
            convs.append((h_conv1, in_node))
            histogram_dict[layer_str + '_wu'] = wu
            histogram_dict[layer_str + '_bu'] = bu
            histogram_dict[layer_str + '_h_upconv'] = h_upconv
            histogram_dict[layer_str + '_h_upconv_concat'] = h_upconv_concat
            histogram_dict[layer_str + '_w1'] = w1
            histogram_dict[layer_str + '_w2'] = w2
            histogram_dict[layer_str + '_w3'] = w3
            histogram_dict[layer_str + '_b1'] = b1
            histogram_dict[layer_str + '_b2'] = b2
            histogram_dict[layer_str + '_b3'] = b3
            histogram_dict[layer_str + '_activations'] = up_h_convs[layer]

            size *= 2
            size -= 4


        # Output map
        # features_root * 2 because there is one less up layer than
        # down layer
        w_o = get_weight_variable('w_o', [5, 5, features_root, 2]) #* 2, 2])
        b_o = get_bias_variable('b_o', [2])
        last_layer = conv2d(in_node, w_o) + b_o


        # Predictions
        self.prediction = tf.nn.sigmoid(last_layer)
        self.binary_prediction = tf.round(self.prediction)

        # Loss
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(last_layer, self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        self.saver = tf.train.Saver()

