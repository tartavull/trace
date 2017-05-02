import tensorflow as tf
from .common import *
from collections import OrderedDict
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from utils import *

FOV = 189
OUTPT = 192
INPT = 380


class UNetArchitecture(Architecture):
    def __init__(self, model_name, output_mode, layers):
        super(UNetArchitecture, self).__init__(model_name, output_mode, '2D')
        self.receptive_field = FOV

        self.layers = layers

        self.fov = 1
        self.z_fov = 1
        
       # for layer in layers:

       #     self.fov += 4
          #CALCULATE THE FOV  




RES_VNET = UNetArchitecture(
    model_name='res_vnet',
    output_mode=AFFINITIES_2D,
    layers=[],
)


UNET_3D = UNetArchitecture(
    model_name='unet_3d',
    output_mode=AFFINITIES_3D,
    layers=[
        UNet3DLayer(layer_name='layer_d1', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=64, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_d2', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=128, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_d3', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=256, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_d4', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=512, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_5', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=1024, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u4', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=512, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u3', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=256, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u2', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=128, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u1', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=64, num_convs=3, is_contracting=False, 
                    is_expanding=False, is_training=False),
    ]
)

UNET_3D_4LAYERS = UNetArchitecture(
    model_name='unet_3d_4layers',
    output_mode=AFFINITIES_3D,
    layers=[
        UNet3DLayer(layer_name='layer_d1', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=64, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_d2', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=128, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_d3', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=256, num_convs=3, is_contracting=True, 
                    is_expanding=False, is_training=False),
        UNet3DLayer(layer_name='layer_4', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=1,
                    n_feature_maps=512, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u3', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=256, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u2', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=128, num_convs=3, is_contracting=False, 
                    is_expanding=True, is_training=False),
        UNet3DLayer(layer_name='layer_u1', is_valid=False, is_residual=True, 
                    uses_max_pool=True, filter_size=3, z_filter_size=3,
                    n_feature_maps=64, num_convs=3, is_contracting=False, 
                    is_expanding=False, is_training=False),
    ]
)

class UNet(Model):
    def __init__(self, architecture, is_training=False):
        super(UNet, self).__init__(architecture)
        prev_layer = self.image
        prev_n_feature_maps = 1

        skip_connections = []
        
        num_layers = len(self.architecture.layers)
        for layer_num, layer in enumerate(self.architecture.layers):
            with tf.variable_scope('layer' + str(layer_num)):
                layer.depth = layer_num
                
                if layer.is_contracting:
                    prev_layer, skip_connect, prev_n_feature_maps = layer.connect(prev_layer, prev_n_feature_maps, dilation_rate=1, is_training=False)
                    skip_connections.append(skip_connect)
                elif layer.is_expanding:
                    if num_layers - layer_num - 1 < len(skip_connections):
                        skip_connect = skip_connections[num_layers - layer_num - 1]
                    else:
                        skip_connect = None
                    prev_layer, prev_n_feature_maps = layer.connect(prev_layer, prev_n_feature_maps, dilation_rate=1, is_training=False, skip_connect=skip_connect)
                else:
                    last_layer = layer.connect(prev_layer, prev_n_feature_maps, dilation_rate=1, is_training=False, skip_connect=skip_connections[0])

        # Predictions
        self.prediction = tf.nn.sigmoid(last_layer)
        self.binary_prediction = tf.round(self.prediction)

        # Loss
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=last_layer,
                                                                                    labels=self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        self.saver = tf.train.Saver()


    def predict(self, session, inputs, pred_batch_shape, mirror_inputs=True):
        print("inputs sizes")
        print(inputs.shape[1])
        print(inputs.shape[2])
        print(inputs.shape[3])
        if mirror_inputs:
            inputs = mirror_aross_borders_3d(inputs, self.fov, self.z_fov)

        predict_affinities = self.__predict_with_evaluation(session, inputs, None, pred_batch_shape, mirror_inputs)
        predict_affinities_flat = predict_affinities.flatten()
        predict_affinities_flat = sorted(predict_affinities_flat)
        num_elements = len(predict_affinities_flat)
        low_boundary_ind = int(0.01*num_elements)
        high_boundary_ind = int(0.8*num_elements)
        print(low_boundary_ind)
        print(high_boundary_ind)
        low_boundary = predict_affinities_flat[low_boundary_ind]
        high_boundary = predict_affinities_flat[high_boundary_ind]
        print (low_boundary)
        print(high_boundary)
        #        predict_segmentation = run_watershed_on_affinities_and_store(predict_affinities)
        return predict_affinities


    def __predict_with_evaluation(self, session, inputs, metrics, pred_tile_shape, mirror_inputs=True):

        print("inputs sizes")
        print(inputs.shape[1])
        print(inputs.shape[2])
        print(inputs.shape[3])
        
        # Extract the tile sizes from the argument
        z_out_patch, y_out_patch, x_out_patch = pred_tile_shape[0], pred_tile_shape[1], pred_tile_shape[2]
        z_in_patch = z_out_patch + self.z_fov - 1
        y_in_patch = y_out_patch + self.fov - 1
        x_in_patch = x_out_patch + self.fov - 1

        # Extract the overall input size.
        z_inp_size, y_inp_size, x_inp_size = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        z_outp_size = z_inp_size - self.z_fov + 1
        y_outp_size = y_inp_size - self.fov + 1
        x_outp_size  = x_inp_size - self.fov + 1

        # Create accumulator for output.
        combined_pred = np.zeros((inputs.shape[0],
                                  z_outp_size, y_outp_size, x_outp_size, 3))
        # Create accumulator for overlaps.
        overlaps = np.zeros((inputs.shape[0], z_outp_size, y_outp_size, x_outp_size, 3))

        dirac = np.zeros((z_out_patch, y_out_patch, x_out_patch, 3))
        dirac[z_out_patch // 2, y_out_patch // 2, x_out_patch // 2, 1] = 1
        gaussian_kernel=gaussian_filter(dirac, (z_out_patch // 6, y_out_patch // 6, x_out_patch // 6, 100))

        import itertools
        n = 4
        final_lst = map(list, itertools.product([0,1], repeat=n))

        sum_predictions = np.zeros((inputs.shape[0], z_outp_size, y_outp_size, x_outp_size, 3))

        final_lst = [[0,0,0,0]]
        for stack, _ in enumerate(inputs):
            # Iterate through the overlapping tiles.
            for rule in final_lst:
                np_stack = inputs[stack, :, :, :, :]
                ans = flip(np_stack, rule)
                for z in range(0, z_inp_size - z_in_patch + 1, z_out_patch - 5) + [z_inp_size - z_in_patch]:
                    print('z: ' + str(z) + '/' + str(z_inp_size))
                    for y in range(0, y_inp_size - y_in_patch + 1, y_out_patch - 50) + [y_inp_size - y_in_patch]:
                        print('y: ' + str(y) + '/' + str(y_inp_size))
                        for x in range(0, x_inp_size - x_in_patch + 1, x_out_patch - 50) + [x_inp_size - x_in_patch]:
                            pred= session.run(self.prediction,
                                           feed_dict={
                                               self.example: inputs[stack:stack + 1, 
                                                                    z:z + z_in_patch, 
                                                                    y:y + y_in_patch,
                                                                    x:x + x_in_patch,
                                                                    :]
                                           }) 

                            '''
                            prev = combined_pred[stack, 
                                                 z:z + z_out_patch,
                                                 y:y + y_out_patch,
                                                 x:x + x_out_patch, :]

                            combined_pred[stack, 
                                      z:z + z_out_patch,
                                      y:y + y_out_patch,
                                      x:x + x_out_patch, :] = np.minimum(prev, pred[0])
                            '''
                            reverted_pred = revert_flip(pred[0], rule)
                            combined_pred[stack, 
                                      z:z + z_out_patch,
                                      y:y + y_out_patch,
                                      x:x + x_out_patch, :] += reverted_pred * gaussian_kernel
                            overlaps[stack,
                                 z:z + z_out_patch,
                                 y:y + y_out_patch,
                                 x:x + x_out_patch, 
                                 :] += gaussian_kernel

            # Normalize the combined prediction by the number of times each
            # voxel was computed in the overlapping computation.
            validation_prediction = np.divide(combined_pred, overlaps)
                 
            return validation_prediction


                

class ResVNet(Model):
    '''
    Creates a new U-Net for the given parametrization.

    :param x: input tensor variable, shape [?, ny, nx, channels]
    :param keep_prob: dropout probability tensor
    :param layers: number of layers in the unet
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolutional kernel
    :param learning_rate: learning rate for the optimizer
    super(UNet, self).__init__(architecture)
    '''
    def __init__(self, architecture, is_training=False, num_layers=5, features_root=64, filter_size=3):
        super(ResVNet, self).__init__(architecture)

        in_node = self.image
        batch_size = tf.shape(in_node)[0]
        in_size = tf.shape(in_node)[1]
        size = in_size

        weights = []
        biases = []
        self.filter_size = kwargs['filter_size']
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
                w1 = get_weight_variable(layer_str + '_w1', [filter_size, filter_size, 1, num_feature_maps])
            else:
                w1 = get_weight_variable(layer_str + '_w1', [filter_size, filter_size, num_feature_maps, num_feature_maps])
            w2 = get_weight_variable(layer_str + '_w2', [filter_size, filter_size, num_feature_maps, num_feature_maps])
            w3 = get_weight_variable(layer_str + '_w3', [filter_size, filter_size, num_feature_maps, num_feature_maps])
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

            wu = get_weight_variable(layer_str + '_wu', [filter_size, filter_size, num_feature_maps, num_feature_maps * 2])
            bu = get_bias_variable(layer_str + '_bu', [num_feature_maps])
            h_upconv = tf.nn.elu(conv2d_transpose(in_node, wu, stride=2) + bu)
            h_upconv_concat = crop_and_concat(dw_h_convs[layer], h_upconv, batch_size)
            upconvs[layer] = h_upconv_concat

            w1 = get_weight_variable(layer_str + '_w1', [filter_size, filter_size, num_feature_maps * 2, num_feature_maps])
            w2 = get_weight_variable(layer_str + '_w2', [filter_size, filter_size, num_feature_maps, num_feature_maps])
            w3 = get_weight_variable(layer_str + '_w3', [filter_size, filter_size, num_feature_maps, num_feature_maps])
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

