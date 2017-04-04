import tensorflow as tf
from .common import *
from collections import OrderedDict
import numpy as np

from utils import *



class UNetModArchitecture(Architecture):
    def __init__(self, model_name, output_mode):
        super(UNetModArchitecture, self).__init__(model_name, output_mode,'3D')

        self.fov = 1
        self.z_fov = 1

       # for layer in layers:

       #     self.fov += 4
          #CALCULATE THE FOV


UNET_MOD = UNetModArchitecture(
    model_name='unet_mod',
    output_mode=AFFINITIES_3D,
    #layers=[],
)


class UNet_Mod(Model):
    def __init__(self, architecture, is_training=False):
        super(UNet_Mod, self).__init__(architecture)
        prev_layer = self.image
        prev_n_feature_maps = 1

        l0 = self.image   #inpt

        w0_f1 = get_weight_variable("w0_f1", [3,3,3,1,64])
        b0_f1 = get_bias_variable("b0_f1", [64])
        l0_f1 = tf.nn.relu(same_conv3d(l0, w0_f1)+ b0_f1)

        w0_f2 = get_weight_variable("w0_f2", [3,3,3,64,64])
        b0_f2 = get_bias_variable("b0_f2", [64])
        l0_f2 = tf.nn.relu(same_conv3d(l0_f1, w0_f2)+ b0_f2)

        l1 = max_pool_3d(l0_f2)

        w1_f1 = get_weight_variable("w1_f1", [3,3,3,64,128])
        b1_f1 = get_bias_variable("b1_f1", [128])
        l1_f1 = tf.nn.relu(same_conv3d(l1, w1_f1)+ b1_f1)

        w1_f2 = get_weight_variable("w1_f2", [3,3,3,128,128])
        b1_f2 = get_bias_variable("b1_f2", [128])
        l1_f2 = tf.nn.relu(same_conv3d(l1_f1, w1_f2)+ b1_f2)

        l2 = max_pool_3d(l1_f2)

        w2_f1 = get_weight_variable("w2_f1", [3,3,3,128,256])
        b2_f1 = get_bias_variable("b2_f1", [256])
        l2_f1 = tf.nn.relu(same_conv3d(l2, w2_f1)+ b2_f1)

        w2_f2 = get_weight_variable("w2_f2", [3,3,3,256,256])
        b2_f2 = get_bias_variable("b2_f2", [256])
        l2_f2 = tf.nn.relu(same_conv3d(l2_f1, w2_f2)+ b2_f2)

        l3 = max_pool_3d(l2_f2)

        w3_f1 = get_weight_variable("w3_f1", [3,3,3,256,512])
        b3_f1 = get_bias_variable("b3_f1", [512])
        l3_f1 = tf.nn.relu(same_conv3d(l3, w3_f1)+ b3_f1)

        w3_f2 = get_weight_variable("w3_f2", [3,3,3,512,512])
        b3_f2 = get_bias_variable("b3_f2", [512])
        l3_f2 = tf.nn.relu(same_conv3d(l3_f1, w3_f2)+ b3_f2)

        l4 = max_pool_3d(l3_f2)

        #UP

        w4_f1 = get_weight_variable("w4_f1", [3,3,3,512,1024])
        b4_f1 = get_bias_variable("b4_f1", [1024])
        l4_f1 = tf.nn.relu(same_conv3d(l4, w4_f1)+ b4_f1)

        w4_f2 = get_weight_variable("w4_f2", [3,3,3,1024,1024])
        b4_f2 = get_bias_variable("b4_f2", [1024])
        l4_f2 = tf.nn.relu(same_conv3d(l4_f1, w4_f2)+ b4_f2)

        w3u = get_weight_variable("w3u", [3,3,3,512,1024])
        b3u = get_bias_variable("b3u", [512])
        l3u = tf.nn.relu(conv3d_transpose(l4_f2, w3u, stride=2) + b3u)


        w3u_f1 = get_weight_variable("w3u_f1", [3,3,3,1024,512])
        b3u_f1 = get_bias_variable("b3u_f1", [512])
        l3u_f1 = tf.nn.relu(same_conv3d(tf.concat([l3_f2,l3u],4), w3u_f1)+ b3u_f1)

        w3u_f2 = get_weight_variable("w3u_f2", [3,3,3,512,512])
        b3u_f2 = get_bias_variable("b3u_f2", [512])
        l3u_f2 = tf.nn.relu(same_conv3d(l3u_f1, w3u_f2)+ b3u_f2)

        w2u = get_weight_variable("w2u", [3,3,3,256,512])
        b2u = get_bias_variable("b2u", [256])
        l2u = tf.nn.relu(conv3d_transpose(l3u_f2, w2u, stride=2) + b2u)


        w2u_f1 = get_weight_variable("w2u_f1", [3,3,3,512,256])
        b2u_f1 = get_bias_variable("b2u_f1", [256])
        l2u_f1 = tf.nn.relu(same_conv3d(tf.concat([l2_f2,l2u],4), w2u_f1)+ b2u_f1)

        w2u_f2 = get_weight_variable("w2u_f2", [3,3,3,256,256])
        b2u_f2 = get_bias_variable("b2u_f2", [256])
        l2u_f2 = tf.nn.relu(same_conv3d(l2u_f1, w2u_f2)+ b2u_f2)

        w1u = get_weight_variable("w1u", [3,3,3,128,256])
        b1u = get_bias_variable("b1u", [128])
        l1u = tf.nn.relu(conv3d_transpose(l2u_f2, w1u, stride=2) + b1u)


        w1u_f1 = get_weight_variable("w1u_f1", [3,3,3,256,128])
        b1u_f1 = get_bias_variable("b1u_f1", [128])
        l1u_f1 = tf.nn.relu(same_conv3d(tf.concat([l1_f2,l1u],4), w1u_f1)+ b1u_f1)

        w1u_f2 = get_weight_variable("w1u_f2", [3,3,3,128,128])
        b1u_f2 = get_bias_variable("b1u_f2", [128])
        l1u_f2 = tf.nn.relu(same_conv3d(l1u_f1, w1u_f2)+ b1u_f2)

        w0u = get_weight_variable("w0u", [3,3,3,64,128])
        b0u = get_bias_variable("b0u", [64])
        l0u = tf.nn.relu(conv3d_transpose(l1u_f2, w0u, stride=2) + b0u)


        w0u_f1 = get_weight_variable("w0u_f1", [3,3,3,128,64])	#wtf
        b0u_f1 = get_bias_variable("b0u_f1", [64])
        l0u_f1 = tf.nn.relu(same_conv3d(tf.concat([l0_f2,l0u],4), w0u_f1)+ b0u_f1)

        w0u_f2 = get_weight_variable("w0u_f2", [3,3,3,64,64])
        b0u_f2 = get_bias_variable("b0u_f2", [64])
        l0u_f2 = tf.nn.relu(same_conv3d(l0u_f1, w0u_f2)+ b0u_f2)

        # final
        wf = get_weight_variable("wf", [3,3,3,64,3])
        bf = get_bias_variable("bf", [3])
        lf = tf.nn.relu(same_conv3d(l0u_f2, wf)+ bf)


        self.logits = lf

        # Predictions
        self.prediction = tf.nn.sigmoid(lf)
        self.binary_prediction = tf.round(self.prediction)

        # Loss
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=lf,
                                                                                    labels=self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        self.saver = tf.train.Saver()


    def predict(self, session, inputs, pred_batch_shape, mirror_inputs=True):
        if mirror_inputs:
            inputs = mirror_aross_borders_3d(inputs, self.fov, self.z_fov)

        return self.__predict_with_evaluation(session, inputs, None, pred_batch_shape, mirror_inputs)


    def __predict_with_evaluation(self, session, inputs, metrics, pred_tile_shape, mirror_inputs=True):
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
        combined_pred = np.ones((inputs.shape[0],
                                  z_outp_size, y_outp_size, x_outp_size, 3))
        # Create accumulator for overlaps.
        overlaps = np.zeros((inputs.shape[0], z_outp_size, y_outp_size, x_outp_size, 3))

        for stack, _ in enumerate(inputs):
            # Iterate through the overlapping tiles.
            for z in range(0, z_inp_size - z_in_patch + 1, z_out_patch - 2) + [z_inp_size - z_in_patch]:
                print('z: ' + str(z) + '/' + str(z_inp_size))
                for y in range(0, y_inp_size - y_in_patch + 1, y_out_patch - 10) + [y_inp_size - y_in_patch]:
                    print('y: ' + str(y) + '/' + str(y_inp_size))
                    for x in range(0, x_inp_size - x_in_patch + 1, x_out_patch - 10) + [x_inp_size - x_in_patch]:
                        pred = session.run(self.prediction,
                                           feed_dict={
                                               self.example: inputs[stack:stack + 1,
                                                                    z:z + z_in_patch,
                                                                    y:y + y_in_patch,
                                                                    x:x + x_in_patch,
                                                                    :]
                                           })

                        prev = combined_pred[stack,
                                                 z:z + z_out_patch,
                                                 y:y + y_out_patch,
                                                 x:x + x_out_patch, :]
                        combined_pred[stack,
                                      z:z + z_out_patch,
                                      y:y + y_out_patch,
                                      x:x + x_out_patch, :] = np.minimum(prev, pred[0])
                        '''
                        combined_pred[stack,
                                      z:z + z_out_patch,
                                      y:y + y_out_patch,
                                      x:x + x_out_patch, :] += pred[0]
                        '''
                        overlaps[stack,
                                 z:z + z_out_patch,
                                 y:y + y_out_patch,
                                 x:x + x_out_patch,
                                 :] += np.ones((z_out_patch, y_out_patch, x_out_patch, 3))

            # Normalize the combined prediction by the number of times each
            # voxel was computed in the overlapping computation.
            #validation_prediction = np.divide(combined_pred, overlaps)
            validation_prediction = combined_pred

            return validation_prediction


