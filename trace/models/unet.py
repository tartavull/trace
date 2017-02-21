import tensorflow as tf
import numpy as np
import pprint
import operator
from .common import *
import em_dataset as em


class UNet:
    def __init__(self, model_name, output_mode, is_training=False):
        self.model_name = model_name
        self.output_mode = output_mode
        
        if output_mode == em.BOUNDARIES_MODE:
            self.n_outputs = 1
        elif output_mode == em.AFFINITIES_2D_MODE:
            self.n_outputs = 2
        elif output_mode == em.AFFINITIES_3D_MODE:
            self.n_outputs = 3

        #convolution variables
        c1=ConvKernel3d(size=(4,4,1), strides=(2,2,1), n_lower=1, n_upper=12)
        c2=ConvKernel3d(size=(4,4,1), strides=(2,2,1), n_lower=12, n_upper=24)
        c3=ConvKernel3d(size=(4,4,4), strides=(2,2,2), n_lower=24, n_upper=48)

        c1t=ConvKernel3d(size=(4,4,1), strides=(2,2,1), n_lower=1, n_upper=12).transpose()
        c2t=ConvKernel3d(size=(4,4,1), strides=(2,2,1), n_lower=12, n_upper=24).transpose()
        c3t=ConvKernel3d(size=(4,4,4), strides=(2,2,2), n_lower=24, n_upper=48).transpose()

        #bias variables
        b0=make_variable([1,1,1,1,1])
        b1=make_variable([1,1,1,1,12])
        b2=make_variable([1,1,1,1,24])
        b3=make_variable([1,1,1,1,48])

        b0t=make_variable([1,1,1,1,1])
        b1t=make_variable([1,1,1,1,12])
        b2t=make_variable([1,1,1,1,24])

        self.queue = tf.FIFOQueue(50, tf.float32)

        # Draw example from the queue and separate
        self.example = tf.placeholder_with_default(self.queue.dequeue(), shape=[None, None, None, None, architecture.n_outputs + 1])
        self.image = self.example[:, :, :, :, :1]

        # Standardize each input image, using map because per_image_standardization takes one image at a time
        standardized_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.image)

        #Basic U-net with relu non-linearities and skip connections
        l0 = standardized_image
        l1 = tf.nn.relu(c1(l0)+b1)
        l2 = tf.nn.relu(c2(l1)+b2)
        l3 = tf.nn.relu(c3(l2)+b3)

        l3t = l3
        l2t = tf.nn.relu(c3t(l3t)+l2+b2t)
        l1t = tf.nn.relu(c2t(l2t)+l1+b1t)
        l0t = tf.nn.relu(c1t(l1t)+l0+b0t)

        self.prediction = tf.nn.sigmoid(l0t)
        self.binary_prediction = tf.round(self.prediction)

        self.saver = tf.train.Saver()

# sess=tf.Session()




# #Tensorflow convention is to represent volumes as (batch, x, y, z, channel)
# inpt = tf.ones((1,128,128,32,1))

# #Purely linear network, no skip connections
# downsampled = c3(c2(c1(inpt)))
# print "Coarsest layer dimensions:", downsampled._shape_as_list()
# upsampled = c1t(c2t(c3t(downsampled)))
# print "Output dimensions:", upsampled._shape_as_list()
# otpt1=upsampled

# otpt2 = l0t

# sess=tf.Session()
# sess.run(tf.initialize_all_variables())
# print "Running linear network"
# print sess.run(otpt1).shape
# print "Running u-net"
# print sess.run(otpt2).shape