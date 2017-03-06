import tensorflow as tf
import numpy as np
dtype=tf.float32
shape_dict3d={}
import pprint
import operator
from .common import *
from utils import *
from collections import OrderedDict

class UNetAltArchitecture(Architecture):
    def __init__(self, model_name, output_mode, layers):
        super(UNetAltArchitecture, self).__init__(model_name, output_mode, layers)
        self.layers = layers
        self.fov = 1
        self.z_fov = 1

U_NET = UNetAltArchitecture(
    model_name='unet_alt',
    output_mode=AFFINITIES_3D,
    layers=[],
)

def default_Unet():
    params = {
        'lr': 0.001,
        'out': 101
    }
    return Unet(params)

def make_variable(shape, val=0.0):
    initial = tf.constant(val, dtype=dtype, shape=shape)
    var = tf.Variable(initial, dtype=dtype)
    return var

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
    def __init__(self, dict_key, size=(4,4,1), strides=(2,2,1), n_lower=1, n_upper=1,stddev=0.5,dtype=dtype):
        initial = tf.truncated_normal([size[0],size[1],size[2],n_lower,n_upper], stddev=stddev, dtype=dtype)
        self.weights=tf.Variable(initial, dtype=dtype)
        self.size=size
        self.strides=[1,strides[0],strides[1],strides[2],1]
        self.n_lower=n_lower
        self.n_upper=n_upper
        self.dict_key = dict_key

        #up_coeff and down_coeff are coefficients meant to keep the magnitude of the output independent\ of stride and size choices                                                                                          
        self.up_coeff = 1.0/np.sqrt(reduce(operator.mul,size)*n_lower)
        self.down_coeff = 1.0/np.sqrt(reduce(operator.mul,size)/(reduce(operator.mul,strides))*n_upper)

        
    def transpose(self):
        return TransposeKernel(self)

    def __call__(self,x):

        with tf.name_scope('conv3d') as scope:
            self.in_shape = tf.shape(x)
            tmp=tf.nn.conv3d(x, self.up_coeff*self.weights, strides=self.strides, padding='VALID')
            x_shape = tf.shape(x)
            shape_dict3d[self.dict_key]=x_shape[1:4]
        return tmp

    def transpose_call(self,x):
        with tf.name_scope('conv3d_t') as scope:
            if not hasattr(self,"in_shape"):
                res = tuple(x._shape_as_list()[1:4])
                self.in_shape=tf.concat([shape_dict3d[self.dict_key], tf.stack([self.n_lower,])],0)
            full_in_shape = tf.concat([tf.shape(x)[:1], self.in_shape], 0)
            ret = tf.nn.conv3d_transpose(x, self.down_coeff*self.weights, output_shape=full_in_shape, strides=self.strides, padding='VALID')

        return tf.reshape(ret, full_in_shape)

class UNet_Alt(Model):
    def __init__(self, architecture, is_training=False):
        super(UNet_Alt, self).__init__(architecture)
        learning_rate = .001
        self.out = 101
        self.fov = 1
        self.inpt = self.fov + 2 * (self.out // 2)
        self.z_fov = 1
        prev_layer = self.image

        #convolution variables                                                                                 
        c0=ConvKernel3d(dict_key = "0", size=(1,1,1), strides=(1,1,1), n_lower=1, n_upper=3)
        c1=ConvKernel3d(dict_key = "1", size=(1,4,4), strides=(1,2,2), n_lower=3, n_upper=12)
        c2=ConvKernel3d(dict_key = "2", size=(1,4,4), strides=(1,2,2), n_lower=12, n_upper=24)
        c3=ConvKernel3d(dict_key = "3", size=(4,4,4), strides=(2,2,2), n_lower=24, n_upper=48)

        c1t=ConvKernel3d(dict_key = "1", size=(1,4,4), strides=(1,2,2), n_lower=3, n_upper=12).transpose()
        c2t=ConvKernel3d(dict_key = "2", size=(1,4,4), strides=(1,2,2), n_lower=12, n_upper=24).transpose()
        c3t=ConvKernel3d(dict_key = "3", size=(4,4,4), strides=(2,2,2), n_lower=24, n_upper=48).transpose()

    #bias variables                                                                                      
        b0_0 = make_variable([1,1,1,1,1])
        b0=make_variable([1,1,1,1,1])
        b1=make_variable([1,1,1,1,12])
        b2=make_variable([1,1,1,1,24])
        b3=make_variable([1,1,1,1,48])

        b0t=make_variable([1,1,1,1,1])
        b1t=make_variable([1,1,1,1,12])
        b2t=make_variable([1,1,1,1,24])

        #Tensorflow convention is to represent volumes as (batch, x, y, z, channel)                            
#        inpt = tf.ones((1,128,128,32,1))                                                                      

#        self.image = tf.placeholder(tf.float32, shape=[1, 128, 128, 32, 1])
#        self.target = tf.placeholder(tf.float32, shape=[None, None, None, None, 3])
        #Basic U-net with relu non-linearities and skip connections                                            
        l0 = self.image   #inpt                                                                                
        l0_0 = tf.nn.relu(c0(l0) + b0_0)
        l1 = tf.nn.relu(c1(l0_0)+b1)
        l2 = tf.nn.relu(c2(l1)+b2)
        l3 = tf.nn.relu(c3(l2)+b3)

        l3t = l3
        l2t = tf.nn.relu(c3t(l3t)+l2+b2t)
        l1t = tf.nn.relu(c2t(l2t)+l1+b1t)
        l0t = tf.nn.relu(c1t(l1t)+l0+b0t)

        otpt2 = l0t
        self.prediction = otpt2
        self.binary_prediction=tf.round(self.prediction)
        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
#        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = tf.reshape(self.prediction\
    #, [-1,2]), labels = tf.reshape(self.target, [-1,2]))) #maybe take out reshaping if you need to          
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l0t, labels = self.target))
            
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        self.saver = tf.train.Saver()
        self.model_name = "unet3d_first"
