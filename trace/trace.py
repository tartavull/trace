# -*- coding: utf-8 -*-
# hsddfasdfadsf ttps://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np

import snemi3d
from augmentation import batch_iterator


FOV = 95
OUTPT = 101
INPT = FOV + 2 * (OUTPT//2)


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
  initial = tf.constant(.1)
  return tf.Variable(initial)

def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def deconv2d(x, W, shape):
  x_shape = tf.shape(x)
  output_shape = tf.pack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3]//2])
  return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='VALID')

# crops x, appends to y to form the output shape, and returns new tensor
def crop_and_concat(x, y, output_shape):
    offsets = tf.zeros(tf.pack([output_shape[0], 2]), dtype=tf.float32)
    y_shape = tf.shape(y)
    size = tf.pack((y_shape[1], y_shape[2]))
    x_crop = tf.image.extract_glimpse(x, size=size, offsets=offsets, centered=True)
    return tf.concat(3, [x_crop, y])

def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def create_network(inpt, out, learning_rate=0.001):
    class UNet:
        # ----------------------------------------------------------------------------------------
        # - Downsample and extraction phase                                                      -
        # ----------------------------------------------------------------------------------------
        # layer 0   - Input
        image = tf.placeholder(tf.float32, shape=[1, inpt, inpt, 1])
        target = tf.placeholder(tf.float32, shape=[1, out, out, 2])

        # layer 1   - 3x3 convoluution 
        W_conv1 = weight_variable([3, 3, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1, dilation=1) + b_conv1)

        # layer 2   - 3x3 convolution -- SAVED
        W_conv2 = weight_variable([3, 3, 48, 48])
        b_conv2 = bias_variable([48])
        # use cropped output from convolution
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, dilation=1) + b_conv2)

        # layer 3   - Downsample max-pool 
        h_conv3 = max_pool(h_conv2, strides=[1,1], dilation=1)

        # layer 4   - 3x3 convolution  
        W_conv4 = weight_variable([3, 3, 48, 96])
        b_conv4 = bias_variable([96])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, dilation=2) + b_conv4)

        # layer 5   - 3x3 convolution  
        W_conv5 = weight_variable([3, 3, 96, 96])
        b_conv5 = bias_variable([96])
        # use cropped output from convolution
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, dilation=2) + b_conv5)

        # layer 6   - Downsample max-pool 
        h_conv6 = max_pool(h_conv5, strides=[1,1], dilation=2)

        # layer 7   - 3x3 convolution  
        W_conv7 = weight_variable([3, 3, 96, 192])
        b_conv7 = bias_variable([192])
        h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7, dilation=4) + b_conv7)

        # layer 8   - 3x3 convolution  
        W_conv8 = weight_variable([3, 3, 192, 192])
        b_conv8 = bias_variable([192])
        # use cropped output from convolution
        h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8, dilation=4) + b_conv8)  

        # layer 9   - Downsample max-pool 
        h_conv9 = max_pool(h_conv8, strides=[1,1], dilation=4)

        # layer 10  - 3x3 convolution  
        W_conv10 = weight_variable([3, 3, 192, 384])
        b_conv10 = bias_variable([384])
        h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10, dilation=8  ) + b_conv10)

        # layer 11  - 3x3 convolution  
        W_conv11 = weight_variable([3, 3, 384, 384])
        b_conv11 = bias_variable([384])
        # use cropped output from convolution
        h_conv11 = tf.nn.relu(conv2d(h_conv10, W_conv11, dilation=8 ) + b_conv11)

        # layer 12   - Downsample max-pool
        h_conv12 = max_pool(h_conv11, strides=[1,1], dilation=8 )

        # layer 13  - 3x3 convolution  
        W_conv13 = weight_variable([3, 3, 384, 768])
        b_conv13 = bias_variable([768])
        h_conv13 = tf.nn.relu(conv2d(h_conv12, W_conv13, dilation=16) + b_conv13)

        # layer 14  - 3x3 convolution  
        W_conv14 = weight_variable([3, 3, 768, 768])
        b_conv14 = bias_variable([768])
        h_conv14 = tf.nn.relu(conv2d(h_conv13, W_conv14, dilation=16) + b_conv14)

        # ----------------------------------------------------------------------------------------
        # - Upsample and reconstruction phase                                                    -
        # ----------------------------------------------------------------------------------------
        # layer 15   - up-sample 2x2 convolution  
        W_conv15 = weight_variable([2, 2, 768, 768])
        b_conv15 = bias_variable([768])
        h_conv15 = tf.nn.relu(deconv2d(h_conv14, W_conv15, [384, 768]) + b_conv15)

        # layer 15.5 - combine upsampling with Step 11.5
        comb_conv15 = crop_and_concat(h_conv11, h_conv15, [1])

        # layer 16   - 3x3 convolution  
        W_conv16 = weight_variable([3, 3, 768, 384])
        b_conv16 = bias_variable([384])
        h_conv16 = tf.nn.relu(conv2d(comb_conv15, W_conv16, dilation=8) + b_conv16)

        # layer 17  - 3x3 convolution  
        W_conv17 = weight_variable([3, 3, 384, 384])
        b_conv17 = bias_variable([384])
        h_conv17 = tf.nn.relu(conv2d(h_conv16, W_conv17, dilation=8) + b_conv17)
        
        # layer 18   - up-sample 2x2 convolution  
        W_conv18 = weight_variable([2, 2, 384, 384])
        b_conv18 = bias_variable([384])
        h_conv18 = tf.nn.relu(deconv2d(h_conv17, W_conv18, [384, 384]) + b_conv18)

        # layer 18.5 - combine upsampling with Step 8.5
        comb_conv18 = crop_and_concat(h_conv8, h_conv18, [1])

        # layer 19   - 3x3 convolution  
        W_conv19 = weight_variable([3, 3, 384, 192])
        b_conv19 = bias_variable([192])
        h_conv19 = tf.nn.relu(conv2d(comb_conv18, W_conv19, dilation=4) + b_conv19)

        # layer 20  - 3x3 convolution  
        W_conv20 = weight_variable([3, 3, 192, 192])
        b_conv20 = bias_variable([192])
        h_conv20 = tf.nn.relu(conv2d(h_conv19, W_conv20, dilation=4) + b_conv20)
        
        # layer 21   - up-sample 2x2 convolution  
        W_conv21 = weight_variable([2, 2, 192, 192])
        b_conv21 = bias_variable([192])
        h_conv21 = tf.nn.relu(deconv2d(h_conv20, W_conv21, [192, 192]) + b_conv21)

        # layer 21.5 - combine upsampling with Step 5.5
        comb_conv21 = crop_and_concat(h_conv5, h_conv21, [1])

        # layer 22   - 3x3 convolution  
        W_conv22 = weight_variable([3, 3, 192, 96])
        b_conv22 = bias_variable([96])
        h_conv22 = tf.nn.relu(conv2d(comb_conv21, W_conv22, dilation=2) + b_conv22)

        # layer 23  - 3x3 convolution  
        W_conv23 = weight_variable([3, 3, 96, 96])
        b_conv23 = bias_variable([96])
        h_conv23 = tf.nn.relu(conv2d(h_conv22, W_conv23, dilation=2) + b_conv23)
        
        # layer 24   - up-sample 2x2 convolution  
        W_conv24 = weight_variable([2, 2, 96, 96])
        b_conv24 = bias_variable([96])
        h_conv24 = tf.nn.relu(deconv2d(h_conv23, W_conv24, [96, 96]) + b_conv24)

        # layer 24.5 - combine upsampling with Step 2.5
        comb_conv24 = crop_and_concat(h_conv2, h_conv24, [1])

        # layer 25   - 3x3 convolution  
        W_conv25 = weight_variable([3, 3, 96, 48])
        b_conv25 = bias_variable([48])
        h_conv25 = tf.nn.relu(conv2d(comb_conv24, W_conv25, dilation=1) + b_conv25)

        # layer 26  - 3x3 convolution  
        W_conv26 = weight_variable([3, 3, 48, 24])
        b_conv26 = bias_variable([24])
        h_conv26 = tf.nn.relu(conv2d(h_conv25, W_conv26, dilation=1) + b_conv26)

        # layer 27   - 1x1 convolution                  
        W_conv27 = weight_variable([1, 1, 24, 2])
        b_conv27 = bias_variable([2])
        prediction = conv2d(h_conv26, W_conv27, dilation=1) + b_conv27
        # ----------------------------------------------------------------------------------------
        sigmoid_prediction = tf.nn.sigmoid(prediction)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,target))
        loss_summary = tf.scalar_summary('cross_entropy', cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    class Net:
        # layer 0
        image = tf.placeholder(tf.float32, shape=[1, inpt, inpt, 1])
        target = tf.placeholder(tf.float32, shape=[1, out, out, 2])

        # layer 1 - original stride 1
        W_conv1 = weight_variable([4, 4, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1, dilation=1) + b_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1,1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, 48, 48])
        b_conv2 = bias_variable([48])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=2) + b_conv2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1,1], dilation=2)

        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, 48, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=4) + b_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1,1], dilation=4)

        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, 48, 48])
        b_conv4 = bias_variable([48])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=8) + b_conv4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(h_conv4, strides=[1,1], dilation=8)

        # layer 9 - original stride 1
        W_fc1 = weight_variable([3, 3, 48, 200])
        b_fc1 = bias_variable([200])
        h_fc1 = tf.nn.relu(conv2d(h_pool4, W_fc1, dilation=16) + b_fc1)

        # layer 10 - original stride 2
        W_fc2 = weight_variable([1, 1, 200, 2])
        b_fc2 = bias_variable([2])
        prediction = conv2d(h_fc1, W_fc2, dilation=16) + b_fc2

        sigmoid_prediction = tf.nn.sigmoid(prediction)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,target))
        loss_summary = tf.scalar_summary('cross_entropy', cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return Net()

def train(n_iterations=10000):

    net = create_network(INPT, OUTPT)
    print ('Run tensorboard to visualize training progress')
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(
                       snemi3d.folder()+'tmp/', graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        for step, (inputs, affinities) in enumerate(batch_iterator(FOV,OUTPT,INPT)):
            sess.run(net.train_step, 
                    feed_dict={net.image: inputs,
                               net.target: affinities})
            
            if step % 10 == 0:
                print ('step :'+str(step))
                summary = sess.run(net.loss_summary, 
                    feed_dict={net.image: inputs,
                               net.target: affinities})
            
                summary_writer.add_summary(summary, step)

            if step % 1000 == 0:
                # Save the variables to disk.
                save_path = net.saver.save(sess, snemi3d.folder()+"tmp/model.ckpt")
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break

def predict():
    from tqdm import tqdm
    net = create_network(INPT, OUTPT)
    with tf.Session() as sess:
        # Restore variables from disk.
        net.saver.restore(sess, snemi3d.folder()+"tmp/model.ckpt")
        print("Model restored.")
        with h5py.File(snemi3d.folder()+'test-input.h5','r') as input_file:
            inpt = input_file['main'][:].astype(np.float32) / 255.0
            with h5py.File(snemi3d.folder()+'test-affinities.h5','w') as output_file:
                output_file.create_dataset('main', shape=(3,)+input_file['main'].shape)
                out = output_file['main']

                #TODO pad the image with zeros so that the ouput covers the whole dataset
                for z in xrange(inpt.shape[0]):
                    print ('z: {} of {}'.format(z,inpt.shape[0]))
                    for y in xrange(0,inpt.shape[1]-INPT, OUTPT):
                        for x in xrange(0,inpt.shape[1]-INPT, OUTPT):
                            pred = sess.run(net.sigmoid_prediction,
                                feed_dict={net.image: inpt[z,y:y+INPT,x:x+INPT].reshape(1,INPT,INPT,1)})
                            reshapedPred = np.zeros(shape=(2, OUTPT, OUTPT))
                            reshapedPred[0] = pred[0,:,:,0].reshape(OUTPT, OUTPT)
                            reshapedPred[1] = pred[0,:,:,1].reshape(OUTPT, OUTPT)
                            out[0:2,
                                z,
                                y+FOV//2:y+FOV//2+OUTPT,
                                x+FOV//2:x+FOV//2+OUTPT] = reshapedPred
