# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
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
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def create_network(inpt, out, learning_rate=0.001):
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

        sess.run(tf.initialize_all_variables())
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
