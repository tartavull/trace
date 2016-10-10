# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function

import h5py
import tensorflow as tf
import numpy as np

import snemi3d
from augmentation import batch_iterator

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

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def create_network(learning_rate=0.001):
    class Net:
        # layer 0
        image = tf.placeholder(tf.float32, shape=[None, 95, 95, 1])
        target = tf.placeholder(tf.float32, shape=[None, 2])

        # layer 1
        W_conv1 = weight_variable([4, 4, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)

        # layer 2
        h_pool1 = max_pool_2x2(h_conv1)

        # layer 3
        W_conv2 = weight_variable([5, 5, 48, 48])
        b_conv2 = bias_variable([48])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # layer 4
        h_pool2 = max_pool_2x2(h_conv2)

        # layer 5
        W_conv3 = weight_variable([4, 4, 48, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        # layer 6
        h_pool3 = max_pool_2x2(h_conv3)

        # layer 7
        W_conv4 = weight_variable([4, 4, 48, 48])
        b_conv4 = bias_variable([48])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

        # layer 6
        h_pool4 = max_pool_2x2(h_conv4)

        # layer 9
        W_fc1 = weight_variable([3 * 3 * 48, 200])
        b_fc1 = bias_variable([200])

        h_pool4_flat = tf.reshape(h_pool4, [-1, 3*3*48])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

        # layer 10
        W_fc2 = weight_variable([200, 2])
        b_fc2 = bias_variable([2])
        prediction = tf.matmul(h_fc1, W_fc2) + b_fc2
        sigmoid_prediction = tf.sigmoid(prediction)

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,target))
        loss_summary = tf.scalar_summary('cross_entropy', cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return Net()

def train(n_iterations=10000):
    net = create_network()
    print ('Run tensorboard to visualize training progress')
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(
                       '/tmp/snemi3d', graph=sess.graph)

        sess.run(tf.initialize_all_variables())
        for step, (inputs, affinities) in enumerate(batch_iterator(50)):
            sess.run(net.train_step, 
                    feed_dict={net.image: inputs,
                               net.target: affinities})
            
            if step % 100 == 0:
                print ('step :'+str(step))
                summary = sess.run(net.loss_summary, 
                    feed_dict={net.image: inputs,
                               net.target: affinities})
            
                summary_writer.add_summary(summary, step)

            if step % 1000 == 0:
                # Save the variables to disk.
                save_path = net.saver.save(sess, "/tmp/snemi3d/model.ckpt")
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break

def predict():
    net = create_network()
    with tf.Session() as sess:
        # Restore variables from disk.
        net.saver.restore(sess, "/tmp/snemi3d/model.ckpt")
        print("Model restored.")
        with h5py.File(snemi3d.folder()+'test-input.h5','r') as input_file:
            inpt = input_file['main']
            with h5py.File(snemi3d.folder()+'test-affinities.h5','w') as output_file:
                output_file.create_dataset('main', shape=(3,)+input_file['main'].shape)
                out = output_file['main']
                for z in xrange(inpt.shape[0]):
                    print ('z: {} of {}'.format(z,inpt.shape[0]))
                    for y in xrange(0,inpt.shape[1]-95):
                        for x in xrange(0,inpt.shape[1]-95):
                            pred = sess.run(net.sigmoid_prediction,
                                feed_dict={net.image: inpt[z,y:y+95,x:x+95].reshape(1,95,95,1)})
                            out[:2,z,y+47,x+47] = pred[0]

