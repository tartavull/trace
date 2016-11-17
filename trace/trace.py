# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np

import snemi3d
from augmentation import batch_iterator
from augmentation import alternating_example_iterator


FOV = 95
OUTPT = 101
INPT = OUTPT + FOV - 1


def weight_variable(name, shape):
  """
  One should generally initialize weights with a small amount of noise
  for symmetry breaking, and to prevent 0 gradients.
  Since we're using ReLU neurons, it is also good practice to initialize
  them with a slightly positive initial bias to avoid "dead neurons".
  """
  #initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def unbiased_bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def create_simple_network(inpt, out, learning_rate=0.1):
    class Net:
        # layer 0
        image = tf.placeholder(tf.float32, shape=[1, inpt, inpt, 1])
        target = tf.placeholder(tf.float32, shape=[1, out, out, 2])
        targetImage = tf.image.resize_images(target[:,:out//2,:out//2,:], (out, out), method=1)
        input_summary = tf.image_summary('input image', image[:,FOV//2:FOV//2+out,FOV//2:FOV//2+out,:])
        target_x_summary = tf.image_summary('target x affinities', targetImage[:,:,:,:1])
        target_y_summary = tf.image_summary('target y affinities', targetImage[:,:,:,1:])

        # layer 1 - original stride 1
        W_conv1 = weight_variable('W_conv1', [FOV, FOV, 1, 2])
        b_conv1 = bias_variable([2])
        prediction = conv2d(image, W_conv1, dilation=1) + b_conv1

        w1_hist = tf.histogram_summary('W_conv1 weights', W_conv1)
        b1_hist = tf.histogram_summary('b_conv1 biases', b_conv1)
        prediction_hist = tf.histogram_summary('prediction activations', prediction)

        sigmoid_prediction = tf.nn.sigmoid(prediction)

        sigmoid_prediction_hist = tf.histogram_summary('sigmoid prediction activations', sigmoid_prediction)
        x_affinity_summary = tf.image_summary('x-affinity predictions', sigmoid_prediction[:,:,:,:1])
        y_affinity_summary = tf.image_summary('y-affinity predictions', sigmoid_prediction[:,:,:,1:])

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,targetImage))
        #cross_entropy = tf.reduce_mean(tf.mul(sigmoid_cross_entropy, (target - 1) * (-9) + 1))

        loss_summary = tf.scalar_summary('cross_entropy', cross_entropy)

        binary_prediction = tf.round(sigmoid_prediction) 
        pixel_error = tf.reduce_mean(tf.cast(tf.abs(binary_prediction - targetImage), tf.float32))
        pixel_error_summary = tf.scalar_summary('pixel_error', pixel_error)
        avg_affinity = tf.reduce_mean(tf.cast(targetImage, tf.float32))
        avg_affinity_summary = tf.scalar_summary('average_affinity', avg_affinity)

        summary_op = tf.merge_all_summaries()

        batch = tf.Variable(0)
        decaying_rate = tf.train.exponential_decay(
                learning_rate,
                batch,
                200,
                0.95,
                staircase=True)

        train_step = tf.train.MomentumOptimizer(decaying_rate, 0.9).minimize(cross_entropy, global_step=batch)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return Net()

def create_network(inpt, out, learning_rate=0.0001):
    class Net:
        # layer 0
        image = tf.placeholder(tf.float32, shape=[1, inpt, inpt, 1])
        target = tf.placeholder(tf.float32, shape=[1, out, out, 2])
        targetImage = tf.image.resize_images(target[:,:out//2,:out//2,:], (out, out), method=1)
        input_summary = tf.image_summary('input image', image[:,FOV//2:FOV//2+out,FOV//2:FOV//2+out,:])
        target_x_summary = tf.image_summary('target x affinities', targetImage[:,:,:,:1])
        target_y_summary = tf.image_summary('target y affinities', targetImage[:,:,:,1:])

        # layer 1 - original stride 1
        W_conv1 = weight_variable('W_conv1', [4, 4, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1, dilation=1) + b_conv1)

        w1_hist = tf.histogram_summary('W_conv1 weights', W_conv1)
        b1_hist = tf.histogram_summary('b_conv1 biases', b_conv1)
        h1_hist = tf.histogram_summary('h_conv1 activations', h_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1,1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable('W_conv2', [5, 5, 48, 48])
        b_conv2 = bias_variable([48])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=2) + b_conv2)

        w2_hist = tf.histogram_summary('W_conv2 weights', W_conv2)
        b2_hist = tf.histogram_summary('b_conv2 biases', b_conv2)
        h2_hist = tf.histogram_summary('h_conv2 activations', h_conv2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1,1], dilation=2)

        # layer 5 - original stride 1
        W_conv3 = weight_variable('W_conv3', [4, 4, 48, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=4) + b_conv3)

        w3_hist = tf.histogram_summary('W_conv3 weights', W_conv3)
        b3_hist = tf.histogram_summary('b_conv3 biases', b_conv3)
        h3_hist = tf.histogram_summary('h_conv3 activations', h_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1,1], dilation=4)


        # layer 7 - original stride 1
        W_conv4 = weight_variable('W_conv4', [4, 4, 48, 48])
        b_conv4 = bias_variable([48])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=8) + b_conv4)

        w4_hist = tf.histogram_summary('W_conv4 weights', W_conv4)
        b4_hist = tf.histogram_summary('b_conv4 biases', b_conv4)
        h4_hist = tf.histogram_summary('h_conv4 activations', h_conv4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(h_conv4, strides=[1,1], dilation=8)


        # layer 9 - original stride 1
        W_fc1 = weight_variable('W_fc1', [3, 3, 48, 200])
        b_fc1 = unbiased_bias_variable([200])
        h_fc1 = tf.nn.relu(conv2d(h_pool4, W_fc1, dilation=16) + b_fc1)

        w_fc1_hist = tf.histogram_summary('W_fc1 weights', W_fc1)
        b_fc1_hist = tf.histogram_summary('b_fc1 biases', b_fc1)
        h_fc1_hist = tf.histogram_summary('h_fc1 activations', h_fc1)

        # layer 10 - original stride 2
        W_fc2 = weight_variable('W_fc2', [1, 1, 200, 2])
        b_fc2 = unbiased_bias_variable([2])
        prediction = conv2d(h_fc1, W_fc2, dilation=16) + b_fc2

        w_fc2_hist = tf.histogram_summary('W_fc2 weights', W_fc2)
        b_fc2_hist = tf.histogram_summary('b_fc2 biases', b_fc2)
        prediction_hist = tf.histogram_summary('prediction activations', prediction)

        sigmoid_prediction = tf.nn.sigmoid(prediction)

        sigmoid_prediction_hist = tf.histogram_summary('sigmoid prediction activations', sigmoid_prediction)
        image_summary = tf.image_summary('x-affinity predictions', sigmoid_prediction[:,:,:,:1])

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,targetImage))
        #cross_entropy = tf.reduce_mean(tf.mul(sigmoid_cross_entropy, (targetImage - 1) * (-9) + 1))
        loss_summary = tf.scalar_summary('cross_entropy', cross_entropy)

        binary_prediction = tf.round(sigmoid_prediction) 
        pixel_error = tf.reduce_mean(tf.cast(tf.abs(binary_prediction - targetImage), tf.float32))
        pixel_error_summary = tf.scalar_summary('pixel_error', pixel_error)
        avg_affinity = tf.reduce_mean(tf.cast(target, tf.float32))
        avg_affinity_summary = tf.scalar_summary('average_affinity', avg_affinity)

        summary_op = tf.merge_all_summaries()

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return Net()

def train(n_iterations=200000):

    net = create_network(INPT, OUTPT)
    print ('Run tensorboard to visualize training progress')
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(
                       snemi3d.folder()+'tmp/longTrain3/', graph=sess.graph)

        sess.run(tf.initialize_all_variables())
        for step, (inputs, affinities) in enumerate(batch_iterator(FOV,OUTPT,INPT)):
            sess.run(net.train_step, 
                    feed_dict={net.image: inputs,
                               net.target: affinities})

            if step % 10 == 0:
                print ('step :'+str(step))
                summary = sess.run(net.summary_op,
                    feed_dict={net.image: inputs,
                               net.target: affinities})
            
                summary_writer.add_summary(summary, step)

            if step % 1000 == 0:
                # Save the variables to disk.
                save_path = net.saver.save(sess, snemi3d.folder()+"tmp/longTrain3/model.ckpt")
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break

def predict():
    from tqdm import tqdm
    with h5py.File(snemi3d.folder()+'test-input.h5','r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        with h5py.File(snemi3d.folder()+'test-affinities.h5','w') as output_file:
            output_file.create_dataset('main', shape=(3,)+input_file['main'].shape)
            out = output_file['main']

            inputShape = inpt.shape[1]
            outputShape = inpt.shape[1] - FOV + 1
            net = create_network(inputShape, outputShape)
            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder()+"tmp/longTrain3/model.ckpt")
                print("Model restored.")

                #TODO pad the image with zeros so that the ouput covers the whole dataset
                for z in xrange(inpt.shape[0]):
                    print ('z: {} of {}'.format(z,inpt.shape[0]))
                    pred = sess.run(net.sigmoid_prediction,
                            feed_dict={net.image: inpt[z].reshape(1, inputShape, inputShape, 1)})
                    reshapedPred = np.zeros(shape=(2, outputShape, outputShape))
                    reshapedPred[0] = pred[0,:,:,0].reshape(outputShape, outputShape)
                    reshapedPred[1] = pred[0,:,:,1].reshape(outputShape, outputShape)
                    out[0:2,
                        z,
                        FOV//2:FOV//2+outputShape,
                        FOV//2:FOV//2+outputShape] = reshapedPred