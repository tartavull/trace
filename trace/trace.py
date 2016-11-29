# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import os
import subprocess
import h5py
import tensorflow as tf
import numpy as np

import snemi3d
from augmentation import batch_iterator
from thirdparty.segascorus import io_utils
from thirdparty.segascorus import utils
from thirdparty.segascorus.metrics import *


FOV = 95
OUTPT = 101
INPT = FOV + 2 * (OUTPT//2)

# Change this for each run to save the results in a new folder
tmp_dir = 'tmp/run2/'

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
        image = tf.placeholder(tf.float32, shape=[None, inpt, inpt, 1])
        target = tf.placeholder(tf.float32, shape=[None, out, out, 2])

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

        binary_prediction = tf.round(sigmoid_prediction) 
        pixel_error = tf.reduce_mean(tf.cast(tf.abs(binary_prediction - target), tf.float32))
        pixel_error_summary = tf.summary.scalar('pixel_error', pixel_error)
        validation_pixel_error_summary = tf.summary.scalar('validation pixel_error', pixel_error)

        rand_f_score = tf.placeholder(tf.float32)
        rand_f_score_merge = tf.placeholder(tf.float32)
        rand_f_score_split = tf.placeholder(tf.float32)
        vi_f_score = tf.placeholder(tf.float32)
        vi_f_score_merge = tf.placeholder(tf.float32)
        vi_f_score_split = tf.placeholder(tf.float32)

        rand_f_score_summary = tf.summary.scalar('rand f score', rand_f_score)
        rand_f_score_merge_summary = tf.summary.scalar('rand f merge score', rand_f_score_merge)
        rand_f_score_split_summary = tf.summary.scalar('rand f split score', rand_f_score_split)
        vi_f_score_summary = tf.summary.scalar('vi f score', vi_f_score)
        vi_f_score_merge_summary = tf.summary.scalar('vi f merge score', vi_f_score_merge)
        vi_f_score_split_summary = tf.summary.scalar('vi f split score', vi_f_score_split)

        score_summary_op = tf.summary.merge([rand_f_score_summary,
                                             rand_f_score_merge_summary,
                                             rand_f_score_split_summary,
                                             vi_f_score_summary,
                                             vi_f_score_merge_summary,
                                             vi_f_score_split_summary
                                            ])

        summary_op = tf.summary.merge([loss_summary,
                                       pixel_error_summary
                                       ])

       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return Net()

# Set validation to true and ensure that the appropriate truncated file is
# present to do validation testing.
def train(n_iterations=10000, validation=False):
    if validation:
        validation_input_file = h5py.File(snemi3d.folder()+'validation-input.h5','r')
        validation_input = validation_input_file['main'][:5,:,:].astype(np.float32) / 255.0
        num_validation_layers = validation_input.shape[0]
        mirrored_validation_input = _mirrorAcrossBorders(validation_input, FOV)
        validation_input_shape = mirrored_validation_input.shape[1]
        validation_output_shape = mirrored_validation_input.shape[1] - FOV + 1
        reshaped_validation_input = mirrored_validation_input.reshape(num_validation_layers, validation_input_shape, validation_input_shape, 1)
        validation_input_file.close()

        validation_label_file = h5py.File(snemi3d.folder()+'validation-affinities.h5','r')
        validation_labels = validation_label_file['main']
        reshaped_labels = np.einsum('dzyx->zyxd', validation_labels[0:2])
        validation_label_file.close()

    with tf.variable_scope('foo'):
        net = create_network(INPT, OUTPT)
    if validation:
        with tf.variable_scope('foo', reuse=True):
            validation_net = create_network(validation_input_shape, validation_output_shape)

    print ('Run tensorboard to visualize training progress')
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(
                       snemi3d.folder()+tmp_dir, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
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
            
            if validation and step % 100 == 0:
                # Measure validation error

                # Compute pixel error

                validation_sigmoid_prediction, validation_pixel_error_summary = \
                        sess.run([validation_net.sigmoid_prediction, validation_net.validation_pixel_error_summary],
                            feed_dict={validation_net.image: reshaped_validation_input,
                                       validation_net.target: reshaped_labels})

                summary_writer.add_summary(validation_pixel_error_summary, step)

                # Calculate rand and VI scores
                scores = _evaluateRandError(validation_sigmoid_prediction, num_validation_layers, validation_output_shape, watershed_high=0.95)
                score_summary = sess.run(net.score_summary_op,
                         feed_dict={net.rand_f_score: scores['Rand F-Score Full'],
                                    net.rand_f_score_merge: scores['Rand F-Score Merge'],
                                    net.rand_f_score_split: scores['Rand F-Score Split'],
                                    net.vi_f_score: scores['VI F-Score Full'],
                                    net.vi_f_score_merge: scores['VI F-Score Merge'],
                                    net.vi_f_score_split: scores['VI F-Score Split'],
                            })

                summary_writer.add_summary(score_summary, step)

            if step % 1000 == 0:
                # Save the variables to disk.
                save_path = net.saver.save(sess, snemi3d.folder()+tmp_dir + 'model.ckpt')
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break


def _mirrorAcrossBorders(data, fov):
    mirrored_data = np.zeros(shape=(data.shape[0], data.shape[1] + fov - 1, data.shape[2] + fov - 1))
    mirrored_data[:,fov//2:-(fov//2),fov//2:-(fov//2)] = data
    for i in range(data.shape[0]):
        # Mirror the left side
        mirrored_data[i,fov//2:-(fov//2),:fov//2] = np.fliplr(data[i,:,:fov//2])
        # Mirror the right side
        mirrored_data[i,fov//2:-(fov//2),-(fov//2):] = np.fliplr(data[i,:,-(fov//2):])
        # Mirror the top side
        mirrored_data[i,:fov//2,fov//2:-(fov//2)] = np.flipud(data[i,:fov//2,:])
        # Mirror the bottom side
        mirrored_data[i,-(fov//2):,fov//2:-(fov//2)] = np.flipud(data[i,-(fov//2):,:])
        # Mirror the top left corner
        mirrored_data[i,:fov//2,:fov//2] = np.fliplr(np.transpose(np.fliplr(np.transpose(data[i,:fov//2,:fov//2]))))
        # Mirror the top right corner
        mirrored_data[i,:fov//2,-(fov//2):] = np.transpose(np.fliplr(np.transpose(np.fliplr(data[i,:fov//2,-(fov//2):]))))
        # Mirror the bottom left corner
        mirrored_data[i,-(fov//2):,:fov//2] = np.transpose(np.fliplr(np.transpose(np.fliplr(data[i,-(fov//2):,:fov//2]))))
        # Mirror the bottom right corner
        mirrored_data[i,-(fov//2):,-(fov//2):] = np.fliplr(np.transpose(np.fliplr(np.transpose(data[i,-(fov//2):,-(fov//2):]))))
    return mirrored_data


def _evaluateRandError(sigmoid_prediction, num_layers, output_shape, watershed_high=0.9, watershed_low=0.3):
    # Save affinities to temporary file
    #TODO pad the image with zeros so that the ouput covers the whole dataset
    tmp_aff_file = 'validation-tmp-affinities.h5'
    tmp_label_file = 'validation-tmp-labels.h5'
    ground_truth_file = 'validation-generated-labels.h5'

    with h5py.File(snemi3d.folder()+tmp_dir+tmp_aff_file,'w') as output_file:
        output_file.create_dataset('main', shape=(3, num_layers, output_shape, output_shape))
        out = output_file['main']

        reshaped_pred = np.einsum('zyxd->dzyx', sigmoid_prediction)
        out[0:2,:,:,:] = reshaped_pred

    # Do watershed segmentation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir+"/thirdparty/watershed/watershed.jl",
                     snemi3d.folder()+tmp_dir+tmp_aff_file,
                     snemi3d.folder()+tmp_dir+tmp_label_file,
                     str(watershed_high),
                     str(watershed_low)])

    # Compute rand f score
    # --------------------

    # Parameters
    calc_rand_score = True
    calc_rand_error = False
    calc_variation_score = True
    calc_variation_information = False
    relabel2d = True
    foreground_restricted = True
    split_0_segment = True
    other = None

    seg1 = io_utils.import_file(snemi3d.folder()+tmp_dir+tmp_label_file)
    seg2 = io_utils.import_file(snemi3d.folder()+ground_truth_file)
    prep = utils.parse_fns(utils.prep_fns,
                            [relabel2d, foreground_restricted])
    seg1, seg2 = utils.run_preprocessing(seg1, seg2, prep)

    om = utils.calc_overlap_matrix(seg1, seg2, split_0_segment)

    #Calculating each desired metric
    metrics = utils.parse_fns( utils.metric_fns,
                                [calc_rand_score,
                                calc_rand_error,
                                calc_variation_score,
                                calc_variation_information] )

    results = {}
    for (name,metric_fn) in metrics:
        if relabel2d:
            full_name = "2D {}".format(name)
        else:
            full_name = name

        (f,m,s) = metric_fn( om, full_name, other )
        results["{} Full".format(name)] = f
        results["{} Merge".format(name)] = m
        results["{} Split".format(name)] = s

    print('Rand F-Score Full: ' + str(results['Rand F-Score Full']))
    print('Rand F-Score Split: ' + str(results['Rand F-Score Split']))
    print('Rand F-Score Merge: ' + str(results['Rand F-Score Merge']))
    print('VI F-Score Full: ' + str(results['VI F-Score Full']))
    print('VI F-Score Split: ' + str(results['VI F-Score Split']))
    print('VI F-Score Merge: ' + str(results['VI F-Score Merge']))

    return results

def predict():
    with h5py.File(snemi3d.folder()+'test-input.h5','r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        mirrored_inpt = _mirrorAcrossBorders(inpt, FOV)
        num_layers = mirrored_inpt.shape[0]
        input_shape = mirrored_inpt.shape[1]
        output_shape = mirrored_inpt.shape[1] - FOV + 1
        with h5py.File(snemi3d.folder()+'test-affinities.h5','w') as output_file:
            output_file.create_dataset('main', shape=(3,)+input_file['main'].shape)
            out = output_file['main']

            with tf.variable_scope('foo'):
                net = create_network(input_shape, output_shape)
            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder()+tmp_dir+'model.ckpt')
                print("Model restored.")

                pred = sess.run(net.sigmoid_prediction,
                        feed_dict={net.image: mirrored_inpt.reshape(num_layers, input_shape, input_shape, 1)})
                reshaped_pred = np.einsum('zyxd->dzyx', pred)
                out[0:2] = reshaped_pred
