# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import subprocess
import h5py
import tifffile
import tensorflow as tf
import numpy as np
import dataprovider.transform as transform

import snemi3d
from augmentation import batch_iterator
from augmentation import alternating_example_iterator
from thirdparty.segascorus import io_utils
from thirdparty.segascorus import utils
from thirdparty.segascorus.metrics import *


FOV = 189
OUTPT = 192
INPT = 380

FULL_FOV = 191
FULL_INPT = 702
FULL_OUTPT = 512

tmp_dir = 'tmp/unet_dropout_varying_lr_0.0001/'


def weight_variable(name, shape):
  """
  One should generally initialize weights with a small amount of noise
  for symmetry breaking, and to prevent 0 gradients.
  Since we're using ReLU neurons, it is also good practice to initialize
  them with a slightly positive initial bias to avoid "dead neurons".
  """
  #initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def bias_variable(name, shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name, initializer=initial)

def unbiased_bias_variable(name, shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name, initializer=initial)

def conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def same_conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='SAME', dilation_rate= [dilation, dilation])

def max_pool(x, dilation=1, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def conv2d_transpose(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def crop_and_concat(x1, x2, batch_size):
    offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
    x2_shape = tf.shape(x2)
    size = tf.pack([x2_shape[1], x2_shape[2]])
    x1_crop = tf.image.extract_glimpse(x1, size=size, offsets=offsets, centered=True)
    return tf.concat(3, [x1_crop, x2])

def dropout(x, keep_prob):
    mask = tf.ones(x.get_shape()[3])
    dropoutMask = tf.nn.dropout(mask, keep_prob)
    return x * dropoutMask



def createHistograms(name2var):
    listOfSummaries = []
    for name, var in name2var.iteritems():
        listOfSummaries.append(tf.summary.histogram(name, var))
    return tf.summary.merge(listOfSummaries)


def create_unet(image, target, keep_prob, layers=5, features_root=64, kernel_size=3, learning_rate=0.0001):
    '''
    Creates a new U-Net for the given parametrization.
    
    :param x: input tensor variable, shape [?, ny, nx, channels]
    :param keep_prob: dropout probability tensor
    :param layers: number of layers in the unet
    :param features_root: number of features in the first layer
    :param kernel_size: size of the convolutional kernel
    :param learning_rate: learning rate for the optimizer
    '''

    class UNet:
        in_node = image
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
        for layer in range(layers):
            num_feature_maps = 2**layer * features_root
            layer_str = 'layer' + str(layer)

            # Input layer maps a 1-channel image to num_feature_maps channels
            if layer == 0:
                w1 = weight_variable(layer_str + '_w1', [kernel_size, kernel_size, 1, num_feature_maps]) 
            else:
                w1 = weight_variable(layer_str + '_w1', [kernel_size, kernel_size, num_feature_maps//2, num_feature_maps]) 
            w2 = weight_variable(layer_str + '_w2', [kernel_size, kernel_size, num_feature_maps, num_feature_maps]) 
            b1 = bias_variable(layer_str + '_b1',  [num_feature_maps])
            b2 = bias_variable(layer_str + '_b2', [num_feature_maps])

            conv1 = conv2d(in_node, w1)
            h_conv1 = dropout(tf.nn.elu(conv1 + b1), 0.9)
            h_conv2 = dropout(tf.nn.elu(conv2d(h_conv1, w2) + b2), 0.9)
            dw_h_convs[layer] = h_conv2

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((h_conv1, h_conv2))
            histogram_dict[layer_str + '_in_node'] = in_node
            histogram_dict[layer_str + '_w1'] = w1
            histogram_dict[layer_str + '_w2'] = w2
            histogram_dict[layer_str + '_b1'] = b1
            histogram_dict[layer_str + '_b2'] = b2
            histogram_dict[layer_str + '_h_conv1'] = h_conv1
            histogram_dict[layer_str + '_h_conv2'] = h_conv2

            size -= 4

            h_conv2_packed = computeGridSummary(h_conv2, num_feature_maps, size)
            image_summaries.append(tf.summary.image(layer_str + '  activations', h_conv2_packed))

            # If not the bottom layer, do a max-pool
            if layer < layers -1:
                pools[layer] = max_pool(h_conv2)
                in_node = pools[layer]
                size //= 2


        in_node = dw_h_convs[layers-1]

        # Up layers
        for layer in range(layers - 2, -1, -1):
            layer_str = 'layer_u' + str(layer)
            num_feature_maps = 2**layer * features_root

            wu = weight_variable(layer_str + '_wu', [kernel_size, kernel_size, num_feature_maps, num_feature_maps * 2])
            bu = bias_variable(layer_str + '_bu', [num_feature_maps])
            h_upconv = tf.nn.elu(conv2d_transpose(in_node, wu, stride=2) + bu)
            h_upconv_concat = crop_and_concat(dw_h_convs[layer], h_upconv, batch_size)
            upconvs[layer] = h_upconv_concat

            w1 = weight_variable(layer_str + '_w1', [kernel_size, kernel_size, num_feature_maps * 2, num_feature_maps])
            w2 = weight_variable(layer_str + '_w2', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            b1 = bias_variable(layer_str + '_b1', [num_feature_maps])
            b2 = bias_variable(layer_str + '_b2', [num_feature_maps])

            h_conv1 = dropout(tf.nn.elu(conv2d(h_upconv_concat, w1) + b1), 0.95)
            in_node = dropout(tf.nn.elu(conv2d(h_conv1, w2) + b2), 0.95)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((h_conv1, in_node))
            histogram_dict[layer_str + '_wu'] = wu
            histogram_dict[layer_str + '_bu'] = bu
            histogram_dict[layer_str + '_h_upconv'] = h_upconv
            histogram_dict[layer_str + '_h_upconv_concat'] = h_upconv_concat
            histogram_dict[layer_str + '_w1'] = w1
            histogram_dict[layer_str + '_w2'] = w2
            histogram_dict[layer_str + '_b1'] = b1
            histogram_dict[layer_str + '_b2'] = b2
            histogram_dict[layer_str + '_h_conv1'] = h_conv1
            histogram_dict[layer_str + '_h_conv2'] = in_node

            size *= 2
            size -= 4

            h_conv2_packed = computeGridSummary(in_node, num_feature_maps, size)
            image_summaries.append(tf.summary.image(layer_str + '  activations', h_conv2_packed))



        # Output map
        w_o = weight_variable('w_o', [5, 5, features_root, 1])
        b_o = bias_variable('b_o', [1])
        prediction = conv2d(in_node, w_o) + b_o
        sigmoid_prediction = tf.nn.sigmoid(prediction)

        histogram_dict['prediction'] = prediction
        histogram_dict['sigmoid prediction'] = sigmoid_prediction

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,target))

        loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)

        padding = (in_size - size) // 2

        image_summaries.append(tf.summary.image('input image', image))
        mirrored_input_summary = tf.summary.image('mirrored input image', image)
        image_summaries.append(tf.summary.image('boundary output patch', image[:,padding:-padding,padding:-padding,:]))
        validation_output_patch_summary = tf.summary.image('validation output patch', image[:,padding:-padding,padding:-padding,:])
        image_summaries.append(tf.summary.image('boundary targets', target))
        validation_target_summary = tf.summary.image('validation boundary targets', target)

        image_summaries.append(tf.summary.image('boundary predictions', sigmoid_prediction))
        validation_boundary_summary = tf.summary.image('validation boundary predictions', sigmoid_prediction)

        binary_prediction = tf.round(sigmoid_prediction) 
        pixel_error = tf.reduce_mean(tf.cast(tf.abs(binary_prediction - target), tf.float32))
        pixel_error_summary = tf.summary.scalar('pixel_error', pixel_error)
        training_pixel_error_summary = tf.summary.scalar('training pixel_error', pixel_error)
        validation_pixel_error_summary = tf.summary.scalar('validation pixel_error', pixel_error)


        rand_f_score = tf.placeholder(tf.float32)
        rand_f_score_merge = tf.placeholder(tf.float32)
        rand_f_score_split = tf.placeholder(tf.float32)
        vi_f_score = tf.placeholder(tf.float32)
        vi_f_score_merge = tf.placeholder(tf.float32)
        vi_f_score_split = tf.placeholder(tf.float32)

        training_rand_f_score_summary = tf.summary.scalar('training rand f score', rand_f_score)
        training_rand_f_score_merge_summary = tf.summary.scalar('training rand f merge score', rand_f_score_merge)
        training_rand_f_score_split_summary = tf.summary.scalar('training rand f split score', rand_f_score_split)
        training_vi_f_score_summary = tf.summary.scalar('training vi f score', vi_f_score)
        training_vi_f_score_merge_summary = tf.summary.scalar('training vi f merge score', vi_f_score_merge)
        training_vi_f_score_split_summary = tf.summary.scalar('training vi f split score', vi_f_score_split)

        validation_rand_f_score_summary = tf.summary.scalar('validation rand f score', rand_f_score)
        validation_rand_f_score_merge_summary = tf.summary.scalar('validation rand f merge score', rand_f_score_merge)
        validation_rand_f_score_split_summary = tf.summary.scalar('validation rand f split score', rand_f_score_split)
        validation_vi_f_score_summary = tf.summary.scalar('validation vi f score', vi_f_score)
        validation_vi_f_score_merge_summary = tf.summary.scalar('validation vi f merge score', vi_f_score_merge)
        validation_vi_f_score_split_summary = tf.summary.scalar('validation vi f split score', vi_f_score_split)


        training_score_summary_op = tf.summary.merge([training_rand_f_score_summary,
                                             training_rand_f_score_merge_summary,
                                             training_rand_f_score_split_summary,
                                             training_vi_f_score_summary,
                                             training_vi_f_score_merge_summary,
                                             training_vi_f_score_split_summary
                                            ])

        validation_score_summary_op = tf.summary.merge([validation_rand_f_score_summary,
                                             validation_rand_f_score_merge_summary,
                                             validation_rand_f_score_split_summary,
                                             validation_vi_f_score_summary,
                                             validation_vi_f_score_merge_summary,
                                             validation_vi_f_score_split_summary
                                            ])


        histograms = createHistograms(histogram_dict)

        summary_op = tf.summary.merge([histograms,
                                       loss_summary,
                                       pixel_error_summary,
                                       ])

        image_summary_op = tf.summary.merge(image_summaries)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return UNet()

def computeGridSummary(h_conv, num_maps, map_size, width=16, height=0):
    cx = width
    if height == 0:
        cy = num_maps // width
    else:
        cy = height

    # Compute image summaries of the num_maps feature maps
    iy = map_size
    ix = iy
    h_conv_packed = tf.reshape(h_conv[0], tf.pack([iy, ix, num_maps]))
    iy += 4
    ix += 4
    h_conv_packed = tf.image.resize_image_with_crop_or_pad(h_conv_packed, iy, ix)
    h_conv_packed = tf.reshape(h_conv_packed, tf.pack([iy, ix, cy, cx]))
    h_conv_packed = tf.transpose(h_conv_packed, (2,0,3,1)) # cy, iy,  cx, ix
    h_conv_packed = tf.reshape(h_conv_packed, tf.pack([1, cy * iy, cx * ix, 1]))
    return h_conv_packed

def import_image(path, sess, isInput, fov):
    with h5py.File(path, 'r') as f:
        # f['main'] has shape [z,y,x]
        image_data = f['main'][:].astype(np.float32) / 255.0
        if isInput:
            image_data = _mirrorAcrossBorders(image_data, fov)
        image_data = image_data[:,:,:,np.newaxis]
        
        image_ph = tf.placeholder(dtype=image_data.dtype,
                                  shape=image_data.shape)
        # Setting trainable=False keeps the variable out of the
        # Graphkeys.TRAINABLE_VARIABLES collection in the graph, so we won't try
        # and update it when we train. Setting colltions = [] keeps the variable
        # out of the GraphKeys.VARIABLES collection used for saving and
        # restoring checkpoints.
        image_t = tf.Variable(image_ph, trainable=False, collections=[])

        sess.run(image_t.initializer,
            feed_dict={image_ph: image_data})
        del image_data

        return image_t


def train(n_iterations=200000):
        inpt_placeholder = tf.placeholder(tf.float32, [1, INPT, INPT, 1])
        inpt = tf.get_variable('input', [1, INPT, INPT, 1])
        assign_input = tf.assign(inpt, inpt_placeholder)

        target_placeholder = tf.placeholder(tf.float32, [1, OUTPT, OUTPT, 1])
        target = tf.get_variable('target', [1, OUTPT, OUTPT, 1])
        assign_target = tf.assign(target, target_placeholder)
        
        with tf.variable_scope('foo'):
            net = create_unet(inpt, target, keep_prob=0.8, learning_rate=0.0001)

        print ('Run tensorboard to visualize training progress')
        with tf.Session() as sess:
            training_input = import_image(snemi3d.folder()+'training-input.h5', sess, isInput=True, fov=FULL_FOV)
            training_labels = import_image(snemi3d.folder()+'training-labels.h5', sess, isInput=False, fov=FULL_FOV)
            validation_input = import_image(snemi3d.folder()+'validation-input.h5', sess, isInput=True, fov=FULL_FOV)
            validation_labels = import_image(snemi3d.folder()+'validation-labels.h5', sess, isInput=False, fov=FULL_FOV)

            num_layers = training_input.get_shape()[0]
            input_size = training_input.get_shape()[1]
            output_size = training_labels.get_shape()[1]

            training_input_slice = tf.Variable(tf.zeros([1, input_size, input_size, 1]), trainable=False, collections=[], name='input-slice')
            training_label_slice = tf.Variable(tf.zeros([1, output_size, output_size, 1]), trainable=False, collections=[], name='label-slice')

            sess.run(training_input_slice.initializer)
            sess.run(training_label_slice.initializer)

            getInputSlice = []
            getLabelSlice = []
            for layer in range(num_layers):
                getInputSlice.append(training_input_slice.assign(tf.gather(training_input, [layer])))
                getLabelSlice.append(training_label_slice.assign(tf.gather(training_labels, [layer])))


            with tf.variable_scope('foo', reuse=True):
                training_net = create_unet(training_input_slice, training_label_slice, keep_prob=1.0)
                validation_net = create_unet(validation_input, validation_labels, keep_prob=1.0)

            summary_writer = tf.train.SummaryWriter(
                           snemi3d.folder()+tmp_dir, graph=sess.graph)

            sess.run(tf.global_variables_initializer())
            for step, (inputs, boundaries) in enumerate(batch_iterator(FOV,OUTPT,INPT)):
                sess.run(assign_input,
                        feed_dict={inpt_placeholder: inputs})
                sess.run(assign_target,
                        feed_dict={target_placeholder: boundaries / 255.0})
                sess.run(net.train_step)

                if step % 10 == 0:
                    print ('step :'+str(step))
                    summary = sess.run(net.summary_op)
                
                    summary_writer.add_summary(summary, step)

                if step % 1000 == 0:
                    image_summary = sess.run(net.image_summary_op)
                
                    summary_writer.add_summary(image_summary, step)

                    # Save the variables to disk.
                    save_path = net.saver.save(sess, snemi3d.folder()+tmp_dir + 'model.ckpt')
                    print("Model saved in file: %s" % save_path)

                    # Measure training errror

                    training_sigmoid_prediction = np.empty((num_layers, output_size, output_size, 1))
                    for layer in range(num_layers):
                        sess.run(getInputSlice[layer])
                        sess.run(getLabelSlice[layer])
                        training_sigmoid_prediction[layer:layer+1] = sess.run(training_net.sigmoid_prediction)


                    training_scores = _evaluateRandError('training', training_sigmoid_prediction, watershed_high=0.95)
                    training_score_summary = sess.run(training_net.training_score_summary_op,
                             feed_dict={training_net.rand_f_score: training_scores['Rand F-Score Full'],
                                        training_net.rand_f_score_merge: training_scores['Rand F-Score Merge'],
                                        training_net.rand_f_score_split: training_scores['Rand F-Score Split'],
                                        training_net.vi_f_score: training_scores['VI F-Score Full'],
                                        training_net.vi_f_score_merge: training_scores['VI F-Score Merge'],
                                        training_net.vi_f_score_split: training_scores['VI F-Score Split'],
                                })

                    summary_writer.add_summary(training_score_summary, step)

                    # Measure validation error

                    validation_sigmoid_prediction, validation_pixel_error_summary, mirrored_input_summary, validation_output_patch_summary, validation_target_summary, validation_boundary_summary = \
                            sess.run([validation_net.sigmoid_prediction, 
                                      validation_net.validation_pixel_error_summary,
                                      validation_net.mirrored_input_summary,
                                      validation_net.validation_output_patch_summary,
                                      validation_net.validation_target_summary,
                                      validation_net.validation_boundary_summary])

                    summary_writer.add_summary(mirrored_input_summary, step)
                    summary_writer.add_summary(validation_pixel_error_summary, step)
                    summary_writer.add_summary(validation_output_patch_summary, step)
                    summary_writer.add_summary(validation_target_summary, step)
                    summary_writer.add_summary(validation_boundary_summary, step)

                    validation_scores = _evaluateRandError('validation', validation_sigmoid_prediction, watershed_high=0.95)
                    validation_score_summary = sess.run(net.validation_score_summary_op,
                             feed_dict={net.rand_f_score: validation_scores['Rand F-Score Full'],
                                        net.rand_f_score_merge: validation_scores['Rand F-Score Merge'],
                                        net.rand_f_score_split: validation_scores['Rand F-Score Split'],
                                        net.vi_f_score: validation_scores['VI F-Score Full'],
                                        net.vi_f_score_merge: validation_scores['VI F-Score Merge'],
                                        net.vi_f_score_split: validation_scores['VI F-Score Split'],
                                })

                    summary_writer.add_summary(validation_score_summary, step)

                if step == n_iterations:
                    break


def _mirrorAcrossBorders(data, fov):
    mirrored_data = np.pad(data, [(0, 0), (fov//2, fov//2), (fov//2, fov//2)], mode='reflect')
    return mirrored_data


def _evaluateRandError(dataset, sigmoid_prediction, watershed_high=0.9, watershed_low=0.3):
    # Save affinities to temporary file
    tmp_aff_file = dataset + '-tmp-affinities.h5'
    tmp_label_file = dataset + '-tmp-labels.h5'
    ground_truth_file = dataset + '-generated-labels.h5'

    reshapedAffs = np.einsum('zyxd->dzyx', sigmoid_prediction)
    affs = np.concatenate([reshapedAffs, reshapedAffs, np.zeros(reshapedAffs.shape)])
    with h5py.File(snemi3d.folder()+tmp_dir+tmp_aff_file,'w') as output_file:
        output_file.create_dataset('main', data=affs)


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
    inpt = tf.get_variable('input', shape=[1, INPT, INPT, 1])
    inpt_placeholder = tf.placeholder(tf.float32, shape=(1, INPT, INPT, 1))
    assign_input = tf.assign(inpt, inpt_placeholder)
    target = tf.get_variable('target', shape=[1, OUTPT, OUTPT, 1]) # This does not do anything - it is just for the initialization of the net.
    with tf.variable_scope('foo'):
        net = create_unet(inpt, target, keep_prob=1.0)
    with h5py.File(snemi3d.folder()+'test-input.h5','r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        mirrored_inpt = _mirrorAcrossBorders(inpt, FOV)
        num_layers = mirrored_inpt.shape[0]
        input_shape = mirrored_inpt.shape[1]
        output_shape = mirrored_inpt.shape[1] - FOV + 1
        with h5py.File(snemi3d.folder()+'test-affinities.h5','w') as output_file:
            output_file.create_dataset('main', shape=(3,) + input_file['main'].shape)
            out = output_file['main']

            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder()+tmp_dir+'model.ckpt')
                print("Model restored.")

                overlappedComputations = np.zeros(input_file['main'].shape)
                for z in xrange(num_layers):
                    print('z: ' + str(z + 1) + '/' + str(num_layers))
                    for y in (range(0, input_shape - INPT + 1, OUTPT) + [input_shape - INPT]):
                        #print('y: ' + str(y + 1) + '/' + str(input_shape - INPT + 1))
                        for x in (range(0, input_shape - INPT + 1, OUTPT) + [input_shape - INPT]):
                            #print('x: ' + str(x + 1) + '/' + str(input_shape - INPT + 1))
                            sess.run(assign_input,
                                    feed_dict={inpt_placeholder: mirrored_inpt[z,y:y+INPT,x:x+INPT].reshape(1, INPT, INPT, 1)})
                            pred = sess.run(net.sigmoid_prediction)
                            out[0,z,y:y+OUTPT,x:x+OUTPT] += pred[0,:,:,0]
                            out[1,z,y:y+OUTPT,x:x+OUTPT] += pred[0,:,:,0]
                            overlappedComputations[z,y:y+OUTPT,x:x+OUTPT] += np.ones((OUTPT, OUTPT))

                outBoundaries = np.divide(out[0], overlappedComputations)
                tifffile.imsave(snemi3d.folder() + 'test-boundaries.tif', outBoundaries)

