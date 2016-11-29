# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np
import subprocess

from thirdparty.segascorus import io_utils
from thirdparty.segascorus import utils
from thirdparty.segascorus.metrics import *

import os

from augmentation import batch_iterator


def train(model, config, n_iterations=10000, validation=True):
    if validation:
        validation_input_file = h5py.File(config.folder + config.validation_input_h5, 'r')
        validation_input = validation_input_file['main'][:5,:,:].astype(np.float32) / 255.0
        num_validation_layers = validation_input.shape[0]
        mirrored_validation_input = _mirror_across_borders(validation_input, model.fov)
        validation_input_shape = mirrored_validation_input.shape[1]
        validation_output_shape = mirrored_validation_input.shape[1] - model.fov + 1
        reshaped_validation_input = mirrored_validation_input.reshape(num_validation_layers, validation_input_shape, validation_input_shape, 1)
        validation_input_file.close()

        validation_label_file = h5py.File(config.folder + 'validation-affinities.h5','r')
        validation_labels = validation_label_file['main']
        reshaped_labels = np.einsum('dzyx->zyxd', validation_labels[0:2])
        validation_label_file.close()

    print('Run tensorboard to visualize training progress')

    ckpt_folder = config.folder + model.model_name + '/'

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(ckpt_folder, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        for step, (inputs, affinities) in enumerate(batch_iterator(config, model.fov, model.out, model.inpt)):

            sess.run(model.train_step, feed_dict={
                model.image: inputs,
                model.target: affinities
            })

            if step % 10 == 0:
                print('step :'+str(step))
                summary = sess.run(model.loss_summary, feed_dict={
                    model.image: inputs,
                    model.target: affinities
                })

                summary_writer.add_summary(summary, step)

            if validation and step % 300 == 0:
                # Measure validation error

                # Compute pixel error

                validation_sigmoid_prediction, validation_pixel_error_summary = \
                    sess.run([model.sigmoid_prediction, model.validation_pixel_error_summary],
                             feed_dict={model.image: reshaped_validation_input,
                                        model.target: reshaped_labels})

                summary_writer.add_summary(validation_pixel_error_summary, step)

                # Calculate rand and VI scores
                scores = _evaluate_rand_error(config, model, validation_sigmoid_prediction, num_validation_layers,
                                              validation_output_shape, watershed_high=0.95)
                score_summary = sess.run(model.score_summary_op,
                                         feed_dict={model.rand_f_score: scores['Rand F-Score Full'],
                                                    model.rand_f_score_merge: scores['Rand F-Score Merge'],
                                                    model.rand_f_score_split: scores['Rand F-Score Split'],
                                                    model.vi_f_score: scores['VI F-Score Full'],
                                                    model.vi_f_score_merge: scores['VI F-Score Merge'],
                                                    model.vi_f_score_split: scores['VI F-Score Split'],
                                                    })

                summary_writer.add_summary(score_summary, step)

            if step % 1000 == 0:
                # Save the variables to disk.
                save_path = model.saver.save(sess, ckpt_folder + 'model.ckpt')
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break


def _mirror_across_borders(data, fov):
    mirrored_data = np.zeros(shape=(data.shape[0], data.shape[1] + fov - 1, data.shape[2] + fov - 1))
    mirrored_data[:, fov // 2:-(fov // 2), fov // 2:-(fov // 2)] = data
    for i in range(data.shape[0]):
        # Mirror the left side
        mirrored_data[i, fov // 2:-(fov // 2), :fov // 2] = np.fliplr(data[i, :, :fov // 2])
        # Mirror the right side
        mirrored_data[i, fov // 2:-(fov // 2), -(fov // 2):] = np.fliplr(data[i, :, -(fov // 2):])
        # Mirror the top side
        mirrored_data[i, :fov // 2, fov // 2:-(fov // 2)] = np.flipud(data[i, :fov // 2, :])
        # Mirror the bottom side
        mirrored_data[i, -(fov // 2):, fov // 2:-(fov // 2)] = np.flipud(data[i, -(fov // 2):, :])
        # Mirror the top left corner
        mirrored_data[i, :fov // 2, :fov // 2] = np.fliplr(
            np.transpose(np.fliplr(np.transpose(data[i, :fov // 2, :fov // 2]))))
        # Mirror the top right corner
        mirrored_data[i, :fov // 2, -(fov // 2):] = np.transpose(
            np.fliplr(np.transpose(np.fliplr(data[i, :fov // 2, -(fov // 2):]))))
        # Mirror the bottom left corner
        mirrored_data[i, -(fov // 2):, :fov // 2] = np.transpose(
            np.fliplr(np.transpose(np.fliplr(data[i, -(fov // 2):, :fov // 2]))))
        # Mirror the bottom right corner
        mirrored_data[i, -(fov // 2):, -(fov // 2):] = np.fliplr(
            np.transpose(np.fliplr(np.transpose(data[i, -(fov // 2):, -(fov // 2):]))))
    return mirrored_data


def _evaluate_rand_error(config, model, sigmoid_prediction, num_layers, output_shape, watershed_high=0.9,
                         watershed_low=0.3):
    # Save affinities to temporary file
    # TODO pad the image with zeros so that the ouput covers the whole dataset
    tmp_aff_file = 'validation-tmp-affinities.h5'
    tmp_label_file = 'validation-tmp-labels.h5'
    ground_truth_file = 'validation-labels.h5'

    base = config.folder + model.model_name + '/'

    with h5py.File(base + tmp_aff_file, 'w') as output_file:
        output_file.create_dataset('main', shape=(3, num_layers, output_shape, output_shape))
        out = output_file['main']

        reshaped_pred = np.einsum('zyxd->dzyx', sigmoid_prediction)
        out[0:2,:,:,:] = reshaped_pred

    # Do watershed segmentation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir+"/thirdparty/watershed/watershed.jl",
                     base + tmp_aff_file,
                     base + tmp_label_file,
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

    seg1 = io_utils.import_file(base + tmp_label_file)
    seg2 = io_utils.import_file(config.folder + ground_truth_file)
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


def predict(model, config, split):
    ckpt_folder = config.folder + model.model_name + '/'
    prefix = config.folder + split

    with h5py.File(prefix + '-input.h5', 'r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        mirrored_inpt = _mirror_across_borders(inpt, model.fov)
        num_layers = mirrored_inpt.shape[0]
        input_shape = mirrored_inpt.shape[1]
        with h5py.File(config.folder + split + '-affinities.h5', 'w') as output_file:
            output_file.create_dataset('main', shape=(3,) + input_file['main'].shape)
            out = output_file['main']

            with tf.Session() as sess:
                # Restore variables from disk.
                model.saver.restore(sess, ckpt_folder + 'model.ckpt')
                print("Model restored.")
                for z in range(num_layers):
                    pred = sess.run(net.sigmoid_prediction,
                                    feed_dict={
                                        model.image: mirrored_inpt[z].reshape(1, input_shape, input_shape, 1)})
                    reshaped_pred = np.einsum('zyxd->dzyx', pred)
                    out[0:2, z] = reshaped_pred[:,0]


def grid_search():
    pass
