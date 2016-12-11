# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import sys

import h5py
import tensorflow as tf
import tifffile

try:
    from thirdparty.segascorus import io_utils
    from thirdparty.segascorus import utils
    from thirdparty.segascorus.metrics import *
except Exception:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")

import models.N4 as n4
import download_data as down
import augmentation as aug
import evaluation


def train(model, data_provider, data_folder, n_iterations=10000):
    results_folder = data_folder + 'results/'
    ckpt_folder = results_folder + model.model_name + '/'

    # Configure validation
    validation_input_file = h5py.File(data_folder + down.VALIDATION_INPUT + down.H5, 'r')
    validation_input = validation_input_file['main'][:5,:,:].astype(np.float32) / 255.0
    num_validation_layers = validation_input.shape[0]
    mirrored_validation_input = aug.mirror_across_borders(validation_input, model.fov)
    validation_input_shape = mirrored_validation_input.shape[1]
    validation_output_shape = mirrored_validation_input.shape[1] - model.fov + 1
    reshaped_validation_input = mirrored_validation_input.reshape(num_validation_layers, validation_input_shape,
                                                                  validation_input_shape, 1)
    validation_input_file.close()

    validation_label_file = h5py.File(data_folder + down.VALIDATION_AFFINITIES + down.H5, 'r')
    validation_labels = validation_label_file['main']
    reshaped_labels = np.einsum('dzyx->zyxd', validation_labels[0:2])
    validation_label_file.close()

    print('Run tensorboard to visualize training progress')

    with tf.Session() as sess:
        # We will write our summaries here
        summary_writer = tf.summary.FileWriter(ckpt_folder + '/events', graph=sess.graph)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the dataset
        for step, (inputs, labels) in enumerate(data_provider.batch_iterator(model.fov, model.out, model.inpt)):

            sess.run(model.train_step, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            if step % 10 == 0:
                print('step :'+str(step))
                summary = sess.run(model.loss_summary, feed_dict={
                    model.image: inputs,
                    model.target: labels
                })

                summary_writer.add_summary(summary, step)

            if step % 500 == 0:
                # Measure validation error

                # Compute pixel error

                validation_sigmoid_prediction, validation_pixel_error_summary = \
                    sess.run([model.sigmoid_prediction, model.validation_pixel_error_summary],
                             feed_dict={model.image: reshaped_validation_input,
                                        model.target: reshaped_labels})

                summary_writer.add_summary(validation_pixel_error_summary, step)

                # Calculate rand and VI scores
                scores = evaluation.rand_error(model, data_folder, validation_sigmoid_prediction, num_validation_layers,
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

    return scores


def predict(model, data_folder, subset):
    # TODO(beisner): refactor such that predictions aren't necessarily made from affinities, i.e. get from DataProvider
    # Where we store the output affinities and map
    results_folder = data_folder + 'results/'

    # Where we store model
    ckpt_folder = results_folder + model.model_name + '/'

    # Where the data is stored
    data_prefix = data_folder + subset

    # Get the h5 input
    with h5py.File(data_prefix + '-input.h5', 'r') as input_file:
        # Scale appropriately, and mirror
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        mirrored_inpt = aug.mirror_across_borders(inpt, model.fov)
        num_layers = mirrored_inpt.shape[0]
        input_shape = mirrored_inpt.shape[1]

        # Create an affinities file
        with h5py.File(ckpt_folder + subset + '-affinities.h5', 'w') as output_file:
            output_file.create_dataset('main', shape=(3,) + input_file['main'].shape)
            out = output_file['main']

            # Make a prediction
            with tf.Session() as sess:
                # Restore variables from disk.
                model.saver.restore(sess, ckpt_folder + 'model.ckpt')
                print("Model restored.")
                for z in range(num_layers):
                    pred = sess.run(model.sigmoid_prediction,
                                    feed_dict={
                                        model.image: mirrored_inpt[z].reshape(1, input_shape, input_shape, 1)})
                    reshaped_pred = np.einsum('zyxd->dzyx', pred)
                    out[0:2, z] = reshaped_pred[:, 0]

            # Our border is the max of the output
            tifffile.imsave(results_folder + model.model_name + '/' + subset + '-map.tif', np.minimum(out[0], out[1]))


def __grid_search(data_provider, data_folder, remaining_params, current_params, results_dict):
    if len(remaining_params) > 0:
        # Get a parameter
        param, values = remaining_params.popitem()

        # For each potential parameter, copy current_params and add the potential parameter to next_params
        for value in values:
            next_params = current_params.copy()
            next_params[param] = value

            # Perform grid search on the remaining params
            __grid_search(data_provider, data_folder, remaining_params=remaining_params.copy(),
                          current_params=next_params, results_dict=results_dict)
    else:
        try:
            print('Training this model:')
            print(current_params)
            model = n4.N4(current_params)
            results_dict[model.model_name] = train(model, data_provider, data_folder, n_iterations=500)  # temp
        except:
            print("Failed to train this model, ", sys.exc_info()[0])


def grid_search(data_provider, data_folder, params_lists):
    tf.Graph().as_default()

    # Mapping between parameter set and metrics.
    results_dict = dict()

    # perform the recursive grid search
    __grid_search(data_provider, data_folder, params_lists, dict(), results_dict)

    return results_dict
