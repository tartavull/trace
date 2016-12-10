# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np
import subprocess
import tifffile

#from thirdparty.segascorus import io_utils
#from thirdparty.segascorus import utils
#from thirdparty.segascorus.metrics import *

import models

import os
import sys

from augmentation import batch_iterator, batch_iterator_bn
import IPython as ipy



def train(model, config, n_iterations=10000, validation=True):

    print('Run tensorboard to visualize training progress')

    ckpt_folder = config.folder + model.model_name + '/'
    loss, pixel = 0, 0
    m = .9
    m1 = 1-m

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(ckpt_folder, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        for step, (inputs, affinities) in enumerate(batch_iterator(config, model.fov, model.out, model.inpt)):

            affinities = np.reshape(affinities, [1,2])
            sess.run([model.train_step,model.cross_entropy], feed_dict={
                model.image: inputs,
                model.target: affinities
            })


            if step % 10 == 0:
                print('step :'+str(step))
                summary,loss_val, pe = sess.run([model.loss_summary,model.cross_entropy,model.pixel_error], 
                    feed_dict={
                    model.image: inputs,
                    model.target: affinities
                })
                loss = m*loss + m1*loss_val
                pixel = m*pixel + m1*pe
                print(loss, pixel)

                summary_writer.add_summary(summary, step)

            # if step % 1000 == 0:
            #     # Save the variables to disk.
            #     save_path = model.saver.save(sess, ckpt_folder + 'model.ckpt')
            #     print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break

    return



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
                    pred = sess.run(model.sigmoid_prediction,
                                    feed_dict={
                                        model.image: mirrored_inpt[z].reshape(1, input_shape, input_shape, 1)})
                    reshaped_pred = np.einsum('zyxd->dzyx', pred)
                    out[0:2, z] = reshaped_pred[:,0]

            # Average x and y affinities to get a probabilistic boundary map
            tifffile.imsave(config.folder + split + '-map.tif', (out[0] + out[1])/2)



