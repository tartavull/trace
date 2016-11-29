# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np

from augmentation import batch_iterator


def train(model, config, n_iterations=10000):
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

            if step % 100 == 0:
                print('step :'+str(step))
                summary = sess.run(model.loss_summary, feed_dict={
                    model.image: inputs,
                    model.target: affinities
                })

                summary_writer.add_summary(summary, step)

            if step % 1000 == 0:
                # Save the variables to disk.
                save_path = model.saver.save(sess, ckpt_folder + 'model.ckpt')
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break


def predict(model, config, split):
    ckpt_folder = config.folder + model.model_name + '/'
    prefix = config.folder + split

    with tf.Session() as sess:
        # Restore variables from disk.
        model.saver.restore(sess, ckpt_folder + 'model.ckpt')
        print("Model restored.")
        with h5py.File(prefix + 'input.h5','r') as input_file:
            inpt = input_file['main'][:].astype(np.float32) / 255.0
            with h5py.File(prefix + '-affinities.h5', 'w') as output_file:
                output_file.create_dataset('main', shape=(3,)+input_file['main'].shape)
                out = output_file['main']

                OUTPT = model.out
                INPT = model.inpt
                FOV = model.fov

                # TODO: pad the image with zeros so that the ouput covers the whole dataset
                for z in xrange(inpt.shape[0]):
                    print('z: {} of {}'.format(z,inpt.shape[0]))
                    for y in xrange(0,inpt.shape[1]-INPT, OUTPT):
                        for x in xrange(0,inpt.shape[1]-INPT, OUTPT):
                            pred = sess.run(model.sigmoid_prediction, feed_dict={
                                model.image: inpt[z,y:y+INPT,x:x+INPT].reshape(1,INPT,INPT,1)
                            })

                            reshapedPred = np.zeros(shape=(2, OUTPT, OUTPT))
                            reshapedPred[0] = pred[0,:,:,0].reshape(OUTPT, OUTPT)
                            reshapedPred[1] = pred[0,:,:,1].reshape(OUTPT, OUTPT)
                            out[0:2,
                                z,
                                y+FOV//2:y+FOV//2+OUTPT,
                                x+FOV//2:x+FOV//2+OUTPT] = reshapedPred


def grid_search():
    pass
