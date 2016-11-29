# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np

import snemi3d
from augmentation import batch_iterator


def train(model, n_iterations=10000):
    print ('Run tensorboard to visualize training progress')
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(
                       download_split.snemi3d_folder()+'tmp/', graph=sess.graph)

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
