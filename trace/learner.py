# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import augmentation as aug

try:
    from thirdparty.segascorus import io_utils
    from thirdparty.segascorus import utils
    from thirdparty.segascorus.metrics import *
except Exception:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")

import evaluation


class LossHook:
    def __init__(self, frequency):
        self.frequency = frequency

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            print('step :' + str(step))
            summary = session.run(model.training_summaries, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            summary_writer.add_summary(summary, step)


class ValidationHook:
    def __init__(self, frequency, data_provider, model, data_folder):
        self.frequency = frequency
        self.data_folder = data_folder

        # Get the inputs and mirror them
        self.reshaped_val_inputs, self.reshaped_val_labels = data_provider.dataset_from_h5py('validation')
        self.reshaped_val_inputs = aug.mirror_across_borders(self.reshaped_val_inputs, model.fov)

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            # Make predictions on the validation set
            validation_prediction, validation_training_summary = session.run(
                [model.prediction, model.training_summaries],
                feed_dict={
                    model.image: self.reshaped_val_inputs,
                    model.target: self.reshaped_val_labels
                })

            summary_writer.add_summary(validation_training_summary, step)

            val_n_layers = self.reshaped_val_inputs.shape[0]
            val_output_dim = self.reshaped_val_inputs.shape[1] - model.fov + 1

            # Calculate rand and VI scores
            scores = evaluation.rand_error(model, self.data_folder, validation_prediction, val_n_layers,
                                           val_output_dim, watershed_high=0.95)

            score_summary = session.run(model.validation_summaries,
                                        feed_dict={model.rand_f_score: scores['Rand F-Score Full'],
                                                   model.rand_f_score_merge: scores['Rand F-Score Merge'],
                                                   model.rand_f_score_split: scores['Rand F-Score Split'],
                                                   model.vi_f_score: scores['VI F-Score Full'],
                                                   model.vi_f_score_merge: scores['VI F-Score Merge'],
                                                   model.vi_f_score_split: scores['VI F-Score Split'],
                                                   })

            summary_writer.add_summary(score_summary, step)


class ModelSaverHook:
    def __init__(self, frequency, ckpt_folder):
        self.frequency = frequency
        self.ckpt_folder = ckpt_folder

    def eval(self, step, model, session, summary_writer, inputs, labels):
        save_path = model.saver.save(session, self.ckpt_folder + 'model.ckpt')
        print("Model saved in file: %s" % save_path)


class Learner:
    def __init__(self, model, ckpt_folder):
        self.model = model
        self.sess = tf.Session()
        self.ckpt_folder = ckpt_folder

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def train(self, data_provider, hooks, n_iterations):

        sess = self.sess
        model = self.model

        # We will write our summaries here
        summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the dataset
        for step, (inputs, labels) in enumerate(data_provider.batch_iterator(model.fov, model.output, model.input)):

            # Run the optimizer
            sess.run(model.optimizer, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            for hook in hooks:
                hook.eval(step, model, sess, summary_writer, inputs, labels)

            # Stop when we've trained enough
            if step == n_iterations:
                break

    def restore(self):
        self.model.saver.restore(self.sess, self.ckpt_folder + 'model.ckpt')
        print("Model restored.")

    def predict(self, inputs):
        # Mirror the inputs
        mirrored_inputs = aug.mirror_across_borders(inputs, self.model.fov)
        return self.sess.run(self.model.prediction, feed_dict={self.model.image: mirrored_inputs})
