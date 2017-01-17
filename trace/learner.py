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


class TrainingParams:
    def __init__(self, optimizer, learning_rate, n_iter, output_size):
        self.output_size = output_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_iter = n_iter


DEFAULT_TRAINING_PARAMS = TrainingParams(
    optimizer=tf.train.AdamOptimizer,
    learning_rate=0.0001,
    n_iter=50000,
    output_size=101,
)


class LossHook:
    def __init__(self, frequency, model):
        self.frequency = frequency
        self.training_summaries = tf.summary.merge([
            tf.summary.scalar('cross_entropy', model.cross_entropy),
            tf.summary.scalar('pixel_error', model.pixel_error),
        ])

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            print('step :' + str(step))

            summary = session.run(self.training_summaries, feed_dict={
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

        self.rand_f_score = tf.placeholder(tf.float32)
        self.rand_f_score_merge = tf.placeholder(tf.float32)
        self.rand_f_score_split = tf.placeholder(tf.float32)
        self.vi_f_score = tf.placeholder(tf.float32)
        self.vi_f_score_merge = tf.placeholder(tf.float32)
        self.vi_f_score_split = tf.placeholder(tf.float32)

        self.training_summaries = tf.summary.merge([
            tf.summary.scalar('cross_entropy', model.cross_entropy),
            tf.summary.scalar('pixel_error', model.pixel_error),
        ])

        self.validation_summaries = tf.summary.merge([
            tf.summary.scalar('rand_score', self.rand_f_score),
            tf.summary.scalar('rand_merge_score', self.rand_f_score_merge),
            tf.summary.scalar('rand_split_score', self.rand_f_score_split),
            tf.summary.scalar('vi_score', self.vi_f_score),
            tf.summary.scalar('vi_merge_score', self.vi_f_score_merge),
            tf.summary.scalar('vi_split_score', self.vi_f_score_split),
        ])

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            # Make predictions on the validation set
            validation_prediction, validation_training_summary = session.run(
                [model.prediction, self.training_summaries],
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

            score_summary = session.run(self.validation_summaries,
                                        feed_dict={self.rand_f_score: scores['Rand F-Score Full'],
                                                   self.rand_f_score_merge: scores['Rand F-Score Merge'],
                                                   self.rand_f_score_split: scores['Rand F-Score Split'],
                                                   self.vi_f_score: scores['VI F-Score Full'],
                                                   self.vi_f_score_merge: scores['VI F-Score Merge'],
                                                   self.vi_f_score_split: scores['VI F-Score Split'],
                                                   })

            summary_writer.add_summary(score_summary, step)


class ModelSaverHook:
    def __init__(self, frequency, ckpt_folder):
        self.frequency = frequency
        self.ckpt_folder = ckpt_folder

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            save_path = model.saver.save(session, self.ckpt_folder + 'model.ckpt')
            print("Model saved in file: %s" % save_path)


class Learner:
    def __init__(self, model, ckpt_folder):
        self.model = model
        self.sess = tf.Session()
        self.ckpt_folder = ckpt_folder

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def train(self, training_params, data_provider, hooks):

        sess = self.sess
        model = self.model

        # We will write our summaries here
        summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        # Definte an optimizer
        optimize_step = training_params.optimizer(training_params.learning_rate).minimize(model.cross_entropy)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        fov = model.fov
        output_size = training_params.output_size
        input_size = fov + 2 * (output_size // 2)

        # Iterate through the dataset
        for step, (inputs, labels) in enumerate(data_provider.batch_iterator(fov, output_size, input_size)):

            # Run the optimizer
            sess.run(optimize_step, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            for hook in hooks:
                hook.eval(step, model, sess, summary_writer, inputs, labels)

            # Stop when we've trained enough
            if step == training_params.n_iter:
                break

    def restore(self):
        self.model.saver.restore(self.sess, self.ckpt_folder + 'model.ckpt')
        print("Model restored.")

    def predict(self, inputs):
        # Mirror the inputs
        mirrored_inputs = aug.mirror_across_borders(inputs, self.model.fov)

        preds = []

        # Break into slices because otherwise tensorflow runs out of memory
        num_slices = mirrored_inputs.shape[0]
        for l in range(num_slices):
            reshaped_slice = np.expand_dims(mirrored_inputs[l], axis=0)
            pred = self.sess.run(self.model.prediction, feed_dict={self.model.image: reshaped_slice})
            preds.append(pred)

        return np.squeeze(np.asarray(preds), axis=1)
