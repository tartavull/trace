# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import math
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
    learning_rate=0.00001,
    n_iter=50000,
    output_size=101,
)


class Hook(object):
    def eval(self, step, model, session, summary_writer, inputs, labels):
        raise NotImplementedError('Abstract Class!')


class LossHook(Hook):
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


class ValidationHook(Hook):
    def __init__(self, frequency, dset, model, data_folder, boundary_mode):
        self.boundary_mode = boundary_mode
        self.frequency = frequency
        self.data_folder = data_folder

        # Get the inputs and mirror them
        self.reshaped_val_inputs, self.reshaped_val_labels = dset.get_validation_set()
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

            reshaped_pred = validation_prediction.squeeze(axis=(3,))
            # Calculate rand and VI scores
            scores = evaluation.rand_error(model, self.data_folder, 'validation-labels.tif',
                                           reshaped_pred, val_n_layers, val_output_dim,
                                           data_type=self.boundary_mode)

            score_summary = session.run(self.validation_summaries,
                                        feed_dict={self.rand_f_score: scores['Rand F-Score Full'],
                                                   self.rand_f_score_merge: scores['Rand F-Score Merge'],
                                                   self.rand_f_score_split: scores['Rand F-Score Split'],
                                                   self.vi_f_score: scores['VI F-Score Full'],
                                                   self.vi_f_score_merge: scores['VI F-Score Merge'],
                                                   self.vi_f_score_split: scores['VI F-Score Split'],
                                                   })

            summary_writer.add_summary(score_summary, step)


class ModelSaverHook(Hook):
    def __init__(self, frequency, ckpt_folder):
        self.frequency = frequency
        self.ckpt_folder = ckpt_folder

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            save_path = model.saver.save(session, self.ckpt_folder + 'model.ckpt')
            print("Model saved in file: %s" % save_path)


class ImageVisualizationHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        with tf.variable_scope('images'):
            self.training_summaries = tf.summary.merge([
                tf.summary.image('input_image', model.image),
                tf.summary.image('output_patch', model.image[:,
                                                 model.fov // 2:-model.fov // 2, model.fov // 2:-model.fov // 2, :]),
                tf.summary.image('output_target', model.target[:, :, :, :1]),
                tf.summary.image('predictions', model.prediction[:, :, :, :1]),
            ])

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            summary = session.run(self.training_summaries, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            summary_writer.add_summary(summary, step)


class HistogramHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        histograms = []
        with tf.variable_scope('histograms', reuse=True):
            for layer in model.architecture.layers:
                layer_str = 'layer ' + str(layer.depth) + ': ' + layer.layer_type
                histograms.append(tf.summary.histogram(layer_str + \
                                                       ' activations', layer.activations))
                if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                    histograms.append(tf.summary.histogram(layer_str + \
                                                           ' weights', layer.weights))
                    histograms.append(tf.summary.histogram(layer_str + \
                                                           ' biases', layer.biases))

            histograms.append(tf.summary.histogram('prediction', model.prediction))

        self.training_summaries = tf.summary.merge(histograms)

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            summary = session.run(self.training_summaries, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            summary_writer.add_summary(summary, step)


class LayerVisualizationHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        summaries = []
        for layer in model.architecture.layers:
            layer_str = 'layer ' + str(layer.depth) + ' ' + layer.layer_type + \
                        ' activations'
            activations = self.computeVisualizationGrid(layer.activations,
                                                        layer.n_feature_maps)
            summaries.append(tf.summary.image(layer_str, activations))

        self.training_summaries = tf.summary.merge(summaries)

    def eval(self, step, model, session, summary_writer, inputs, labels):
        if step % self.frequency == 0:
            summary = session.run(self.training_summaries, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            summary_writer.add_summary(summary, step)

    def computeVisualizationGrid(self, activations, num_maps, width=16, height=0):
        if num_maps % width == 0:
            cx = width
            if height == 0:
                cy = num_maps // width
            else:
                cy = height
        else:
            cx = int(math.sqrt(num_maps))
            while num_maps % cx != 0:
                cx -= 1
            cy = num_maps // cx

        # Arrange the feature maps into a grid
        # -------------------

        # Choose first example from batch
        reshaped = activations[0, :, :, :]
        dim = tf.shape(reshaped)
        map_size = dim[0]  # size of feature map

        border_thickness = 4
        ix = map_size + border_thickness
        iy = ix

        # Add a border to each image
        padded = tf.image.resize_image_with_crop_or_pad(reshaped, iy, ix)

        # Separate the feature maps into rows and columns
        grid = tf.reshape(padded, tf.pack([iy, ix, cy, cx]))

        # Swap the order that the dimensions are iterated through upon reshape.
        # First, the first (iy) row (ix) of pixels in the first (cy) row (cx) of
        # feature maps is iterated through, followed by the first row in the
        # second feature map until the first row of feature maps is completed.
        # Then this procedure is repeated for the second row of pixels in
        # the first row of feature maps, and then the third row, etc, until
        # the first row of feature maps is completed. Then we move to the
        # second row of feature maps, and so on.
        grid = tf.transpose(grid, (2, 0, 3, 1))  # cy, iy, cx, ix

        # Reshape into final grid image.
        grid = tf.reshape(grid, tf.pack([1, cy * iy, cx * ix, 1]))

        return grid


class Learner:
    def __init__(self, model, ckpt_folder):
        self.model = model
        self.sess = tf.Session()
        self.ckpt_folder = ckpt_folder

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def train(self, training_params, dset, hooks):

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
        input_size = output_size + fov - 1

        # Iterate through the dataset
        for step in range(training_params.n_iter):

            inputs, labels = dset.random_sample(input_size)
            # Run the optimizer

            # Crop the model to the appropriate field of view
            labels = labels[:, fov // 2:-(fov // 2), fov // 2:-(fov // 2), :]

            sess.run(optimize_step, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            for hook in hooks:
                hook.eval(step, model, sess, summary_writer, inputs, labels)

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
