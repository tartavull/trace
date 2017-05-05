from __future__ import print_function
from __future__ import division

import math
import tensorflow as tf

import trace.sampling.augmentation as aug

try:
    from trace.thirdparty.segascorus import io_utils
    from trace.thirdparty.segascorus import utils
    from trace.thirdparty.segascorus.metrics import *
except ImportError:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")

import trace.evaluation as eva
from trace.common import *


class Hook(object):
    def eval(self, step, model, session, summary_writer):
        raise NotImplementedError('Abstract Class!')


class LossHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        self.training_summaries = tf.summary.merge([
            tf.summary.scalar('cross_entropy', model.cross_entropy),
            tf.summary.scalar('pixel_error', model.pixel_error),
        ])

    def eval(self, step, model, session, summary_writer):
        if step % self.frequency == 0:
            print('step :' + str(step))

            summary = session.run(self.training_summaries)

            summary_writer.add_summary(summary, step)


class ValidationHook(Hook):
    def __init__(self, frequency, dset_sampler, model, data_folder, boundary_mode, inference_params):
        self.inference_params = inference_params
        self.dset_sampler = dset_sampler
        self.boundary_mode = boundary_mode
        self.frequency = frequency
        self.data_folder = data_folder

        # Get the inputs from dataset
        self.val_inputs, self.val_labels, self.val_targets = dset_sampler.get_validation_set()

        self.val_examples = np.concatenate([self.val_inputs, self.val_targets], axis=CHANNEL_AXIS)
        self.mirrored_val_examples = aug.mirror_across_borders_3d(self.val_examples, model.fov, model.z_fov)

        # Set up placeholders for other metrics
        self.validation_cross_entropy = tf.placeholder(tf.float32)
        self.validation_pixel_error = tf.placeholder(tf.float32)
        self.rand_f_score = tf.placeholder(tf.float32)
        self.rand_f_score_merge = tf.placeholder(tf.float32)
        self.rand_f_score_split = tf.placeholder(tf.float32)
        self.vi_f_score = tf.placeholder(tf.float32)
        self.vi_f_score_merge = tf.placeholder(tf.float32)
        self.vi_f_score_split = tf.placeholder(tf.float32)

        # Create validation summaries
        self.validation_summaries = tf.summary.merge([
            tf.summary.scalar('validation_cross_entropy', self.validation_cross_entropy),
            tf.summary.scalar('validation_pixel_error', self.validation_pixel_error),
            tf.summary.scalar('rand_score', self.rand_f_score),
            tf.summary.scalar('rand_merge_score', self.rand_f_score_merge),
            tf.summary.scalar('rand_split_score', self.rand_f_score_split),
            # tf.summary.scalar('vi_score', self.vi_f_score),
            # tf.summary.scalar('vi_merge_score', self.vi_f_score_merge),
            # tf.summary.scalar('vi_split_score', self.vi_f_score_split),
        ])

        # Create image summaries
        with tf.variable_scope('validation_images'):
            if model.fov == 1:
                summary_image = model.image[0]
            else:
                summary_image = model.image[0, model.z_fov // 2:(model.z_fov // 2) + 3,
                                            model.fov // 2:-(model.fov // 2), model.fov // 2:-(model.fov // 2), :]

            val_output_patch_summary = tf.summary.image('validation_output_patch', summary_image)
            self.validation_image_summaries = tf.summary.merge([
                tf.summary.image('validation_input_image', model.image[0]),
                val_output_patch_summary,
                tf.summary.image('validation_output_target', model.target[0, :3, :, :, :]),
                tf.summary.image('validation_predictions', model.prediction[0, :3, :, :, :]),
            ])

    def eval(self, step, model, session, summary_writer):
        if step % self.frequency == 0:
            print('eval validation hook')
            # Make the prediction

            validation_prediction = model.predict(session, self.mirrored_val_examples, self.inference_params,
                                                  mirror_inputs=False)

            diff = np.absolute(validation_prediction - self.val_targets)
            validation_cross_entropy = -np.mean(diff * np.log(diff))

            validation_binary_prediction = np.round(validation_prediction)
            validation_pixel_error = np.mean(np.absolute(validation_binary_prediction - self.val_targets))

            # Run an image summary
            summary_im = self.mirrored_val_examples[:, :16, :400, :400, :]
            validation_image_summary = session.run(self.validation_image_summaries,
                                                   feed_dict={model.example: summary_im})

            summary_writer.add_summary(validation_image_summary, step)

            # Calculate rand and VI scores
            scores = eva.rand_error_from_prediction(self.val_labels[0, :, :, :, 0],
                                                    validation_prediction[0],
                                                    pred_type=model.architecture.output_mode)

            score_summary = session.run(self.validation_summaries,
                                        feed_dict={self.validation_cross_entropy: validation_cross_entropy,
                                                   self.validation_pixel_error: validation_pixel_error,
                                                   self.rand_f_score: scores['Rand F-Score Full'],
                                                   self.rand_f_score_merge: scores['Rand F-Score Merge'],
                                                   self.rand_f_score_split: scores['Rand F-Score Split'],
                                                   # self.vi_f_score: scores['VI F-Score Full'],
                                                   # self.vi_f_score_merge: scores['VI F-Score Merge'],
                                                   # self.vi_f_score_split: scores['VI F-Score Split'],
                                                   })

            summary_writer.add_summary(score_summary, step)


class ModelSaverHook(Hook):
    def __init__(self, frequency, ckpt_folder):
        self.frequency = frequency
        self.ckpt_folder = ckpt_folder

    def eval(self, step, model, session, summary_writer):
        if step % self.frequency == 0:
            save_path = model.saver.save(session, self.ckpt_folder + 'model.ckpt')
            print("Model saved in file: %s" % save_path)


class ImageVisualizationHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        with tf.variable_scope('images'):
            if model.fov == 1:
                output_patch_summary = tf.summary.image('output_patch', model.raw_image[0, :3, :, :, :])
            else:
                output_patch = model.raw_image[0,
                                               model.z_fov // 2:(model.z_fov // 2) + 3,
                                               model.fov // 2:-(model.fov // 2),
                                               model.fov // 2:-(model.fov // 2),
                                               :]
                output_patch_summary = tf.summary.image('output_patch', output_patch),

            self.training_summaries = tf.summary.merge([
                tf.summary.image('input_image', model.raw_image[0, model.z_fov // 2:(model.z_fov // 2) + 3]),
                output_patch_summary,
                tf.summary.image('output_target', model.target[0, :3, :, :, :]),
                tf.summary.image('predictions', model.prediction[0, :3, :, :, :]),
            ])

    def eval(self, step, model, session, summary_writer):
        if step % self.frequency == 0:
            print('image vis hook')
            summary = session.run(self.training_summaries)

            summary_writer.add_summary(summary, step)


class HistogramHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        histograms = []
        with tf.variable_scope('histograms', reuse=True):
            for layer in model.architecture.layers:
                layer_str = 'layer ' + str(layer.depth) + ': ' + layer.layer_type
                histograms.append(tf.summary.histogram(layer_str + ' activations', layer.activations))
                if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                    histograms.append(tf.summary.histogram(layer_str + ' weights', layer.weights))
                    histograms.append(tf.summary.histogram(layer_str + ' biases', layer.biases))

            histograms.append(tf.summary.histogram('prediction', model.prediction))

        self.training_summaries = tf.summary.merge(histograms)

    def eval(self, step, model, session, summary_writer):
        if step % self.frequency == 0:
            summary = session.run(self.training_summaries)

            summary_writer.add_summary(summary, step)


class LayerVisualizationHook(Hook):
    def __init__(self, frequency, model):
        self.frequency = frequency
        summaries = []
        for layer in model.architecture.layers:
            layer_str = 'layer ' + str(layer.depth) + ' ' + layer.layer_type + \
                        ' activations'
            activations = self.compute_visualization_grid(layer.activations,
                                                          layer.n_feature_maps)
            summaries.append(tf.summary.image(layer_str, activations))

        self.training_summaries = tf.summary.merge(summaries)

    def eval(self, step, model, session, summary_writer):
        if step % self.frequency == 0:
            summary = session.run(self.training_summaries)

            summary_writer.add_summary(summary, step)

    @staticmethod
    def compute_visualization_grid(activations, num_maps, width=16, height=0):
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
        grid = tf.reshape(padded, tf.stack([iy, ix, cy, cx]))

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
        grid = tf.reshape(grid, tf.stack([1, cy * iy, cx * ix, 1]))

        return grid