import math

import tensorflow as tf
import numpy as np

from trace.common import *
from trace.utils import convert_between_label_types, expand_3d_to_5d, tf_convert_between_label_types, cond_apply

from . import tf_augmentation as tfaug


class EMDatasetSampler(object):
    def __init__(self, dataset, sample_shape, batch_size, augmentation_config, label_output_type=BOUNDARIES):
        """Helper for sampling an EM dataset. The field self.training_example_op is the only field that should be
        accessed outside this class for training.

        :param sample_shape: The shape of output of a sample in the form [z_size, y_size, x_size]
        :param dataset: An instance of Dataset, namely SNEMI3DDataset, ISBIDataset, or CREMIDataset
        :param batch_size: The number of samples produced with each execution of the example op
        :param label_output_type: The format in which the dataset labels should be sampled, i.e. for training, taking
        on values 'boundaries', 'affinities-2d', etc.
        """

        # All inputs and labels come in with the shape: [n_images, x_dim, y_dim]
        # In order to generalize we, expand into 5 dimensions: [batch_size, z_dim, x_dim, y_dim, n_channels]

        # Extract the inputs and labels from the dataset
        self.__train_inputs = expand_3d_to_5d(dataset.train_inputs)
        self.__train_labels = expand_3d_to_5d(dataset.train_labels)
        self.__train_targets = convert_between_label_types(dataset.label_type, label_output_type,
                                                           expand_3d_to_5d(dataset.train_labels))

        # Crop to get rid of edge affinities
        self.__train_inputs = self.__train_inputs[:, 1:, 1:, 1:, :]
        self.__train_labels = self.__train_labels[:, 1:, 1:, 1:, :]
        self.__train_targets = self.__train_targets[:, 1:, 1:, 1:, :]

        self.__validation_inputs = expand_3d_to_5d(dataset.validation_inputs)
        self.__validation_labels = expand_3d_to_5d(dataset.validation_labels)
        self.__validation_targets = convert_between_label_types(dataset.label_type, label_output_type,
                                                                expand_3d_to_5d(dataset.validation_labels))

        # Crop to get rid of edge affinities
        self.__validation_inputs = self.__validation_inputs[:, 1:, 1:, 1:, :]
        self.__validation_labels = self.__validation_labels[:, 1:, 1:, 1:, :]
        self.__validation_targets = self.__validation_targets[:, 1:, 1:, 1:, :]

        self.__test_inputs = expand_3d_to_5d(dataset.test_inputs)

        # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs
        train_stacked = np.concatenate((self.__train_inputs, self.__train_labels), axis=CHANNEL_AXIS)

        # Determine whether or not we need to apply crop padding
        apply_crop_padding = augmentation_config.apply_rotation

        # Some of the augmentations we use distort at the edges, so we want to add a buffer that we can crop away
        if apply_crop_padding:
            crop_pad = sample_shape[1] // 10 * 4
            z_crop_pad = sample_shape[0] // 4 * 2
        else:
            crop_pad = 0
            z_crop_pad = 0

        # Resulting in patch sizes that are a bit larger (we will strip away later)
        patch_size = sample_shape[1] + crop_pad
        z_patch_size = sample_shape[0] + z_crop_pad

        # Create dataset, and pad the dataset with mirroring
        self.__padded_dataset = np.pad(train_stacked,
                                       [[0, 0], [z_crop_pad, z_crop_pad], [crop_pad, crop_pad], [crop_pad, crop_pad],
                                        [0, 0]], mode='reflect')

        with tf.device('/cpu:0'):
            # The dataset is loaded into a constant variable from a placeholder
            # because a tf.constant cannot hold a dataset that is over 2GB.
            self.__image_ph = tf.placeholder(dtype=tf.float32, shape=self.__padded_dataset.shape)
            self.__dataset_constant = tf.Variable(self.__image_ph, trainable=False, collections=[])

            # Sample and squeeze the dataset in multiple batches, squeezing so that we can perform the distortions
            crop_size = [1, z_patch_size, patch_size, patch_size, train_stacked.shape[4]]
            samples = []
            for i in range(batch_size):
                samples.append(tf.random_crop(self.__dataset_constant, size=crop_size))

            # samples will now have shape [batch_size, z_patch_size, patch_size, patch_size, n_channels + 1]
            samples = tf.squeeze(samples, axis=1)

            # Spatial Augmentations

            # Randomly rotate each stack
            mirrored_sample = cond_apply(samples,
                                         tfaug.tf_randomly_mirror_each_stack,
                                         lambda stacks: stacks,
                                         augmentation_config.apply_mirroring)

            # Randomly flip each stack
            flipped_sample = cond_apply(mirrored_sample,
                                        tfaug.tf_randomly_flip_each_stack,
                                        lambda stacks: stacks,
                                        augmentation_config.apply_flipping)

            # Randomly rotate each stack
            rotated_sample = cond_apply(flipped_sample,
                                        tfaug.tf_randomly_rotate_each_stack,
                                        lambda stacks: stacks,
                                        augmentation_config.apply_rotation)

            # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute
            # elastically_deformed_sample = tf.elastic_deformation(rotated_sample)
            # Right now we do nothing
            elastically_deformed_sample = cond_apply(rotated_sample,
                                                     lambda stacks: stacks,
                                                     lambda stacks: stacks,
                                                     True)

            # Separate the image from the labels
            deformed_inputs = elastically_deformed_sample[:, :, :, :, :1]
            deformed_labels = elastically_deformed_sample[:, :, :, :, 1:]

            # Image Augmentations

            # Randomly blur the inputs
            blurred_inputs = cond_apply(deformed_inputs,
                                        tfaug.tf_randomly_blur_each_stack,
                                        lambda stacks: stacks,
                                        augmentation_config.apply_blur)

            converted_labels = tf_convert_between_label_types(dataset.label_type, label_output_type, deformed_labels)

            # Crop the image, to remove the padding that was added to allow safe augmentation.
            z_begin = z_crop_pad // 2
            z_end = tf.shape(blurred_inputs)[1] - z_begin

            xy_begin = crop_pad // 2
            xy_end = tf.shape(blurred_inputs)[2] - xy_begin

            cropped_inputs = blurred_inputs[:, z_begin:z_end, xy_begin:xy_end, xy_begin:xy_end, :]
            cropped_labels = converted_labels[:, z_begin:z_end, xy_begin:xy_end, xy_begin:xy_end, :]

            # Re-stack the image and labels
            self.__training_example_op = tf.concat([cropped_inputs, cropped_labels], axis=CHANNEL_AXIS)

    def initialize_session_variables(self, sess):
        sess.run(self.__dataset_constant.initializer, feed_dict={self.__image_ph: self.__padded_dataset})
        del self.__padded_dataset

    def get_sample_op(self):
        return self.__training_example_op

    def get_full_training_set(self):
        return self.__train_inputs, self.__train_labels, self.__train_targets

    def get_validation_set(self):
        return self.__validation_inputs, self.__validation_labels, self.__validation_targets

    def get_test_set(self):
        return self.__test_inputs
