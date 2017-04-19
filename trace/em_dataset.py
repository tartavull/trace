from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

# I/O
import tifffile as tiff
import cremi.io as cremiio
import h5py

import download_data as down

from augmentation import *
from utils import *


class Dataset(object):
    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        raise NotImplementedError

    @staticmethod
    def prepare_predictions_for_neuroglancer(results_folder, split, predictions, label_type):
        """Prepare the provided labels to be visualized using Neuroglancer.
        :param label_type: The format of the predictions passed in
        :param results_folder: The location where we should save the h5 file
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'split']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        predictions = convert_between_label_types(label_type, SEGMENTATION_3D, predictions[0])
        # Create an affinities file
        with h5py.File(results_folder + split + '-predictions.h5', 'w') as output_file:
            output_file.create_dataset('main', shape=predictions.shape)

            # Reformat our predictions
            out = output_file['main']

            # Reshape and set in out
            '''
            for i in range(predictions.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(predictions[i], axis=0))
                out[0:2, i] = reshaped_pred[:, 0]
            '''
            out[:] = predictions

    @staticmethod
    def prepare_predictions_for_neuroglancer_affinities(results_folder, split, predictions, label_type):
        """Prepare the provided affinities to be visualized using Neuroglancer.
        :param label_type: The format of the predictions passed in
        :param results_folder: The location where we should save the h5 file
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'split']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        pred_affinities = convert_between_label_types(label_type, AFFINITIES_3D, predictions[0])
        sha = pred_affinities.shape
        # Create an affinities file
        with h5py.File(results_folder + split + '-pred-affinities.h5', 'w') as output_file:
            # Create the dataset in the file
            new_shape = (3, sha[0], sha[1], sha[2])

            output_file.create_dataset('main', shape=new_shape)

            # Reformat our predictions
            out = output_file['main']

            '''
            for i in range(pred_affinities.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(pred_affinities[i], axis=0))
                # Copy over as many affinities as we got
                out[0:sha[3], i] = reshaped_pred[:, 0]
            '''
            reshaped_pred = np.einsum('zyxd->dzyx', pred_affinities)
            # Copy over as many affinities as we got
            out[0:sha[3]] = reshaped_pred[:]


class ISBIDataset(Dataset):
    label_type = BOUNDARIES

    def __init__(self, data_folder):
        """Wrapper for the ISBI dataset. The ISBI dataset as downloaded (via download_data.py) is as follows:
        train_input.tif: Training data for a stack of EM images from fish in the shape [num_images, x_size, y_size],
        where each value is found on the interval [0, 255], representing a stack of greyscale images.
        train_labels.tif: Training labels for the above EM images. Represent the ground truth of where the boundaries of
        all structures in the EM images exist.
        validation_input.tif, validation_labels.tif: Same as above, except represent a partitioned subset of the
        original training set.
        test_input.tif: Labelless testing images in the same format as above.
        :param data_folder: Path to where the ISBI data is found
        """
        self.data_folder = data_folder

        self.train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
        self.train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF) / 255.0

        self.validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
        self.validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF) / 255.0

        self.test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)

    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        """Prepare the provided labels
        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param results_folder: The location where we should save the dataset
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions)
        tiff.imsave(results_folder + split + '-predictions.tif', trans_predictions)


class SNEMI3DDataset(Dataset):
    label_type = SEGMENTATION_3D

    def __init__(self, data_folder):
        """Wrapper for the SNEMI3D dataset. The SNEMI3D dataset as downloaded (via download_data.py) is as follows:
        train_input.tif: Training data for a stack of EM images from fly in the shape [num_images, x_size, y_size],
        where each value is found on the interval [0, 255], representing a stack of greyscale images.
        train_labels.tif: Training labels for the above EM images. Represent the ground truth segmentation for the
        training data (each region has a unique ID).
        validation_input.tif, validation_labels.tif: Same as above, except represent a partitioned subset of the
        original training set.
        test_input.tif: Labelless testing images in the same format as above.
        :param data_folder: Path to where the SNEMI3D data is found
        """

        self.data_folder = data_folder

        self.train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
        self.train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF)

        self.validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
        self.validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF)

        self.test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)

    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        """Prepare the provided labels to submit on the SNEMI3D competition.
        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param results_folder: The location where we should save the dataset
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions)
        tiff.imsave(results_folder + split + '-predictions.tif', trans_predictions)


class CREMIDataset(Dataset):
    label_type = SEGMENTATION_3D

    def __init__(self, data_folder):
        """Wrapper for the CREMI dataset. The CREMI dataset as downloaded (via download_data.py) is as follows:
        train.hdf: Training data for a stack of EM images from fly. Can be used to derive the inputs,
        in the shape [num_images, x_size, y_size] where each value is found on the interval [0, 255] representing
        a stack of greyscale images, and the labels, in the shape [num_images, x_size, y_size] where each value
        represents the unique id of the object at that position.
        validation.hdf: Same as above, except represents a partitioned subset of the original training set.
        test.hdf: Labelless testing set in the same format as above.
        :param data_folder: Path to where the CREMI data is found
        """
        self.data_folder = data_folder
        self.name = down.CREMI

        train_file = cremiio.CremiFile(data_folder + 'train.hdf', 'r')
        self.train_inputs = train_file.read_raw().data.value
        if down.CLEFTS in self.data_folder:
            self.train_masks = train_file.read_neuron_ids().data.value
            self.train_labels = train_file.read_clefts().data.value
        else:
            self.train_labels = train_file.read_neuron_ids().data.value
        train_file.close()

        validation_file = cremiio.CremiFile(data_folder + 'validation.hdf', 'r')
        self.validation_inputs = validation_file.read_raw().data.value
        if down.CLEFTS in self.data_folder:
            self.validation_labels = validation_file.read_clefts().data.value
        else:
            self.validation_labels = validation_file.read_neuron_ids().data.value
        validation_file.close()

        # TODO(beisner): Decide if we need to load the test file every time (probably don't)
        test_dir = data_folder.replace(down.CLEFTS, '')
        test_file = cremiio.CremiFile(test_dir + 'test.hdf', 'r')
        self.test_inputs = test_file.read_raw().data.value
        test_file.close()


    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        """Prepare a given segmentation prediction for submission to the CREMI competiton
        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param results_folder: The location where we should save the dataset.
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions)

        # Get the input we used
        input_file = cremiio.CremiFile(self.data_folder + split + '.hdf', 'r')
        raw = input_file.read_raw()
        inputs = raw.data.value
        resolution = raw.resolution

        input_file.close()

        pred_file = cremiio.CremiFile(results_folder + split + '-predictions.hdf', 'w')
        pred_file.write_raw(cremiio.Volume(inputs, resolution=resolution))
        pred_file.write_neuron_ids(cremiio.Volume(trans_predictions, resolution=resolution))
        pred_file.close()


class EMDatasetSampler(object):

    def __init__(self, dataset, input_size, z_input_size, batch_size=1, label_output_type=BOUNDARIES):
        """Helper for sampling an EM dataset. The field self.training_example_op is the only field that should be
        accessed outside this class for training.
        :param input_size: The size of the field of view
        :param z_input_size: The size of the field of view in the z-direction
        :param batch_size: The number of images to stack together in a batch
        :param dataset: An instance of Dataset, namely SNEMI3DDataset, ISBIDataset, or CREMIDataset
        :param label_output_type: The format in which the dataset labels should be sampled, i.e. for training, taking
        on values 'boundaries', 'affinities-2d', etc.
        """
        # All inputs and labels come in with the shape: [n_images, x_dim, y_dim]
        # In order to generalize we, expand into 5 dimensions: [batch_size, z_dim, x_dim, y_dim, n_channels]
   
        # Extract the inputs and labels from the dataset
        '''TODO: MAKE SURE TO CHANGE IT SUCH THAT BOUNDARY CONVERSION IS HANDLED MORE NEATLY'''
        self.__train_inputs = expand_3d_to_5d(dataset.train_inputs)
        self.__train_labels = convert_between_label_types(dataset.name, dataset.label_type, label_output_type,
            dataset.train_labels)
        self.__train_labels = expand_3d_to_5d(self.__train_labels)

        self.__train_targets = convert_between_label_types(dataset.name, dataset.label_type, label_output_type,
            dataset.train_labels)
        self.__train_targets = expand_3d_to_5d(self.__train_targets)

        # Crop to get rid of edge affinities
        self.__train_inputs = self.__train_inputs[:, 1:, 1:, 1:, :]
        self.__train_labels = self.__train_labels[:, 1:, 1:, 1:, :]
        self.__train_targets = self.__train_targets[:, 1:, 1:, 1:, :]

        self.__validation_inputs = expand_3d_to_5d(dataset.validation_inputs)

        self.__validation_labels = convert_between_label_types(dataset.name, dataset.label_type, label_output_type,
            dataset.validation_labels)
        self.__validation_labels = expand_3d_to_5d(self.__validation_labels)

        self.__validation_targets = convert_between_label_types(dataset.name, dataset.label_type, label_output_type,
            dataset.validation_labels)
        self.__validation_targets = expand_3d_to_5d(self.__validation_targets)

        # Crop to get rid of edge affinities
        self.__validation_inputs = self.__validation_inputs[:, 1:, 1:, 1:, :]
        self.__validation_labels = self.__validation_labels[:, 1:, 1:, 1:, :]
        self.__validation_targets = self.__validation_targets[:, 1:, 1:, 1:, :]

        self.__test_inputs = expand_3d_to_5d(dataset.test_inputs)

        # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs

        # If computing for clefts, include ops for masks as well
        if dataset.train_masks.any():
            self.__train_masks = 1 - convert_between_label_types(dataset.name, dataset.label_type, label_output_type,
                        dataset.train_masks)
            self.__train_masks = expand_3d_to_5d(self.__train_masks)
            self.__train_masks = self.__train_masks[:, 1:, 1:, 1:, :]
            train_stacked = np.concatenate((self.__train_inputs, self.__train_labels, self.__train_masks), axis=CHANNEL_AXIS)
        else:
            train_stacked = np.concatenate((self.__train_inputs, self.__train_labels), axis=CHANNEL_AXIS)

        # Define inputs to the graph
        crop_pad = input_size // 10 * 4
        z_crop_pad = z_input_size // 4 * 2
        patch_size = input_size + crop_pad
        z_patch_size = z_input_size + z_crop_pad

        # Create dataset, and pad the dataset with mirroring
        self.__padded_dataset = np.pad(train_stacked, [[0, 0], [z_crop_pad, z_crop_pad], [crop_pad, crop_pad], [crop_pad, crop_pad], [0, 0]], mode='reflect')

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

            samples = tf.squeeze(samples, axis=1)

            # Flip a coin, and apply an op to sample (sample can be 5d or 4d)
            # Prob is the denominator of the probability (1 in prob chance)
            def randomly_map_and_apply_op(data, op, prob=2):
                should_apply = tf.random_uniform(shape=(), minval=0, maxval=prob, dtype=tf.int32)

                def tf_if(ex):
                    return tf.cond(tf.equal(0, should_apply), lambda: op(ex), lambda: ex)

                return tf.map_fn(tf_if, data)

            def randomly_apply_op(data, op, prob=2):
                should_apply = tf.random_uniform(shape=(), minval=0, maxval=prob, dtype=tf.int32)

                return tf.cond(tf.equal(0, should_apply), lambda: op(data), lambda: data)

            # Perform random mirroring, by applying the same mirroring to each image in the stack
            def mirror_each_image_in_stack_op(stack):
                return tf.map_fn(lambda img: tf.image.flip_left_right(img), stack)
            mirrored_sample = randomly_map_and_apply_op(samples, mirror_each_image_in_stack_op)

            # Randomly flip the 3D shape upside down
            flipped_sample = randomly_map_and_apply_op(mirrored_sample, lambda stack: tf.reverse(stack, axis=[0]))

            # Apply a random rotation to each stack
            def apply_random_rotation_to_stack(stack):
                # Get the random angle
                angle = tf.random_uniform(shape=(), minval=0, maxval=2 * math.pi)

                # Rotate each image by that angle
                return tf.map_fn(lambda img: tf.contrib.image.rotate(img, angle), stack)

            rotated_sample = tf.map_fn(apply_random_rotation_to_stack, flipped_sample)

            # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute
            # elastically_deformed_sample = tf.elastic_deformation(rotated_sample)
            elastically_deformed_sample = rotated_sample


            # Separate the image from the labels
            #deformed_inputs = elastically_deformed_sample[:, :, :, :, :1]
            #deformed_labels = elastically_deformed_sample[:, :, :, :, 1:]
            deformed_inputs = samples[:, :, :, :, :1]
            deformed_labels = samples[:, :, :, :, 1:2]

            # Apply random gaussian blurring to the image
            def apply_random_blur_to_stack(stack):
                def apply_random_blur_to_slice(img):
                    sigma = tf.random_uniform(shape=(), minval=2, maxval=5, dtype=tf.float32)
                    return tf_gaussian_blur(img, sigma, size=5)

                return tf.map_fn(lambda img: randomly_apply_op(img, apply_random_blur_to_slice, prob=5), stack)

            blurred_inputs = tf.map_fn(lambda stack: apply_random_blur_to_stack(stack),
                                       deformed_inputs)

            # Mess with the levels
            # leveled_image = tf.image.random_brightness(deformed_image, max_delta=0.15)
            # leveled_image = tf.image.random_contrast(leveled_image, lower=0.5, upper=1.5)
            leveled_inputs = blurred_inputs

            # Affinitize the labels if applicable
            # TODO (ffjiang): Do the if applicable part
            if label_output_type == AFFINITIES_3D:
                deformed_labels = affinitize(deformed_labels)

            # Crop the image, to remove the padding that was added to allow safe augmentation.
            cropped_inputs = leveled_inputs[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
            cropped_labels = deformed_labels[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
            


            # Re-stack the image and labels
            if dataset.train_masks.any():
                # Include masks if they exist
                deformed_masks = samples[:, :, :, :, 2:]
                if label_output_type == AFFINITIES_3D:
                    deformed_masks = affinitize(deformed_masks)
                cropped_masks = deformed_masks[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
                self.training_example_op = tf.concat([tf.concat([cropped_inputs, cropped_labels, cropped_masks], axis=CHANNEL_AXIS)] * batch_size, axis=BATCH_AXIS)
                                                    
            else:
                self.training_example_op = tf.concat([tf.concat([cropped_inputs, cropped_labels], axis=CHANNEL_AXIS)] * batch_size, axis=BATCH_AXIS)

    def initialize_session_variables(self, sess):
        sess.run(self.__dataset_constant.initializer, feed_dict={self.__image_ph: self.__padded_dataset})
        del self.__padded_dataset

    def get_full_training_set(self):
        return self.__train_inputs, self.__train_labels, self.__train_targets

    def get_validation_set(self):
        return self.__validation_inputs, self.__validation_labels, self.__validation_targets

    def get_test_set(self):
        return self.__test_inputs

DATASET_DICT = {
    down.CREMI_A: CREMIDataset,
    down.CREMI_B: CREMIDataset,
    down.CREMI_C: CREMIDataset,
    down.CREMI_A_CLEFTS: CREMIDataset,
    down.CREMI_B_CLEFTS: CREMIDataset,
    down.CREMI_C_CLEFTS: CREMIDataset,
    down.ISBI: ISBIDataset,
    down.SNEMI3D: SNEMI3DDataset,
}