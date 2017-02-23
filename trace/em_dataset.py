import tifffile as tiff
import tensorflow as tf
import numpy as np
import cremi.io as cremiio

import download_data as down

from utils import *

import dataprovider.transform as trans
import augmentation as aug
import h5py


class Dataset(object):
    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        raise NotImplementedError

    def prepare_predictions_for_neuroglancer(self, results_folder, split, predictions, label_type):
        """Prepare the provided labels to be visualized using Neuroglancer.

        :param label_type: The format of the predictions passed in
        :param results_folder: The location where we should save the h5 file
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'split']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """

        # Create an affinities file
        with h5py.File(results_folder + split + '-predictions.h5', 'w') as output_file:
            output_file.create_dataset('main', shape=predictions.shape)

            # Reformat our predictions
            out = output_file['main']

            # Reshape and set in out
            for i in range(predictions.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(predictions[i], axis=0))
                out[0:2, i] = reshaped_pred[:, 0]

    def prepare_predictions_for_neuroglancer_affinities(self, results_folder, split, predictions, label_type):
        """Prepare the provided affinities to be visualized using Neuroglancer.

        :param label_type: The format of the predictions passed in
        :param results_folder: The location where we should save the h5 file
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'split']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        pred_affinities = convert_between_label_types(label_type, AFFINITIES_3D, predictions)
        sha = pred_affinities.shape
        # Create an affinities file
        with h5py.File(results_folder + split + '-pred-affinities.h5', 'w') as output_file:
            # Create the dataset in the file
            new_shape = (3, sha[0], sha[1], sha[2])

            output_file.create_dataset('main', shape=new_shape)

            # Reformat our predictions
            out = output_file['main']

            for i in range(pred_affinities.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(pred_affinities[i], axis=0))
                # Copy over as many affinities as we got
                out[0:sha[3], i] = reshaped_pred[:, 0]


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
        self.train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF)

        self.validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
        self.validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF)

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

        train_file = cremiio.CremiFile(data_folder + 'train.hdf', 'r')
        self.train_inputs = train_file.read_raw().data.value
        self.train_labels = train_file.read_neuron_ids().data.value
        train_file.close()

        validation_file = cremiio.CremiFile(data_folder + 'validation.hdf', 'r')
        self.validation_inputs = validation_file.read_raw().data.value
        self.validation_labels = validation_file.read_neuron_ids().data.value
        validation_file.close()

        # TODO(beisner): Decide if we need to load the test file every time (probably don't)

        test_file = cremiio.CremiFile(data_folder + 'train.hdf', 'r')
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
    def __init__(self, dataset, label_output_type=BOUNDARIES):
        """Helper for sampling an EM dataset

        :param dataset: An instance of Dataset, namely SNEMI3DDataset, ISBIDataset, or CREMIDataset
        :param label_output_type: The format in which the dataset labels should be sampled, i.e. for training, taking
        on values 'boundaries', 'affinities-2d', etc.
        """
        raise NotImplementedError("THIS COMMIT IS VERY BROKEN, BUT I WANT TO PULL")
        self.__dataset = dataset

        # Extract the inputs and labels from the dataset
        train_inputs = dataset.train_inputs
        train_labels = convert_between_label_types(dataset.label_type, label_output_type, dataset.train_labels)

        self.validation_inputs = dataset.validation_inputs
        self.validation_labels = convert_between_label_types(dataset.label_type, label_output_type, dataset.validation_labels)

        self.test_inputs = dataset.test_inputs

        # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs
        train_stacked = np.concatenate((train_inputs, train_labels), axis=3)

        # Define inputs to the graph
        self.crop_padding = 40
        self.patch_size_placeholder = tf.placeholder(dtype=tf.int32, shape=(), name='FOV')
        self.patch_size = tf.Variable(self.patch_size_placeholder, name='patch_size') + self.crop_padding

        # Create dataset, and pad the dataset with mirroring
        dataset = tf.constant(train_stacked, dtype=tf.float32)

        pad = tf.floordiv(self.patch_size, 2)
        padded_dataset = tf.pad(dataset, [[0, 0], tf.stack([pad, pad]), tf.stack([pad, pad]), [0, 0]], mode="REFLECT")

        # Sample and squeeze the dataset, squeezing so that we can perform the distortions
        sample = tf.random_crop(padded_dataset, size=[1, self.patch_size, self.patch_size, train_stacked.shape[3]])
        squeezed_sample = tf.squeeze(sample)

        # Perform the first transformation
        distorted_sample = tf.image.random_flip_left_right(squeezed_sample)
        self.distorted_sample = tf.image.random_flip_up_down(distorted_sample)

        # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute

        # Independently, feed in warped image
        # self.elastically_deformed_image = tf.placeholder(np.float64, shape=[None, None, 1], name="elas_deform_input")
        self.elastically_deformed_image = self.distorted_sample

        # self.standardized_image = tf.image.per_image_standardization(self.elastically_deformed_image)

        distorted_image = tf.image.random_brightness(self.elastically_deformed_image, max_delta=0.15)
        self.distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)

        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

    def get_validation_set(self):
        return self.validation_inputs, self.validation_labels

    def get_test_set(self):
        return self.test_inputs

    def generate_random_samples(self, model):
        # Create op for generation and enqueueing of random samples.

        # The distortion causes weird things at the boundaries, so we pad our sample and crop to get desired patch size


        # sigma = np.random.randint(low=35, high=100)

        # Apply elastic deformation

        # TODO(beisner): Move affinitization after elastic deformation, or think about it...
        # el_image, el_labels = aug.elastic_transform(separated_image, separated_labels, alpha=2000, sigma=sigma)

        # with tf.device('/cpu:0'):
        crop_padding = self.crop_padding
        cropped_image = self.distorted_image[crop_padding // 2:-crop_padding // 2,
                        crop_padding // 2:-crop_padding // 2, :1]
        cropped_labels = self.elastically_deformed_image[crop_padding // 2:-crop_padding // 2,
                         crop_padding // 2:-crop_padding // 2, 1:]

        training_example = tf.concat(3, [tf.expand_dims(cropped_image, 0), tf.expand_dims(cropped_labels, 0)])

        enqueue_op = model.queue.enqueue(training_example)

        return enqueue_op








#
# class EMDataset(object):
#     def __init__(self, data_folder, input_mode='boundary-output', output_mode='boundaries'):
#         self.data_folder = data_folder
#         self.input_mode = input_mode
#         self.output_mode = output_mode
#
#         # Read in the dataset, which we assume to be in TIF format
#         train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
#         train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF)
#
#         validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
#         validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF)
#
#         test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)
#
#         # If the dataset we are using
#         if self.input_mode == BOUNDARY_INPUT:
#
#
#
#
#        # with tf.device('/cpu:0'):
#
#
#         # Read in the datasets
#         train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
#         train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF)
#
#         validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
#         validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF)
#
#         test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)
#
#         # All inputs have one channel
#         train_inputs = np.expand_dims(train_inputs, 3)
#         self.validation_inputs = np.expand_dims(validation_inputs, 3)
#         self.test_inputs = np.expand_dims(test_inputs, 3)
#
#         # Transform the labels based on the mode we are using
#         if output_mode == BOUNDARY_OUTPUT:
#             # Expand dimensions from [None, None, None] -> [None, None, None, 1]
#             train_labels = np.expand_dims(train_labels, 3) // 255.0
#             self.validation_labels = np.expand_dims(validation_labels, 3) // 255.0
#
#         elif output_mode == AFFINITIES_2D_OUTPUT:
#             # Affinitize in 2 dimensions
#
#             def aff_and_reshape_2d(dset):
#
#                 # Affinitize
#                 aff_dset = trans.affinitize(dset)
#
#                 # Reshape [3, None, None, None] -> [None, None, None, 3]
#                 rearranged = np.einsum('abcd->bcda', aff_dset)
#
#                 # Remove the third dimension
#                 return rearranged[:, :, :, 0:2]
#
#             train_labels = aff_and_reshape_2d(train_labels)
#             self.validation_labels = aff_and_reshape_2d(validation_labels)
#
#         elif output_mode == AFFINITIES_3D_OUTPUT:
#             # Affinitize in 3 dimensions
#
#             def aff_and_reshape_3d(dset):
#
#                 # Affinitize
#                 aff_dset = trans.affinitize(dset)
#
#                 # Reshape [3, None, None, None] -> [None, None, None, 3]
#                 return np.einsum('abcd->bcda', aff_dset)
#
#             train_labels = aff_and_reshape_3d(train_labels)
#             self.validation_labels = aff_and_reshape_3d(validation_labels)
#
#         else:
#             raise Exception("Invalid output_mode!")
#
#         # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs
#         train_stacked = np.concatenate((train_inputs, train_labels), axis=3)
#
#         # Define inputs to the graph
#         self.crop_padding = 40
#         self.patch_size_placeholder = tf.placeholder(dtype=tf.int32, shape=(), name='FOV')
#         self.patch_size = tf.Variable(self.patch_size_placeholder, name='patch_size') + self.crop_padding
#
#         # Create dataset, and pad the dataset with mirroring
#         dataset = tf.constant(train_stacked, dtype=tf.float32)
#
#         pad = tf.floordiv(self.patch_size, 2)
#         padded_dataset = tf.pad(dataset, [[0, 0], tf.stack([pad, pad]), tf.stack([pad, pad]), [0, 0]], mode="REFLECT")
#
#         # Sample and squeeze the dataset, squeezing so that we can perform the distortions
#         sample = tf.random_crop(padded_dataset, size=[1, self.patch_size, self.patch_size, train_stacked.shape[3]])
#         squeezed_sample = tf.squeeze(sample)
#
#         # Perform the first transformation
#         distorted_sample = tf.image.random_flip_left_right(squeezed_sample)
#         self.distorted_sample = tf.image.random_flip_up_down(distorted_sample)
#
#         # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute
#
#         # Independently, feed in warped image
#         #self.elastically_deformed_image = tf.placeholder(np.float64, shape=[None, None, 1], name="elas_deform_input")
#         self.elastically_deformed_image = self.distorted_sample
#
#         # self.standardized_image = tf.image.per_image_standardization(self.elastically_deformed_image)
#
#         distorted_image = tf.image.random_brightness(self.elastically_deformed_image, max_delta=0.15)
#         self.distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
#
#         #self.sess = tf.Session()
#         #self.sess.run(tf.global_variables_initializer())
#
#     def get_validation_set(self):
#         return self.validation_inputs, self.validation_labels
#
#     def get_test_set(self):
#         return self.test_inputs
#
#     def generate_random_samples(self, model):
#         # Create op for generation and enqueueing of random samples.
#
#         # The distortion causes weird things at the boundaries, so we pad our sample and crop to get desired patch size
#
#
#         #sigma = np.random.randint(low=35, high=100)
#
#         # Apply elastic deformation
#
#         # TODO(beisner): Move affinitization after elastic deformation, or think about it...
#         #el_image, el_labels = aug.elastic_transform(separated_image, separated_labels, alpha=2000, sigma=sigma)
#
#         #with tf.device('/cpu:0'):
#         crop_padding = self.crop_padding
#         cropped_image = self.distorted_image[crop_padding // 2:-crop_padding // 2,
#                 crop_padding // 2:-crop_padding // 2, :1]
#         cropped_labels = self.elastically_deformed_image[crop_padding // 2:-crop_padding // 2,
#                 crop_padding // 2:-crop_padding // 2, 1:]
#
#         training_example = tf.concat(3, [tf.expand_dims(cropped_image, 0), tf.expand_dims(cropped_labels, 0)])
#
#         enqueue_op = model.queue.enqueue(training_example)
#
#         return enqueue_op
