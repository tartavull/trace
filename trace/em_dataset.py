import tensorflow as tf
import numpy as np

# I/O
import tifffile as tiff
import cremi.io as cremiio
import h5py

import download_data as down
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
        predictions = convert_between_label_types(label_type, SEGMENTATION_3D, predictions)
        # Create an affinities file
        with h5py.File(results_folder + split + '-predictions.h5', 'w') as output_file:
            output_file.create_dataset('main', shape=predictions.shape)

            # Reformat our predictions
            out = output_file['main']

            # Reshape and set in out
            for i in range(predictions.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(predictions[i], axis=0))
                out[0:2, i] = reshaped_pred[:, 0]

    @staticmethod
    def prepare_predictions_for_neuroglancer_affinities(results_folder, split, predictions, label_type):
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

        train_input_file = cremiio.CremiFile(data_folder + 'train_input.h5', 'r')
        train_label_file = cremiio.CremiFile(data_folder + 'train_labels.h5', 'r')
        self.train_inputs = train_input_file.read_raw().data.value
        self.train_labels = train_label_file.read_neuron_ids().data.value
        train_input_file.close()
        train_label_file.close()

        validation_input_file = cremiio.CremiFile(data_folder + 'validation_input.h5', 'r')
        validation_label_file = cremiio.CremiFile(data_folder + 'validation_labels.h5', 'r')
        self.validation_inputs = validation_input_file.read_raw().data.value
        self.validation_labels = validation_labels_file.read_neuron_ids().data.value
        validation_input_file.close()
        validation_label_file.close()

        # TODO(beisner): Decide if we need to load the test file every time (probably don't)
        test_file = cremiio.CremiFile(data_folder + 'test.hdf', 'r')
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
        :param batch_size: The number of images to stack together in a batch
        :param dataset: An instance of Dataset, namely SNEMI3DDataset, ISBIDataset, or CREMIDataset
        :param label_output_type: The format in which the dataset labels should be sampled, i.e. for training, taking
        on values 'boundaries', 'affinities-2d', etc.
        """

        # Extract the inputs and labels from the dataset
        self.train_inputs = dataset.train_inputs
        self.train_labels = convert_between_label_types(dataset.label_type, label_output_type, dataset.train_labels)

        self.validation_inputs = dataset.validation_inputs
        self.validation_labels = convert_between_label_types(dataset.label_type, label_output_type,
                                                             dataset.validation_labels)
        self.test_inputs = dataset.test_inputs

        self.train_inputs = np.expand_dims(self.train_inputs, axis=3)
        if label_output_type == BOUNDARIES:
            self.dim = 2
            self.train_labels = np.expand_dims(self.train_labels, axis=3)
        elif label_output_type == AFFINITIES_2D:
            self.dim = 2
            self.train_labels = np.einsum('dzyx->zyxd', self.train_labels[:2])
        elif label_output_type == AFFINITIES_3D:
            self.dim = 3
            self.train_labels = np.einsum('dzyx->zyxd', self.train_labels)

        # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs
        train_stacked = np.concatenate((self.train_inputs, self.train_labels), axis=3)
        if self.dim == 3:
            train_stacked = np.expand_dims(train_stacked, axis=0)

        # Define inputs to the graph
        crop_pad = input_size // 4
        z_crop_pad = z_input_size // 2
        patch_size = input_size + crop_pad
        z_patch_size = z_input_size + z_crop_pad

        # Create dataset, and pad the dataset with mirroring
        pad = input_size // 2
        z_pad = z_input_size // 2
        pad_dims = [[pad, pad], [pad, pad]]
        if self.dim == 3:
            pad_dims = [[z_pad, z_pad]] + pad_dims
        self.padded_dataset = np.pad(train_stacked, [[0, 0]] + pad_dims + [[0, 0]], mode='reflect')

        # The dataset is loaded into a constant variable from a placeholder
        # because a tf.constant cannot hold a dataset that is over 2GB.
        image_ph = tf.placeholder(dtype=tf.float32, shape=self.padded_dataset.shape, name='image_ph')
        #dataset_constant = tf.Variable(image_ph, trainable=False, collections=[])
        print(self.padded_dataset.shape)
        print(self.padded_dataset[:, :self.padded_dataset.shape[1] // 2, :self.padded_dataset.shape[2] // 2, :self.padded_dataset.shape[3] // 2].shape)
        dataset_constant = tf.constant(self.padded_dataset[:, :self.padded_dataset.shape[1] // 2, :self.padded_dataset.shape[2] // 2, :self.padded_dataset.shape[3] // 2], dtype=tf.float32)

        with tf.device('/cpu:0'):
            # Sample and squeeze the dataset, squeezing so that we can perform the distortions
            patch_size_dims = [patch_size, patch_size]
            if self.dim == 3:
                patch_size_dims = [z_patch_size] + patch_size_dims
            sample = tf.random_crop(dataset_constant, size=[batch_size] + patch_size_dims + [train_stacked.shape[self.dim + 1]])

            # Perform random flips
            #flipped_sample = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), sample)
            #flipped_sample = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), flipped_sample)
            flipped_sample = sample

            # Apply a random rotation
            # angle = tf.random_uniform(shape=(), minval=0, maxval=6.28)
            # rotated_sample = tf.contrib.image.rotate(flipped_sample, angle)
            rotated_sample = flipped_sample

            # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute
            # elastically_deformed_sample = tf.elastic_deformation(rotated_sample)
            elastically_deformed_sample = rotated_sample

            # Separate the image from the labels
            if self.dim == 2:
                deformed_image = elastically_deformed_sample[:, :, :, :1]
                deformed_labels = elastically_deformed_sample[:, :, :, 1:]
            elif self.dim == 3:
                deformed_image = elastically_deformed_sample[:, :, :, :, :1]
                deformed_labels = elastically_deformed_sample[:, :, :, :, 1:]


            # Mess with the levels
            #leveled_image = tf.image.random_brightness(deformed_image, max_delta=0.15)
            #leveled_image = tf.image.random_contrast(leveled_image, lower=0.5, upper=1.5)
            leveled_image = deformed_image

            # leveled_image = tf.Print(leveled_image, [tf.shape(leveled_image)])

            # Crop the image, to remove the padding that was added to allow safe
            # augmentation.
            if self.dim == 2:
                cropped_image = leveled_image[:, crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
                cropped_labels = deformed_labels[:, crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
            elif self.dim == 3:
                cropped_image = leveled_image[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
                cropped_labels = deformed_labels[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]

            # Re-stack the image and labels
            self.training_example_op = tf.concat([cropped_image, cropped_labels], axis=self.dim + 1)

    def get_full_training_set(self):
        return self.train_inputs, self.train_labels

    def get_validation_set(self):
        return self.validation_inputs, self.validation_labels

    def get_test_set(self):
        return self.test_inputs

DATASET_DICT = {
    down.CREMI_A: CREMIDataset,
    down.CREMI_B: CREMIDataset,
    down.CREMI_C: CREMIDataset,
    down.ISBI: ISBIDataset,
    down.SNEMI3D: SNEMI3DDataset,
}