from __future__ import print_function

import cremi.io as cremiio

from trace.common import *
from trace.utils import convert_between_label_types
from .dataset import Dataset


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
