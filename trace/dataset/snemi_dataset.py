from __future__ import print_function

import tifffile as tiff

from trace.common import *
from trace.utils import convert_between_label_types

from .dataset import Dataset


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

        self.train_inputs = tiff.imread(data_folder + TRAIN_INPUT + TIF)
        self.train_labels = tiff.imread(data_folder + TRAIN_LABELS + TIF)

        self.validation_inputs = tiff.imread(data_folder + VALIDATION_INPUT + TIF)
        self.validation_labels = tiff.imread(data_folder + VALIDATION_LABELS + TIF)

        self.test_inputs = tiff.imread(data_folder + TEST_INPUT + TIF)

    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        """Prepare the provided labels to submit on the SNEMI3D competition.

        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param results_folder: The location where we should save the dataset
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions)
        tiff.imsave(results_folder + split + '-predictions.tif', trans_predictions)
