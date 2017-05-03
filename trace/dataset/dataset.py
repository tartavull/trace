from __future__ import print_function

import numpy as np
import h5py

from trace.common import *
from trace.utils import convert_between_label_types


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
