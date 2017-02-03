from __future__ import print_function
from __future__ import division

import os

import h5py
import subprocess
import skimage.measure as meas
import skimage.morphology as morph
import tifffile as tiff

import cv2

try:
    from thirdparty.segascorus import io_utils
    from thirdparty.segascorus import utils
    from thirdparty.segascorus.metrics import *
except Exception:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")


def __rand_error(true_seg, pred_seg, calc_rand_score=True, calc_rand_error=False, calc_variation_score=True,
                 calc_variation_information=False, relabel2d=True, foreground_restricted=True, split_0_segment=True,
                 other=None):

    # Preprocess
    # relabel2d: if True, relabel the segments such that individual slices contain unique IDs
    # foreground_restrict: if True, from both images remove the border pixels of the true segmentation
    prep = utils.parse_fns(utils.prep_fns, [relabel2d, foreground_restricted])
    pred_seg, true_seg = utils.run_preprocessing(pred_seg, true_seg, prep)

    # Calculate the overlap matrix
    om = utils.calc_overlap_matrix(pred_seg, true_seg, split_0_segment)

    # Determine which metrics to execute
    metrics = utils.parse_fns(utils.metric_fns, [calc_rand_score, calc_rand_error, calc_variation_score,
                                                 calc_variation_information])

    # Prepare a results dictionary
    results = {}
    for (name, metric_fn) in metrics:
        if relabel2d:
            full_name = "2D {}".format(name)
        else:
            full_name = name

        (f, m, s) = metric_fn(om, full_name, other)
        results["{} Full".format(name)] = f
        results["{} Merge".format(name)] = m
        results["{} Split".format(name)] = s

    return results


def __prepare_true_segmentation_for_rand(images):
    # Add a 1-pixel boundary, otherwise segmentation will ignore segments that are cut off by boundaries
    bordered = np.zeros((images.shape[0], images.shape[1] + 2, images.shape[2] + 2), dtype=np.uint8)
    bordered[:, 1:-1, 1:-1] = images

    # Segment each image in the stack
    segs = np.zeros(images.shape, dtype=np.uint8)
    for im in range(images.shape[0]):
        segs[im] = meas.label(bordered[im], background=0, connectivity=1)[1:-1, 1:-1]

    return segs


def __prepare_probabilistic_segmentation_for_rand(thresh, images):
    # For each image in the stack
    segs = np.zeros(images.shape, dtype=np.uint8)
    for im in range(images.shape[0]):
        # Threshold at a given value
        _, threshed = cv2.threshold(images[im], thresh, 255, cv2.THRESH_BINARY)

        # Add a 1-px boundary
        bordered = np.zeros((images.shape[1] + 2, images.shape[2] + 2), dtype=np.uint8)
        bordered[1:-1, 1:-1] = threshed

        # Thin the borders
        skel = 255 - morph.skeletonize_3d(255 - bordered)

        # Segment
        segs[im] = meas.label(skel, background=0, connectivity=1)[1:-1, 1:-1]

    return segs


def __rand_error_boundaries(true_labels, pred_labels):
    # Make sure in the shape [n_images, im_dim_1, im_dim_2]
    assert (len(true_labels.shape) == 3)
    assert (len(pred_labels.shape) == 3)

    # Get the true segmentation
    true_seg = __prepare_true_segmentation_for_rand(true_labels)

    # Get the rand scores for each threshold
    scores = dict()

    for thresh in np.arange(0, 1, 0.1):
        # print('Calculating at thresh %0.1f' % thresh)
        pred_seg = __prepare_probabilistic_segmentation_for_rand(thresh, pred_labels)

        # TODO(beisner): Currently, VI doesn't work, gets a divide by zero error
        # Convert to uint32 here because otherwise we get an error about type mismatch in the C++ code
        scores[thresh] = __rand_error(true_seg.astype(np.uint32), pred_seg.astype(np.uint32), calc_variation_information=False)

    max_score_thresh = max(scores.iterkeys(), key=(lambda key: scores[key]['Rand F-Score Full']))

    # TODO(beisner): remove when VI is fixed
    scores[max_score_thresh]['VI F-Score Full'] = 0
    scores[max_score_thresh]['VI F-Score Merge'] = 0
    scores[max_score_thresh]['VI F-Score Split'] = 0
    return scores[max_score_thresh]


def __rand_error_affinities(model, data_folder, sigmoid_prediction, num_layers, output_shape, watershed_high=0.9, watershed_low=0.3):
    # Save affinities to temporary file
    # TODO pad the image with zeros so that the output covers the whole dataset

    tmp_aff_file = 'validation-tmp-affinities.h5'
    tmp_label_file = 'validation-tmp-labels.h5'
    ground_truth_file = 'validation-labels.h5'

    base = data_folder + 'results/' + model.model_name + '/'

    # Write predictions to a temporary file
    with h5py.File(base + tmp_aff_file, 'w') as output_file:
        output_file.create_dataset('main', shape=(3, num_layers, output_shape, output_shape))
        out = output_file['main']

        reshaped_pred = np.einsum('zyxd->dzyx', sigmoid_prediction)
        out[0:2, :, :, :] = reshaped_pred

    # Do watershed segmentation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     base + tmp_aff_file,
                     base + tmp_label_file,
                     str(watershed_high),
                     str(watershed_low)])

    # Load the results of watershedding
    pred_seg = io_utils.import_file(base + tmp_label_file)
    true_seg = io_utils.import_file(data_folder + ground_truth_file)

    return __rand_error(true_seg, pred_seg)


def rand_error(model, data_folder, labels_file, sigmoid_prediction, num_layers, output_shape, data_type='boundaries'):

    if data_type == 'boundaries':
        # Squeeze the single layer into one
        reshaped_pred = sigmoid_prediction.squeeze(axis=(3,))
        true_labels = tiff.imread(data_folder + labels_file)
        return __rand_error_boundaries(true_labels, reshaped_pred)
    else:
        return __rand_error_affinities(model, data_folder, sigmoid_prediction, num_layers, output_shape)
