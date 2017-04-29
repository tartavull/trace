from __future__ import print_function
from __future__ import division

import os

import h5py
import subprocess
import skimage.measure as meas
import skimage.morphology as morph
import tifffile as tiff

import cv2
from utils import *

import cremi.io as cremiio
import clefts
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

    # Segascorus demands uint32
    true_seg = np.squeeze(true_seg.astype(dtype=np.uint32))
    pred_seg = pred_seg.astype(dtype=np.uint32)

    # Preprocess
    # relabel2d: if True, relabel the segments such that individual slices contain unique IDs
    # foreground_restrict: if True, from both images remove the border pixels of the true segmentation
    prep = utils.parse_fns(utils.prep_fns, [relabel2d, foreground_restricted])
    pred_seg, true_seg = utils.run_preprocessing(pred_seg, true_seg, prep)

    # Calculate the overlap matrix
    om = utils.calc_overlap_matrix(pred_seg, true_seg, split_0_segment)

    # Determine which metrics to execute
    # TODO(beisner): fix VOI
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

    # TODO(beisner): fix VOI
    results["{} Full".format('VI')] = 0
    results["{} Merge".format('VI')] = 0
    results["{} Split".format('VI')] = 0

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
        scores[thresh] = __rand_error(true_seg, pred_seg, calc_variation_information=False)

    max_score_thresh = max(scores.iterkeys(), key=(lambda key: scores[key]['Rand F-Score Full']))

    return scores[max_score_thresh]

def __rand_error_affinities(pred_affinities, true_seg, aff_type=AFFINITIES_3D):
    # If the affinities type we're being passed in is 2D, we can only generate a 2D segmentation,
    # so we must relabel
    relabel2d = (aff_type == AFFINITIES_2D)

    # Generate a 3D segmentation from the affinities, regardless of whether they are 2D or 3D, because that's
    # how segascorus works
    pred_segmentation = convert_between_label_types(input_type=aff_type, output_type=SEGMENTATION_3D,
                                                    original_labels=pred_affinities)

    return __rand_error(true_seg, pred_segmentation, calc_variation_information=False, calc_variation_score=False, relabel2d=relabel2d)

def cleft_stats_optimize(pred, truth, thresh_pos=200, thresh_neg=200, thresh_true=200):
    pred_file = cremiio.CremiFile(pred, 'r')
    truth_file = cremiio.CremiFile(truth, 'r')

    temp_file = cremiio.CremiFile('temp.hdf', 'w')
    truth_cleft = truth_file.read_clefts()
    truth_cleft_data = truth_cleft.data.value
    truth_cleft_res = truth_cleft.resolution
    truth_cleft_data = np.where(truth_cleft_data == 0., truth_cleft_data, 1)
    truth_cleft_data = np.where(truth_cleft_data == 1., truth_cleft_data, 0xffffffffffffffff)

    temp_file.write_clefts(cremiio.Volume(truth_cleft_data, resolution=truth_cleft_res))
    clefts_eval = clefts.Clefts(pred_file.read_clefts(), temp_file.read_clefts())

    truth_file.close()
    temp_file.close()
    pred_file.close()
    os.remove('temp.hdf')

    num_false_pos = clefts_eval.count_false_positives(threshold=thresh_pos)
    num_false_neg = clefts_eval.count_false_negatives(threshold=thresh_neg)
    num_true_pos = clefts_eval.count_true_positives(threshold=thresh_true)
    f_score = clefts_eval.f_score(num_true_pos, num_false_neg, num_false_pos)

    return f_score, num_true_pos, num_false_pos, num_false_neg


def cleft_stats(pred, truth, thresh_pos, thresh_neg, thresh_true):
    pred_file = cremiio.CremiFile(pred, 'r')
    truth_file = cremiio.CremiFile(truth, 'r')

    temp_file = cremiio.CremiFile('temp.hdf', 'w')
    truth_cleft = truth_file.read_clefts()
    truth_cleft_data = truth_cleft.data.value
    truth_cleft_res = truth_cleft.resolution
    truth_cleft_data = np.where(truth_cleft_data == 0., truth_cleft_data, 1)
    truth_cleft_data = np.where(truth_cleft_data == 1., truth_cleft_data, 0xffffffffffffffff)

    temp_file.write_clefts(cremiio.Volume(truth_cleft_data, resolution=truth_cleft_res))
    clefts_eval = clefts.Clefts(pred_file.read_clefts(), temp_file.read_clefts())

    pred_file.close()
    truth_file.close()
    temp_file.close()
    os.remove('temp.hdf')

    num_false_pos = clefts_eval.count_false_positives(threshold=thresh_pos)
    num_false_neg = clefts_eval.count_false_negatives(threshold=thresh_neg)
    num_true_pos = clefts_eval.count_true_positives(threshold=thresh_true)
    acc_false_neg = clefts_eval.acc_false_negatives()
    acc_false_pos = clefts_eval.acc_false_positives()

    return num_true_pos, num_false_pos, num_false_neg, acc_false_pos, acc_false_neg

def rand_error_from_prediction(true_labels, pred_values, pred_type=BOUNDARIES):
    """ Predict the rand error and variation of information from a given prediction. Based on the prediction type, we
    generate a segmentation and evaluate based on that. Only accept one at a time, rather than in batches

    :param true_labels: The true labels, either in BOUNDARIES mode or SEGMENTATION mode (depends on dataset). Shape must
                        be [z, y, x], since labels are 1-dimensional
    :param pred_values: The predictions, either in BOUNDARIES, SEGMENTATION, or AFFINITIES mode. Shape must be
                        [z, y, x, chan], since labels are either 1, 2, or 3 dimensional
    :param pred_type: The label format of the prediction, either BOUNDARIES, AFFINITIES, or SEGMENTATION
    :return: A list of all the calculated scores
    """
    #assert(len(true_labels.shape) == 3)
    #assert(len(pred_values.shape) == 4)

    if pred_type == BOUNDARIES:
        true_labels = np.squeeze(true_labels, axis=3)
        pred_values = np.squeeze(pred_values, axis=3)
        return __rand_error_boundaries(true_labels, pred_values)
    elif pred_type == AFFINITIES_2D or pred_type == AFFINITIES_3D:
        return __rand_error_affinities(pred_values, true_labels, aff_type=pred_type)
    elif pred_type == SEGMENTATION_2D or pred_type == SEGMENTATION_3D:
        return __rand_error(true_labels, pred_values, relabel2d=(pred_type == SEGMENTATION_2D))
    else:
        raise NotImplementedError('Invalid pred_type %s' % pred_type)
