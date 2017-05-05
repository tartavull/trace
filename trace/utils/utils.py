import subprocess
import shutil
import h5py
import time
import os
import tensorflow as tf
import numpy as np

import dataprovider.transform as trans

from trace.common import *
import trace.thirdparty.watershed as wshed

try:
    from trace.thirdparty.segascorus import io_utils
    from trace.thirdparty.segascorus import utils
except ImportError:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")


# seg is the batch of segmentations to be affinitized. seg should be a 5D tensor
# dst is the stride of the affinity in each diretion
def tf_affinitize(seg, dst=(1, 1, 1)):
    seg_shape = tf.shape(seg)
    (dz, dy, dx) = dst

    # z-affinity
    if dz > 0:
        connected = tf.equal(seg[:, dz:, :, :, :], seg[:, :-dz, :, :, :])
        background = tf.greater(seg[:, dz:, :, :, :], 0)
        zero_pad = tf.zeros(tf.concat([(seg_shape[BATCH_AXIS], dz), seg_shape[Y_AXIS:]], axis=0))
        z_aff = tf.concat([zero_pad, tf.cast(tf.logical_and(connected, background), tf.float32)], axis=Z_AXIS)

    # y-affinity
    if dy > 0:
        connected = tf.equal(seg[:, :, dy:, :, :], seg[:, :, :-dy, :, :])
        background = tf.greater(seg[:, :, dy:, :, :], 0)
        zero_pad = tf.zeros(tf.concat([seg_shape[:Y_AXIS], (dy,), seg_shape[X_AXIS:]], axis=0))
        y_aff = tf.concat([zero_pad, tf.cast(tf.logical_and(connected, background), tf.float32)], axis=Y_AXIS)

    # x-affinity
    if dx > 0:
        connected = tf.equal(seg[:, :, :, dx:, :], seg[:, :, :, :-dx, :])
        background = tf.greater(seg[:, :, :, dx:, :], 0)
        zero_pad = tf.zeros(tf.concat([seg_shape[:X_AXIS], (dx, seg_shape[CHANNEL_AXIS])], axis=0))
        x_aff = tf.concat([zero_pad, tf.cast(tf.logical_and(connected, background), tf.float32)], axis=X_AXIS)

    aff = tf.concat([x_aff, y_aff, z_aff], axis=CHANNEL_AXIS)

    return aff


def cond_apply(inp, fn_true, fn_false, cond):
    if cond:
        return fn_true(inp)
    else:
        return fn_false(inp)


def expand_3d_to_5d(data):
    # Add a batch dimension and a channel dimension
    data = np.expand_dims(data, axis=BATCH_AXIS)
    data = np.expand_dims(data, axis=CHANNEL_AXIS)

    return data


def run_watershed_on_affinities(affinities, relabel2d=False, low=0.9, hi=0.9995):
    # tmp_aff_file = 'tmp-affinities.h5'
    # tmp_label_file = 'tmp-labels.h5'
    #
    # base = './tmp/' + str(int(round(time.time() * 1000))) + '/'
    #
    # os.makedirs(base)
    #
    # # Move to the front
    # reshaped_aff = np.einsum('zyxd->dzyx', affinities)
    #
    # shape = reshaped_aff.shape
    #
    # # Write predictions to a temporary file
    # with h5py.File(base + tmp_aff_file, 'w') as output_file:
    #     output_file.create_dataset('main', shape=(3, shape[1], shape[2], shape[3]))
    #     out = output_file['main']
    #     out[:shape[0], :, :, :] = reshaped_aff
    #
    # # Do watershed segmentation
    # print('segmenting')
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # subprocess.call(["julia",
    #                  current_dir + "/thirdparty/watershed/watershed.jl",
    #                  base + tmp_aff_file,
    #                  base + tmp_label_file,
    #                  str(hi),
    #                  str(low)])
    #
    # print('segmentation complete')
    #
    # # Load the results of watershedding, and maybe relabel
    # pred_seg = io_utils.import_file(base + tmp_label_file)

    print('Begin segmentation')

    pred_seg, _ = wshed.watershed(affinities, low, hi)


    print('Segmentation complete')

    prep = utils.parse_fns(utils.prep_fns, [relabel2d, False])
    pred_seg, _ = utils.run_preprocessing(pred_seg, pred_seg, prep)


    return pred_seg


def convert_between_label_types(input_type, output_type, original_labels):
    # No augmentation needed, as we're basically doing e2e learning
    if input_type == output_type:
        return original_labels

    # This looks like a shit show, but conversion is hard.
    # Also, we will implement this as we go.
    # Alternatively, we could convert to some intermediate form (3D Affinities), and then convert to a final form

    if input_type == BOUNDARIES:
        if output_type == AFFINITIES_2D:
            raise NotImplementedError('Boundaries->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            raise NotImplementedError('Boundaries->Aff3d not implemented')
        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Boundaries->Seg2d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Boundaries->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_2D:
        if output_type == BOUNDARIES:
            # Take the average of each affinity in the x and y direction
            return np.mean(original_labels, axis=3)
        elif output_type == AFFINITIES_3D:
            # There are no z-direction affinities, so just make the z-affinity 0
            sha = original_labels.shape
            dtype = original_labels.dtype
            return np.concatenate((original_labels, np.zeros([sha[0], sha[1], sha[2], 1], dtype=dtype)), axis=3)
        elif output_type == SEGMENTATION_2D:
            # Run watershed, and relabel segmentation so each slice has unique labels
            return run_watershed_on_affinities(original_labels, relabel2d=True)
        elif output_type == SEGMENTATION_3D:
            # Run watershed
            return run_watershed_on_affinities(original_labels)
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_3D:
        if output_type == BOUNDARIES:
            # Take the average of each affinity in the x, y, and z direction
            return np.mean(original_labels, axis=3)
        elif output_type == AFFINITIES_2D:
            # Discard the affinities in the z direction
            return original_labels[:, :, :, 0:2]
        elif output_type == SEGMENTATION_2D:
            # Run watershed, and relabel segmentation so each slice has unique labels
            return run_watershed_on_affinities(original_labels, relabel2d=True)
        elif output_type == SEGMENTATION_3D:
            # Run watershed
            return run_watershed_on_affinities(original_labels)
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_2D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg2d->Boundaries not implemented')
        elif output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg2d->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            raise NotImplementedError('Seg2d->Aff3d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Seg2d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_3D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg3d->Boundaries not implemented')
        elif output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg3d->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:

            # For each batch of stacks, affinitize and reshape

            def aff_and_reshape(labs):
                # Affinitize takes a 3d tensor, so we just take the first index
                return np.einsum('dzyx->zyxd', trans.affinitize(labs[:, :, :, 0]))

            return np.array(map(aff_and_reshape, original_labels))

        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Seg3d->Seg2d not implemented')
        else:
            raise Exception('Invalid output_type')
    else:
        raise Exception('Invalid input_type')


def tf_convert_between_label_types(input_type, output_type, original_labels):
    # Don't convert if it's unnecessary
    if input_type == output_type:
        return original_labels

    if input_type == BOUNDARIES:
        if output_type == AFFINITIES_2D:
            raise NotImplementedError('Boundaries->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            raise NotImplementedError('Boundaries->Aff3d not implemented')
        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Boundaries->Seg2d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Boundaries->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_2D:
        if output_type == BOUNDARIES:
            # Take the average of each affinity in the x and y direction
            return np.mean(original_labels, axis=3)
        elif output_type == AFFINITIES_3D:
            # There are no z-direction affinities, so just make the z-affinity 0
            sha = original_labels.shape
            dtype = original_labels.dtype
            return np.concatenate((original_labels, np.zeros([sha[0], sha[1], sha[2], 1], dtype=dtype)), axis=3)
        elif output_type == SEGMENTATION_2D:
            # Run watershed, and relabel segmentation so each slice has unique labels
            return run_watershed_on_affinities(original_labels, relabel2d=True)
        elif output_type == SEGMENTATION_3D:
            # Run watershed
            return run_watershed_on_affinities(original_labels)
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_3D:
        if output_type == BOUNDARIES:
            # Take the average of each affinity in the x, y, and z direction
            return np.mean(original_labels, axis=3)
        elif output_type == AFFINITIES_2D:
            # Discard the affinities in the z direction
            return original_labels[:, :, :, 0:2]
        elif output_type == SEGMENTATION_2D:
            # Run watershed, and relabel segmentation so each slice has unique labels
            return run_watershed_on_affinities(original_labels, relabel2d=True)
        elif output_type == SEGMENTATION_3D:
            # Run watershed
            return run_watershed_on_affinities(original_labels)
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_2D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg2d->Boundaries not implemented')
        elif output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg2d->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            raise NotImplementedError('Seg2d->Aff3d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Seg2d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_3D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg3d->Boundaries not implemented')
        elif output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg3d->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            return tf_affinitize(original_labels)

        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Seg3d->Seg2d not implemented')
        else:
            raise Exception('Invalid output_type')
    else:
        raise Exception('Invalid input_type')