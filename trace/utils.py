import subprocess
import os

import numpy as np

import tensorflow as tf

import h5py
import time

import dataprovider.transform as trans

try:
    from thirdparty.segascorus import io_utils
    from thirdparty.segascorus import utils
except Exception:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")

# LABEL MODES
BOUNDARIES = 'boundaries'
AFFINITIES_2D = 'affinities-2d'
AFFINITIES_3D = 'affinities-3d'
SEGMENTATION_2D = 'segmentation-2d'
SEGMENTATION_3D = 'segmentation-3d'

BATCH_AXIS = 0
Z_AXIS = 1
Y_AXIS = 2
X_AXIS = 3
CHANNEL_AXIS = 4

SPLIT = ['train', 'validation', 'test']


def expand_3d_to_5d(data):
    # Add a batch dimension and a channel dimension
    data = np.expand_dims(data, axis=BATCH_AXIS)
    data = np.expand_dims(data, axis=CHANNEL_AXIS)

    return data


def run_watershed_on_affinities(affinities, relabel2d=False, low=0.9, hi=0.95):
    tmp_aff_file = 'tmp-affinities.h5'
    tmp_label_file = 'tmp-labels.h5'

    base = './tmp/' + str(int(round(time.time() * 1000))) + '/'
    os.makedirs(base)

    # Move to the front
    reshaped_aff = np.einsum('zyxd->dzyx', affinities)

    shape = reshaped_aff.shape

    # Write predictions to a temporary file
    with h5py.File(base + tmp_aff_file, 'w') as output_file:
        output_file.create_dataset('main', shape=(3, shape[1], shape[2], shape[3]))
        out = output_file['main']
        out[:shape[0], :, :, :] = reshaped_aff

    # Do watershed segmentation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     base + tmp_aff_file,
                     base + tmp_label_file,
                     str(hi),
                     str(low)])

    # Load the results of watershedding, and maybe relabel
    pred_seg = io_utils.import_file(base + tmp_label_file)

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



