import subprocess
import os

import numpy as np

import shutil

import tensorflow as tf

import h5py
import time

import os

import dataprovider.transform as trans


import cremi.io as cremiio
#import cremi.evaluation as cremival
from cremi.evaluation import NeuronIds

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

SPLIT = ['train', 'validation', 'test', 'aligned']

def check_tensor(data):
    """Ensure that data is numpy 4D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError('data must be a numpy 4D array')

    assert data.ndim==4
    return data

def flip(data, rule):
    """Flip data according to a specified rule.
Args:
data: 3D numpy array to be transformed.
rule: Transform rule, specified as a Boolean array.
     [z reflection,
      y reflection,
      x reflection,
      xy transpose]
Returns:
data: Transformed data.
"""
    data = check_tensor(data)

    assert np.size(rule)==4

    # z reflection
    if rule[0]:
        data = data[::-1,:,:,:]
        # y reflection
    if rule[1]:
        data = data[:,::-1,:,:]
    # x reflection
    if rule[2]:
        data = data[:,:,::-1,:]
    # Transpose in xy.
    if rule[3]:
        data = data.transpose(0,2,1,3)

    return data

def revert_flip(data, rule, dst=[1,1,1]):
    """
TODO(kisuk): Documentation.
"""
    data = check_tensor(data)
    data = np.einsum('zyxd->dzyx', data)

    assert np.size(rule)==4

    # Special treat for affinity.
    is_affinity = False if dst is None else True
    if is_affinity:
        (dz,dy,dx) = dst
        assert data.shape[-4]==3
        assert dx and abs(dx) < data.shape[-1]
        assert dy and abs(dy) < data.shape[-2]
        assert dz and abs(dz) < data.shape[-3]

        # Transpose in xy.
    if rule[3]:
        data = data.transpose(0,1,3,2)
        # Swap x/y-affinity maps.
        if is_affinity:
            data[[0,1],...] = data[[1,0],...]   #fix this!!!
    # x reflection
    if rule[2]:
        data = data[:,:,:,::-1]
        # Special treatment for x-affinity.
        if is_affinity:
            if dx > 0:
                data[0,:,:,dx:] = data[0,:,:,:-dx]
                data[0,:,:,:dx].fill(0)
            else:
                dx = abs(dx)
                data[0,:,:,:-dx] = data[0,:,:,dx:]
                data[0,:,:,-dx:].fill(0)
    # y reflection
    if rule[1]:
        data = data[:,:,::-1,:]
        # Special treatment for y-affinity.
        if is_affinity:
            if dy > 0:
                data[1,:,dy:,:] = data[1,:,:-dy,:]
                data[1,:,:dy,:].fill(0)
            else:
                dy = abs(dy)
                data[1,:,:-dy,:] = data[1,:,dy:,:]
                data[1,:,-dy:,:].fill(0)
    # z reflection
    if rule[0]:
        data = data[:,::-1,:,:]
        # Special treatment for z-affinity.
        if is_affinity:
            if dz > 0:
                data[2,dz:,:,:] = data[2,:-dz,:,:]
                data[2,:dz,:,:].fill(0)
            else:
                dz = abs(dz)
                data[2,:-dz,:,:] = data[2,dz:,:,:]
                data[2,-dz:,:,:].fill(0)

    data = np.einsum('dzyx->zyxd', data)
    return data

def expand_3d_to_5d(data):
    # Add a batch dimension and a channel dimension
    data = np.expand_dims(data, axis=BATCH_AXIS)
    data = np.expand_dims(data, axis=CHANNEL_AXIS)

    return data

def cremi_calc():
    test = cremiio.CremiFile('mean_affinity_segm0.298.hdf', 'r')
    truth = cremiio.CremiFile('cremi/a/validation.hdf', 'r')

    inputs = truth.read_raw().data.value
    labels = truth.read_neuron_ids().data.value
    res = truth.read_raw().resolution
    inputs = inputs[1:, 1:, 1:]
    labels = labels[1:, 1:, 1:]
    truth_trimmed = cremiio.CremiFile('validation_trimmed.hdf', 'w')
    truth_trimmed.write_raw(cremiio.Volume(inputs, resolution = res))
    truth_trimmed.write_neuron_ids(cremiio.Volume(labels, resolution = res))
    truth_trimmed.close()

    truth_trimmed_read = cremiio.CremiFile('validation_trimmed.hdf', 'r')
    print (truth_trimmed_read.read_neuron_ids().data.value.shape)
    print(test.read_neuron_ids().data.value.shape)
    neuron_ids_evaluation = NeuronIds(truth_trimmed_read.read_neuron_ids())
    (voi_split, voi_merge) = neuron_ids_evaluation.voi(test.read_neuron_ids())
    adapted_rand = neuron_ids_evaluation.adapted_rand(test.read_neuron_ids())
    print(voi_split)
    print(voi_merge)
    print(adapted_rand)
    
def write_predictions(low=0.9, hi=0.9995):
 #   aff_file = 'mean_affinity_segm0.298.h5'
    segment_file = 'mean_affinity_segm0.298.h5'
    segment_file_hdf = 'mean_affinity_segm0.298.hdf'
    current_dir = os.path.dirname(os.path.abspath(__file__))

    o_train_file = cremiio.CremiFile('cremi/a/validation.hdf', 'r')
    o_input_volume = o_train_file.read_raw()
    raw = o_train_file.read_raw()
    inputs = raw.data.value
    o_labels_res = o_input_volume.resolution
    o_train_file.close()
    
#    subprocess.call(["julia"
#                     current_dir + "/thirdparty/watershed/watershed.jl",
#                     aff_file,
#                     segment_file,
#                     str(hi),
#                     str(low)])
    with h5py.File(segment_file, 'r') as f:
        arr = f['main'][:]
        test_file = cremiio.CremiFile(segment_file_hdf, 'w')
        test_file.write_raw(cremiio.Volume(inputs, resolution=o_labels_res))
        test_file.write_neuron_ids(cremiio.Volume(arr, resolution=o_labels_res))
        test_file.close()
    
def run_watershed_on_affinities_and_store(affinities, run_name, split, relabel2d=False, low=0.9, hi=0.9999995):
    tmp_aff_file = 'cremi/a/results/unet_3d_4layers/run-' + run_name + '/' + split + '-pred-affinities.h5'
#    tmp_aff_file ='cremi/a/results/unet_3d_4layers/run-' + run_name + '/combined_map.h5'
    label_file = 'cremi/a/results/unet_3d_4layers/run-' + run_name + '/' + split + '-predictions.h5'
#    label_file = 'validation-labels-new-gaussian-new-bounds.h5'
    final_aff_file = 'final-affinities2.h5'
    final_seg_file = 'validation-final.h5'

#    reshaped_final_aff = np.einsum('zyxd->xyzd', affinities[0])
#    shape_final = reshaped_final_aff.shape

#    with h5py.File(final_aff_file, 'w') as output_file:
#        output_file.create_dataset('main', shape =(shape_final[0],shape_final[1], shape_final[2], shape_final[3]))
#        out = output_file['main']
#        out[:, :, :, :shape_final[0]] = reshaped_final_aff
#    print(shape_final)
        
    
    # Move to the front

    #temporary comment out
#tmp    reshaped_aff = np.einsum('zyxd->dzyx', affinities)
#tmp    shape = reshaped_aff.shape
    
#tmp    with h5py.File(tmp_aff_file, 'w') as output_file:
#tmp        output_file.create_dataset('main', shape=(3, shape[1], shape[2], shape[3]))
#tmp        out = output_file['main']
#tmp        out[:shape[0], :, :, :] = reshaped_aff

    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     tmp_aff_file,
                     label_file,
                     str(hi),
                     str(low)])

#might comment out later
    pred_seg = io_utils.import_file(label_file)
    prep = utils.parse_fns(utils.prep_fns, [relabel2d, False])
    pred_seg, _ = utils.run_preprocessing(pred_seg, pred_seg, prep)
    with h5py.File('cremi/a/results/unet_3d_4layers/run-' + run_name + '/test-predictions.h5', 'w') as output_file:
        output_file.create_dataset('main', data=pred_seg)

    
#tmp    original_shape = affinities[0].shape
#tmp    with h5py.File(final_aff_file, 'w') as final_output:
#tmp        final_output.create_dataset('main', shape=(shape[3], shape[2], shape[1], 3))
#tmp        final_out = final_output['main']
#tmp        reshaped_final_aff = np.einsum('zyxd->xyzd', affinities[0])
#tmp        final_out[:,:,:, :3] = reshaped_final_aff

#tmp    with h5py.File('validation-labels.h5', 'r') as f:
#tmp        arr = f['main'][:]
#tmp       with h5py.File(final_seg_file, 'w') as final_seg_output:
#tmp            final_seg_output.create_dataset('main', shape =(shape[3], shape[2], shape[1]))
#tmp            final_seg_out = final_seg_output['main']
#tmp            reshaped_final_seg_aff = np.einsum('zyx->xyz', arr)
#tmp            final_seg_out[:,:,:] = reshaped_final_seg_aff
        
    
def run_watershed_on_affinities(affinities, relabel2d=False, low=0.9, hi=0.9995):
    tmp_aff_file = 'tmp-affinities.h5'
    tmp_label_file = 'tmp-labels.h5'

    base = './tmp2/' + str(int(round(time.time() * 1000))) + '/'

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

    shutil.rmtree('./tmp2/')

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
        elif output_type == SEGMENTATION_3D:
            return original_labels
        else:
            raise Exception('Invalid output_type')
    else:
        print(input_type)
        raise Exception('Invalid input_type')



