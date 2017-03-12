"""
Download and decompress SNEMI3D, ISBI, and CREMI
"""

from __future__ import print_function
import os
import os.path
import urllib
import zipfile
import h5py
import tifffile as tif
import dataprovider.transform as transform
import subprocess
import numpy as np

import cremi.io as cremiio
import cremi.evaluation as cremival

TRAIN_INPUT = 'train-input'
TRAIN_LABELS = 'train-labels'
TRAIN_AFFINITIES = 'train-affinities'
VALIDATION_INPUT = 'validation-input'
VALIDATION_LABELS = 'validation-labels'
VALIDATION_AFFINITIES = 'validation-affinities'
TEST_INPUT = 'test-input'
TEST_LABELS = 'test-labels'
TEST_AFFINITIES = 'test-affinities'

ZIP = '.zip'
TIF = '.tif'
H5 = '.h5'

SNEMI3D = 'snemi3d'
ISBI = 'isbi'
CREMI = 'cremi'
CREMI_A = 'cremi/a'
CREMI_B = 'cremi/b'
CREMI_C = 'cremi/c'


def __maybe_download(base_url, remote_filename, dest_folder, dest_filename):
    full_url = base_url + remote_filename
    full_path = dest_folder + dest_filename
    if not os.path.exists(full_path):
        print(full_path + " Does not exist")
        print("downloading " + full_url)
        urllib.urlretrieve(full_url, full_path)


def __maybe_create_hdf5_from_tif(folder, base_fn):
    full_path = folder + base_fn + H5

    if not os.path.exists(full_path):
        with tif.TiffFile(folder + base_fn + TIF) as file:
            arr = file.asarray()
            with h5py.File(folder + base_fn + H5) as f:
                print('created ' + base_fn + H5)
                f.create_dataset('main', data=arr)


def __maybe_unzip(folder, base_fn):
    full_path = folder + base_fn + ZIP

    if not os.path.exists(folder + base_fn + TIF):
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()


def __maybe_split(folder, train_fraction):
    train_input_tif = TRAIN_INPUT + TIF
    train_labels_tif = TRAIN_LABELS + TIF
    val_input_tif = VALIDATION_INPUT + TIF
    val_labels_tif = VALIDATION_LABELS + TIF

    if not os.path.exists(folder + val_input_tif):
        print(str.format('splitting {} into {}% training,  {}% into validation', train_input_tif, 100 * train_fraction,
                         100 * (1 - train_fraction)))

        # For Input and Labels
        for train_fn, val_fn in [(train_input_tif, val_input_tif), (train_labels_tif, val_labels_tif)]:
            print(folder + train_fn)
            with tif.TiffFile(folder + train_fn) as file:
                total_set = file.asarray()
                num_slices = total_set.shape[0]

                train_slices = int(num_slices * train_fraction)

                print(train_slices)
                train_set = file.asarray()[:train_slices, :, :]
                validation_set = file.asarray()[train_slices:, :, :]

                tif.imsave(folder + train_fn, train_set)
                tif.imsave(folder + val_fn, validation_set)


def __maybe_split_cremi(folder, train_fraction):
    if not os.path.exists(folder + 'validation.hdf'):
        print(str.format('splitting {} into {}% training,  {}% into validation, and adding borders',
                         folder + 'train-full.hdf', 100 * train_fraction, 100 * (1 - train_fraction)))

        # Extract the input and labels from the hdf
        o_train_file = cremiio.CremiFile(folder + 'train-full.hdf', 'r')
        o_input_volume = o_train_file.read_raw()
        o_input = o_input_volume.data.value
        o_input_res = o_input_volume.resolution

        o_labels_volume = o_train_file.read_neuron_ids()
        o_labels = o_labels_volume.data.value
        o_labels_res = o_input_volume.resolution

        # Add a boundary to original labels (with id 0), so that we can train well on affinities
        o_bounded_labels = np.zeros(o_labels.shape, dtype=np.int32)
        cremival.create_border_mask(input_data=o_labels, target=o_bounded_labels, max_dist=2, background_label=0)

        o_train_file.close()

        # Split
        num_slices = o_input.shape[0]
        train_slices = int(num_slices * train_fraction)

        train_input = o_input[:train_slices, :, :]
        validation_input = o_input[train_slices:, :, :]

        train_labels = o_bounded_labels[:train_slices, :, :]
        validation_labels = o_bounded_labels[train_slices:, :, :]

        train_file = cremiio.CremiFile(folder + 'train.hdf', 'w')
        train_file.write_raw(cremiio.Volume(train_input, resolution=o_input_res))
        train_file.write_neuron_ids(cremiio.Volume(train_labels, resolution=o_labels_res))
        train_file.close()

        validation_file = cremiio.CremiFile(folder + 'validation.hdf', 'w')
        validation_file.write_raw(cremiio.Volume(validation_input, resolution=o_input_res))
        validation_file.write_neuron_ids(cremiio.Volume(validation_labels, resolution=o_labels_res))
        validation_file.close()


def __maybe_create_isbi(dest_folder, train_frac):
    base_url = 'http://brainiac2.mit.edu/isbi_challenge/sites/default/files/'

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    # Since we rename from *-volume to *-input, we check to see if we've performed the rename already
    __maybe_download(base_url, 'train-volume.tif', dest_folder, TRAIN_INPUT + TIF)
    __maybe_download(base_url, 'train-labels.tif', dest_folder, TRAIN_LABELS + TIF)
    __maybe_download(base_url, 'test-volume.tif', dest_folder, TEST_INPUT + TIF)

    # Split into train and validation sets
    __maybe_split(dest_folder, train_frac)

    # Convert each dataset into h5 files, for viewing and use with DataProvider/Neuroglancer
    __maybe_create_hdf5_from_tif(dest_folder, TRAIN_INPUT)
    __maybe_create_hdf5_from_tif(dest_folder, TRAIN_LABELS)
    __maybe_create_hdf5_from_tif(dest_folder, VALIDATION_INPUT)
    __maybe_create_hdf5_from_tif(dest_folder, VALIDATION_LABELS)
    __maybe_create_hdf5_from_tif(dest_folder, TEST_INPUT)

    # For isbi, if we want to load the dataset as affinities, we need to generate them in the first place
    for labels_fn, affinities_fn in [(TRAIN_LABELS + H5, TRAIN_AFFINITIES + H5),
                                     (VALIDATION_LABELS + H5, VALIDATION_AFFINITIES + H5)]:
        with h5py.File(dest_folder + labels_fn, 'r') as file:
            data = file['main'][:]
            aff = transform.affinitize(data)
            aff[2] = 0

        with h5py.File(dest_folder + affinities_fn, 'w') as file:
            file.create_dataset('main', data=aff)

        # Run watershed to recreate affinities
        current_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.call(["julia", current_dir + "/thirdparty/watershed/watershed.jl",
                         dest_folder + affinities_fn,
                         dest_folder + labels_fn,
                         str(0.9), str(0.3)])


def __maybe_create_snemi3d(dest_folder, train_frac):
    base_url = 'http://brainiac2.mit.edu/SNEMI3D/sites/default/files/'

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    __maybe_download(base_url, TRAIN_INPUT + ZIP, dest_folder, TRAIN_INPUT + ZIP)
    __maybe_download(base_url, TRAIN_LABELS + ZIP, dest_folder, TRAIN_LABELS + ZIP)
    __maybe_download(base_url, TEST_INPUT + ZIP, dest_folder, TEST_INPUT + ZIP)

    __maybe_unzip(dest_folder, TRAIN_INPUT)
    __maybe_unzip(dest_folder, TRAIN_LABELS)
    __maybe_unzip(dest_folder, TEST_INPUT)

    __maybe_split(dest_folder, train_frac)

    __maybe_create_hdf5_from_tif(dest_folder, TRAIN_INPUT)
    __maybe_create_hdf5_from_tif(dest_folder, TRAIN_LABELS)
    __maybe_create_hdf5_from_tif(dest_folder, VALIDATION_INPUT)
    __maybe_create_hdf5_from_tif(dest_folder, VALIDATION_LABELS)
    __maybe_create_hdf5_from_tif(dest_folder, TEST_INPUT)


def __maybe_create_cremi(dest_folder, train_frac):
    base_url = 'https://cremi.org/static/data/'

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    # For now, only download the un-padded versions
    a_train_fn = 'sample_A_20160501.hdf'
    a_test_fn = 'sample_A%2B_20160601.hdf'

    b_train_fn = 'sample_B_20160501.hdf'
    b_test_fn = 'sample_B%2B_20160601.hdf'

    c_train_fn = 'sample_C_20160501.hdf'
    c_test_fn = 'sample_C%2B_20160601.hdf'

    __maybe_download(base_url, a_train_fn, dest_folder + 'a/', 'train-full.hdf')
    __maybe_download(base_url, a_test_fn, dest_folder + 'a/', 'test.hdf')

    __maybe_download(base_url, b_train_fn, dest_folder + 'b/', 'train-full.hdf')
    __maybe_download(base_url, b_test_fn, dest_folder + 'b/', 'test.hdf')

    __maybe_download(base_url, c_train_fn, dest_folder + 'c/', 'train-full.hdf')
    __maybe_download(base_url, c_test_fn, dest_folder + 'c/', 'test.hdf')

    __maybe_split_cremi(dest_folder + 'a/', train_frac)
    __maybe_split_cremi(dest_folder + 'b/', train_frac)
    __maybe_split_cremi(dest_folder + 'c/', train_frac)




def maybe_create_all_datasets(trace_folder, train_frac):
    __maybe_create_snemi3d(trace_folder + SNEMI3D + '/', train_frac)
    __maybe_create_isbi(trace_folder + ISBI + '/', train_frac)
    __maybe_create_cremi(trace_folder + CREMI + '/', 0.8)


if __name__ == '__main__':
    current_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    maybe_create_all_datasets(current_folder, 0.9)
