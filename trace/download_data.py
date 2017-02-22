"""
Download and decompress SNEMI3D and ISBI
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
from libtiff import TIFF
from cremi.io import CremiFile

TRAIN_INPUT = 'train-input'
TRAIN_LABELS = 'train-labels'
TRAIN_AFFINITIES = 'train-affinities'
VALIDATION_INPUT = 'validation-input'
VALIDATION_LABELS = 'validation-labels'
VALIDATION_AFFINITIES = 'validation-affinities'
TEST_INPUT = 'test-input'
TEST_LABELS = 'test-labels'
TEST_AFFINITIES = 'test-affinities'

TRAIN_A = 'training_A'
TRAIN_B = 'training_B'
TRAIN_C = 'training_C'
TEST_A = 'test_A'
TEST_B = 'test_B'
TEST_C = 'test_C'

ZIP = '.zip'
TIF = '.tif'
TIFF = '.tiff'
H5 = '.h5'
HDF = '.hdf'

SNEMI3D = 'snemi3d'
ISBI = 'isbi'
CREMI = 'cremi'

DATASET_NAMES = [SNEMI3D, ISBI, CREMI]

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
            with h5py.File(full_path) as f:
                print('created ' + base_fn + H5)
                f.create_dataset('main', data=arr)

def __maybe_create_tif_from_hdf5(folder, base_fn):
    full_path = folder + base_fn + TIFF

    if not os.path.exists(full_path):
        print(folder + base_fn + HDF)
        file = CremiFile(folder + base_fn + HDF, 'r')
        arr = file.read_raw().data
        with TIFF.open(full_path) as f:
            print('created ' + base_fn + TIF)
            f.write_image(arr)

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

                train_set = file.asarray()[:train_slices, :, :]
                validation_set = file.asarray()[train_slices:, :, :]

                tif.imsave(folder + train_fn, train_set)
                tif.imsave(folder + val_fn, validation_set)


def __maybe_create_isbi(dest_folder, train_frac):
    base_url = 'http://brainiac2.mit.edu/isbi_challenge/sites/default/files/'

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
    train_A_sc = 'sample_A_20160501.hdf'
    train_B_sc = 'sample_B_20160501.hdf'
    train_C_sc = 'sample_C_20160501.hdf'
    test_A_sc = 'sample_A%2B_20160601.hdf'
    test_B_sc = 'sample_B%2B_20160601.hdf'
    test_C_sc = 'sample_C%2B_20160601.hdf'

    __maybe_download(base_url, train_A_sc, dest_folder, TRAIN_A + HDF)
    __maybe_download(base_url, train_B_sc, dest_folder, TRAIN_B + HDF)
    __maybe_download(base_url, train_C_sc, dest_folder, TRAIN_C + HDF)
    __maybe_download(base_url, test_A_sc, dest_folder, TEST_A + HDF)
    __maybe_download(base_url, test_B_sc, dest_folder, TEST_B + HDF)
    __maybe_download(base_url, test_C_sc, dest_folder, TEST_C + HDF)

    __maybe_create_tif_from_hdf5(dest_folder, TRAIN_A)

def maybe_create_all_datasets(trace_folder, train_frac):
    __maybe_create_snemi3d(trace_folder + SNEMI3D + '/', train_frac)
    __maybe_create_isbi(trace_folder + ISBI + '/', train_frac)
    __maybe_create_cremi(trace_folder + CREMI + '/', train_frac)


if __name__ == '__main__':
    current_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    maybe_create_all_datasets(current_folder, 0.9)
