"""
Download and decompress SNEMI3D
"""

from __future__ import print_function
import os.path
import urllib
import zipfile
import h5py
import tifffile as tif
import dataprovider.transform as transform
import subprocess


def snemi3d_config():
    return DatasetConfig(
        dataset='snemi3d',
        base_url='http://brainiac2.mit.edu/SNEMI3D/sites/default/files/',
        train_input_base='train-input',
        validation_input_base='validation-input',
        train_labels_base='train-labels',
        validation_labels_base='validation-labels',
        test_input_base='test-input',
        needs_unzipping=True,
        needs_affinitizing=False
    )


def isbi_config():
    return DatasetConfig(
        dataset='isbi',
        base_url='http://brainiac2.mit.edu/SNEMI3D/sites/default/files/',
        train_input_base='train-volume',
        validation_input_base='validation-volume',
        train_labels_base='train-labels',
        validation_labels_base='validation-labels',
        test_input_base='test-volume',
        needs_unzipping=False,
        needs_affinitizing=True
    )


class DatasetConfig:
    def __init__(self, dataset, base_url, train_input_base, validation_input_base, train_labels_base,
                 validation_labels_base, test_input_base, needs_unzipping, needs_affinitizing):
        self.dataset = dataset
        self.base_url = base_url

        self.train_input_zip = train_input_base + '.zip'
        self.train_input_tif = train_input_base + '.tif'
        self.train_input_h5 = train_input_base + '.h5'

        self.validation_input_zip = validation_input_base + '.zip'
        self.validation_input_tif = validation_input_base + '.tif'
        self.validation_input_h5 = validation_input_base + '.h5'

        self.train_labels_zip = train_labels_base + '.zip'
        self.train_labels_tif = train_labels_base + '.tif'
        self.train_labels_h5 = train_labels_base + '.h5'

        self.validation_labels_zip = validation_labels_base + '.zip'
        self.validation_labels_tif = validation_labels_base + '.tif'
        self.validation_labels_h5 = validation_labels_base + '.h5'

        self.test_input_zip = test_input_base + '.zip'
        self.test_input_tif = test_input_base + '.tif'
        self.test_input_h5 = test_input_base + '.h5'

        self.folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset + '/'
        self.needs_unzipping = needs_unzipping


def maybe_download(base_url, folder, filename):
  full_url = base_url + filename
  full_path = folder + filename
  if not os.path.exists(full_path):
    print("downloading "+full_url)
    urllib.urlretrieve(full_url, full_path)


def maybe_create_hdf5(folder, filename):
  full_path = folder + filename
  path_without_ext , ext = os.path.splitext(full_path)

  if not os.path.exists(path_without_ext+'.h5'):
    with tif.TiffFile(path_without_ext+'.tif') as file:
      arr = file.asarray()
      with h5py.File(path_without_ext+'.h5') as f:
        print('created '+path_without_ext+'.h5')
        f.create_dataset('main', data=arr)


def maybe_unzip(folder, filename):
    full_path = folder + filename
    path_without_ext , ext = os.path.splitext(full_path)

    if not os.path.exists(path_without_ext+'.tif'):
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()


def maybe_split(folder, training_fn, validation_fn, label_fn, label_validation_fn, train_fraction):
    if not os.path.exists(folder + validation_fn):
        print(str.format('splitting {} into {}% training,  {}% into validation', training_fn, 100*train_fraction,
                         100*(1-train_fraction)))
        with tif.TiffFile(folder + training_fn) as file:
            total_set = file.asarray()
            num_slices = total_set.shape[0]

            train_slices = int(num_slices * train_fraction)

            train_set = file.asarray()[:train_slices, :, :]
            validation_set = file.asarray()[train_slices:, :, :]

            tif.imsave(folder + training_fn, train_set)
            tif.imsave(folder + validation_fn, validation_set)

        with tif.TiffFile(folder + label_fn) as file:

            total_set = file.asarray()

            num_slices = total_set.shape[0]

            train_slices = int(num_slices * train_fraction)

            train_label_set = file.asarray()[:train_slices, :, :]
            validation_label_set = file.asarray()[train_slices:, :, :]

            tif.imsave(folder + label_fn, train_label_set)
            tif.imsave(folder + label_validation_fn, validation_label_set)


def maybe_create_dataset(config, train_frac):
    if not os.path.exists(config.folder):
        os.mkdir(config.folder)

    # Download the dataset
    if config.needs_unzipping:
        maybe_download(config.base_url, config.folder, config.train_input_zip)
        maybe_download(config.base_url, config.folder, config.train_labels_zip)
        maybe_download(config.base_url, config.folder, config.test_input_zip)

        # Unzip the dataset
        maybe_unzip(config.folder, config.train_input_zip)
        maybe_unzip(config.folder, config.train_labels_zip)
        maybe_unzip(config.folder, config.test_input_zip)
    else:
        maybe_download(config.base_url, config.folder, config.train_input_tif)
        maybe_download(config.base_url, config.folder, config.train_labels_tif)
        maybe_download(config.base_url, config.folder, config.test_input_tif)

    # Split the training dataset into train and validation
    maybe_split(config.folder, config.train_input_tif, config.validation_input_tif, config.train_labels_tif,
                config.validation_labels_tif, train_frac)

    # Create hdf files
    maybe_create_hdf5(config.folder, config.train_input_tif)
    maybe_create_hdf5(config.folder, config.validation_input_tif)
    maybe_create_hdf5(config.folder, config.train_labels_tif)
    maybe_create_hdf5(config.folder, config.validation_labels_tif)
    maybe_create_hdf5(config.folder, config.test_input_tif)


# Ugly hack for transforming into proper dataset
def affinitize_isbi(folder, train_labels_h5, validation_labels_h5):
    train_aff = 'train-affinities.h5'
    validation_aff = 'validation-affinities.h5'
    # Training set
    with h5py.File(folder + train_labels_h5, 'r') as file:
        data = file['main'][:]
        aff = transform.affinitize(data)

    with h5py.File(folder + train_aff, 'w') as file:
        file.create_dataset('main', data=aff)

    # Validation set
    with h5py.File(folder + validation_labels_h5, 'r') as file:
        data = file['main'][:]
        aff = transform.affinitize(data)

    with h5py.File(folder + validation_aff, 'w') as file:
        file.create_dataset('main', data=aff)


    # Run julia
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     folder + train_aff,
                     folder + "train-labels.h5",
                     str(0.9),
                     str(0.3)])

    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     folder + validation_aff,
                     folder + "validation-labels.h5",
                     str(0.9),
                     str(0.3)])


def maybe_create_all_datasets(train_frac):
    # snemi3d
    maybe_create_dataset(snemi3d_config(), train_frac)

    # isbi
    isbi_conf = isbi_config()
    maybe_create_dataset(isbi_conf, train_frac)
    affinitize_isbi(isbi_conf.folder, isbi_conf.train_labels_h5, isbi_conf.validation_labels_h5)



if __name__ == '__main__':
    maybe_create_all_datasets(0.9)
