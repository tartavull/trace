from __future__ import print_function
from __future__ import division
import os.path
import configparser as cp
import numpy as np

import h5py

from dataprovider.data_provider import VolumeDataProvider


def mirror_across_borders(data, fov):
    mirrored_data = np.pad(data, [(0, 0), (fov//2, fov//2), (fov//2, fov//2)], mode='reflect')
    return mirrored_data


def maybe_create_affinities(dataset_prefix, num_examples):
    """
    Args:
        dataset_prefix (str): either train or test

        It should recompute them in the case that the labels changes
    """
    if os.path.exists(dataset_prefix + '-affinities.h5'):
        print('Affinties already exists.')
        return

    if not os.path.exists(dataset_prefix+'-labels.h5'):
        print('No labels from where to compute affinities.')
        return

    # there is a little bug in DataProvider that doesn't let us do it
    # for the actual size of the dataset 100,1024,1024
    patch_size = (num_examples, 1023, 1023)

    net_spec = {
        'label': patch_size
    }

    params = {
        'augment': [],
        'drange': [0]
    }

    spec = dataset_prefix + '.spec'

    config = cp.RawConfigParser()
    config.read(spec)

    config.set('files', 'img', dataset_prefix + '-input.h5')
    config.set('files', 'lbl', dataset_prefix + '-labels.h5')

    # Writing our modified configuration file
    with open(spec, 'wb') as f:
        config.write(f)

    dp = VolumeDataProvider(spec, net_spec, params)
    affinities = dp.random_sample()['label']
    with h5py.File(dataset_prefix + '-affinities.h5','w') as f:
        f.create_dataset('main', data=affinities)

