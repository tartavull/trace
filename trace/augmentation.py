from __future__ import print_function
from __future__ import division
from ConfigParser import RawConfigParser
import os.path

import h5py
import numpy as np

from dataprovider.data_provider import VolumeDataProvider


def set_path_to_config(dataset_prefix):
    config = RawConfigParser()
    configpath = dataset_prefix + '.spec'
    config.read(configpath)
    config.set('files','img', dataset_prefix + '-input.h5')
    config.set('files','lbl', dataset_prefix + '-labels.h5')
    # Writing our modified configuration file
    with open(configpath, 'wb') as f:
        config.write(f)


def maybe_create_affinities(dataset_prefix):
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
    patch_size = (99, 1023, 1023)

    net_spec = {
        'label': patch_size
    }

    params = {
        'augment': [],
        'drange': [0]
    }

    set_path_to_config(dataset_prefix)
    spec = dataset_prefix + '.spec'
    dp = VolumeDataProvider(spec, net_spec, params)
    affinities = dp.random_sample()['label']
    with h5py.File(dataset_prefix + '-affinities.h5','w') as f:
        f.create_dataset('main', data=affinities)


def batch_iterator(config, fov, output_patch, input_patch):
    split = 'train'
    dataset_prefix = config.folder + split
    set_path_to_config(dataset_prefix)

    spec = dataset_prefix + '.spec'

    print(spec)

    net_spec = {
        'label': (1, input_patch, input_patch),
        'input': (1, input_patch, input_patch)
    }

    params = {
        'augment': [],
        'drange': [0]
    }

    dp = VolumeDataProvider(spec, net_spec, params)

    while True:
        sample = dp.random_sample()
        inpt, label = sample['input'], sample['label']
        inpt = inpt.reshape(1,input_patch,input_patch,1)
        label = label[0:2,0,fov//2:fov//2+output_patch,fov//2:fov//2+output_patch]
        reshapedLabel = np.zeros(shape=(1, output_patch, output_patch, 2))

        #central output patch, only x,y affinities
        reshapedLabel[0,:,:,0] = label[0]
        reshapedLabel[0,:,:,1] = label[1]
        yield inpt, reshapedLabel

def batch_iterator_bn(config, fov, output_patch, input_patch, batch_size = 10):
    split = 'train'
    dataset_prefix = config.folder + split
    set_path_to_config(dataset_prefix)

    spec = dataset_prefix + '.spec'

    print(spec)

    net_spec = {
        'label': (1, input_patch, input_patch),
        'input': (1, input_patch, input_patch)
    }

    params = {
        'augment': [],
        'drange': [0]
    }

    dp = VolumeDataProvider(spec, net_spec, params)

    while True:
        input_batch = np.zeros([batch_size, input_patch, input_patch,1])
        label_batch = np.zeros([batch_size, output_patch, output_patch, 2])
        ## TODO ask ignacio how to optomize this/if there is a built in
        ## batch operator in dataprovider
        for i in range(batch_size):
            sample = dp.random_sample()
            inpt, label = sample['input'], sample['label']
            inpt = inpt.reshape(1,input_patch,input_patch,1)
            label = label[0:2,0,fov//2:fov//2+output_patch,fov//2:fov//2+output_patch]
            reshapedLabel = np.zeros(shape=(1, output_patch, output_patch, 2))

            #central output patch, only x,y affinities
            reshapedLabel[0,:,:,0] = label[0]
            reshapedLabel[0,:,:,1] = label[1]
            input_batch[i,:,:,:] = inpt
            label_batch[i,:,:,:] = reshapedLabel

        yield input_batch, label_batch

if __name__ == '__main__':
    batch_iterator(1).next()
