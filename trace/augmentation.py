from __future__ import print_function
from __future__ import division
import os.path
import configparser as cp
import numpy as np

import h5py
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from dataprovider.data_provider import VolumeDataProvider


# Taken from: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, labels, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    assert(len(image.shape) == 3)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[0:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')


    # Missing section augmentation
    # Data blurring
    # Misalignment (learning linear transformation)
    
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    # print(image.shape)
    # def mapper(t):
    #     print(t.shape)
    #     return map_coordinates(t, indices, order=1).reshape(shape)
    #
    # el_image = np.apply_along_axis(mapper, axis=2, arr=image)
    # el_label = np.apply_along_axis(mapper, axis=2, arr=labels)

    el_image = np.zeros(shape=image.shape)
    for i in range(image.shape[2]):
        el_image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape)

    el_label = np.zeros(shape=labels.shape)
    for i in range(labels.shape[2]):
        el_label[:, :, i] = map_coordinates(labels[:, :, i], indices, order=1).reshape(shape)

    return el_image, el_label


def mirror_across_borders(data, fov):
    mirrored_data = np.pad(data, [(0, 0), (fov // 2, fov // 2), (fov // 2, fov // 2), (0, 0)], mode='reflect')
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

    if not os.path.exists(dataset_prefix + '-labels.h5'):
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
    with h5py.File(dataset_prefix + '-affinities.h5', 'w') as f:
        f.create_dataset('main', data=affinities)
