from __future__ import print_function
from __future__ import division
import os.path
import configparser as cp
import numpy as np
import tensorflow as tf

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


def mirror_across_borders_3d(data, fov, z_fov):
    half = fov // 2
    z_half = z_fov // 2
    return np.pad(data, [(0, 0), (z_half, z_half), (half, half), (half, half), (0, 0)], mode='reflect')


# Image is a 3D tensor.
#
# Sigma is the standard deviation in pixels - that is, the distance from the
# center to reach one standard deviation above the mean.
#
# Size is the length of one side of the gaussian filter. Assuming size is odd
def tf_gaussian_blur(image, sigma, size=5):
    padding = tf.cast(size // 2, tf.float32)
    # Create grid of points to evaluate gaussian function at.
    indices = tf.linspace(-padding, padding, size)
    X, Y = tf.meshgrid(indices, indices)
    padding = tf.cast(padding, tf.int32)

    # Create gaussian filter, of size [size, size]
    g_filter = tf.exp(-tf.cast(X * X + Y * Y, tf.float32)/(2 * sigma * sigma))

    # Normalize to 1 over truncated filter
    normalized_gaussian_filter = g_filter / tf.reduce_sum(g_filter)

    # Expand/tile the filter to shape [size, size, in_channels, out_channels], required for tf.convolution
    num_channels = image.get_shape()[-1]
    blur_filter = tf.expand_dims(tf.expand_dims(normalized_gaussian_filter, axis=2), axis=3)
    blur_filter = tf.tile(blur_filter, tf.stack([1, 1, num_channels, num_channels]))

    # Reflect image at borders to create padding for the filter.
    padding = tf.cast(padding, tf.int32)
    mirrored_image = tf.pad(image, tf.stack([[padding, padding], [padding, padding], [0, 0]]), 'REFLECT')

    # Expand the tensor from [x, y, chan] -> [batch, x, y, chan], because tf.convolution requires
    mirrored_image = tf.expand_dims(mirrored_image, axis=0)

    # Apply the gaussian filter.
    filtered_image = tf.nn.convolution(mirrored_image, blur_filter, strides=[1, 1], padding='VALID')

    # Reduce dimensions of the output image, to put it back in 3D
    squeezed_image = tf.squeeze(filtered_image, axis=0)

    return squeezed_image
