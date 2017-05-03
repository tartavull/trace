from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.stats as st

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


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


def gkern(kernlen=21, nsig=3.0):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
