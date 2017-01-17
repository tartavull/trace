import tensorflow as tf
import tifffile as tif
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from numpy.random import random_integers
from scipy.signal import convolve2d

import math


def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma

    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float

    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = np.zeros((dim, dim), dtype=np.float16)

    # calculate the center point
    center = dim / 2

    # calculate the variance
    variance = sigma ** 2

    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val ** 2 + y_val ** 2
            denom = 2 * variance

            kernel[x, y] = coeff * np.exp(-1. * numerator / denom)

    # normalise it
    return kernel / sum(sum(kernel))


def elastic_transform_2(image, label, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """
    This method performs elastic transformations on an image by convolving
    with a gaussian kernel.
    NOTE: Image dimensions should be a sqaure image

    :param image: the input image
    :type image: a numpy nd array
    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int
    :param sigma: standard deviation of the kernel
    :type sigma: float
    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float
    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean
    :returns: a nd array transformed image
    """

    # convert the image to single channel if it is multi channel one

    # check if the image is a negated one

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")

    # check if kernel dimesnion is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create random displacement fields
    displacement_field_x = np.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                                     for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = np.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                                     for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields

    def warp(im):

        # create an empty image
        result = np.zeros(image.shape)

        for row in xrange(im.shape[1]):
            for col in xrange(im.shape[0]):
                low_ii = row + int(math.floor(displacement_field_x[row, col]))
                high_ii = row + int(math.ceil(displacement_field_x[row, col]))

                low_jj = col + int(math.floor(displacement_field_y[row, col]))
                high_jj = col + int(math.ceil(displacement_field_y[row, col]))

                if low_ii < 0 or low_jj < 0 or high_ii >= im.shape[1] - 1 \
                    or high_jj >= im.shape[0] - 1:
                    continue

                res = im[low_ii, low_jj] / 4 + im[low_ii, high_jj] / 4 + \
                      im[high_ii, low_jj] / 4 + im[high_ii, high_jj] / 4

                result[row, col] = res

        return result

    r1 = warp(image)
    r2 = warp(label)

    return r1, r2


# Taken from: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, labels, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    print(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    el_image = map_coordinates(image, indices, order=1).reshape(shape)
    el_label = map_coordinates(labels, indices, order=1).reshape(shape)

    return el_image, el_label


class Sampler(object):
    def __init__(self, input_fn, labels_fn, affinitize=False):
        # Read the files in
        inputs = tif.imread(input_fn)
        labels = tif.imread(labels_fn)

        # Stack the two datasets together, so we can sample effectively
        dset = np.stack((inputs, labels), axis=3)

        # Define inputs to the graph
        self.fov = tf.placeholder(tf.int32, name="FOV")
        self.n_channels = tf.placeholder(tf.int32, name="n_channels")
        self.n_images = tf.placeholder(tf.int32, name="n_images")

        # Create dataset, and pad the dataset with mirroring
        dataset = tf.constant(dset)
        pad = tf.floordiv(self.fov, 2)
        padded_dataset = tf.pad(dataset, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")

        # Sample and squeeze the dataset, squeezing so that we can perform the distortions
        sample = tf.random_crop(padded_dataset, size=[self.n_images, self.fov, self.fov, self.n_channels])
        squeezed_sample = tf.squeeze(sample)

        # Perform the first transformation
        self.distorted_sample = tf.image.random_flip_left_right(squeezed_sample)
        self.distorted_sample = tf.image.random_flip_up_down(self.distorted_sample)

        # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute

        # Independently, feed in warped image
        self.elastically_deformed_image = tf.placeholder(np.float64, shape=[None, None, 1], name="elas_deform_input")

        distorted_image = tf.image.random_brightness(self.elastically_deformed_image, max_delta=0.15)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)

        self.standardized_image = tf.image.per_image_standardization(distorted_image)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_image(self, fov, n_channels, n_images):
        # Generate a distorted sample

        crop_padding = 20

        adjusted_fov = fov + crop_padding
        intermediate = self.sess.run(self.distorted_sample, feed_dict={
            self.fov: adjusted_fov,
            self.n_channels: n_channels,
            self.n_images: n_images,
        })

        separated_image = np.squeeze(intermediate[:, :, 0:1])
        separated_labels = np.squeeze(intermediate[:, :, 1:])

        sigma = np.random.randint(low=35, high=100)

        # Apply elastic deformation
        el_image, el_labels = elastic_transform(separated_image, separated_labels, alpha=2000, sigma=sigma)
        # el_image, el_labels = elastic_transform_2(separated_image, separated_labels, kernel_dim=25, alpha=8, sigma=35)

        el_image = np.expand_dims(el_image, axis=2)
        el_labels = np.expand_dims(el_labels, axis=2)

        image_sample = self.sess.run(self.standardized_image, feed_dict={
            self.elastically_deformed_image: el_image
        })

        cropped_image = el_image[crop_padding//2:fov + crop_padding//2, crop_padding//2:fov + crop_padding//2]

        return image_sample, el_image, el_labels, separated_image, separated_labels, cropped_image
