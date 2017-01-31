import tensorflow as tf
import tifffile as tif
import numpy as np

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

        # The distortion causes weird things at the boundaries, so we pad our sample and crop to get the desired fov
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

        cropped_image = image_sample[crop_padding//2:fov + crop_padding//2, crop_padding//2:fov + crop_padding//2]
        cropped_labels = el_labels[crop_padding // 2:fov + crop_padding // 2, crop_padding // 2:fov + crop_padding // 2]

        return cropped_image, cropped_labels
