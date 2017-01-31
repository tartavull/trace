import tifffile as tiff
import tensorflow as tf
import numpy as np

import download_data as down

import dataprovider.transform as trans
import augmentation as aug

BOUNDARIES_MODE = 'boundaries'
AFFINITIES_2D_MODE = 'affinities-2d'
AFFINITIES_3D_MODE = 'affinities-3d'


class EMDataset(object):
    def __init__(self, data_folder, output_mode='boundaries'):
        self.data_folder = data_folder
        self.output_mode = output_mode

        # Read in the datasets
        train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
        train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF)

        validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
        validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF)

        test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)

        # All inputs have one channel
        train_inputs = np.expand_dims(train_inputs, 3)
        self.validation_inputs = np.expand_dims(validation_inputs, 3)
        self.test_inputs = np.expand_dims(test_inputs, 3)

        # Transform the labels based on the mode we are using
        if output_mode == BOUNDARIES_MODE:
            # Expand dimensions from [None, None, None] -> [None, None, None, 1]
            train_labels = np.expand_dims(train_labels, 3)
            self.validation_labels = np.expand_dims(validation_labels, 3)

        elif output_mode == AFFINITIES_2D_MODE:
            # Affinitize in 2 dimensions

            def aff_and_reshape_2d(dset):

                # Affinitize
                aff_dset = trans.affinitize(dset)

                # Reshape [3, None, None, None] -> [None, None, None, 3]
                rearranged = np.einsum('abcd->bcda', aff_dset)

                # Remove the third dimension
                return rearranged[:, :, :, 0:2]

            train_labels = aff_and_reshape_2d(train_labels)
            self.validation_labels = aff_and_reshape_2d(validation_labels)

        elif output_mode == AFFINITIES_3D_MODE:
            # Affinitize in 3 dimensions

            def aff_and_reshape_3d(dset):

                # Affinitize
                aff_dset = trans.affinitize(dset)

                # Reshape [3, None, None, None] -> [None, None, None, 3]
                return np.einsum('abcd->bcda', aff_dset)

            train_labels = aff_and_reshape_3d(train_labels)
            self.validation_labels = aff_and_reshape_3d(validation_labels)

        else:
            raise Exception("Invalid output_mode!")

        # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs
        train_stacked = np.concatenate((train_inputs, train_labels), axis=3)

        # Define inputs to the graph
        self.patch_size = tf.placeholder(tf.int32, name="FOV")

        # Create dataset, and pad the dataset with mirroring
        dataset = tf.constant(train_stacked)
        pad = tf.floordiv(self.patch_size, 2)
        padded_dataset = tf.pad(dataset, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")

        # Sample and squeeze the dataset, squeezing so that we can perform the distortions
        sample = tf.random_crop(padded_dataset, size=[1, self.patch_size, self.patch_size, train_stacked.shape[3]])
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

    def get_validation_set(self):
        return self.validation_inputs, self.validation_labels

    def get_test_set(self):
        return self.test_inputs

    def random_sample(self, patch_size):
        # Generate a distorted sample

        # The distortion causes weird things at the boundaries, so we pad our sample and crop to get desired patch size
        crop_padding = 20

        adjusted_patch_size = patch_size + crop_padding
        intermediate = self.sess.run(self.distorted_sample, feed_dict={
            self.patch_size: adjusted_patch_size,
        })

        separated_image = intermediate[:, :, 0:1]
        separated_labels = intermediate[:, :, 1:]

        sigma = np.random.randint(low=35, high=100)

        # Apply elastic deformation

        # TODO(beisner): Move affinitization after elastic deformation, or think about it...
        el_image, el_labels = aug.elastic_transform(separated_image, separated_labels, alpha=2000, sigma=sigma)

        im_sample = self.sess.run(self.standardized_image, feed_dict={
            self.elastically_deformed_image: el_image
        })

        cropped_image = im_sample[crop_padding // 2:patch_size + crop_padding // 2,
                        crop_padding // 2:patch_size + crop_padding // 2]
        cropped_labels = el_labels[crop_padding // 2:patch_size + crop_padding // 2,
                         crop_padding // 2:patch_size + crop_padding // 2]

        return np.expand_dims(cropped_image, axis=0), np.expand_dims(cropped_labels, axis=0)