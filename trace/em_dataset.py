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
    def __init__(self, data_folder, output_mode='boundaries', suffix=''):
       # with tf.device('/cpu:0'):
        self.data_folder = data_folder
        self.output_mode = output_mode

        # Read in the datasets
        train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + suffix + down.TIF)
        train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + suffix + down.TIF)

        validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + suffix + down.TIF)
        validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + suffix + down.TIF)

        test_inputs = tiff.imread(data_folder + down.TEST_INPUT + suffix + down.TIF)

        # All inputs have one channel
        train_inputs = np.expand_dims(train_inputs, 3)
        self.validation_inputs = np.expand_dims(validation_inputs, 3)
        self.test_inputs = np.expand_dims(test_inputs, 3)

        # Transform the labels based on the mode we are using
        if output_mode == BOUNDARIES_MODE:
            # Expand dimensions from [None, None, None] -> [None, None, None, 1]
            train_labels = np.expand_dims(train_labels, 3) // 255.0
            self.validation_labels = np.expand_dims(validation_labels, 3) // 255.0

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
        self.crop_padding = 40
        self.patch_size_placeholder = tf.placeholder(dtype=tf.int32, shape=(), name='FOV')
        self.patch_size = tf.Variable(self.patch_size_placeholder, name='patch_size') + self.crop_padding

        # Create dataset, and pad the dataset with mirroring
        dataset = tf.constant(train_stacked, dtype=tf.float32)

        pad = tf.floordiv(self.patch_size, 2)
        padded_dataset = tf.pad(dataset, [[0, 0], tf.stack([pad, pad]), tf.stack([pad, pad]), [0, 0]], mode="REFLECT")

        # Sample and squeeze the dataset, squeezing so that we can perform the distortions
        sample = tf.random_crop(padded_dataset, size=[1, self.patch_size, self.patch_size, train_stacked.shape[3]])
        squeezed_sample = tf.squeeze(sample)

        # Perform the first transformation
        distorted_sample = tf.image.random_flip_left_right(squeezed_sample)
        self.distorted_sample = tf.image.random_flip_up_down(distorted_sample)

        # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute

        # Independently, feed in warped image
        #self.elastically_deformed_image = tf.placeholder(np.float64, shape=[None, None, 1], name="elas_deform_input")
        self.elastically_deformed_image = self.distorted_sample

        # self.standardized_image = tf.image.per_image_standardization(self.elastically_deformed_image)

        distorted_image = tf.image.random_brightness(self.elastically_deformed_image, max_delta=0.15)
        self.distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)

        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())

    def get_validation_set(self):
        return self.validation_inputs, self.validation_labels

    def get_test_set(self):
        return self.test_inputs

    def generate_random_samples(self, model):
        # Create op for generation and enqueueing of random samples.

        # The distortion causes weird things at the boundaries, so we pad our sample and crop to get desired patch size


        #sigma = np.random.randint(low=35, high=100)

        # Apply elastic deformation

        # TODO(beisner): Move affinitization after elastic deformation, or think about it...
        #el_image, el_labels = aug.elastic_transform(separated_image, separated_labels, alpha=2000, sigma=sigma)

        #with tf.device('/cpu:0'):
        crop_padding = self.crop_padding
        cropped_image = self.distorted_image[crop_padding // 2:-crop_padding // 2,
                crop_padding // 2:-crop_padding // 2, :1]
        cropped_labels = self.elastically_deformed_image[crop_padding // 2:-crop_padding // 2,
                crop_padding // 2:-crop_padding // 2, 1:]

        training_example = tf.concat(3, [tf.expand_dims(cropped_image, 0), tf.expand_dims(cropped_labels, 0)])

        enqueue_op = model.queue.enqueue(training_example)

        return enqueue_op
