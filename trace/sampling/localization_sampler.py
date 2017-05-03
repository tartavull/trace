import math

import tensorflow as tf
import tensorflow.contrib.image as tfim

from trace.utils import cond_apply, expand_3d_to_5d


class SimpleEMInputSampler(object):
    def __init__(self, dataset, sample_shape, batch_size):
        self.__train_inputs = expand_3d_to_5d(dataset.train_inputs)
        self.__validation_inputs = expand_3d_to_5d(dataset.validation_inputs)
        self.__test_inputs = expand_3d_to_5d(dataset.test_inputs)

        with tf.device('/cpu:0'):
            # The dataset is loaded into a constant variable from a placeholder
            # because a tf.constant cannot hold a dataset that is over 2GB.
            self.__image_ph = tf.placeholder(dtype=tf.float32, shape=self.__train_inputs.shape)
            self.__dataset_constant = tf.Variable(self.__image_ph, trainable=False, collections=[])

            samples = []
            for i in range(batch_size):
                # Crop a 5d section
                samples.append(tf.random_crop(self.__dataset_constant, size=sample_shape))

            # samples will now have shape [batch_size, z_patch_size, patch_size, patch_size, 1]
            self.__samples = tf.squeeze(samples, axis=1)

    def initialize_session_variables(self, sess):
        sess.run(self.__dataset_constant.initializer, feed_dict={self.__image_ph: self.__train_inputs})

    def get_sample_op(self):
        return self.__samples

    def get_full_training_set(self):
        return self.__train_inputs

    def get_validation_set(self):
        return self.__validation_inputs

    def get_test_set(self):
        return self.__test_inputs


class LocalizationExample(object):
    def __init__(self, ref_slice, off_slice, corr_off_slice, transformation):
        self.ref_slice = ref_slice
        self.off_slice = off_slice
        self.corr_off_slice = corr_off_slice
        self.transformation = transformation


class LocalizationSampler(object):
    def __init__(self, dset_sampler, rotation_aug=True, max_angle=math.pi / 8, translation_aug=True,
                 max_shift=50):

        self.dset_sampler = dset_sampler

        # We have no use for batches here, so just get the first one, and just the first channel!
        sample_op = self.dset_sampler.get_sample_op()
        train_sample = sample_op[0, :, :, :, 0:1]

        # So we could feed in others
        self.aligned_sample = tf.placeholder_with_default(train_sample, [None, None, None, 1])

        # Pad the image on all 4 sides, so that when we rotate and shift we don't lose any information
        # pad = int((aligned_stack.shape[1] * (math.sqrt(2) - 1)) / 2 + max_shift)
        # padded_sample = tf.pad(aligned_sample, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

        # Break the sample into two slices
        reference_slice = self.aligned_sample[0:1]
        secondary_slice = self.aligned_sample[1:]

        # tf.contrib.image.transform defines what the transform should look like
        identity_transform = tf.constant([1., 0., 0.,
                                          0., 1., 0.,
                                          0., 0.])

        def add_rotation_to_transform(transform, th):
            cos_th = tf.cos(th)
            sin_th = tf.sin(th)

            rotation_matrix = tf.stack([[cos_th, -sin_th],
                                        [sin_th, cos_th]], axis=0)

            original_rotation = tf.stack([[transform[0], transform[1]],
                                          [transform[3], transform[4]]], axis=0)

            rotated = tf.matmul(rotation_matrix, original_rotation)

            # Construct a new transformation
            return tf.stack([rotated[0, 0], rotated[0, 1], transform[2], rotated[1, 0], rotated[1, 1], transform[5],
                             transform[6], transform[7]], axis=0)

        def add_translation_to_transform(transform, dx, dy):
            return tf.concat([transform[0:2], [transform[2] + dx], transform[3:5], [transform[5] + dy], transform[6:]],
                             axis=0)

        def add_random_rotation_to_transform(transform, min_ang, max_ang):
            # Define a random angle
            th = tf.random_uniform(shape=(), minval=min_ang, maxval=max_ang)

            return add_rotation_to_transform(transform, th), th

        def add_random_translation_to_transform(transform, x_max_trans, y_max_trans):
            # Define a random x and y translation
            dx = tf.round(tf.random_uniform(shape=(), minval=-x_max_trans, maxval=x_max_trans))
            dy = tf.round(tf.random_uniform(shape=(), minval=-y_max_trans, maxval=y_max_trans))

            return add_translation_to_transform(transform, dx, dy), dx, dy

        # Randomly augment the reference slice
        ref_trans, ref_rot = cond_apply(identity_transform,
                                        lambda t: add_random_rotation_to_transform(t, -max_angle, max_angle),
                                        lambda t: (t, 0.0),
                                        rotation_aug)

        ref_trans, ref_dx, ref_dy = cond_apply(ref_trans,
                                               lambda t: add_random_translation_to_transform(t, max_shift, max_shift),
                                               lambda t: (t, 0.0, 0.0),
                                               translation_aug)

        # Randomly augment the secondary slice
        sec_trans, sec_rot = cond_apply(identity_transform,
                                        lambda t: add_random_rotation_to_transform(t, -max_angle, max_angle),
                                        lambda t: (t, 0.0),
                                        rotation_aug)

        sec_trans, sec_dx, sec_dy = cond_apply(sec_trans,
                                               lambda t: add_random_translation_to_transform(t, max_shift, max_shift),
                                               lambda t: (t, 0.0, 0.0),
                                               translation_aug)

        # ref_trans = tf.Print(ref_trans, [ref_trans], message='REF_TRANS', summarize=8)
        # sec_trans = tf.Print(sec_trans, [sec_trans], message='SEC_TRANS', summarize=8)

        # Get the relative transformation
        relative_rot = ref_rot - sec_rot
        relative_dx = ref_dx - sec_dx
        relative_dy = ref_dy - sec_dy

        relative_transform = add_rotation_to_transform(identity_transform, relative_rot)
        relative_transform = add_translation_to_transform(relative_transform, relative_dx, relative_dy)

        # Apply the computed transformations
        transformed_ref = tfim.transform(reference_slice, ref_trans)
        transformed_sec = tfim.transform(secondary_slice, sec_trans)

        # relative_transform = tf.Print(relative_transform, [relative_transform], message='RELATIVE_TRANS', summarize=8)

        # Apply a transformation to secondary that aligns it to the transformed reference
        realigned_sec = tfim.transform(transformed_sec, relative_transform)

        # The reference slice, randomly transformed
        self.__transformed_reference = transformed_ref
        # The secondary slice, randomly transformed
        self.__transformed_secondary = transformed_sec
        # The secondary slice, transformed with the same transformations as the reference slice
        self.__secondary_label = realigned_sec
        # The relative transformation applied
        self.__relative_transform = relative_transform

    def get_sample_funcs(self):
        return self.__transformed_reference, self.__transformed_secondary, self.__secondary_label, \
               self.__relative_transform

    def initialize_session_variables(self, sess):
        self.dset_sampler.initialize_session_variables(sess)

    def sample(self, sess):
        return sess.run([self.__transformed_reference, self.__transformed_secondary, self.__secondary_label,
                         self.__relative_transform])


