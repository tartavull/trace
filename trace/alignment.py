import tensorflow as tf
import numpy as np
import math
import models.common as com
import thirdparty.tf_models.spatial_transformer as spat
import utils as uti

import tensorflow.contrib.image as tfim

debug = False


class CorrespondencePredictor(object):
    def __init__(self):
        pass

    def train(self, inputs, correspondences, n_iter):
        """Train our model based on pairs of slices, and their corresponding correspondences.

        :param inputs: A stack of training inputs of shape [n_examples, 2, x_dim, y_dim, n_channels], where each
               example consists of 2 adjacent slices
        :param correspondences: A stack of correspondences between the 2 adjacent slices in each example, with shape
               [n_examples, 2, x_dim, y_dim, n_channels]. The correspondences should be float values on the interval
               [0, 1], where 0 indicates no correspondence and 1 indicates perfect correlation. The corre
        :param n_iter:

        """
        pass

    def predict(self, input_pair):
        pass


def misalign(aligned_images, padding_size):
    # Create a padding around the original images
    padded_aligned = np.pad(aligned_images, pad_width=padding_size, mode='constant', constant_values=0)

    # Apply an arbitrary rotation and translation


def conv3d_same(inp, kernel_shape, in_dim, out_dim, activation_fn=lambda x: x, name='unnamed'):
    filter_shape = kernel_shape + [in_dim, out_dim]
    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_dim]))

    conv = tf.nn.convolution(inp, w, padding='SAME')
    layer = activation_fn(conv + b)

    if debug:
        print(name)
        print(layer.get_shape())
    return layer


def conv3d_valid(inp, kernel_shape, in_dim, out_dim, activation_fn=lambda x: x, name='unnamed'):
    filter_shape = kernel_shape + [in_dim, out_dim]
    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_dim]))

    conv = tf.nn.convolution(inp, w, padding='VALID')
    layer = activation_fn(conv + b)

    if debug:
        print(name)
        print(layer.get_shape())
    return layer


def fully_connected(inp, in_dim, out_dim):
    w = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_dim]))

    act = tf.nn.relu(tf.matmul(inp, w) + b)

    return act


class Realigner(object):
    def __init__(self):
        # 2d reference image, to which we will realign

        # make sure it's on 0-1 scale

        self.reference_image = tf.placeholder(tf.float32, shape=[None, 1, None, None, 1])

        self.unaligned_image = tf.placeholder(tf.float32, shape=[None, 1, None, None, 1])

        self.target_image = tf.placeholder(tf.float32, shape=[None, 1, None, None, 1])

        l0 = tf.concat([self.reference_image, self.unaligned_image], axis=1)

        if debug:
            print('l0')
            print(l0.get_shape())

        # Layer 1: (2) 2x3x3 convolutions, then a relu
        l1 = conv3d_same(l0, kernel_shape=[2, 3, 3], in_dim=1, out_dim=64, name='l1')
        l2 = conv3d_same(l1, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l2')

        # Layer 2: (2) 2x3x3 convolutions, then a relu
        l3 = conv3d_same(l2, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l3')
        l4 = conv3d_same(l3, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l4')

        # Layer 3: (2) 2x3x3 convolutions, then a relu
        l5 = conv3d_same(l4, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l5')
        l6 = conv3d_same(l5, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l6')

        # Layer 3: (2) 2x3x3 convolutions, then a relu
        l7 = conv3d_same(l6, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l7')
        l8 = conv3d_same(l7, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l8')

        # Layer 4: (2) 2x3x3 convolutions, then a relu
        l9 = conv3d_same(l8, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l9')
        l10 = conv3d_same(l9, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l10')

        # Layer 5: (2) 2x3x3 convolutions, then a relu
        l11 = conv3d_same(l10, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l11')
        l12 = conv3d_same(l11, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l2')

        # Layer 6: (2) 2x3x3 convolutions, then a relu
        l13 = conv3d_same(l12, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l13')
        l14 = conv3d_same(l13, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l14')

        # Layer 7: (2) 2x3x3 convolutions, then a valid conv to reduce dims between two images, then a relu
        l15 = conv3d_same(l14, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, name='l15')
        l16 = conv3d_same(l15, kernel_shape=[2, 3, 3], in_dim=64, out_dim=64, activation_fn=tf.nn.relu, name='l16')
        l17 = conv3d_valid(l16, kernel_shape=[2, 1, 1], in_dim=64, out_dim=1, activation_fn=tf.nn.relu, name='l17')

        # A fully connected layer is not reasonable
        # # Layer 6: (1) fully connected layer, but only on the second feature map
        # reshaped = tf.reshape(l12, [-1])
        # dim = reshaped.get_shape()[0].value
        # full = fully_connected(reshaped, in_dim=dim, out_dim=(dim / 2))
        #
        # pre_sig_image = tf.reshape(full, shape=[1, y_dim, x_dim])

        pre_sig_image = l17

        self.pred_image = tf.sigmoid(pre_sig_image)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_image, logits=pre_sig_image))

        self.pixel_error = tf.reduce_mean(tf.abs(self.pred_image - self.target_image))

    def train(self, aligned_images, n_iter):
        # 4d tensor [z_dim, y_dim, x_dim, n_chan]
        assert (len(aligned_images.shape) == 4)

        tf_images = tf.constant(aligned_images)

        # Sample 2 adjacent images from the aligned images
        print(aligned_images.shape[1:])
        sample = tf.random_crop(tf_images, size=[2] + list(aligned_images.shape[1:]))

        # aligned_images
        #
        # padded_sample = tf.pad(sample, paddings=[0, ])
        #
        #
        #
        #
        # # Rotate randomly up to 45 degrees
        # angle = tf.random_uniform(shape=(), minval=-math.pi/4, maxval=math.pi/4)
        #
        # bounding_box =

        # Induce a transformation and a unique rotation on both of them

        # Create the target as the original second image with the first transformation applied

        pass


class SpatialTransformer(object):
    def __call__(self, ref_slice, off_slice):
        raise NotImplementedError('Abstract class, the inheritor must implement.')


class AffineSpatialTransformer(SpatialTransformer):
    def __init__(self, in_dim, is_trainable=True):
        self.in_dim = in_dim
        # Declare all the variables for a simple convolution

        with tf.variable_scope('aff_trans'):

            with tf.variable_scope('layer1'):

                self.__histogram1 = []

                # Layer 1 (724->360)
                self.__w11 = com.get_weight_variable('w11', [2, 3, 3, 1, 24], dtype=tf.float32, trainable=is_trainable)
                self.__w12 = com.get_weight_variable('w12', [2, 3, 3, 24, 24], dtype=tf.float32, trainable=is_trainable)
                self.__b1 = com.get_bias_variable('b1', [24], dtype=tf.float32, trainable=is_trainable)

                self.__histogram1.append(tf.summary.histogram('w11', self.__w11))
                self.__histogram1.append(tf.summary.histogram('w12', self.__w12))
                self.__histogram1.append(tf.summary.histogram('wb1', self.__b1))

                new_dim = in_dim/2 + in_dim % 2
                print(new_dim)



            with tf.variable_scope('layer2'):

                self.__histogram2 = []

                # Layer 2 (360->178)
                self.__w21 = com.get_weight_variable('w21', [2, 3, 3, 24, 36], dtype=tf.float32, trainable=is_trainable)
                self.__w22 = com.get_weight_variable('w22', [2, 3, 3, 36, 36], dtype=tf.float32, trainable=is_trainable)
                self.__w23 = com.get_weight_variable('w23', [1, 2, 2, 36, 36], dtype=tf.float32, trainable=is_trainable)
                self.__b2 = com.get_bias_variable('b2', [36], dtype=tf.float32, trainable=is_trainable)

                self.__histogram2.append(tf.summary.histogram('w21', self.__w21))
                self.__histogram2.append(tf.summary.histogram('w22', self.__w22))
                self.__histogram2.append(tf.summary.histogram('w23', self.__w23))
                self.__histogram2.append(tf.summary.histogram('b2', self.__b2))

                new_dim = (new_dim - 2+1)/2 + (new_dim - 2+1) % 2
                print(new_dim)

            with tf.variable_scope('layer3'):
                self.__histogram3 = []

                # Layer 3 (178->87)
                self.__w31 = com.get_weight_variable('w31', [2, 3, 3, 36, 48], dtype=tf.float32, trainable=is_trainable)
                self.__w32 = com.get_weight_variable('w32', [2, 3, 3, 48, 48], dtype=tf.float32, trainable=is_trainable)
                self.__b3 = com.get_bias_variable('b3', [48], dtype=tf.float32, trainable=is_trainable)

                self.__histogram3.append(tf.summary.histogram('w31', self.__w31))
                self.__histogram3.append(tf.summary.histogram('w32', self.__w32))
                self.__histogram3.append(tf.summary.histogram('b3', self.__b3))

                new_dim = new_dim/2 + new_dim % 2
                print(new_dim)

            with tf.variable_scope('layer4'):

                self.__histogram4 = []

                # Layer 4 (87->42)
                self.__w41 = com.get_weight_variable('w41', [2, 3, 3, 48, 64], dtype=tf.float32, trainable=is_trainable)
                self.__w42 = com.get_weight_variable('w42', [2, 3, 3, 64, 64], dtype=tf.float32, trainable=is_trainable)
                self.__w43 = com.get_weight_variable('w43', [2, 2, 2, 64, 64], dtype=tf.float32, trainable=is_trainable)
                self.__b4 = com.get_bias_variable('b4', [64], dtype=tf.float32, trainable=is_trainable)

                new_dim = new_dim/2 + new_dim % 2
                print(new_dim)

                self.__histogram4.append(tf.summary.histogram('w41', self.__w41))
                self.__histogram4.append(tf.summary.histogram('w42', self.__w42))
                self.__histogram4.append(tf.summary.histogram('w43', self.__w43))
                self.__histogram4.append(tf.summary.histogram('wb4', self.__b4))

            with tf.variable_scope('layer_fc1'):

                self.__histogram5 = []

                # Fully connected 1
                self.__wfc1 = com.get_weight_variable('wfc1', [2 * new_dim * new_dim * 64, 512], dtype=tf.float32, trainable=is_trainable)
                self.__bfc1 = com.get_bias_variable('bfc1', [512], dtype=tf.float32, trainable=is_trainable)

                self.__histogram5.append(tf.summary.histogram('wfc1', self.__wfc1))
                self.__histogram5.append(tf.summary.histogram('bfc1', self.__bfc1))


            with tf.variable_scope('layer_fc2'):

                self.__histogram6 = []

                # Fully connected 2
                self.__wfc2 = com.get_weight_variable('wfc2', [512, 6], dtype=tf.float32, trainable=is_trainable)

                # Initialize the bias function to the identity
                initial = np.array([[1., 0, 0], [0, 1., 0]])
                initial = initial.astype('float32')
                initial = initial.flatten()
                self.__bfc2 = tf.get_variable('bfc2', initializer=initial, dtype=tf.float32, trainable=is_trainable)
                self.__bfc2 = tf.Print(self.__bfc2, [self.__bfc2], message='bfc2', summarize=6)

                self.__histogram6.append(tf.summary.histogram('wfc2', self.__wfc2))
                self.__histogram6.append(tf.summary.histogram('bfc2', self.__bfc2))

        self.saver = tf.train.Saver()

    def __call__(self, ref_slice, off_slice):
        # Stack so that they have the shape [z, y, x], then expand so they have [batch, z, y, x, chan]
        assert(len(ref_slice.get_shape()) == 4)
        assert (len(off_slice.get_shape()) == 4)

        # Stack on the z dim, then expand to batch=1
        stacked = tf.expand_dims(tf.concat([ref_slice, off_slice], axis=0), axis=0)

        # Layer 1
        l11 = tf.nn.convolution(stacked, self.__w11, padding='SAME')
        l12 = tf.nn.convolution(l11, self.__w12, padding='SAME')
        l1 = tf.nn.sigmoid(l12 + self.__b1)

        self.__histogram1.append(tf.summary.histogram('l1', l1))
        histogram1 = tf.summary.merge(self.__histogram1)

        p1 = tf.nn.pool(l1, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p1.get_shape())

        # Layer 2
        l21 = tf.nn.convolution(p1, self.__w21, padding='SAME')
        l22 = tf.nn.convolution(l21, self.__w22, padding='SAME')
        l23 = tf.nn.convolution(l22, self.__w23, padding='VALID')
        l2 = tf.nn.sigmoid(l23 + self.__b2)

        self.__histogram2.append(tf.summary.histogram('l2', l2))
        histogram2 = tf.summary.merge(self.__histogram2)

        p2 = tf.nn.pool(l2, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p2.get_shape())

        # Layer 3
        l31 = tf.nn.convolution(p2, self.__w31, padding='SAME')
        l32 = tf.nn.convolution(l31, self.__w32, padding='SAME')
        l3 = tf.nn.sigmoid(l32 + self.__b3)

        self.__histogram3.append(tf.summary.histogram('l3', l3))
        histogram3 = tf.summary.merge(self.__histogram3)

        p3 = tf.nn.pool(l3, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p3.get_shape())

        # Layer 4
        l41 = tf.nn.convolution(p3, self.__w41, padding='SAME')
        l42 = tf.nn.convolution(l41, self.__w42, padding='SAME')
        l43 = tf.nn.convolution(l42, self.__w43, padding='SAME')
        l4 = tf.nn.sigmoid(l43 + self.__b4)

        self.__histogram4.append(tf.summary.histogram('l4', l4))
        histogram4 = tf.summary.merge(self.__histogram4)

        p4 = tf.nn.pool(l4, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        print(p4.get_shape())

        l4_flat = tf.reshape(p4, [1, -1])

        # Fully connected 1
        fc1 = tf.nn.sigmoid(tf.matmul(l4_flat, self.__wfc1) + self.__bfc1)

        self.__histogram5.append(tf.summary.histogram('l5', fc1))
        histogram5 = tf.summary.merge(self.__histogram5)

        # Fully connected 2, w/ sigmoid
        theta = tf.nn.softmax(tf.matmul(fc1, self.__wfc2) + self.__bfc2)

        self.__histogram6.append(tf.summary.histogram('theta', theta))
        histogram6 = tf.summary.merge(self.__histogram6)

        theta = tf.reshape(theta, [-1])

        theta = tf.Print(theta, [theta], message='theta: ', summarize=6)

        transformed = spat.transformer(off_slice, theta, out_size=(self.in_dim, self.in_dim))

        self.histograms = tf.summary.merge([histogram1, histogram2, histogram3, histogram4, histogram5, histogram6])

        # Squish it down to 2d, so that our transformer layer can process it
        return tf.reshape(transformed, shape=[self.in_dim, self.in_dim])


def transformer_layer(stack, mode='AFFINE', custom_fn=None):
    if mode == 'AFFINE':
        transformer = AffineSpatialTransformer()
        align_apply = transformer
    elif mode == 'CUSTOM':
        align_apply = custom_fn
    else:
        raise ValueError('Mode [%s] is not a valid mode' % mode)

    # We align to the top of the stack, for better or worse
    reference_image = stack[0]
    unaligned = stack[1:]

    # Iterate through the stack and align each image successively to the previous image
    # tf.scan produces
    realigned_stack = tf.scan(align_apply, unaligned, initializer=reference_image)

    complete_stack = tf.concat([tf.expand_dims(reference_image, axis=0), realigned_stack], axis=0)

    return complete_stack


class LocalizationSampler(object):
    def __init__(self, aligned_stack, rotation_aug=True, max_angle=math.pi / 8, translation_aug=True, max_shift=50):
        # 4d tensor [z_dim, y_dim, x_dim, n_chan]
        assert (len(aligned_stack.shape) == 4)

        # Put the stack into a constant
        tf_images = tf.constant(aligned_stack, dtype=tf.float32)

        # Sample 2 adjacent images from the aligned images
        aligned_sample = tf.random_crop(tf_images, size=(2,) + aligned_stack.shape[1:])

        # Pad the image on all 4 sides, so that when we rotate and shift we don't lose any information
        pad = int((aligned_stack.shape[1] * (math.sqrt(2) - 1)) / 2 + max_shift)
        padded_sample = tf.pad(aligned_sample, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

        # Break the sample into two slices
        reference_slice = padded_sample[0:1]
        secondary_slice = padded_sample[1:]

        # Function to apply a random rotation on inp with angle on interval [min_angle, max_angle]
        def tf_apply_random_rotation(inp, min_angle, max_angle):
            # Define a random angle
            angle = tf.random_uniform(shape=(), minval=min_angle, maxval=max_angle)

            # Rotate the image by that angle
            return tfim.rotate(inp, angle), angle

        # Function to apply a random translation on inp with dx on interval [-x_max_trans, x_max_trans] and dy
        # on interval [-y_max_trans, y_max_trans
        def tf_apply_random_translation(inp, x_max_trans, y_max_trans):
            # Define a random x and y translation
            x_trans = tf.random_uniform(shape=(), minval=-x_max_trans, maxval=x_max_trans)
            y_trans = tf.random_uniform(shape=(), minval=-y_max_trans, maxval=y_max_trans)

            # If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the output point (x, y)
            # to a transformed input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
            # where k = c0 x + c1 y + 1.
            #
            # From https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform

            # We only want to shift x by x_trans and y by y_trans
            transforms = [1, 0, x_trans, 0, 1, y_trans, 0, 0]

            # Perform the transforms
            return tfim.transform(inp, transforms), transforms

        ref_trans = dict()

        def save_trans(trans_dict, name, output, transform, ):
            trans_dict[name] = transform
            return output

        # APPLY RANDOM AUGMENTATIONS to the reference slice
        rotated_ref = uti.cond_apply(reference_slice,
                                     lambda inp: save_trans(ref_trans, 'rotation',
                                                            *tf_apply_random_rotation(inp, -max_angle, max_angle)),
                                     rotation_aug)

        translated_ref = uti.cond_apply(rotated_ref,
                                        lambda inp: save_trans(ref_trans, 'translation',
                                                               *tf_apply_random_translation(inp, max_shift, max_shift)),
                                        translation_aug)

        # APPLY DIFFERENT RANDOM AUGMENTATIONS to the secondary slice, only save the first output
        rotated_sec = uti.cond_apply(secondary_slice,
                                     lambda inp: tf_apply_random_rotation(inp, -max_angle, max_angle)[0],
                                     rotation_aug)

        translated_sec = uti.cond_apply(rotated_sec,
                                        lambda inp: tf_apply_random_translation(inp, max_shift, max_shift)[0],
                                        translation_aug)

        # APPLY THE SAME AUGMENTATION as the reference slice to the secondary slice, producing trans_lab
        rotated_lab = uti.cond_apply(secondary_slice,
                                     lambda inp: tfim.rotate(inp, ref_trans['rotation']), rotation_aug)

        trans_lab = uti.cond_apply(rotated_lab, lambda inp: tfim.transform(inp, ref_trans['translation']), translation_aug)

        # The reference slice, randomly transformed
        self.__transformed_reference = translated_ref
        # The secondary slice, randomly transformed
        self.__transformed_secondary = translated_sec
        # The secondary slice, transformed with the same transformations as the reference slice
        self.__secondary_label = trans_lab

    def get_sample_funcs(self):
        return self.__transformed_reference, self.__transformed_secondary, self.__secondary_label

    def sample(self):
        with tf.Session() as sess:
            t_ref, t_sec, sec_l = sess.run(
                [self.__transformed_reference, self.__transformed_secondary, self.__secondary_label])

        return t_ref, t_sec, sec_l


class LocalizationTrainer(object):
    def __init__(self, transformer, ckpt_folder):
        self.transformer = transformer
        self.ckpt_folder = ckpt_folder

    def train(self, sess, aligned_stack, n_iter, rotation_aug=True, translation_aug=True):
        assert(len(aligned_stack.shape) == 4)
        assert(np.amax(aligned_stack) <= 1.0)

        dim = aligned_stack.shape[1]

        sampler = LocalizationSampler(aligned_stack, rotation_aug=rotation_aug, translation_aug=translation_aug)

        ref_op, sec_op, true_realign_op = sampler.get_sample_funcs()

        pred_realign = self.transformer(ref_op, sec_op)

        pred_realign = tf.expand_dims(tf.expand_dims(pred_realign, axis=0), axis=3)

        pixel_error = tf.reduce_mean(tf.abs(pred_realign - true_realign_op))

        optimizer = tf.train.AdamOptimizer(0.001).minimize(pixel_error)

        summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        loss_summary = tf.summary.scalar('pixel_error', pixel_error)

        ref_summary = tf.summary.image('reference_image', ref_op)
        sec_summary = tf.summary.image('misaligned_image', sec_op)
        true_realign_summary = tf.summary.image('true_realignment', true_realign_op)
        pred_realign_summary = tf.summary.image('pred_realignment', pred_realign)

        image_summaries = tf.summary.merge([ref_summary, sec_summary, true_realign_summary, pred_realign_summary])

        sess.run(tf.initialize_all_variables())

        for i in range(n_iter):
            # Run the optimizer
            sess.run(optimizer)

            print(i)


            # Output to tensorboard
            if i % 10 == 0:
                loss_summary_rep, weight_summary = sess.run([loss_summary, self.transformer.histograms])
                summary_writer.add_summary(loss_summary_rep, i)
                summary_writer.add_summary(weight_summary, i)
                print('Loss summary written')

            # Image summary
            if i % 100 == 0:
                loss_summary_rep = sess.run(image_summaries)
                summary_writer.add_summary(loss_summary_rep, i)
                print('Computed images')

            # Model saver
            if i % 1000 == 0:
                self.transformer.saver.save(sess, self.ckpt_folder + 'model.ckpt')
                print('Model saved in %smodel.ckpt' % self.ckpt_folder)


# class SpatialTransformAligner(object):
#
#     def __init__(self, ):
