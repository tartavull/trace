import tensorflow as tf

from .transform import TranslationTransform
from .localization import FCLocalizationNetwork, Conv2DLocalizationNetwork


class SpatialTransformer(object):
    def __call__(self, ref_slice, off_slice):
        raise NotImplementedError('Abstract class, the inheritor must implement.')


class TranslationSpatialTransformer(SpatialTransformer):
    def __init__(self, in_dim, max_shift):
        self.in_dim = in_dim
        self.max_shift = max_shift

        # Reduce the size of the inputs so that there is no 'black region' by 2 * max_shift
        self.reduced_dim = in_dim - 2 * max_shift

        # We introduce a scale factor so that our translation is within a known bound
        shift_fraction = float(max_shift) / in_dim

        # Initialize a TranslationOperation, but not a localization_network
        self.transformer = TranslationTransform(shift_fraction)
        self.localization_network = None

    def __call__(self, ref_slice, off_slice):
        # Crop the slices down to their reduced dim

        crop_start = self.max_shift
        crop_end = crop_start + self.reduced_dim

        cropped_ref = ref_slice[:, crop_start:crop_end, crop_start:crop_end, :]
        cropped_off = off_slice[:, crop_start:crop_end, crop_start:crop_end, :]

        # Compute the parameters
        params, self.histograms = self.localization_network(cropped_ref, cropped_off)
        realigned_off, theta = self.transformer(off_slice, params)

        return realigned_off, theta, self.reduced_dim


class FCTranslationSpatialTransformer(TranslationSpatialTransformer):
    def __init__(self, in_dim, fc1_units, max_shift, trainable):
        super(FCTranslationSpatialTransformer, self).__init__(in_dim, max_shift)

        self.localization_network = FCLocalizationNetwork(self.reduced_dim, self.transformer.params_dim, fc1_units,
                                                          trainable)
        self.saver = tf.train.Saver()

        assert (self.localization_network.params_dim == self.transformer.params_dim)


class ConvTranslationSpatialTransformer(TranslationSpatialTransformer):
    def __init__(self, in_dim, l1_n, l2_n, l3_n, l4_n, fc1_n, max_shift, trainable):
        super(ConvTranslationSpatialTransformer, self).__init__(in_dim, max_shift)

        self.localization_network = Conv2DLocalizationNetwork(self.reduced_dim, self.transformer.params_dim,
                                                              l1_n, l2_n, l3_n, l4_n, fc1_n, trainable)
        self.saver = tf.train.Saver()

        assert (self.localization_network.params_dim == self.transformer.params_dim)


def transformer_layer(stack, transformer):
    # We align to the top of the stack, for better or worse
    reference_image = stack[0]
    unaligned = stack[1:]

    # Iterate through the stack and align each image successively to the previous image
    # tf.scan produces
    realigned_stack = tf.scan(transformer, unaligned, initializer=reference_image)

    complete_stack = tf.concat([tf.expand_dims(reference_image, axis=0), realigned_stack], axis=0)

    return complete_stack
