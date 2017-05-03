import tensorflow as tf

import trace.thirdparty.tf_models.spatial_transformer as spat


class Transform2D(object):
    params_dim = None

    def __call__(self, off_slice, params):
        raise NotImplementedError('Abstract class, the inheritor must implement')


class TranslationTransform(Transform2D):
    params_dim = 2

    def __init__(self, max_shift_factor):
        self.__max_shift_factor = max_shift_factor

    def __call__(self, off_slice, params):

        # Scale the shift by the 2x scale factor, so that we limit the amount of translation to within some bound (2x)
        params *= (2 * self.__max_shift_factor)

        out_size = tf.shape(off_slice)[1]

        trans = tf.reshape([
            [1, 0, params[0]],
            [0, 1, params[1]],
        ], [-1])

        transformed = spat.transformer(off_slice, trans, out_size=(out_size, out_size))

        # Squish down to 2d so that the scan transformation works in the batch aligner
        return tf.reshape(transformed, shape=[out_size, out_size]), trans


class RotationTransform(Transform2D):
    params_dim = 1

    def __init__(self, max_rotation):
        self.__max_rotation = max_rotation

    def __call__(self, off_slice, params):
        # Parameter should be [cos(th), -sin(th), 0, sin(th), cos(th), 0]

        # Limit theta to some max angle, since params should be
        params *= self.__max_rotation

        out_size = tf.shape(off_slice)[1]

        sin_th = tf.sin(params[0])
        cos_th = tf.cos(params[0])

        trans = tf.reshape([
            [cos_th, -sin_th, 0],
            [sin_th, cos_th, 0],
        ], [-1])

        transformed = spat.transformer(off_slice, trans, out_size=(out_size, out_size))

        # Squish down to 2d so that the scan transformation works in the batch aligner
        return tf.reshape(transformed, shape=[out_size, out_size]), trans


class RigidTransform(Transform2D):
    params_dim = 3

    def __init__(self, max_rotation, max_shift_factor):
        self.__max_rotation = max_rotation
        self.__max_shift_factor = max_shift_factor

    def __call__(self, off_slice, params):
        # Parameter should be [cos(th), -sin(th), dx, sin(th), cos(th), dy]

        out_size = tf.shape(off_slice)[1]

        th = params[0] * self.__max_rotation
        dx = params[1] * ((1 - self.__max_shift_factor) * 2)
        dy = params[2] * ((1 - self.__max_shift_factor) * 2)

        sin_th = tf.sin(th)
        cos_th = tf.cos(th)

        trans = tf.reshape([
            [cos_th, -sin_th, dx],
            [sin_th, cos_th, dy],
        ], [-1])

        transformed = spat.transformer(off_slice, trans, out_size=(out_size, out_size))

        # Squish down to 2d so that the scan transformation works in the batch aligner
        return tf.reshape(transformed, shape=[out_size, out_size]), trans


class AffineTransform(Transform2D):
    params_dim = 6

    def __call__(self, off_slice, params):
        pass
