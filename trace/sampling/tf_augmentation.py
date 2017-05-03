import tensorflow as tf
import math

# Flip a coin, and apply an op to sample (sample can be 5d or 4d)
# Prob is the denominator of the probability (1 in prob chance)
def tf_randomly_map_and_apply_op(data, op, prob=2):
    should_apply = tf.random_uniform(shape=(), minval=0, maxval=prob, dtype=tf.int32)

    def tf_if(ex):
        return tf.cond(tf.equal(0, should_apply), lambda: op(ex), lambda: ex)

    return tf.map_fn(tf_if, data)


def tf_randomly_apply_op(data, op, prob=2):
    should_apply = tf.random_uniform(shape=(), minval=0, maxval=prob, dtype=tf.int32)

    return tf.cond(tf.equal(0, should_apply), lambda: op(data), lambda: data)


# Perform random mirroring, by applying the same mirroring to each image in the stack
def tf_mirror_each_image_in_stack_op(stack):
    return tf.map_fn(lambda img: tf.image.flip_left_right(img), stack)


# Apply a random rotation to each stack
def tf_apply_random_rotation_to_stack(stack, min_rot, max_rot):
    # Get the random angle
    angle = tf.random_uniform(shape=(), minval=min_rot, maxval=max_rot)

    # Rotate each image by that angle
    return tf.map_fn(lambda img: tf.contrib.image.rotate(img, angle), stack)


# Apply random gaussian blurring to the image
def tf_apply_random_blur_to_stack(stack, min_sigma, max_sigma, prob):
    def apply_random_blur_to_slice(img):
        sigma = tf.random_uniform(shape=(), minval=min_sigma, maxval=max_sigma, dtype=tf.float32)
        return tf_gaussian_blur(img, sigma, size=5)

    return tf.map_fn(lambda img: tf_randomly_apply_op(img, apply_random_blur_to_slice, prob=prob), stack)


def tf_randomly_blur_each_stack(stacks, min_sig=2, max_sig=5, prob=2):
    return tf.map_fn(lambda stack: tf_apply_random_blur_to_stack(stack, min_sig, max_sig, prob), stacks)


def tf_randomly_mirror_each_stack(stacks):
    return tf_randomly_map_and_apply_op(stacks, tf_mirror_each_image_in_stack_op)


def tf_randomly_flip_each_stack(stacks):
    return tf_randomly_map_and_apply_op(stacks, lambda stack: tf.reverse(stack, axis=[0]))


def tf_randomly_rotate_each_stack(stacks, min_rot=0, max_rot=2 * math.pi):
    return tf.map_fn(lambda stack: tf_apply_random_rotation_to_stack(stack, min_rot, max_rot), stacks)


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

