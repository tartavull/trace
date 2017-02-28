import numpy as np
import tensorflow as tf

# LABEL MODES
BOUNDARIES = 'boundaries'
AFFINITIES_2D = 'affinities-2d'
AFFINITIES_3D = 'affinities-3d'
SEGMENTATION_2D = 'segmentation-2d'
SEGMENTATION_3D = 'segmentation-3d'

SPLIT = ['train', 'validation', 'test']

import dataprovider.transform as trans


def convert_between_label_types(input_type, output_type, original_labels):
    # No augmentation needed, as we're basically doing e2e learning
    if input_type == output_type:
        return original_labels

    # This looks like a shit show, but conversion is hard.
    # Also, we will implement this as we go.
    # Alternatively, we could convert to some intermediate form (3D Affinities), and then convert to a final form

    if input_type == BOUNDARIES:
        if output_type == AFFINITIES_2D:
            raise NotImplementedError('Boundaries->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            raise NotImplementedError('Boundaries->Aff3d not implemented')
        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Boundaries->Seg2d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Boundaries->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_2D:
        if output_type == BOUNDARIES:
            # Take the average of each affinity in the x and y direction
            return np.mean(original_labels, axis=3)
        elif output_type == AFFINITIES_3D:
            # There are no z-direction affinities, so just make the z-affinity 0
            sha = original_labels.shape
            dtype = original_labels.dtype
            return np.concatenate((original_labels, np.zeros([sha[0], sha[1], sha[2], 1], dtype=dtype)), axis=3)
        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Aff2d->Seg2d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Aff2d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_3D:
        if output_type == BOUNDARIES:
            # Take the average of each affinity in the x, y, and z direction
            return np.mean(original_labels, axis=3)
        elif output_type == AFFINITIES_2D:
            # Discard the affinities in the z direction
            return original_labels[:, :, :, 0:2]
        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Aff3d->Seg2d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Aff3d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_2D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg2d->Boundaries not implemented')
        elif output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg2d->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            raise NotImplementedError('Seg2d->Aff3d not implemented')
        elif output_type == SEGMENTATION_3D:
            raise NotImplementedError('Seg2d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_3D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg3d->Boundaries not implemented')
        elif output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg3d->Aff2d not implemented')
        elif output_type == AFFINITIES_3D:
            return trans.affinitize(original_labels)
        elif output_type == SEGMENTATION_2D:
            raise NotImplementedError('Seg3d->Seg2d not implemented')
        else:
            raise Exception('Invalid output_type')
    else:
        raise Exception('Invalid input_type')


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

    # Create gaussian filter.
    gaussian_filter = tf.exp(-tf.cast(X * X + Y * Y, tf.float32)/(2 * sigma * sigma))

    # Normalize to 1 over truncated filter
    normalized_gaussian_filter = gaussian_filter / tf.reduce_sum(gaussian_filter)
    
    num_channels = image.get_shape()[-1]
    blur_filter = tf.expand_dims(tf.expand_dims(normalized_gaussian_filter, axis=2), axis=3)
    blur_filter = tf.tile(blur_filter, tf.stack([1, 1, num_channels, num_channels]))

    # Reflect image at borders to create padding for the filter.
    padding = tf.cast(padding, tf.int32)
    mirrored_image = tf.pad(image, 
                tf.stack([[padding, padding], [padding, padding], [0, 0]]), 'REFLECT')
    mirrored_image = tf.expand_dims(mirrored_image, axis=0)
    
    # Apply the gaussian filter.
    filtered_image = tf.nn.convolution(mirrored_image, blur_filter,
                strides = [1, 1], padding='VALID')

    return filtered_image[0]
