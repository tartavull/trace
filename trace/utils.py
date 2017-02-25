import numpy as np

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
