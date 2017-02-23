import h5py
import numpy as np
import tifffile

# LABEL MODES
BOUNDARIES = 'boundaries'
AFFINITIES_2D = 'affinities-2d'
AFFINITIES_3D = 'affinities-3d'
SEGMENTATION_2D = 'segmentation-2d'
SEGMENTATION_3D = 'segmentation-3d'


def convert_between_label_types(input_type, output_type, original_labels):

    # No augmentation needed, as we're basically doing e2e learning
    if input_type == output_type:
        return original_labels

    # This looks like a shit show, but conversion is hard.
    # Also, we will implement this as we go.
    # Alternatively, we could convert to some intermediate form, and then convert to a final form

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
            raise NotImplementedError('Aff2d->Boundaries not implemented')
        if output_type == AFFINITIES_3D:
            raise NotImplementedError('Aff2d->Aff3d not implemented')
        if output_type == SEGMENTATION_2D:
            raise NotImplementedError('Aff2d->Seg2d not implemented')
        if output_type == SEGMENTATION_3D:
            raise NotImplementedError('Aff2d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == AFFINITIES_3D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Aff3d->Boundaries not implemented')
        if output_type == AFFINITIES_2D:
            raise NotImplementedError('Aff3d->Aff2d not implemented')
        if output_type == SEGMENTATION_2D:
            raise NotImplementedError('Aff3d->Seg2d not implemented')
        if output_type == SEGMENTATION_3D:
            raise NotImplementedError('Aff3d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_2D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg2d->Boundaries not implemented')
        if output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg2d->Aff2d not implemented')
        if output_type == AFFINITIES_3D:
            raise NotImplementedError('Seg2d->Aff3d not implemented')
        if output_type == SEGMENTATION_3D:
            raise NotImplementedError('Seg2d->Seg3d not implemented')
        else:
            raise Exception('Invalid output_type')
    elif input_type == SEGMENTATION_3D:
        if output_type == BOUNDARIES:
            raise NotImplementedError('Seg3d->Boundaries not implemented')
        if output_type == AFFINITIES_2D:
            raise NotImplementedError('Seg3d->Aff2d not implemented')
        if output_type == AFFINITIES_3D:
            raise NotImplementedError('Seg3d->Aff3d not implemented')
        if output_type == SEGMENTATION_2D:
            raise NotImplementedError('Seg3d->Seg2d not implemented')
        else:
            raise Exception('Invalid output_type')
    else:
        raise Exception('Invalid input_type')





def generate_files_from_predictions(ckpt_folder, data_prefix, predictions):
    assert (data_prefix == 'train' or data_prefix == 'validation' or data_prefix == 'test')

    sha = predictions.shape
    # Create an affinities file
    with h5py.File(ckpt_folder + data_prefix + '-affinities.h5', 'w') as output_file:
        # Create the dataset in the file
        new_shape = (3, sha[0], sha[1], sha[2])

        output_file.create_dataset('main', shape=new_shape)

        # Reformat our predictions
        out = output_file['main']

        for i in range(predictions.shape[0]):
            reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(predictions[i], axis=0))
            out[0:2, i] = reshaped_pred[:, 0]

        # Our border is the max of the output
        tifffile.imsave(ckpt_folder + data_prefix + '-map.tif', np.minimum(out[0], out[1]))


def __grid_search(data_provider, data_folder, remaining_params, current_params, results_dict):
    if len(remaining_params) > 0:
        # Get a parameter
        param, values = remaining_params.popitem()

        # For each potential parameter, copy current_params and add the potential parameter to next_params
        for value in values:
            next_params = current_params.copy()
            next_params[param] = value

            # Perform grid search on the remaining params
            __grid_search(data_provider, data_folder, remaining_params=remaining_params.copy(),
                          current_params=next_params, results_dict=results_dict)
    else:
        try:
            print('Training this model:')
            print(current_params)
            model = n4.N4(current_params)
            results_dict[model.model_name] = train(model, data_provider, data_folder, n_iterations=500)  # temp
        except:
            print("Failed to train this model, ", sys.exc_info()[0])


def grid_search(data_provider, data_folder, params_lists):
    tf.Graph().as_default()

    # Mapping between parameter set and metrics.
    results_dict = dict()

    # perform the recursive grid search
    __grid_search(data_provider, data_folder, params_lists, dict(), results_dict)

    return results_dict
