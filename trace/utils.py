import h5py
import numpy as np
import tifffile


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
