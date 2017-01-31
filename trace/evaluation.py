from __future__ import print_function
from __future__ import division

import os

import h5py
import tensorflow as tf
import subprocess
from tqdm import tqdm

try:
    from thirdparty.segascorus import io_utils
    from thirdparty.segascorus import utils
    from thirdparty.segascorus.metrics import *
except Exception:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")



def rand_error(model, data_folder, sigmoid_prediction, num_layers, output_shape, watershed_high=0.9, watershed_low=0.3):
    # Save affinities to temporary file
    # TODO pad the image with zeros so that the output covers the whole dataset
    tmp_aff_file = 'validation-tmp-affinities.h5'
    tmp_label_file = 'validation-tmp-labels.h5'
    ground_truth_file = 'validation-labels.h5'

    base = data_folder + 'results/' + model.model_name + '/'

    with h5py.File(base + tmp_aff_file, 'w') as output_file:
        output_file.create_dataset('main', shape=(3, num_layers, output_shape, output_shape))
        out = output_file['main']

        reshaped_pred = np.einsum('zyxd->dzyx', sigmoid_prediction)
        out[0:2, :, :, :] = reshaped_pred

    # Do watershed segmentation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     base + tmp_aff_file,
                     base + tmp_label_file,
                     str(watershed_high),
                     str(watershed_low)])

    # Compute rand f score
    # --------------------

    # Parameters
    calc_rand_score = True
    calc_rand_error = False
    calc_variation_score = True
    calc_variation_information = False
    relabel2d = True
    foreground_restricted = True
    split_0_segment = True
    other = None

    seg1 = io_utils.import_file(base + tmp_label_file)
    seg2 = io_utils.import_file(data_folder + ground_truth_file)
    prep = utils.parse_fns(utils.prep_fns,
                           [relabel2d, foreground_restricted])
    seg1, seg2 = utils.run_preprocessing(seg1, seg2, prep)

    om = utils.calc_overlap_matrix(seg1, seg2, split_0_segment)

    # Calculating each desired metric
    metrics = utils.parse_fns(utils.metric_fns,
                              [calc_rand_score,
                               calc_rand_error,
                               calc_variation_score,
                               calc_variation_information])

    results = {}
    for (name, metric_fn) in metrics:
        if relabel2d:
            full_name = "2D {}".format(name)
        else:
            full_name = name

        (f, m, s) = metric_fn(om, full_name, other)
        results["{} Full".format(name)] = f
        results["{} Merge".format(name)] = m
        results["{} Split".format(name)] = s

    print('Rand F-Score Full: ' + str(results['Rand F-Score Full']))
    print('Rand F-Score Split: ' + str(results['Rand F-Score Split']))
    print('Rand F-Score Merge: ' + str(results['Rand F-Score Merge']))
    print('VI F-Score Full: ' + str(results['VI F-Score Full']))
    print('VI F-Score Split: ' + str(results['VI F-Score Split']))
    print('VI F-Score Merge: ' + str(results['VI F-Score Merge']))

    return results


def evaluate_pixel_error(dataset):
    with h5py.File(snemi3d.folder() + dataset + '-input.h5', 'r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        with h5py.File(snemi3d.folder() + dataset + '-affinities.h5', 'r') as label_file:
            inputShape = inpt.shape[1]
            outputShape = inpt.shape[1] - FOV + 1
            labels = label_file['main']

            with tf.variable_scope('foo'):
                net = create_network()
            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder() + tmp_dir + 'model.ckpt')
                print("Model restored.")

                # TODO pad the image with zeros so that the ouput covers the whole dataset
                totalPixelError = 0.0
                for z in tqdm(xrange(inpt.shape[0])):
                    print('z: {} of {}'.format(z, inpt.shape[0]))
                    reshapedLabel = np.einsum('dzyx->zyxd', labels[0:2, z:z + 1, FOV // 2:FOV // 2 + outputShape,
                                                            FOV // 2:FOV // 2 + outputShape])

                    pixelError = sess.run(net.pixel_error,
                                          feed_dict={net.image: inpt[z].reshape(1, inputShape, inputShape, 1),
                                                     net.target: reshapedLabel})

                    totalPixelError += pixelError

                print('Average pixel error: ' + str(totalPixelError / inpt.shape[0]))
