# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import webbrowser
import subprocess
import click
import tensorflow as tf

import em_dataset as em
import download_data
import ensemble as ens
import learner

from utils import *
import viewer_utils as vu

from em_dataset import DATASET_DICT
from models import MODEL_DICT, PARAMS_DICT
from ensemble import ENSEMBLE_METHOD_DICT, ENSEMBLE_PARAMS_DICT


@click.group()
def cli():
    pass


@cli.command()
def download():
    current_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    download_data.maybe_create_all_datasets(current_folder, 0.9)


@cli.command()
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
@click.option('--aff/--no-aff', default=False, help="Display only the affinities.")
@click.argument('params_type', type=click.Choice(PARAMS_DICT.keys()))
@click.argument('run_name', type=str, default='1')
@click.option('--ip', default='172.17.0.2', help="IP address for serving")
@click.option('--port', default=4125, help="Port for serving")
@click.option('--remote', help="IP address of AWS machine")
def visualize(dataset_name, split, params_type, run_name, aff, ip, port, remote):
    """
    Opens a tab in your webbrowser showing the chosen dataset
    """
    import neuroglancer

    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'

    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])
    if aff:
        vu.add_affinities(data_folder, split + '-affinities', viewer)
    else:
        vu.add_file(data_folder, split + '-input', viewer)
        print(data_folder)
        if split == 'test':
            vu.add_file(data_folder + 'results/' + params_type + '/' +  'run-' + run_name + '/', split+'-predictions', viewer)
            print(data_folder + 'results/' + params_type + '/' +  'run-' + run_name + '/')
        else:
            vu.add_file(data_folder, split + '-labels', viewer)

    print('open your brower at:')
    print(viewer.__str__().replace('172.17.0.2', remote))
    webbrowser.open(viewer.__str__())
    print("press any key to exit")
    input()

@cli.command()
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
@click.option('--high', type=float, default=0.9)
@click.option('--low', type=float, default=0.3)
@click.option('--dust', type=int, default=250)
def watershed(dataset_name, split, high, low, dust):
    """
    TODO Explain what each argument is, dust is currently ignored
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir +"/thirdparty/watershed/watershed.jl",
                     current_dir + '/' + dataset_name + '/' + split + "-affinities.h5",
                     current_dir + '/' + dataset_name + '/' + split + "-labels.h5",
                     str(high),
                     str(low)])


@cli.command()
@click.argument('model_type', type=click.Choice(MODEL_DICT.keys()))
@click.argument('params_type', type=click.Choice(PARAMS_DICT.keys()))
@click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
@click.argument('n_iter', type=int, default=10000)
@click.argument('run_name', type=str, default='1')
def train(model_type, params_type, dataset_name, n_iter, run_name):
    """
    Train an N4 models to predict affinities
    """
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'

    model_constructor = MODEL_DICT[model_type]
    params = PARAMS_DICT[params_type]
    model = model_constructor(params, is_training=True)

    batch_size = 1

    training_params = learner.TrainingParams(
        optimizer=tf.train.AdamOptimizer,
        learning_rate=0.0002,
        n_iter=n_iter,
        output_size=120,
        z_output_size=16,
        batch_size=batch_size
    )

    # Determine the input size to be sampled from the dataset
    input_size = training_params.output_size + model.fov - 1
    z_input_size = training_params.z_output_size + model.z_fov - 1

    # Construct the dataset sampler
    dset_constructor = DATASET_DICT[dataset_name]
    dataset = dset_constructor(data_folder)
    dset_sampler = em.EMDatasetSampler(dataset, input_size, z_input_size, batch_size=batch_size, label_output_type=params.output_mode)

    ckpt_folder = data_folder + 'results/' + model.model_name + '/run-' + run_name + '/'

    classifier = learner.Learner(model, ckpt_folder)

    hooks = [
        learner.LossHook(10, model),
        learner.ModelSaverHook(5000, ckpt_folder),
        learner.ValidationHook(500, dset_sampler, model, data_folder, params.output_mode, [training_params.z_output_size, training_params.output_size, training_params.output_size]),
        learner.ImageVisualizationHook(2000, model),
        # learner.HistogramHook(100, model),
        # learner.LayerVisualizationHook(500, model),
    ]

    # Train the model
    print('Training for %d iterations' % n_iter)
    classifier.train(training_params, dset_sampler, hooks)


@cli.command()
@click.argument('model_type', type=click.Choice(MODEL_DICT.keys()))
@click.argument('params_type', type=click.Choice(PARAMS_DICT.keys()))
@click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('run_name', type=str, default='1')
def predict(model_type, params_type, dataset_name, split, run_name):
    """
    Realods a model previously trained
    """
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'

    # Create the model
    model_constructor = MODEL_DICT[model_type]
    params = PARAMS_DICT[params_type]
    model = model_constructor(params, is_training=False)

    dset_constructor = DATASET_DICT[dataset_name]
    dataset = dset_constructor(data_folder)

    # Input size doesn't matter for us; neither does batch size
    dset_sampler = em.EMDatasetSampler(dataset, input_size=model.fov + 1, z_input_size=model.z_fov + 1, label_output_type=params.output_mode)

    if split == 'train':
        inputs, _ = dset_sampler.get_full_training_set()
    elif split == 'validation':
        inputs, _ = dset_sampler.get_validation_set()
    else:
        inputs = dset_sampler.get_test_set()

    # Define results folder
    ckpt_folder = data_folder + 'results/' + model.model_name + '/run-' + run_name + '/'

    # Create and restore the classifier
    classifier = learner.Learner(model, ckpt_folder)
    classifier.restore()

    # Predict on the classifier
    predictions = classifier.predict(inputs, [16, 120, 120])

    # Prepare the predictions for submission for this particular dataset
    # Only send in the first dimension of predictions, because theoretically predict can predict on many stacks
    dataset.prepare_predictions_for_submission(ckpt_folder, split, predictions[0], params.output_mode)


@cli.command()
@click.argument('ensemble_method', type=click.Choice(ENSEMBLE_METHOD_DICT.keys()))
@click.argument('ensemble_params', type=click.Choice(ENSEMBLE_PARAMS_DICT.keys()))
@click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
@click.argument('run_name', type=str, default='1')
def ens_train(ensemble_method, ensemble_params, dataset_name, run_name):
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'

    # Construct the dataset sampler
    dset_constructor = DATASET_DICT[dataset_name]
    dataset = dset_constructor(data_folder)

    ensemble_method = ENSEMBLE_METHOD_DICT[ensemble_method]
    p_name = ensemble_params
    ensemble_params = ENSEMBLE_PARAMS_DICT[ensemble_params]

    classifier = ens.EnsembleLearner(ensemble_params, p_name, ensemble_method, data_folder, run_name)

    print('Training the ensemble...')
    classifier.train(dataset)


@cli.command()
@click.argument('ensemble_method', type=click.Choice(ENSEMBLE_METHOD_DICT.keys()))
@click.argument('ensemble_params', type=click.Choice(ENSEMBLE_PARAMS_DICT.keys()))
@click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('run_name', type=str, default='1')
def ens_predict(ensemble_method, ensemble_params, dataset_name, split, run_name):
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'

    # Construct the dataset sampler
    dset_constructor = DATASET_DICT[dataset_name]
    dataset = dset_constructor(data_folder)

    ensemble_method = ENSEMBLE_METHOD_DICT[ensemble_method]
    p_name = ensemble_params
    ensemble_params = ENSEMBLE_PARAMS_DICT[ensemble_params]

    # Input size doesn't matter for us; neither does batch size
    # TODO(beisner): Generalize ensemble_params so that it's not just an array, but a struct itself
    dset_sampler = em.EMDatasetSampler(dataset, input_size=100, z_input_size=model.z_fov + 1, label_output_type=ensemble_params[0].output_mode)

    # Inputs we will use
    if split == 'train':
        inputs, _ = dset_sampler.get_full_training_set()
    elif split == 'validation':
        inputs, _ = dset_sampler.get_validation_set()
    else:
        inputs = dset_sampler.get_test_set()

    # Create the classifier
    classifier = ens.EnsembleLearner(ensemble_params, p_name, ensemble_method, data_folder, run_name)

    # Make the predictions
    predictions = classifier.predict(inputs, [16, 120, 120])

    # Prepare the predictions for submission for this particular dataset
    # Only take the first of the predictions
    dataset.prepare_predictions_for_submission(classifier.ensembler_folder, split, predictions[0],
                                               ensemble_params[0].output_mode)

if __name__ == '__main__':
    cli()
