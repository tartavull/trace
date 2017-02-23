# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import webbrowser
import subprocess

import h5py
import click
import utils

import tensorflow as tf

import em_dataset as em

import download_data
import ensemble as ens
import learner
from dp_transformer import DPTransformer
from models import MODEL_DICT, PARAMS_DICT
from ensemble import ENSEMBLE_METHOD_DICT, ENSEMBLE_PARAMS_DICT

SPLIT = ['train', 'validation', 'test']


@click.group()
def cli():
    pass


@cli.command()
def download():
    current_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    download_data.maybe_create_all_datasets(current_folder, 0.9)


@cli.command()
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('dataset', type=click.Choice(download_data.DATASET_NAMES))
@click.option('--aff/--no-aff', default=False, help="Display only the affinities.")
@click.option('--ip', default='172.17.0.2', help="IP address for serving")
@click.option('--port', default=4125, help="Port for serving")
def visualize(dataset, split, aff, ip, port):
    """
    Opens a tab in your webbrowser showing the chosen dataset
    """
    import neuroglancer

    # config = config_dict(dataset)

    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])
    if aff:
        import augmentation
        augmentation.maybe_create_affinities(split)
        add_affinities(config.folder, split + '-affinities', viewer)
    else:
        add_file(config.folder, split + '-input', viewer)
        add_file(config.folder, split + '-labels', viewer)

    print('open your brower at:')
    print(viewer.__str__().replace('172.17.0.2', '54.166.106.209')) # Replace the second argument with your own server's ip address
    webbrowser.open(viewer.__str__())
    print("press any key to exit")
    input()


def add_file(folder, filename, viewer):
    try:
        with h5py.File(folder+filename+'.h5','r') as f:
            arr = f['main'][:]
            viewer.add(arr, name=filename)
    except IOError:
        print(filename+' not found')


def add_affinities(folder, filename, viewer):
    """
    This is holding all the affinities in RAM,
    it would be easy to modify so that it is
    reading from disk directly.
    """
    try:
        with h5py.File(folder+filename+'.h5','r') as f:
            x_aff = f['main'][0,:,:,:]
            viewer.add(x_aff, name=filename+'-x', shader="""
            void main() {
              emitRGB(
                    vec3(1.0 - toNormalized(getDataValue(0)),
                         0,
                         0)
                      );
            }
            """)
            y_aff = f['main'][1,:,:,:]
            viewer.add(y_aff, name=filename+'-y', shader="""
            void main() {
              emitRGB(
                    vec3(0,
                         1.0 - toNormalized(getDataValue(0)),
                         0)
                      );
            }
            """)
            z_aff = f['main'][2,:,:,:]
            viewer.add(z_aff, name=filename+'-z', shader="""
            void main() {
              emitRGB(
                    vec3(0,
                         0,
                         1.0 - toNormalized(getDataValue(0)))
                      );
            }
            """)
    except IOError:
        print(filename+'.h5 not found')


@cli.command()
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('dataset', type=click.Choice(download_data.DATASET_NAMES))
@click.option('--high', type=float, default=0.9)
@click.option('--low', type=float, default=0.3)
@click.option('--dust', type=int, default=250)
def watershed(dataset, split, high, low, dust):
    """
    TODO Explain what each argument is, dust is currently ignored
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir +"/thirdparty/watershed/watershed.jl",
                     current_dir + '/' + dataset + '/' + split + "-affinities.h5",
                     current_dir + '/' + dataset + '/' + split + "-labels.h5",
                     str(high),
                     str(low)])


@cli.command()
@click.argument('model_type', type=click.Choice(MODEL_DICT.keys()))
@click.argument('params_type', type=click.Choice(PARAMS_DICT.keys()))
@click.argument('dataset', type=click.Choice(download_data.DATASET_NAMES))
@click.argument('suffix', type=str, default='')
@click.argument('n_iter', type=int, default=10000)
@click.argument('run_name', type=str, default='1')
def train(model_type, params_type, dataset, suffix, n_iter, run_name):
    """
    Train an N4 models to predict affinities
    """
    dataset = dataset + suffix

    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset + '/'

    model_constructor = MODEL_DICT[model_type]
    params = PARAMS_DICT[params_type]
    model = model_constructor(params, is_training=True)

    dset = em.EMDataset(data_folder=data_folder, output_mode=params.output_mode)

    ckpt_folder = data_folder + 'results/' + model.model_name + '/run-' + run_name + '/'

    classifier = learner.Learner(model, ckpt_folder)

    hooks = [
        learner.LossHook(50, model),
        learner.ModelSaverHook(1000, ckpt_folder),
        learner.ValidationHook(500, dset, model, data_folder, params.output_mode),
        learner.ImageVisualizationHook(500, model),
        learner.HistogramHook(100, model),
        learner.LayerVisualizationHook(500, model),
    ]

    training_params = learner.TrainingParams(
        optimizer=tf.train.AdamOptimizer,
        learning_rate=0.0001,
        n_iter=n_iter,
        output_size=101,
    )

    # Train the model
    print('Training for %d iterations' % n_iter)
    classifier.train(training_params, dset, hooks)


@cli.command()
@click.argument('model_type', type=click.Choice(MODEL_DICT.keys()))
@click.argument('params_type', type=click.Choice(PARAMS_DICT.keys()))
@click.argument('dataset', type=click.Choice(download_data.DATASET_NAMES))
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('run_name', type=str, default='1')
def predict(model_type, params_type, dataset, split, run_name):
    """
    Realods a model previously trained
    """
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset + '/'
    data_provider = DPTransformer(data_folder, 'train.spec')

    # Create the model
    model_constructor = MODEL_DICT[model_type]
    params = PARAMS_DICT[params_type]
    model = model_constructor(params, is_training=False)

    # Inputs we will use
    inputs, _ = data_provider.dataset_from_h5py(split)

    # Define results folder
    ckpt_folder = data_folder + 'results/' + model.model_name + '/run-' + run_name + '/'

    # Create and restore the classifier
    classifier = learner.Learner(model, ckpt_folder)
    classifier.restore()

    # Predict on the classifier
    predictions = classifier.predict(inputs)

    # Save the outputs
    utils.generate_files_from_predictions(ckpt_folder, split, predictions)


@cli.command()
@click.argument('ensemble_method', type=click.Choice(ENSEMBLE_METHOD_DICT.keys()))
@click.argument('ensemble_params', type=click.Choice(ENSEMBLE_PARAMS_DICT.keys()))
@click.argument('dataset', type=click.Choice(download_data.DATASET_NAMES))
@click.argument('run_name', type=str, default='1')
def ens_train(ensemble_method, ensemble_params, dataset, run_name):
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset + '/'
    data_provider = DPTransformer(data_folder, 'train.spec')

    ensemble_method = ENSEMBLE_METHOD_DICT[ensemble_method]
    p_name = ensemble_params
    ensemble_params = ENSEMBLE_PARAMS_DICT[ensemble_params]

    classifier = ens.EnsembleLearner(ensemble_params, p_name, ensemble_method, data_folder, run_name)

    print('Training the ensemble...')
    classifier.train(data_provider)


@cli.command()
@click.argument('ensemble_method', type=click.Choice(ENSEMBLE_METHOD_DICT.keys()))
@click.argument('ensemble_params', type=click.Choice(ENSEMBLE_PARAMS_DICT.keys()))
@click.argument('dataset', type=click.Choice(download_data.DATASET_NAMES))
@click.argument('split', type=click.Choice(SPLIT))
@click.argument('run_name', type=str, default='1')
def ens_predict(ensemble_method, ensemble_params, dataset, split, run_name):
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset + '/'
    data_provider = DPTransformer(data_folder, 'train.spec')

    ensemble_method = ENSEMBLE_METHOD_DICT[ensemble_method]

    p_name = ensemble_params
    ensemble_params = ENSEMBLE_PARAMS_DICT[ensemble_params]

    # Inputs we will use
    inputs, _ = data_provider.dataset_from_h5py(split)

    # Create the classifier
    classifier = ens.EnsembleLearner(ensemble_params, p_name, ensemble_method, data_folder, run_name)

    # Make the predictions
    predictions = classifier.predict(inputs)

    # Generate output files
    utils.generate_files_from_predictions(classifier.results_folder, split, predictions)

if __name__ == '__main__':
    cli()
