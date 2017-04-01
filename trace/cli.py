# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import webbrowser
import subprocess
import click
import tensorflow as tf
import h5py

import em_dataset as em
import download_data
# import ensemble as ens
import learner
import params as par

from utils import *
import viewer_utils as vu

import hooks

from em_dataset import DATASET_DICT
from models import MODEL_DICT, ARCH_DICT


# from ensemble import ENSEMBLE_METHOD_DICT, ENSEMBLE_PARAMS_DICT


@click.group()
@click.option('--dset', type=click.Choice(DATASET_DICT.keys()), default='isbi')
@click.option('--arch', type=click.Choice(ARCH_DICT.keys()), default='vd2d_bound')
@click.option('--model', type=click.Choice(MODEL_DICT.keys()), default='conv')
@click.option('--split', type=click.Choice(SPLIT), default='test',
              help='Dataset split for visualization and prediction')
@click.option('--n-iter', type=int, default=10000, help="Number of training iterations")
@click.option('--run-name', type=str, default='1', help="Run name, to be preprended with 'run'")
@click.option('--batch', type=int, default=1, help="Batch size for training.")
@click.option('--train-shape', type=click.Tuple([int, int, int]), default=(16, 160, 160),
              help="Output patch size for training (net input will be calculated based on the output patch size)")
@click.option('--inf-shape', type=click.Tuple([int, int, int]), default=(16, 160, 160),
              help="Output patch size for inference (net input will be calculated based on the output patch size)")
@click.option('--lr', type=float, default=0.0001, help="Learning rate for the optimizer.")
@click.pass_context
def cli(ctx, dset, arch, model, split, n_iter, run_name, batch, train_shape, inf_shape, lr):
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dset + '/'

    model_constructor = MODEL_DICT[model]
    arch = ARCH_DICT[arch]
    dataset_constructor = DATASET_DICT[dset]

    pipeline = par.PipelineConfig(
        data_path=data_folder,
        dataset_constructor=dataset_constructor,
        model_constructor=model_constructor,
        model_arch=arch,
        training_params=par.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=lr,
            n_iterations=n_iter,
            patch_shape=train_shape,
            batch_size=batch
        ),
        inference_params=par.InferenceParams(
            patch_shape=inf_shape
        ))

    ctx.obj['PIPELINE'] = pipeline
    ctx.obj['SPLIT'] = split
    ctx.obj['RUN_NAME'] = run_name

    pass


@cli.command()
def download():
    current_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    download_data.maybe_create_all_datasets(current_folder, 0.9)


@cli.command()
@click.option('--aff/--no-aff', default=False, help="Display the affinities as well.")
@click.option('--ip', default='172.17.0.2', help="IP address for serving")
@click.option('--port', default=4125, help="Port for serving")
@click.option('--remote', default='127.0.0.1', help="IP address of AWS machine")
@click.pass_context
def visualize(ctx, aff, ip, port, remote):
    """
    Opens a tab in your webbrowser showing the chosen dataset
    """
    import neuroglancer

    pipeline = ctx.obj['PIPELINE']
    split = ctx.obj['SPLIT']
    run_name = ctx.obj['RUN_NAME']
    data_folder = pipeline.data_path
    model_name = pipeline.pipeline_name

    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])

    vu.add_file(data_folder, split + '-input', viewer)
    if aff:
        vu.add_affinities(data_folder + 'results/' + model_name + '/' + 'run-' + run_name + '/',
                          split + '-pred-affinities', viewer)
    vu.add_labels(data_folder + 'results/' + model_name + '/' + 'run-' + run_name + '/', split + '-predictions', viewer)
    if split != 'test':
        vu.add_labels(data_folder, split, viewer)

    print('open your brower at:')
    print(viewer.__str__().replace('172.17.0.2', remote))
    webbrowser.open(viewer.__str__())
    print("press any key to exit")
    input()


@cli.command()
@click.option('--high', type=float, default=0.9)
@click.option('--low', type=float, default=0.3)
@click.option('--dust', type=int, default=250)
@click.pass_context
def watershed(ctx, high, low, dust):
    """
    TODO Explain what each argument is, dust is currently ignored
    """

    pipeline = ctx.obj['PIPELINE']
    split = ctx.obj['SPLIT']
    data_folder = pipeline.data_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir + "/thirdparty/watershed/watershed.jl",
                     data_folder + '/' + split + "-affinities.h5",
                     data_folder + '/' + split + "-labels.h5",
                     str(high),
                     str(low)])


@cli.command()
@click.option('--cont/--no-cont', default=False, help="Continue training using a saved model.")
@click.pass_context
def train(ctx, cont):
    """
    Train a model.
    """

    # Extract parameters
    pipeline = ctx.obj['PIPELINE']
    run_name = ctx.obj['RUN_NAME']
    train_params = pipeline.training_params

    # Create model
    arch = pipeline.model_arch
    model_const = pipeline.model_constructor
    model = model_const(arch)

    # Determine the input size to be sampled from the dataset
    sample_shape = np.asarray(train_params.patch_shape) + np.asarray(arch.fov_shape) - 1

    # Construct the dataset sampler
    dataset = pipeline.dataset_constructor(pipeline.data_path)
    dset_sampler = em.EMDatasetSampler(dataset, sample_shape=sample_shape, batch_size=train_params.batch_size,
                                       label_output_type=arch.output_mode)

    ckpt_folder = pipeline.data_path + 'results/' + model.model_name + '/run-' + run_name + '/'

    classifier = learner.Learner(model, ckpt_folder)

    hooks_list = [
        hooks.LossHook(50, model),
        hooks.ModelSaverHook(500, ckpt_folder),
        hooks.ValidationHook(100, dset_sampler, model, pipeline.data_path, arch.output_mode, pipeline.inference_params),
        hooks.ImageVisualizationHook(2000, model),
        # hooks.HistogramHook(100, model),
        # hooks.LayerVisualizationHook(500, model),
    ]

    # Train the model
    print('Training for %d iterations' % train_params.n_iterations)
    classifier.train(train_params, dset_sampler, hooks_list, continue_training=cont)


@cli.command()
@click.pass_context
def predict(ctx):
    """
    Realods a model previously trained
    """
    # Extract parameters
    pipeline = ctx.obj['PIPELINE']
    run_name = ctx.obj['RUN_NAME']
    split = ctx.obj['SPLIT']
    train_params = pipeline.training_params

    # Create model
    arch = pipeline.model_arch
    model_const = pipeline.model_constructor
    model = model_const(arch)

    # Determine the input size to be sampled from the dataset
    sample_shape = np.asarray(train_params.patch_shape) + np.asarray(arch.fov_shape) - 1

    # Create the dataset sampler
    dataset = pipeline.dataset_constructor(pipeline.data_path)
    dset_sampler = em.EMDatasetSampler(dataset, sample_shape=sample_shape, batch_size=train_params.batch_size,
                                       label_output_type=arch.output_mode)

    if split == 'train':
        inputs, _, _ = dset_sampler.get_full_training_set()
    elif split == 'validation':
        inputs, _, _ = dset_sampler.get_validation_set()
    else:
        inputs = dset_sampler.get_test_set()

    # Define results folder
    ckpt_folder = pipeline.data_path + 'results/' + model.model_name + '/run-' + run_name + '/'

    # Create and restore the classifier
    classifier = learner.Learner(model, ckpt_folder)
    classifier.restore()

    # Predict on the classifier
    predictions = classifier.predict(inputs, pipeline.inference_params)

    # Save the predicted affinities for viewing in neuroglancer.
    # TODO(beisner): Fix this eventually
    # dataset.prepare_predictions_for_neuroglancer()
    # dataset.prepare_predictions_for_neuroglancer_affinities(ckpt_folder, split, predictions, arch.output_mode)

    # Prepare the predictions for submission for this particular dataset
    # Only send in the first dimension of predictions, because theoretically predict can predict on many stacks
    dataset.prepare_predictions_for_submission(ckpt_folder, split, predictions[0], arch.output_mode)


# @cli.command()
# @click.argument('ensemble_method', type=click.Choice(ENSEMBLE_METHOD_DICT.keys()))
# @click.argument('ensemble_params', type=click.Choice(ENSEMBLE_PARAMS_DICT.keys()))
# @click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
# @click.argument('run_name', type=str, default='1')
# def ens_train(ensemble_method, ensemble_params, dataset_name, run_name):
#     data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'
#
#     # Construct the dataset sampler
#     dset_constructor = DATASET_DICT[dataset_name]
#     dataset = dset_constructor(data_folder)
#
#     ensemble_method = ENSEMBLE_METHOD_DICT[ensemble_method]
#     p_name = ensemble_params
#     ensemble_params = ENSEMBLE_PARAMS_DICT[ensemble_params]
#
#     classifier = ens.EnsembleLearner(ensemble_params, p_name, ensemble_method, data_folder, run_name)
#
#     print('Training the ensemble...')
#     classifier.train(dataset)
#
#
# @cli.command()
# @click.argument('ensemble_method', type=click.Choice(ENSEMBLE_METHOD_DICT.keys()))
# @click.argument('ensemble_params', type=click.Choice(ENSEMBLE_PARAMS_DICT.keys()))
# @click.argument('dataset_name', type=click.Choice(DATASET_DICT.keys()))
# @click.argument('split', type=click.Choice(SPLIT))
# @click.argument('run_name', type=str, default='1')
# def ens_predict(ensemble_method, ensemble_params, dataset_name, split, run_name):
#     data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset_name + '/'
#
#     # Construct the dataset sampler
#     dset_constructor = DATASET_DICT[dataset_name]
#     dataset = dset_constructor(data_folder)
#
#     ensemble_method = ENSEMBLE_METHOD_DICT[ensemble_method]
#     p_name = ensemble_params
#     ensemble_params = ENSEMBLE_PARAMS_DICT[ensemble_params]
#
#     # Input size doesn't matter for us; neither does batch size
#     # TODO(beisner): Generalize ensemble_params so that it's not just an array, but a struct itself
#     dset_sampler = em.EMDatasetSampler(dataset, input_size=100, z_input_size=model.z_fov + 1, label_output_type=ensemble_params[0].output_mode)
#
#     # Inputs we will use
#     if split == 'train':
#         inputs, _ = dset_sampler.get_full_training_set()
#     elif split == 'validation':
#         inputs, _ = dset_sampler.get_validation_set()
#     else:
#         inputs = dset_sampler.get_test_set()
#
#     # Create the classifier
#     classifier = ens.EnsembleLearner(ensemble_params, p_name, ensemble_method, data_folder, run_name)
#
#     # Make the predictions
#     predictions = classifier.predict(inputs, [16, 160, 160])
#
#     # Prepare the predictions for submission for this particular dataset
#     # Only take the first of the predictions
#     dataset.prepare_predictions_for_submission(classifier.ensembler_folder, split, predictions[0],
#                                                ensemble_params[0].output_mode)

if __name__ == '__main__':
    cli(obj={})
