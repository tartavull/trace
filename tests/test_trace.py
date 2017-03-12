#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_trace
----------------------------------

Tests for `trace` module.
"""

import os.path
import os
import sys

# monkey with the path
sys.path.insert(0, os.path.abspath('./trace'))

from click.testing import CliRunner

import trace.cli as cli
import trace.download_data as download
import trace.augmentation as augmentation
import trace.learner as learner
import trace.em_dataset as em

from trace.models.conv_net import *
from trace.models.unet import *


class TestTrace(object):
    @classmethod
    def setup_class(cls):
        pass

    def test_something(self):
        pass

    def test_command_line_interface(self):
        runner = CliRunner()
        result = runner.invoke(cli.cli)
        assert result.exit_code == 0
        help_result = runner.invoke(cli.cli, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_comand_line_download_data(self):
        runner = CliRunner()
        runner.invoke(cli.cli, ['download'])
        current_folder = os.path.dirname(os.path.abspath(__file__)) + '/../trace/'

        assert os.path.exists(current_folder + download.ISBI + '/' + download.TEST_INPUT + download.H5)
        assert os.path.exists(current_folder + download.SNEMI3D + '/' + download.TEST_INPUT + download.H5)

    # With SNEMI3D, uses too much memory, and haven't implemented ISBI stuff yet.
    # TODO(beisner): Reintroduce this test
    # def test_train(self):
    #     """
    #     Train model for 10 steps and verify a model was created
    #     """
    #
    #     data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../trace/snemi3d/'
    #
    #     model = UNet(UNET_3D_4LAYERS, is_training=True)
    #
    #     run_name = 'test'
    #
    #     batch_size = 1
    #
    #     training_params = learner.TrainingParams(
    #         optimizer=tf.train.AdamOptimizer,
    #         learning_rate=0.0002,
    #         n_iter=10,
    #         output_size=120,
    #         z_output_size=16,
    #         batch_size=batch_size
    #     )
    #
    #     # Determine the input size to be sampled from the dataset
    #     input_size = training_params.output_size + model.fov - 1
    #     z_input_size = training_params.z_output_size + model.z_fov - 1
    #
    #     # Construct the dataset sampler
    #     dataset = em.SNEMI3DDataset(data_folder)
    #     dset_sampler = em.EMDatasetSampler(dataset, input_size, z_input_size, batch_size=batch_size,
    #                                        label_output_type=N4_3D.output_mode)
    #
    #     ckpt_folder = data_folder + 'results/' + model.model_name + '/run-' + run_name + '/'
    #
    #     classifier = learner.Learner(model, ckpt_folder)
    #
    #     # Train the model
    #     classifier.train(training_params, dset_sampler, [])


    @classmethod
    def teardown_class(cls):
        pass
