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

    def test_watershed(self):
        """
        Create affinities from train-labels and save them as
        test-affinities.h5
        And then run watershed on it using the cli
        """
        runner = CliRunner()
        runner.invoke(cli.cli, ['download'])
        current_folder = os.path.dirname(os.path.abspath(__file__)) + '/../trace/' + download.SNEMI3D + '/'
        augmentation.maybe_create_affinities(current_folder + 'train', 89)

        os.rename(current_folder + download.TRAIN_AFFINITIES + download.H5,
                  current_folder + download.TEST_AFFINITIES + download.H5)

        result = runner.invoke(cli.cli, ['watershed', 'test', 'snemi3d'])
        assert result.exit_code == 0
        assert os.path.exists(current_folder + download.TEST_LABELS + download.H5)

    def test_train(self):
        """
        Train model for 10 steps and verify a model was created
        """

        model_params = N4

        run_name = 'test'

        model = ConvNet(model_params)
        data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../trace/isbi/'

        dset = em.EMDataset(data_folder=data_folder, output_mode=model_params.output_mode)

        ckpt_folder = data_folder + 'results/' + model.model_name + '/run-' + run_name + '/'

        classifier = learner.Learner(model, ckpt_folder)

        training_params = learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.00001,
            n_iter=10,
            output_size=101, )

        classifier.train(training_params, dset, [])

    @classmethod
    def teardown_class(cls):
        pass
