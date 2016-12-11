#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_trace
----------------------------------

Tests for `trace` module.
"""

import os.path
import os

from click.testing import CliRunner

import trace.cli as cli
import trace.download_data as download
import trace.augmentation as augmentation
import trace.learner as learner

from trace.dp_transformer import DPTransformer
from trace.models.N4 import default_N4


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
        model = default_N4()
        data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../trace/isbi/'
        data_provider = DPTransformer(data_folder, 'train.spec')

        learner.train(model, data_provider, data_folder, n_iterations=10)

    @classmethod
    def teardown_class(cls):
        pass

