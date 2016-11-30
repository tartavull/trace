#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_trace
----------------------------------

Tests for `trace` module.
"""

import os.path
import os

import pytest
import models

from contextlib import contextmanager
from click.testing import CliRunner

from trace import trace
from trace import cli
from trace import dataset_config
from trace import augmentation

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

    def test_comand_line_downloads_dataset(self):
        runner = CliRunner()
        runner.invoke(cli.cli,['download'])
        snemi3d_config = dataset_config.snemi3d_config()
        isbi_config = dataset_config.isbi_config()
        assert os.path.exists(snemi3d_config.folder + snemi3d_config.test_input_h5)
        assert os.path.exists(isbi_config.folder + isbi_config.test_input_h5)

    def test_watershed(self):
        """
        Create affinities from train-labels and save them as
        test-affinities.h5
        And then run wateshed on it using the cli
        """

        snemi3d_config = dataset_config.snemi3d_config()

        dataset_config.maybe_create_all_datasets(0.9)
        augmentation.maybe_create_affinities(dataset_config.folder + 'train')
        os.rename(snemi3d_config.folder + "train-affinities.h5", snemi3d_config.folder + "test-affinities.h5")
        runner = CliRunner()
        result = runner.invoke(cli.cli,['watershed'])
        assert result.exit_code == 0
        assert os.path.exists(snemi3d_config.folder+'test-labels.h5')

    def test_train(self):
        """
        Train model for 10 steps and verify a model was created
        """
        model = models.default_N4()
        config = dataset_config.snemi3d_config()
        trace.train(model, config)

    @classmethod
    def teardown_class(cls):
        pass

