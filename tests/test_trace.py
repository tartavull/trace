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

from contextlib import contextmanager
from click.testing import CliRunner

from trace import trace
from trace import cli
from trace import snemi3d
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
        assert os.path.exists(snemi3d.folder()+'/test-input.h5')

    def test_watershed(self):
        """
        Create affinities from train-labels and save them as
        test-affinities.h5
        And then run wateshed on it using the cli
        """
        snemi3d.maybe_create_dataset()
        augmentation.maybe_create_affinities('train')
        os.rename(snemi3d.folder()+"train-affinities.h5",snemi3d.folder()+"test-affinities.h5")
        runner = CliRunner()
        result = runner.invoke(cli.cli,['watershed'])
        assert result.exit_code == 0
        assert os.path.exists(snemi3d.folder()+'test-labels.h5')

    def test_train(self):
        """      
        Train model for 10 steps and verify a model was created
        """
        trace.train(10)

    @classmethod
    def teardown_class(cls):
        pass

