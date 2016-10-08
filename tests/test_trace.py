#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_trace
----------------------------------

Tests for `trace` module.
"""

import pytest

from contextlib import contextmanager
from click.testing import CliRunner

from trace import trace
from trace import cli
from trace import snemi3d

import os.path

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

    @classmethod
    def teardown_class(cls):
        pass

