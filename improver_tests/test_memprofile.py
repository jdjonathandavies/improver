# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for optional Memray stage profiling."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from improver.memprofile import memray_stage


def test_disabled_without_memray(monkeypatch):
    """Normal execution needs neither Memray nor an output directory."""
    monkeypatch.delenv("IMPROVER_MEMRAY_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "memray", None)
    with memray_stage("loading"):
        pass


def test_enabled_and_exception_cleanup(monkeypatch, tmp_path):
    """The native tracker exits on errors and the original error propagates."""
    directory = tmp_path / "captures"
    monkeypatch.setenv("IMPROVER_MEMRAY_DIR", str(directory))
    tracker = MagicMock()
    tracker.return_value.__exit__.return_value = False
    monkeypatch.setitem(sys.modules, "memray", SimpleNamespace(Tracker=tracker))

    with pytest.raises(ValueError, match="training failed"):
        with memray_stage("training_arrays"):
            tracker.return_value.__enter__.assert_called_once()
            raise ValueError("training failed")

    assert directory.is_dir()
    tracker.assert_called_once_with(
        str(directory / "training_arrays.bin"), native_traces=True
    )
    tracker.return_value.__exit__.assert_called_once()
