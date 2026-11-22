"""Fixtures for the trainer-core unit tests.

Helpers live in :mod:`trainer_core_support` (unique module name) so importing
them cannot collide with another directory's ``conftest`` when several test
directories are collected together.
"""

from __future__ import annotations

import pytest

from trainer_core_support import make_batch, make_model, tiny_config  # noqa: F401


@pytest.fixture
def batch_factory():
    return make_batch
