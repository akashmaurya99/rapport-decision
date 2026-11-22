"""Fixtures for the data-splits unit tests.

The example factory lives in :mod:`splits_test_support` (unique module name)
so it is import-safe no matter which other directories pytest collects.
"""

from __future__ import annotations

import pytest

from splits_test_support import make_example  # noqa: F401


@pytest.fixture
def example_factory():
    return make_example
