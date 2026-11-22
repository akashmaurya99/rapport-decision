"""Backwards-compatible re-export of the e2e test helpers.

The implementation lives in :mod:`e2e_test_support` so it has a unique module
name; see that module for why ``from conftest import ...`` is unsafe when
several test directories are collected together.
"""

from e2e_test_support import (  # noqa: F401
    ACCEPTANCE,
    REPO_ROOT,
    SCRIPT,
    STARTER,
    only_run_dir,
    read_json,
    run_e2e,
)
