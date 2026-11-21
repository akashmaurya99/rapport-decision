"""Test-local import path setup (mirrors tests/unit/model/core/conftest.py).

The source tree is imported as ``src.*`` from the repository root, so this
conftest puts the repo root on ``sys.path``. Shared helpers live in
``head_helpers.py`` (imported as a plain top-level module, pytest's default
"prepend" import mode puts this directory on ``sys.path``).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
