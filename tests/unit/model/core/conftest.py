"""Test-local import path setup for core model tests.

The source tree is imported as ``src.*`` from the repository root. ``parents[4]``
from this file is the repo root (core → model → unit → tests → repo), so running
pytest with an absolute test path from any CWD resolves the imports correctly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
