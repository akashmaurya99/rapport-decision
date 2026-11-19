"""Test-local import path setup.

The source tree is imported as ``src.config`` from the repository root. The
``src/__init__.py`` package marker is owned by TASK-02, so this conftest —
which this task owns — puts the repo root on ``sys.path`` and relies on PEP
420 namespace packages until TASK-02 lands the marker file.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
