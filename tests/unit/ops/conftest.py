"""Test-local import path setup.

The source tree is imported as ``src.ops`` from the repository root. The
``src/ops`` package marker is owned by this task; this conftest puts the repo
root on ``sys.path`` mirroring the other unit-test suites.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
