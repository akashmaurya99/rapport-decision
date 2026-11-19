"""Test-local import path setup.

The source tree is imported as ``src.training.multitask`` from the repository
root, mirroring the other unit-test suites.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
