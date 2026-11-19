"""Test-local import path setup: the repo root goes on ``sys.path`` so the
source tree is imported as ``src.calibration`` from the repository root."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
