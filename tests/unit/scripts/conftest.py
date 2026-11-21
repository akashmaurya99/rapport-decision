"""Shared subprocess helpers for the end-to-end acceptance CLI tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT = REPO_ROOT / "scripts" / "e2e.py"
ACCEPTANCE = REPO_ROOT / "configs" / "experiments" / "acceptance.yaml"
STARTER = REPO_ROOT / "configs" / "experiments" / "starter.yaml"


def run_e2e(
    runs_dir: Path,
    *extra: str,
    config: Path = ACCEPTANCE,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a fresh process; returns the completed process."""
    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(config),
        "--synthetic",
        "--runs-dir",
        str(runs_dir),
        *extra,
    ]
    environment = dict(os.environ)
    environment.update(env or {})
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def only_run_dir(runs_dir: Path) -> Path:
    directories = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(directories) == 1, directories
    return directories[0]
