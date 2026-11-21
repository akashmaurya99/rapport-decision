"""Module-boundary audit tests.

* the calibration package must not import the evaluation module (FR-CALIB-02:
  "the calibration module computes its own ... and does not depend on the
  evaluation module");
* no fitting entry point accepts XSTest rows (FR-CALIB-03: the XSTest safe
  set is never used for fitting).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

CALIBRATION_DIR = Path(__file__).resolve().parents[3] / "src" / "calibration"

_IMPORT_RE = re.compile(
    r"^\s*(from|import)\s+([.\w]+)", re.MULTILINE
)


def _import_modules() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in sorted(CALIBRATION_DIR.glob("*.py")):
        for match in _IMPORT_RE.finditer(path.read_text(encoding="utf-8")):
            pairs.append((path.name, match.group(2)))
    return pairs


def test_calibration_never_imports_evaluation():
    for filename, module in _import_modules():
        parts = module.split(".")
        assert "evaluation" not in parts, (
            f"{filename} imports {module!r} — the calibration module must not "
            "depend on the evaluation module"
        )


def test_calibration_imports_resolve_to_stdlib_numpy_or_self():
    allowed_roots = {"__future__", "math", "json", "cmath", "dataclasses", "typing",
                     "pathlib", "collections", "enum", "functools", "itertools",
                     "numpy", "abc", "io", "re", "sys", "os", "warnings"}
    for filename, module in _import_modules():
        if module.startswith("."):  # intra-package relative import
            continue
        root = module.split(".", 1)[0]
        if root == "src":
            parts = module.split(".")
            assert parts[:2] == ["src", "calibration"], (
                f"{filename} imports {module!r} — outside the calibration package"
            )
            continue
        assert root in allowed_roots, f"{filename} imports unexpected module {module!r}"


def test_no_fitting_entry_point_accepts_xstest():
    import src.calibration as cal

    for name in ("fit_temperature_map", "fit_risk_thresholds"):
        params = inspect.signature(getattr(cal, name)).parameters
        assert not any("xstest" in p.lower() for p in params)


def test_map_audit_flag_is_false(tmp_path):
    import src.calibration as cal

    logits = [[2.0, 0.1], [0.2, 1.5], [1.0, 1.0], [3.0, 0.0], [0.0, 2.0],
              [1.5, 0.5], [0.5, 1.5], [2.5, 1.0]]
    labels = [0, 1, 0, 0, 1, 0, 1, 0]
    buckets = [cal.make_bucket_key_for_count("choice", 2)] * len(logits)
    groups = [f"g{i}" for i in range(len(logits))]
    m = cal.fit_temperature_map(logits, labels, buckets, 2, groups)
    assert m.xstest_used_for_fitting is False
    payload = m.to_dict()
    assert payload["xstest_used_for_fitting"] is False
    payload["xstest_used_for_fitting"] = True
    with pytest.raises(cal.CalibrationMapError):
        cal.CalibrationMap.from_dict(payload)
