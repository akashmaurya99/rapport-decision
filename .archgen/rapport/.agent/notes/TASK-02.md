# TASK-02 notes — Repository skeleton, toolchain, CI and T4 probe

Done: 2026-11-19. Single pass, one wave.

## Files shipped
- pyproject.toml — name `rapport`, version 0.1.0 (kept in sync with `src/rapport/__init__.py`), requires-python >=3.11, deps exactly per contract (torch, transformers, datasets, onnx, onnxruntime, scikit-learn, pyyaml, tqdm, psutil, numpy); `dev` extras = pytest/ruff/mypy. src-layout: `[tool.setuptools] package-dir={""="src"}` + `packages.find where=["src"]`. ruff line-length 100 (`[tool.ruff.lint] select = ["E","F","I"]`), mypy `ignore_missing_imports`, pytest `testpaths = ["tests"]`.
- src/__init__.py — docstring-only source-root marker.
- src/rapport/__init__.py — only file under `src/rapport/`; sets `__version__ = "0.1.0"`. This is the `import rapport` carrier required by the acceptance; the floor's flat `src/<module>/` layout is unchanged.
- tests/__init__.py — docstring-only test-tree marker.
- Makefile — `probe` is an inline `python -c` one-liner (no module); `data split train calibrate report export bench e2e` delegate to `python -m scripts.e2e --stage <name>` (scripts/e2e.py is TASK-25's — delegation target does not exist yet, by design). `PYTHON ?= $(shell command -v python … || command -v python3)`; `config=<path>` passes `--config` through.
- .github/workflows/ci.yml — push-triggered job on ubuntu-latest, Python 3.11, pip cache keyed on pyproject.toml; steps: install `-e .[dev]`, `ruff check .`, `mypy src`, `pytest -q`. `on:` quoted as `'on':` for YAML 1.1 parsers. No GPU assumption.
- README.md — <60 lines: what RAPPORT is (from docs/prd.md), status "plan phase — implementation in progress", pointer to `.archgen/rapport/`, quickstart with `pip install -e .[dev]`, `make probe`, `make e2e`, explicit FR-CONFIG-02 note.

## Package-layout ruling
Followed the contract's FINAL RULING: feature modules stay flat under `src/` (floor's source_root), `src/rapport/__init__.py` is the version carrier so `import rapport` works after `pip install -e .`. Verified: editable install succeeded, `print(rapport.__version__)` → `0.1.0`.

## Probe semantics (FR-CONFIG-02)
`make probe` prints cuda_available, device_name, compute_capability (via `torch.cuda.get_device_capability()`), total_vram_gb (via `get_device_properties().total_memory`), SDPA backend flags (flash / mem_efficient / math / cudnn), torch_version and cuda_version. It writes the resolved values to `artifacts/run-manifest.json`. No-CUDA machines (CI, this laptop) degrade gracefully: capability/VRAM print as `None`/`n/a`, manifest records nulls, exit 0.

SDPA flag names: torch 2.15.0 renamed `torch.backends.cuda.sdp_{flash,mem_efficient,math}_attention_enabled()` → `flash_sdp_enabled()` / `mem_efficient_sdp_enabled()` / `math_sdp_enabled()` / `cudnn_sdp_enabled()`. The probe uses `getattr` fallbacks so both name generations work.

`artifacts/run-manifest.json` is the probe's own snapshot. The full per-run manifest (git sha, config hash, data fingerprint, seed, library versions) is TASK-01's `src/config/manifest.py` (wave-1 parallel; not importable when this task was written) — TASK-01 and TASK-24 own stamping/comparing per-run manifests; this task does not touch `src/config/**`.

## Verification (shared venv `.archgen/rapport/.agent/venv`, Python 3.12.12, torch 2.15.0 CPU-only, no CUDA)
- `pip install -e .` then `-e .[dev]` — SUCCESS (network OK; deps resolved from PyPI).
- `import rapport; rapport.__version__` → `0.1.0` — OK.
- `ruff check .` → exit 0. `mypy src` → exit 0 (8 source files, includes TASK-01's in-progress files). `pytest -q` → exit 0, 30 passed 1 skipped (TASK-01's wave-1 suite was already on disk).
- `make probe` with venv on PATH → prints all facts, writes artifacts/run-manifest.json, exit 0. Plain `make probe` on this machine fails at `import torch` because bare `python`/`python3` lack torch until you install — same constraint as any fresh checkout; README quickstart installs first. CI: `python` is setup-python's, after `pip install -e .[dev]` torch is importable.

## Blockers / honest caveats
- No GPU in this environment → compute capability / VRAM path was verified only for the CPU-degraded branch (correct nulls, exit 0). The CUDA branch is the same code path with `cuda=True` and was not executed here.
- `pytest -q` initially expected to exit 5 (no tests yet); after TASK-01's suite landed in the shared tree it exits 0, so the CI pytest job was left strict (no exit-5 tolerance) and there is no skip shim.
- `artifacts/run-manifest.json` written by a local probe run remains as an untracked artifact — not committed (git left to the orchestrator).
- The python in the venv is 3.12.12 vs the contracted 3.11 target; `requires-python = ">=3.11"` covers both; CI runs 3.11.

## Out of scope honoured
No PyPI packaging, no Docker, no src/config edits, no other tasks' globs.
