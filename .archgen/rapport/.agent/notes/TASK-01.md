# TASK-01 — notes

## What was built
- `src/config/errors.py` — typed errors with stable `code` attributes (CONFIG_UNKNOWN_KEY, CONFIG_FILE_UNREADABLE, CONFIG_YAML_INVALID, CONFIG_TYPE_MISMATCH, CONFIG_BAD_VALUE, CONFIG_BAD_ENV_VALUE, DEPENDENCY_MISSING, MANIFEST_WRITE_FAILED, …). `str(exc)` includes the code and the offending key.
- `src/config/schema.py` — flat declarative schema (one `KeySpec` row per key: kind/default/domain check). 12 keys: seed, device, data_dir, runs_dir, fp16, determinism, epochs, batch_size, learning_rate, max_length, teacher, data_fingerprint. Adding a key = one row + one line in base.yaml.
- `src/config/loader.py` — pure `load_config(path, env=None) -> ResolvedConfig`. Precedence defaults < YAML < RAPPORT_<KEY>. Unknown YAML keys fail before coercion with CONFIG_UNKNOWN_KEY naming the key. bool never accepted as int; float slots accept int YAML scalars.
- `src/config/seeding.py` — `set_seed(seed, determinism)` seeds random, numpy, torch and CUDA (`torch.cuda.manual_seed_all` when available). `strict` mode sets `CUBLAS_WORKSPACE_CONFIG` (setdefault, operator value wins), `use_deterministic_algorithms(True)`, cudnn deterministic/benchmark=False. Missing numpy/torch → DEPENDENCY_MISSING naming the package.
- `src/config/manifest.py` — `build_run_manifest` (git sha + dirty flag via subprocess; dirty is recorded, never refused; git failure → sha "unknown"), `config_hash` = sha256 of canonical YAML of the resolved config, `run_id` = UTC timestamp + hash prefix, device string + compute capability (None on CPU), versions of python/torch/transformers/datasets/onnxruntime via importlib.metadata with null on absence. `write_run_manifest` writes runs/<run_id>/manifest.json.
- `configs/base.yaml` — 12 declared keys only, no model specifics.

## Verification (venv at repo root `.venv/`, Python 3.12 — repo targets 3.11, nothing 3.12-specific used; `.venv/` is gitignored)
- `.venv/bin/python -m pytest tests/unit/config -q` → 30 passed, 1 skipped (CUDA test skips; no GPU here).
- `ruff check src/config tests/unit/config` → clean. `mypy src/config` → clean.
- Sample manifest printed: git_sha 179cc7d…, git_dirty=true, config_hash sha256, fingerprint from RAPPORT_DATA_FINGERPRINT, determinism=strict, versions recorded with null for absent transformers/datasets/onnxruntime.

## Environment caveats
- Host Python is 3.12.12 and had no pytest/numpy/yaml/torch; created venv at repo root `.venv/` and installed pytest, numpy, pyyaml, torch (CPU wheel), ruff, mypy there. (An earlier in-tree copy at `.archgen/rapport/.agent/venv` was relocated to `.venv/` by the orchestrator; `.agent/` is for notes and decisions only.) `transformers`/`datasets` deliberately NOT installed — manifest records them as null, which the spec allows ("graceful absence").
- CUDA test skips on this host; set_seed's CUDA path is implemented and will run on the T4.

## Import path strategy (TASK-02 owns src/__init__.py)
- Did NOT create src/__init__.py. `tests/unit/config/conftest.py` (owned) puts repo root on sys.path; `import src.config…` works via PEP 420 namespace packages until TASK-02 lands the marker.
