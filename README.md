# RAPPORT

RAPPORT is a compact, non-autoregressive decision model. It reads a bounded
decision state plus a variable set of candidate options and returns a choice,
per-option scores, a calibrated probability, a confidence estimate, an
abstention decision and a risk vector — in one forward pass, with no
generated text.

It is a bounded decision model, not a general assistant, and it is not aimed
at frontier-LLM parity.

## Status

**Implemented and verified; controlled training not yet run.** The architecture
contract and task plan live under [`.archgen/rapport/`](.archgen/rapport/)
(read `architecture.yaml` for the module contract and `tasks.yaml` for the
phases and waves). All modules and the end-to-end runner are present. The unit
suite is green and the synthetic starter smoke passes end to end; no real
115.6k training or T4 latency/VRAM measurement has been performed in this
repository. Acceptance verdicts fail closed: with no real measured evidence
the acceptance command exits non-zero, by design.

Known limits, stated rather than hidden:

* The trunk trains **from scratch** (`init: scratch`); no pretrained-trunk
  loader exists, so the A1 init comparison is not executable.
* Mixture ordering is the deterministic **fingerprint order**; the curriculum
  ordering arm (A7) is not implemented.
* The runner **refuses** any experiment knob it cannot honor instead of
  ignoring it, and reports declared-but-unexecuted alternates in its output.
* FP16 ONNX conversion is refused when the converter produces an invalid
  graph (it does on the current ONNX build); FP32 export and the INT8 gate
  remain available.

## Layout

- `src/` — production code (source root). Feature modules live directly under
  it (`src/config`, `src/model`, ...); `src/rapport/` carries the importable
  distribution package and its version.
- `tests/unit/` — unit tests mirroring `src/` (sibling of the source tree).
- `.archgen/rapport/` — plan, architecture contract, PRD and per-task notes.

## Quickstart

```sh
python -m pip install -e '.[dev]' -c requirements.lock  # install pinned env
python -c 'import rapport; print(rapport.__version__)'
make probe                        # capability, VRAM, SDPA backends
make e2e                          # full pipeline via scripts.e2e
```

The installed distribution exposes `src.*` plus `rapport`; the mixture presets
ship as package data, so `load_preset("starter")` works from a built wheel.

`make probe` re-verifies the runtime facts the T4 run depends on — CUDA
availability, device name, compute capability, total VRAM, SDPA backends
(flash / mem-efficient / math / cuDNN), plus the torch, CUDA and Python
builds — prints them, and writes the resolved values to
`artifacts/run-manifest.json`. It degrades gracefully on CPU-only machines
(e.g. CI). The per-run manifest with git sha, config hash and data
fingerprint is written by the config module (TASK-01); `make probe` provides
the hardware half of FR-CONFIG-02 that both compare against.

Other pipeline entry points (`data`, `split`, `train`, `calibrate`, `report`,
`export`, `bench`, `e2e`) delegate to `python -m scripts.e2e --stage <name>`
(`data` aliases the acquire+clean+adapt chain). Pass a config with
`make train config=configs/experiments/starter.yaml`. Use `--starter` for the
fast smoke path and `--acceptance` for honest fail-closed acceptance reporting;
synthetic evidence never produces real acceptance passes. Shared settings can
be overridden with the documented `RAPPORT_*` environment variables (CLI
`--seed` wins over `RAPPORT_SEED`, which wins over the config's first seed).

Training data is bound from the **train split only**; dev/test/OOD rows can
never enter optimization, and every declared source must be explicitly
admitted by its acquisition manifest before a mixture can be built.

## Development

```sh
ruff check .   # lint
mypy src       # typecheck
pytest -q      # unit tests
```

CI (`.github/workflows/ci.yml`) runs lint, typecheck and pytest on push.
