# RAPPORT

RAPPORT is a compact, non-autoregressive decision model. It reads a bounded
decision state plus a variable set of candidate options and returns a choice,
per-option scores, a calibrated probability, a confidence estimate, an
abstention decision and a risk vector — in one forward pass, with no
generated text.

It is a bounded decision model, not a general assistant, and it is not aimed
at frontier-LLM parity.

## Status

**Plan phase — implementation in progress.** The architecture contract and
task plan live under [`.archgen/rapport/`](.archgen/rapport/) (read
`architecture.yaml` for the module contract and `plans/rapport.md` for the
phases and waves). Modules are landing wave by wave; anything not described
there does not exist yet.

## Layout

- `src/` — production code (source root). Feature modules live directly under
  it (`src/config`, `src/model`, ...); `src/rapport/` carries the importable
  distribution package and its version.
- `tests/unit/` — unit tests mirroring `src/` (sibling of the source tree).
- `.archgen/rapport/` — plan, architecture contract, PRD and per-task notes.

## Quickstart

```sh
python -m pip install -e .[dev]   # install RAPPORT + dev tools
python -c 'import rapport; print(rapport.__version__)'
make probe                        # capability, VRAM, SDPA backends
make e2e                          # full pipeline via scripts.e2e (lands with TASK-25)
```

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
and fail until that module lands. Pass a config with
`make train config=configs/experiments/starter.yaml`.

## Development

```sh
ruff check .   # lint
mypy src       # typecheck
pytest -q      # unit tests (none yet in wave 1)
```

CI (`.github/workflows/ci.yml`) runs lint, typecheck and pytest on push.
