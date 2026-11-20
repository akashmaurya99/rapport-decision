#!/usr/bin/env python3
"""End-to-end acceptance runner and report (TASK-25).

One command drives every stage of the RAPPORT pipeline:

    scripts/e2e.py --config configs/experiments/acceptance.yaml --resume auto

The staged flow is acquire → clean → adapt → split → sample → train →
calibrate → report → export → bench. Every stage is idempotent with respect to
a run fingerprint recorded in ``runs/<run_id>/state.json``; the training stage
is additionally checkpointed by TASK-12's :class:`Trainer` and can resume in a
fresh process after ``os._exit(1)`` preemption (``--kill-after-step N``).

Reality check: this repository does not download a corpus. When neither
``--synthetic`` nor a real data directory is available, the runner says so out
loud and uses the deterministic synthetic fixture from ``e2e_fixtures``. It
never presents synthetic evidence as a real measurement, never fabricates CUDA
timings, and never turns a ``not-measured`` verdict into a pass. The CLI exit
code is exactly the one returned by ``src.evaluation.report.build_report``.

``--starter`` selects the 55k smoke path (with tiny synthetic rows) and emits
no acceptance verdict; it is not part of the acceptance decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The acceptance command invokes this file directly, so scripts/ is sys.path[0].
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from scripts import e2e_fixtures  # noqa: E402
from src.config.loader import ResolvedConfig  # noqa: E402
from src.config.manifest import (  # noqa: E402
    RunManifest,
    compute_config_hash,
    read_versions,
    write_run_manifest,
)
from src.data.clean.pipeline import clean_examples  # noqa: E402
from src.data.core.schema import DecisionExample  # noqa: E402
from src.data.mixture import load_preset  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    REPORT_JSON_NAME,
    Y_CRITERIA,
    build_report,
)
from src.model.core import RapportConfig, SpecConstants  # noqa: E402
from src.model.head import RapportModel  # noqa: E402
from src.ops import JsonlLogger, RunRegistry  # noqa: E402
from src.ops.registry import RunRegistryError  # noqa: E402
from src.training.core import (  # noqa: E402
    CheckpointError,
    Trainer,
    TrainingConfig,
    verify_checkpoint,
)
from src.training.core.checkpoint import step_from_filename  # noqa: E402

STAGES: tuple[str, ...] = (
    "acquire",
    "clean",
    "adapt",
    "split",
    "sample",
    "train",
    "calibrate",
    "report",
    "export",
    "bench",
)
RESUME_TOLERANCE = 1e-4
DEFAULT_SYNTHETIC_STEPS = 50
DEFAULT_STARTER_STEPS = 8


class E2EError(Exception):
    """Base typed error for the acceptance runner."""

    code = "E2E_ERROR"

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.key = key


class ConfigFingerprintMismatchError(E2EError):
    """A checkpoint belongs to a different experiment/configuration."""

    code = "E2E_CONFIG_FINGERPRINT_MISMATCH"


class StageFailureError(E2EError):
    """A pipeline stage refused to produce a consistent artifact."""

    code = "E2E_STAGE_FAILED"


@dataclass(frozen=True)
class Args:
    """Fully resolved command-line arguments."""

    config_path: Path
    runs_dir: Path
    data_dir: Path | None
    resume: str
    starter: bool
    synthetic: bool
    rows_per_family: int
    steps: int
    kill_after_step: int | None
    seed: int
    evidence_file: Path | None
    reference_losses: Path | None


@dataclass(frozen=True)
class Experiment:
    """The experiment YAML, parsed and validated at the CLI boundary."""

    path: Path
    document: dict[str, Any]
    fingerprint: str

    @property
    def arm(self) -> str:
        return str(self.document["arm"])

    @property
    def mixture(self) -> str:
        return str(self.document["mixture"])

    @property
    def mixture_rows(self) -> int:
        return int(self.document["mixture_rows"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.document["seeds"])

    def value(self, key: str, default: Any) -> Any:
        return self.document.get(key, default)


@dataclass
class RunContext:
    """Everything the stage functions need, created once per invocation."""

    args: Args
    experiment: Experiment
    mode: str
    rows: list[DecisionExample]
    data_fingerprint: str
    run_fingerprint: str
    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    resolved: ResolvedConfig
    config_hash: str
    state_path: Path
    state: dict[str, Any]
    logger: JsonlLogger
    train_logger: JsonlLogger
    resumed_from: Path | None = None

    @property
    def starter(self) -> bool:
        return self.args.starter

    @property
    def ood_source(self) -> str:
        return str(self.experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE))


@dataclass(frozen=True)
class TrainOutcome:
    """Result of the (possibly resumed) training stage."""

    model: RapportModel
    step: int
    resumed_step: int
    trajectory: dict[str, float]
    final_loss: float | None
    completed: bool


# ---------------------------------------------------------------------------
# boundary: arguments and experiment config
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> Args:
    """Parse and validate every CLI flag before touching the filesystem."""
    default_config = str(REPO_ROOT / "configs" / "experiments" / "acceptance.yaml")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=default_config, help="experiment YAML path")
    parser.add_argument(
        "--resume",
        nargs="?",
        choices=("auto", "fresh"),
        const="auto",
        default="fresh",
        help="auto discovers the newest verified checkpoint; fresh ignores old runs",
    )
    parser.add_argument("--starter", action="store_true", help="55k smoke path, no verdict")
    parser.add_argument("--synthetic", action="store_true", help="force deterministic fixtures")
    parser.add_argument("--kill-after-step", type=int, default=None, help="simulate preemption")
    parser.add_argument("--runs-dir", default=None, help="override the runs/ root")
    parser.add_argument("--data-dir", default=None, help="optional real corpus directory")
    parser.add_argument("--rows-per-family", type=int, default=40)
    parser.add_argument("--steps", type=int, default=None, help="synthetic optimizer steps")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--evidence-file", default=None, help="JSON with recorded evidence")
    parser.add_argument("--reference-losses", default=None, help="uninterrupted loss JSON")
    ns = parser.parse_args(argv)

    config_path = Path(ns.config)
    if ns.starter and ns.config == default_config:
        config_path = REPO_ROOT / "configs" / "experiments" / "starter.yaml"
    steps = ns.steps
    if steps is None:
        steps = DEFAULT_STARTER_STEPS if ns.starter else DEFAULT_SYNTHETIC_STEPS
    if not isinstance(steps, int) or steps < 1:
        raise E2EError("--steps must be an integer >= 1", key="steps")
    if not isinstance(ns.rows_per_family, int) or ns.rows_per_family < 2:
        raise E2EError("--rows-per-family must be an integer >= 2", key="rows_per_family")
    if ns.kill_after_step is not None:
        if not isinstance(ns.kill_after_step, int) or not 1 <= ns.kill_after_step < steps:
            raise E2EError(
                "--kill-after-step must be in [1, steps)", key="kill_after_step"
            )
    if ns.seed is not None and not isinstance(ns.seed, int):
        raise E2EError("--seed must be an integer", key="seed")
    runs_dir = Path(ns.runs_dir) if ns.runs_dir else REPO_ROOT / "runs"
    return Args(
        config_path=config_path,
        runs_dir=runs_dir,
        data_dir=Path(ns.data_dir) if ns.data_dir else None,
        resume=ns.resume,
        starter=bool(ns.starter),
        synthetic=bool(ns.synthetic),
        rows_per_family=ns.rows_per_family,
        steps=steps,
        kill_after_step=ns.kill_after_step,
        seed=ns.seed if ns.seed is not None else 13,
        evidence_file=Path(ns.evidence_file) if ns.evidence_file else None,
        reference_losses=Path(ns.reference_losses) if ns.reference_losses else None,
    )


def load_experiment(path: Path) -> Experiment:
    """Parse one experiment YAML, refusing malformed or partial documents."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise E2EError(f"cannot read experiment config {path}: {exc}", key="config") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise E2EError(f"invalid YAML in {path}: {exc}", key="config") from exc
    if not isinstance(document, dict):
        raise E2EError(f"experiment config {path} must be a mapping", key="config")
    required = ("arm", "mixture", "mixture_rows", "seeds")
    missing = [key for key in required if key not in document]
    if missing:
        raise E2EError(f"experiment config {path} is missing {missing}", key=missing[0])
    if not isinstance(document["mixture_rows"], int) or document["mixture_rows"] < 1:
        raise E2EError("mixture_rows must be an integer >= 1", key="mixture_rows")
    seeds = document["seeds"]
    if not isinstance(seeds, list) or not seeds or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
    ):
        raise E2EError("seeds must be a non-empty list of integers", key="seeds")
    canonical = json.dumps(document, sort_keys=True, default=str)
    return Experiment(
        path=path,
        document=document,
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


# ---------------------------------------------------------------------------
# run identity, state and resume discovery
# ---------------------------------------------------------------------------


def _quantum_rng_state() -> None:
    """Seed the three global generators; torch RNG is also checkpointed."""
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)


def _real_rows(data_dir: Path) -> list[DecisionExample]:
    """Read parquet rows from a real data directory, if one was supplied."""
    from src.data.core.parquet import read_parquet

    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise E2EError(f"data directory {data_dir} carries no parquet shards", key="data_dir")
    rows: list[DecisionExample] = []
    for path in files:
        rows.extend(read_parquet(path))
    if not rows:
        raise E2EError(f"data directory {data_dir} yielded no rows", key="data_dir")
    return rows


def prepare_data(args: Args, experiment: Experiment) -> tuple[str, list[DecisionExample]]:
    """Choose real or synthetic data and announce the fallback when used."""
    if args.synthetic:
        return "synthetic", e2e_fixtures.make_rows(
            args.rows_per_family,
            ood_source=str(experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE)),
        )
    if args.data_dir is not None and args.data_dir.is_dir():
        return "real", _real_rows(args.data_dir)
    message = (
        "no real data directory was supplied and --synthetic was not set; "
        "using the deterministic synthetic fallback (NOT real corpus evidence)"
    )
    print(f"[e2e] {message}", flush=True)
    return "synthetic", e2e_fixtures.make_rows(
        args.rows_per_family,
        ood_source=str(experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE)),
    )


def _tiny_model_config() -> RapportConfig:
    """A one-layer trunk keeps the CPU smoke run short without new modules."""
    empty_spec = SpecConstants(
        attention_total=0,
        swiglu_total=0,
        norms_total=0,
        trunk=0,
        embedding=0,
        heads_total=0,
        total=0,
        rows=(),
    )
    return RapportConfig(
        variant="e2e-smoke",
        layers=1,
        d_model=32,
        heads=2,
        head_dim=16,
        ffn=64,
        vocab_size=30_522,
        embedding_dim=32,
        spec=empty_spec,
    )


def build_context(args: Args, experiment: Experiment, mode: str,
                  rows: list[DecisionExample]) -> RunContext:
    """Create the run directory, manifest, state file and JSONL loggers."""
    seed = args.seed if args.seed != 13 else int(experiment.seeds[0])
    fingerprint = (
        e2e_fixtures.data_fingerprint(rows)
        if mode == "synthetic"
        else hashlib.sha256(
            "\n".join(row.example_id for row in rows).encode("utf-8")
        ).hexdigest()
    )
    run_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "experiment": experiment.fingerprint,
                "mode": mode,
                "rows_per_family": args.rows_per_family,
                "steps": args.steps,
                "seed": seed,
                "starter": args.starter,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    run_id = f"{experiment.arm}-{mode}-{run_fingerprint[:12]}"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_fp16 = bool(experiment.value("fp16", True))
    resolved = ResolvedConfig(
        seed=seed,
        device=device,
        data_dir=str(args.data_dir) if args.data_dir else "data",
        runs_dir=str(args.runs_dir),
        fp16=requested_fp16 and device == "cuda",
        determinism="strict",
        epochs=int(experiment.value("epochs", 8)),
        batch_size=int(experiment.value("batch_size", 64)),
        learning_rate=float(experiment.value("lr", 3.0e-4)),
        max_length=int(experiment.value("max_length", 512)),
        teacher=str(experiment.value("teacher", "Qwen/Qwen3-1.7B")),
        data_fingerprint=fingerprint,
    )
    config_hash = compute_config_hash(resolved)
    manifest = RunManifest(
        run_id=run_id,
        git_sha="unknown",  # no git invocation: provenance is recorded honestly
        git_dirty=None,
        config_hash=config_hash,
        data_fingerprint=fingerprint,
        seed=seed,
        device=device,
        compute_capability=None,
        determinism="strict",
        versions=read_versions(),
    )
    write_run_manifest(manifest, args.runs_dir)
    state_path = run_dir / "state.json"
    state = {
        "experiment_fingerprint": experiment.fingerprint,
        "run_fingerprint": run_fingerprint,
        "config_hash": config_hash,
        "data_fingerprint": fingerprint,
        "arm": experiment.arm,
        "mode": mode,
        "starter": args.starter,
        "mixture": experiment.mixture,
        "mixture_rows": experiment.mixture_rows,
        "planned_steps": _planned_steps(experiment),
        "steps": args.steps,
        "seed": seed,
        "device": device,
        "fp16": resolved.fp16,
        "ood_source": str(experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE)),
        "stages": {},
    }
    # Reuse stage markers from an identical run: re-invoking a finished run (or
    # resuming after a kill) must not redo the deterministic pre-training work.
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if (
            isinstance(previous, dict)
            and previous.get("run_fingerprint") == run_fingerprint
            and previous.get("config_hash") == config_hash
            and isinstance(previous.get("stages"), dict)
        ):
            state["stages"] = dict(previous["stages"])
    context = RunContext(
        args=args,
        experiment=experiment,
        mode=mode,
        rows=rows,
        data_fingerprint=fingerprint,
        run_fingerprint=run_fingerprint,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        resolved=resolved,
        config_hash=config_hash,
        state_path=state_path,
        state=state,
        logger=JsonlLogger(run_dir / "e2e.jsonl", run_id=run_id),
        train_logger=JsonlLogger(run_dir / "train.jsonl", run_id=run_id),
    )
    write_state(context)
    return context


def _planned_steps(experiment: Experiment) -> int:
    """The real-data step budget implied by the config (never faked)."""
    rows = experiment.mixture_rows
    batch = int(experiment.value("batch_size", 64))
    epochs = int(experiment.value("epochs", 8))
    return ((rows + batch - 1) // batch) * epochs


def stage_fingerprint(ctx: RunContext, name: str) -> str:
    return hashlib.sha256(
        f"{ctx.run_fingerprint}|{name}|{ctx.config_hash}".encode("utf-8")
    ).hexdigest()[:16]


def stage_done(ctx: RunContext, name: str) -> bool:
    stages = ctx.state.get("stages")
    return isinstance(stages, dict) and stages.get(name) == stage_fingerprint(ctx, name)


def write_state(ctx: RunContext) -> None:
    ctx.state_path.write_text(
        json.dumps(ctx.state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def mark_stage(ctx: RunContext, name: str) -> None:
    stages = ctx.state.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise StageFailureError("state stages block is not an object", key="stages")
    stages[name] = stage_fingerprint(ctx, name)
    write_state(ctx)


def write_artifact(ctx: RunContext, name: str, payload: Mapping[str, Any]) -> None:
    target = ctx.run_dir / name
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_artifact(ctx: RunContext, name: str) -> dict[str, Any]:
    path = ctx.run_dir / name
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageFailureError(f"cannot read artifact {path}: {exc}", key=name) from exc
    if not isinstance(parsed, dict):
        raise StageFailureError(f"artifact {path} is not an object", key=name)
    return parsed


def run_stage(
    ctx: RunContext,
    name: str,
    fn: Callable[[RunContext], Any],
    artifact: str | None = None,
) -> Any:
    """Run a deterministic stage once per fingerprint, then record it."""
    done = stage_done(ctx, name)
    ctx.logger.log(
        event="stage_skipped" if done else "stage_start",
        step=0,
        split="eval",
        metrics={"stage": name, "fingerprint": stage_fingerprint(ctx, name)},
    )
    result = fn(ctx)
    if not done:
        if artifact is not None and isinstance(result, Mapping):
            write_artifact(ctx, artifact, result)
        mark_stage(ctx, name)
        ctx.logger.log(event="stage_complete", step=0, split="eval",
                       metrics={"stage": name, "ok": 1})
    return result


def discover_resume(ctx: RunContext) -> Path | None:
    """Find the newest verified checkpoint whose state matches this run.

    A run with a checkpoint but a different experiment fingerprint is refused
    with a typed error rather than silently starting a fresh training run. A
    checkpoint with a missing or mismatched ``.sha256`` sidecar also surfaces
    its typed TASK-12 error before any deserialization.
    """
    if not ctx.args.runs_dir.is_dir():
        return None
    registry = RunRegistry(ctx.args.runs_dir)
    matches: list[tuple[int, Path]] = []
    mismatches: list[str] = []
    for entry in registry.entries:
        run_dir = entry.manifest_path.parent
        checkpoints = [
            path for path in run_dir.rglob("checkpoint_step*.pt") if path.is_file()
        ]
        if not checkpoints:
            continue
        state_path = run_dir / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(state, dict):
            continue
        same = (
            state.get("experiment_fingerprint") == ctx.experiment.fingerprint
            and state.get("config_hash") == ctx.config_hash
            and state.get("run_fingerprint") == ctx.run_fingerprint
        )
        if not same:
            mismatches.append(str(run_dir))
            continue
        newest = max(checkpoints, key=lambda path: step_from_filename(path) or -1)
        verify_checkpoint(newest)  # typed error on bad sidecar; never retrain silently
        step = step_from_filename(newest)
        if step is None:
            continue
        matches.append((step, newest))
    if matches:
        _, path = max(matches, key=lambda item: item[0])
        return path
    if mismatches:
        raise ConfigFingerprintMismatchError(
            "checkpoint(s) found only under a different config: " + ", ".join(mismatches),
            key="config_hash",
        )
    return None


# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------


def stage_acquire(ctx: RunContext) -> dict[str, Any]:
    families = sorted({row.task_family for row in ctx.rows})
    sources = sorted({row.source for row in ctx.rows})
    return {
        "mode": ctx.mode,
        "rows": len(ctx.rows),
        "families": families,
        "sources": sources,
        "ood_source": ctx.ood_source,
        "ood_rows": sum(1 for row in ctx.rows if row.source == ctx.ood_source),
        "fingerprint": ctx.data_fingerprint,
        "mixture_rows_requested": ctx.experiment.mixture_rows,
    }


def stage_clean(ctx: RunContext) -> dict[str, Any]:
    result = clean_examples(ctx.rows)
    return {
        "input_rows": len(ctx.rows),
        "kept_rows": len(result.kept),
        "contested_rows": len(result.contested),
        "exact_removed": result.report.exact_removed,
        "near_removed": result.report.near_removed,
        "per_rule": dict(result.report.per_rule),
    }


def stage_adapt(ctx: RunContext) -> dict[str, Any]:
    adapters = {
        "choice": "ChoiceAdapter",
        "preference": "PreferenceAdapter",
        "scoring": "ScoringAdapter",
        "probability": "ProbabilityAdapter",
        "abstention": "AbstentionAdapter",
        "risk": "RiskAdapter",
    }
    counts: dict[str, int] = {}
    for row in ctx.rows:
        counts[row.task_family] = counts.get(row.task_family, 0) + 1
    unknown = sorted(set(counts) - set(adapters))
    if unknown:
        raise StageFailureError(f"no adapter for families {unknown}", key="family")
    return {"adapters": adapters, "rows_per_family": counts, "rows": len(ctx.rows)}


def stage_split(ctx: RunContext) -> dict[str, Any]:
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    ood: list[str] = []
    for row in ctx.rows:
        if row.source == ctx.ood_source:
            ood.append(row.example_id)
            continue
        splits[row.split].append(row.example_id)
    if not splits["train"]:
        raise StageFailureError("split produced no train rows", key="train")
    return {
        "counts": {name: len(ids) for name, ids in splits.items()},
        "ood_rows": len(ood),
        "ood_source": ctx.ood_source,
        "fingerprint": ctx.data_fingerprint,
        "ids": splits,
    }


def stage_sample(ctx: RunContext) -> dict[str, Any]:
    preset = load_preset(ctx.experiment.mixture)
    return {
        "preset": preset.name,
        "preset_rows": preset.total_rows,
        "requested_rows": ctx.experiment.mixture_rows,
        "planned_steps": _planned_steps(ctx.experiment),
        "executed_steps": ctx.args.steps,
        "scale": "synthetic smoke" if ctx.mode == "synthetic" else "real corpus",
    }


def _training_config(ctx: RunContext) -> TrainingConfig:
    experiment = ctx.experiment
    return TrainingConfig(
        steps=ctx.args.steps,
        epochs=1,
        batch_size=int(experiment.value("batch_size", 64)),
        seed=ctx.resolved.seed,
        device=ctx.resolved.device,
        learning_rate=float(experiment.value("lr", 3.0e-4)),
        min_learning_rate=1.0e-5,
        weight_decay=float(experiment.value("weight_decay", 0.01)),
        warmup_ratio=float(experiment.value("warmup_ratio", 0.05)),
        grad_clip_norm=float(experiment.value("grad_clip", 1.0)),
        fp16=ctx.resolved.fp16,
        checkpoint_every=max(1, ctx.args.steps),
        checkpoint_dir=str(ctx.checkpoint_dir),
    )


def _load_trajectory(ctx: RunContext) -> dict[str, float]:
    path = ctx.run_dir / "loss_trajectory.json"
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(value) for key, value in parsed.items()}


def _save_trajectory(ctx: RunContext, trajectory: Mapping[str, float]) -> None:
    path = ctx.run_dir / "loss_trajectory.json"
    path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_train(ctx: RunContext) -> TrainOutcome:
    """Train with TASK-12's Trainer, resuming a verified checkpoint if asked."""
    train_path = ctx.run_dir / "train.json"
    if stage_done(ctx, "train") and train_path.is_file():
        artifact = read_artifact(ctx, "train.json")
        print(f"[e2e] train stage already complete at step {artifact.get('step')}", flush=True)
        return TrainOutcome(
            model=RapportModel(_tiny_model_config()),
            step=int(artifact.get("step", ctx.args.steps)),
            resumed_step=int(artifact.get("resumed_step", 0)),
            trajectory=_load_trajectory(ctx),
            final_loss=artifact.get("final_loss"),
            completed=True,
        )

    config = _training_config(ctx)
    _quantum_rng_state()
    model = RapportModel(_tiny_model_config())
    trainer = Trainer(config, model, logger=ctx.train_logger)
    resumed_step = 0
    if ctx.args.resume == "auto":
        checkpoint = discover_resume(ctx)
        if checkpoint is None:
            print("[e2e] training path: FRESH (--resume auto found no matching checkpoint)",
                  flush=True)
        else:
            # A deliberately different init makes a successful load meaningful.
            torch.manual_seed(999)
            recovered = RapportModel(_tiny_model_config())
            trainer = Trainer(config, recovered, logger=ctx.train_logger)
            resumed_step = trainer.load(checkpoint)
            ctx.resumed_from = checkpoint
            print(
                f"[e2e] training path: RESUMED from {checkpoint} at step {resumed_step}",
                flush=True,
            )
    else:
        print("[e2e] training path: FRESH (--resume auto not requested)", flush=True)

    if not ctx.resolved.fp16:
        print("[e2e] FP16 disabled: no CUDA device; training in FP32 (never fabricated)",
              flush=True)

    trajectory = _load_trajectory(ctx)
    base_seed = 10_000 + ctx.resolved.seed
    final_loss: float | None = None
    for step in range(resumed_step, ctx.args.steps):
        batch = e2e_fixtures.make_batch(base_seed + step)
        result = trainer.train_batch(batch)
        if result.loss is not None:
            trajectory[str(result.step)] = result.loss
            final_loss = result.loss
        _save_trajectory(ctx, trajectory)
        if ctx.args.kill_after_step is not None and result.step == ctx.args.kill_after_step:
            trainer.save()  # guaranteed checkpoint at the preemption step
            ctx.train_logger.log(
                event="simulated_kill",
                step=result.step,
                split="train",
                metrics={"kill_after_step": ctx.args.kill_after_step, "exit_code": 1},
            )
            print(
                f"[e2e] simulated preemption after step {result.step}; "
                "checkpoint saved; exiting 1",
                flush=True,
            )
            os._exit(1)
    trainer.save()
    outcome = TrainOutcome(
        model=trainer.model if isinstance(trainer.model, RapportModel) else model,
        step=trainer.step,
        resumed_step=resumed_step,
        trajectory=trajectory,
        final_loss=final_loss,
        completed=trainer.step >= ctx.args.steps,
    )
    write_artifact(
        ctx,
        "train.json",
        {
            "step": outcome.step,
            "resumed_step": outcome.resumed_step,
            "final_loss": outcome.final_loss,
            "completed": outcome.completed,
        },
    )
    if not outcome.completed:
        raise StageFailureError(
            f"training stopped at step {outcome.step} of {ctx.args.steps}", key="train"
        )
    return outcome


def stage_calibrate(ctx: RunContext, outcome: TrainOutcome) -> dict[str, Any]:
    return {
        "status": "not-measured",
        "reason": "synthetic smoke evidence; per-bucket calibration needs recorded dev rows",
        "steps": outcome.step,
    }


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            target[key] = dict(value) if isinstance(value, Mapping) else value
    return target


def _reference_abs_diff(ctx: RunContext, trajectory: Mapping[str, float]) -> float | None:
    if ctx.args.reference_losses is None:
        return None
    try:
        reference = json.loads(ctx.args.reference_losses.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E2EError(f"cannot read reference losses: {exc}", key="reference_losses") from exc
    if not isinstance(reference, dict):
        raise E2EError("reference losses must be a JSON object", key="reference_losses")
    shared = set(reference) & set(trajectory)
    if not shared:
        return None
    return max(abs(float(reference[key]) - float(trajectory[key])) for key in shared)


def _run_manifest(ctx: RunContext, outcome: TrainOutcome,
                  split_artifact: Mapping[str, Any]) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in outcome.model.parameters())
    evidence: dict[str, Any] = {}
    supplied: dict[str, Any] = {}
    abs_diff = _reference_abs_diff(ctx, outcome.trajectory)
    if abs_diff is not None:
        evidence["resume"] = {
            "abs_diff": abs_diff,
            "passed": abs_diff <= RESUME_TOLERANCE,
        }
    if ctx.args.evidence_file is not None:
        try:
            supplied = json.loads(ctx.args.evidence_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise E2EError(f"cannot read evidence file: {exc}", key="evidence_file") from exc
        if not isinstance(supplied, dict):
            raise E2EError("evidence file must be a JSON object", key="evidence_file")
        _deep_merge(evidence, supplied.get("evidence") or {})
    robustness: dict[str, Any] = {
        "ood": {
            "source": ctx.ood_source,
            "n_rows": int(split_artifact.get("ood_rows", 0)),
            "split_manifest_fingerprint": ctx.data_fingerprint,
            "empty": int(split_artifact.get("ood_rows", 0)) == 0,
        }
    }
    supplied_robustness = supplied.get("robustness") if ctx.args.evidence_file else None
    if isinstance(supplied_robustness, Mapping):
        _deep_merge(robustness, supplied_robustness)
    benchmark: dict[str, Any] = {
        "parameters": parameters,
        "fp32_size_bytes": parameters * 4,
        "device": ctx.resolved.device,
    }
    supplied_benchmark = supplied.get("benchmark") if ctx.args.evidence_file else None
    if isinstance(supplied_benchmark, Mapping):
        _deep_merge(benchmark, supplied_benchmark)
    baselines = supplied.get("baselines", {}) if ctx.args.evidence_file else {}
    return {
        "run_id": ctx.run_id,
        "config_hash": ctx.config_hash,
        "data_fingerprint": ctx.data_fingerprint,
        "seed": ctx.resolved.seed,
        "arm": "main",
        "device": ctx.resolved.device,
        "versions": read_versions(),
        "evidence": evidence,
        "robustness": robustness,
        "benchmark": benchmark,
        "baselines": baselines,
    }


def stage_report(ctx: RunContext, outcome: TrainOutcome,
                 split_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Render the single report; its exit code is the CLI's exit code."""
    report_json = ctx.run_dir / REPORT_JSON_NAME
    if stage_done(ctx, "report") and report_json.is_file():
        try:
            stored = json.loads(report_json.read_text(encoding="utf-8"))
            code = int(stored.get("exit_code", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise StageFailureError(f"cannot read stored report: {exc}", key="report") from exc
        return {"exit_code": code, "reused": True}

    run = _run_manifest(ctx, outcome, split_artifact)
    result = build_report(
        runs=[run],
        baselines=run["baselines"],
        ablations={},
        out_dir=ctx.run_dir,
        negative_results=None,
    )
    write_artifact(
        ctx,
        "report_summary.json",
        {
            "exit_code": result.exit_code,
            "verdicts": [row.to_dict() for row in result.verdicts],
            "negative_results": list(result.negative_results),
        },
    )
    for row in result.verdicts:
        print(f"[e2e] {row.id}: {row.status} — {row.detail}", flush=True)
    print(
        f"[e2e] report: {result.report_md_path} "
        f"({len(result.verdicts)}/{len(Y_CRITERIA)} verdict rows)",
        flush=True,
    )
    return {"exit_code": result.exit_code, "reused": False}


def stage_export(ctx: RunContext, outcome: TrainOutcome) -> dict[str, Any]:
    return {
        "status": "not-measured",
        "reason": "synthetic smoke weights are not a shippable ONNX/INT8 artifact",
        "parameters": sum(parameter.numel() for parameter in outcome.model.parameters()),
    }


def stage_bench(ctx: RunContext, outcome: TrainOutcome) -> dict[str, Any]:
    device = ctx.resolved.device
    return {
        "device": device,
        "latency_measured": False,
        "reason": "no T4/CUDA device present; T4 FP16 latency is deliberately not measured",
        "parameters": sum(parameter.numel() for parameter in outcome.model.parameters()),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def execute(args: Args) -> int:
    """Run the pipeline; return the report's exit code (0 for starter smoke)."""
    experiment = load_experiment(args.config_path)
    mode, rows = prepare_data(args, experiment)
    ctx = build_context(args, experiment, mode, rows)
    print(
        f"[e2e] run_id={ctx.run_id} arm={experiment.arm} mode={mode} "
        f"mixture={experiment.mixture} mixture_rows={experiment.mixture_rows} "
        f"planned_steps={_planned_steps(experiment)} executed_steps={args.steps}",
        flush=True,
    )
    if ctx.starter:
        print(
            "[e2e] --starter: 55k-shaped smoke path; no acceptance verdict is produced",
            flush=True,
        )
    if args.resume == "auto":
        checkpoint = discover_resume(ctx)
        if checkpoint is None:
            print("[e2e] resume mode: FRESH (no matching checkpoint found)", flush=True)
        else:
            print(f"[e2e] resume mode: RESUME ({checkpoint})", flush=True)

    split_artifact: dict[str, Any] = {}
    for name, fn, artifact in (
        ("acquire", stage_acquire, "acquire.json"),
        ("clean", stage_clean, "clean.json"),
        ("adapt", stage_adapt, "adapt.json"),
        ("split", stage_split, "split.json"),
        ("sample", stage_sample, "sample.json"),
    ):
        result = run_stage(ctx, name, fn, artifact)
        if name == "split":
            split_artifact = result

    outcome: TrainOutcome = run_stage(ctx, "train", stage_train, "train.json")
    run_stage(ctx, "calibrate", lambda _: stage_calibrate(_, outcome), "calibrate.json")

    if ctx.starter:
        write_artifact(
            ctx,
            "smoke.json",
            {
                "status": "ok",
                "starter": True,
                "verdict": False,
                "steps": outcome.step,
                "mixture_rows": experiment.mixture_rows,
            },
        )
        print("[e2e] starter smoke complete; no verdict", flush=True)
        return 0

    report = run_stage(
        ctx,
        "report",
        lambda _: stage_report(_, outcome, split_artifact),
        "report_result.json",
    )
    run_stage(ctx, "export", lambda _: stage_export(_, outcome), "export.json")
    run_stage(ctx, "bench", lambda _: stage_bench(_, outcome), "bench.json")
    exit_code = int(report.get("exit_code", 0))
    print(f"[e2e] done; exit_code={exit_code} (from report.exit_code)", flush=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """CLI boundary: typed errors print a stable code and exit non-zero."""
    try:
        return execute(parse_args(argv))
    except (E2EError, CheckpointError, RunRegistryError, OSError, ValueError) as exc:
        code = getattr(exc, "code", "E2E_UNEXPECTED")
        print(f"ERROR {code}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
