#!/usr/bin/env python3
"""End-to-end acceptance runner and report (TASK-25).

One command drives the pipeline:

    scripts/e2e.py --config configs/experiments/acceptance.yaml --resume auto

The staged flow is acquire → adapt → clean → adapt → split → sample → train →
calibrate → report → export → bench. Every stage is idempotent with respect to
a run fingerprint recorded in ``runs/<run_id>/state.json``; the training stage
is additionally checkpointed by TASK-12's :class:`Trainer` and can resume in a
fresh process after ``os._exit(1)`` preemption (``--kill-after-step N``).

Reality check
-------------
This repository does not download a corpus. When neither ``--synthetic`` nor a
real data directory is available, the runner says so out loud and uses the
deterministic synthetic fixture from ``e2e_fixtures``. Synthetic runs are
reported to ``build_report(..., starter=True)``: the report is marked
``mode: starter``, carries **no acceptance verdict**, and can never be
presented as real evidence. Acceptance-mode runs with no real measurements
fail closed (non-zero exit, ``not-measured`` verdicts). CUDA/timings are never
fabricated on CPU.

``--stage NAME`` runs the pipeline *through* that stage and stops; ``data`` is
an alias for the acquire+clean+adapt chain and ``e2e`` (the default) runs all
stages. A configured list of seeds is executed as the first seed only, and the
fact is recorded in state/report — a single run never claims multi-seed
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The acceptance command invokes this file directly, so scripts/ is sys.path[0].
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import e2e_fixtures  # noqa: E402
from src.config.errors import ConfigError  # noqa: E402
from src.config.loader import ResolvedConfig, load_experiment_config  # noqa: E402
from src.config.manifest import (  # noqa: E402
    RunManifest,
    compute_config_hash,
    read_dependency_lock_digest,
    read_git_state,
    read_versions,
    write_run_manifest,
)
from src.config.seeding import set_seed  # noqa: E402
from src.data.adapters import (  # noqa: E402
    AbstentionAdapter,
    Adapter,
    ChoiceAdapter,
    PreferenceAdapter,
    ProbabilityAdapter,
    RiskAdapter,
    ScoringAdapter,
)
from src.data.clean import clean_examples  # noqa: E402
from src.data.core.normalize import normalize_example  # noqa: E402
from src.data.core.schema import DecisionExample  # noqa: E402
from src.data.mixture import (  # noqa: E402
    MixtureDeclaration,
    availability_rows_by_id,
    load_preset,
)
from src.data.splits import SplitResult, assert_no_leakage, split  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    REPORT_JSON_NAME,
    Y_CRITERIA,
    build_report,
)
from src.model.core import RapportConfig  # noqa: E402
from src.model.head import RapportModel  # noqa: E402
from src.ops import JsonlLogger, RunRegistry, verify_resume  # noqa: E402
from src.ops.logging import iter_records  # noqa: E402
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
#: ``data`` selects the whole acquire+clean+adapt chain; ``e2e`` is all stages.
STAGE_CHOICES: tuple[str, ...] = (*STAGES, "data", "e2e")
RESUME_TOLERANCE = 1e-4
DEFAULT_SYNTHETIC_STEPS = 50
DEFAULT_STARTER_STEPS = 8
REPLAY_STEPS = 8  # bounded, network-free reproducibility harness gate
#: One portable-name rule shared with the config/manifest/registry validators.
_ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: The six production adapters, keyed by the task family each one emits.
ADAPTERS: dict[str, Callable[..., Adapter]] = {
    "choice": ChoiceAdapter,
    "preference": PreferenceAdapter,
    "scoring": ScoringAdapter,
    "probability": ProbabilityAdapter,
    "abstention": AbstentionAdapter,
    "risk": RiskAdapter,
}

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
    acceptance: bool
    synthetic: bool
    rows_per_family: int
    steps: int
    kill_after_step: int | None
    seed: int | None  # None is the "no CLI override" sentinel
    evidence_file: Path | None
    reference_losses: Path | None
    stage: str
    steps_explicit: bool = False
    skip_near_dedup: bool = False


@dataclass(frozen=True)
class Experiment:
    """The validated experiment config behind the agreed config-wave API."""

    path: Path
    config: Any
    document: dict[str, Any]
    fingerprint: str

    @property
    def arm(self) -> str:
        return str(self.value("arm", "")).strip()

    @property
    def mixture(self) -> str:
        return str(self.value("mixture", "full"))

    @property
    def mixture_rows(self) -> int:
        return int(self.value("mixture_rows", 0))

    @property
    def seeds(self) -> tuple[int, ...]:
        raw = self.value("seeds", [13])
        if not isinstance(raw, (list, tuple)) or not raw:
            raise E2EError("seeds must be a non-empty list of integers", key="seeds")
        return tuple(int(seed) for seed in raw)

    def value(self, key: str, default: Any) -> Any:
        """Read one key through ``ExperimentConfig.value`` (or its bridge)."""
        getter = getattr(self.config, "value", None)
        if callable(getter):
            return getter(key, default)
        return self.document.get(key, default)

    def to_resolved(self, overrides: Mapping[str, Any] | None = None) -> ResolvedConfig:
        # Positional: the agreed API calls this parameter ``overrides`` while
        # the first implementation named it ``explicit_overrides``.
        return self.config.to_resolved(dict(overrides or {}))


def _load_experiment(path: Path) -> Experiment:
    """Load an experiment YAML through the validated config API."""
    config = load_experiment_config(path)
    document = dict(config.document)
    canonical = json.dumps(document, sort_keys=True, default=str)
    return Experiment(
        path=path,
        config=config,
        document=document,
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass
class PipelineData:
    """The real data chain, computed once and shared by the data stages."""

    raw_rows: list[Mapping[str, Any]] = field(default_factory=list)
    adapted_rows: list[DecisionExample] = field(default_factory=list)
    clean_rows: list[DecisionExample] = field(default_factory=list)
    split_result: SplitResult | None = None
    mixture_order: Any = None
    adapt: dict[str, Any] = field(default_factory=dict)
    clean: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Everything the stage functions need, created once per invocation."""

    args: Args
    experiment: Experiment
    mode: str
    data: PipelineData
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
        # Synthetic data is never acceptance evidence. --acceptance forces the
        # acceptance-mode report only so the fail-closed behaviour is testable.
        return self.args.starter or (self.mode == "synthetic" and not self.args.acceptance)

    @property
    def ood_source(self) -> str:
        return str(self.experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE))

    @property
    def split_manifest_fingerprint(self) -> str:
        return str(self.state["split_manifest_fingerprint"])


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
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="force acceptance-mode reporting even on synthetic data; with no "
             "real measurements this fails closed with not-measured verdicts",
    )
    parser.add_argument("--synthetic", action="store_true", help="force deterministic fixtures")
    parser.add_argument("--kill-after-step", type=int, default=None, help="simulate preemption")
    parser.add_argument("--runs-dir", default=None, help="override the runs/ root")
    parser.add_argument("--data-dir", default=None, help="optional real corpus directory")
    parser.add_argument("--rows-per-family", type=int, default=40)
    parser.add_argument("--steps", type=int, default=None, help="synthetic optimizer steps")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--evidence-file", default=None, help="JSON with recorded evidence")
    parser.add_argument("--reference-losses", default=None, help="uninterrupted loss JSON")
    parser.add_argument(
        "--skip-near-dedup",
        action="store_true",
        help="skip the O(n^2) near-duplicate scan for curated, already-unique corpora",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="e2e",
        help="run the pipeline through this stage and stop (default: e2e); "
             "'data' is the acquire+clean+adapt chain",
    )
    ns = parser.parse_args(argv)
    if ns.acceptance and ns.starter:
        raise E2EError("--acceptance and --starter are mutually exclusive", key="mode")

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
    if ns.seed is not None and (isinstance(ns.seed, bool) or not isinstance(ns.seed, int)):
        raise E2EError("--seed must be an integer", key="seed")
    runs_dir = Path(ns.runs_dir) if ns.runs_dir else REPO_ROOT / "runs"
    return Args(
        config_path=config_path,
        runs_dir=runs_dir,
        data_dir=Path(ns.data_dir) if ns.data_dir else None,
        resume=ns.resume,
        starter=bool(ns.starter),
        acceptance=bool(ns.acceptance),
        synthetic=bool(ns.synthetic),
        rows_per_family=ns.rows_per_family,
        steps=steps,
        kill_after_step=ns.kill_after_step,
        seed=ns.seed,
        evidence_file=Path(ns.evidence_file) if ns.evidence_file else None,
        reference_losses=Path(ns.reference_losses) if ns.reference_losses else None,
        stage=ns.stage,
        steps_explicit=ns.steps is not None,
        skip_near_dedup=bool(ns.skip_near_dedup),
    )


def _validated_arm(experiment: Experiment) -> str:
    """Reject an arm that is not a single safe path component."""
    arm = experiment.arm
    if (
        not arm
        or arm in {".", ".."}
        or ".." in arm
        or "/" in arm
        or "\\" in arm
        or arm.endswith(".")
        or not _ARM_RE.fullmatch(arm)
    ):
        raise E2EError(
            f"unsafe experiment arm {arm!r}: expected a single safe path component", key="arm"
        )
    return arm


def _git_sha() -> str:
    """Resolve the repository's HEAD sha from the repo root, honestly."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        sha = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        sha = ""
    if not sha:
        sha, _ = read_git_state()
    return sha or "unknown"


def _device_info() -> tuple[str, str, str | None]:
    """Return (device key, descriptive name, compute capability)."""
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = f"cuda:{index} {torch.cuda.get_device_name(index)}"
        major, minor = torch.cuda.get_device_capability(index)
        return "cuda", name, f"{major}.{minor}"
    return "cpu", "cpu (no CUDA device; timings not measured)", None


def _code_fingerprint() -> str:
    """Hash every executable/data input that can change training semantics."""
    sources = sorted((REPO_ROOT / "src").rglob("*.py"))
    sources += sorted((REPO_ROOT / "scripts").glob("*.py"))
    sources += sorted((REPO_ROOT / "src" / "data" / "mixture" / "presets").glob("*.yaml"))
    lock = REPO_ROOT / "requirements.lock"
    if lock.is_file():
        sources.append(lock)
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def _rows_fingerprint(rows: Sequence[DecisionExample]) -> str:
    """Content-complete training identity, including labels and split membership."""
    from src.data.core.serialize import to_dict

    canonical = sorted(
        json.dumps(to_dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# acquire → adapt → clean → split → sample (all on REAL rows)
# ---------------------------------------------------------------------------


def _raw_rows_from_parquet(data_dir: Path) -> list[dict[str, Any]]:
    """Read raw row mappings from a real data directory."""
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise E2EError(f"data directory {data_dir} carries no parquet shards", key="data_dir")
    rows: list[dict[str, Any]] = []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise E2EError(
            "pyarrow is required to read real parquet shards: "
            "pass --synthetic for the fixture path",
            key="data_dir",
        ) from exc
    for path in files:
        try:
            table = pq.read_table(str(path))
        except Exception as exc:
            raise E2EError(f"cannot read parquet shard {path}: {exc}", key="data_dir") from exc
        for row in table.to_pylist():
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise E2EError(f"data directory {data_dir} yielded no rows", key="data_dir")
    return rows


def prepare_data(args: Args, experiment: Experiment) -> tuple[str, list[dict[str, Any]]]:
    """Choose real or synthetic raw rows and announce the fallback when used."""
    ood_source = str(experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE))
    if args.synthetic:
        return "synthetic", e2e_fixtures.make_raw_rows(
            args.rows_per_family, ood_source=ood_source
        )
    if args.data_dir is not None and args.data_dir.is_dir():
        return "real", _raw_rows_from_parquet(args.data_dir)
    message = (
        "no real data directory was supplied and --synthetic was not set; "
        "using the deterministic synthetic fallback (NOT real corpus evidence)"
    )
    print(f"[e2e] {message}", flush=True)
    return "synthetic", e2e_fixtures.make_raw_rows(
        args.rows_per_family, ood_source=ood_source
    )


def _is_unified(row: Mapping[str, Any]) -> bool:
    return all(key in row for key in ("task_family", "state", "options"))


def _adapt_rows(
    mode: str, raw_rows: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[DecisionExample], dict[str, Any]]:
    """Adapt raw rows with the six production adapters (real) or schema-finalize
    synthetic fixture rows. Never fabricates an adapter conversion."""
    if mode == "synthetic":
        normalized = [normalize_example(dict(row)) for row in raw_rows]
        return normalized, {
            "status": "synthetic-fixture-schema-finalized",
            "real_adapters_applied": [],
            "adapters_available": sorted(ADAPTERS),
            "rows": len(normalized),
            "note": (
                "fixtures carry unified-layout rows; "
                "source-specific adapters are not applicable"
            ),
        }

    by_family: dict[str, list[Mapping[str, Any]]] = {key: [] for key in ADAPTERS}
    unified: list[DecisionExample] = []
    for row in raw_rows:
        family = str(row.get("task_family", "")).strip().lower()
        if _is_unified(row):
            unified.append(normalize_example(dict(row)))
        elif family in by_family:
            by_family[family].append(row)
    examples: list[DecisionExample] = list(unified)
    reports: dict[str, Any] = {}
    rejected: dict[str, int] = {}
    for family, adapter_cls in ADAPTERS.items():
        rows = by_family[family]
        adapter = adapter_cls(seed=seed)
        result = adapter.to_decision_example(rows)
        examples.extend(result.examples)
        reports[family] = result.report
        rejected[family] = len(result.rejected)
    if not examples:
        raise StageFailureError(
            "real adapters converted no rows; supply raw source shards or use --synthetic",
            key="adapt",
        )
    return examples, {
        "status": "real-adapters-applied",
        "adapters": {family: cls.__name__ for family, cls in ADAPTERS.items()},
        "real_adapters_applied": sorted(ADAPTERS),
        "rows": len(examples),
        "unified_rows": len(unified),
        "rejected_per_family": rejected,
        "reports": reports,
    }


def prepare_pipeline_data(
    args: Args, experiment: Experiment, mode: str, raw_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> PipelineData:
    """Run acquire, adapt, clean, split and sample on real rows."""
    data = PipelineData(raw_rows=list(raw_rows))
    data.adapted_rows, data.adapt = _adapt_rows(mode, raw_rows, seed)
    cleaned = clean_examples(data.adapted_rows, skip_near_dedup=args.skip_near_dedup)
    data.clean_rows = list(cleaned.kept)
    data.clean = {
        "input_rows": len(data.adapted_rows),
        "kept_rows": len(cleaned.kept),
        "contested_rows": len(cleaned.contested),
        "exact_removed": cleaned.report.exact_removed,
        "near_removed": cleaned.report.near_removed,
        "per_rule": dict(cleaned.report.per_rule),
    }
    if not data.clean_rows:
        raise StageFailureError("cleaning removed every row; nothing to split", key="clean")
    result = split(
        data.clean_rows,
        seed,
        ood_source=str(experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE)),
    )
    assert_no_leakage(result.examples)
    data.split_result = result
    data.split = {
        "counts": {name: len(result.by_split(name)) for name in ("train", "dev", "test")},
        "ood_rows": len(result.ood),
        "ood_source": result.manifest.ood_source,
        "fingerprint": result.fingerprint,
        "ids": {
            name: [row.example_id for row in result.by_split(name)]
            for name in ("train", "dev", "test")
        },
        "manifest": result.manifest.to_dict(),
    }
    preset: MixtureDeclaration = load_preset(experiment.mixture)
    if mode == "synthetic":
        order = preset.sample(seed, result.fingerprint)
        bound = False
    else:
        # Only train rows may enter optimization; dev/test/OOD remain held out.
        # The acquisition root is scanned for source manifests and every source
        # must be explicitly admitted (missing provenance fails closed).
        order = preset.sample(
            seed,
            result.fingerprint,
            available_rows=list(result.train),
            source_status=args.data_dir,
        )
        bound = True
    data.mixture_order = order
    data.sample = {
        "preset": preset.name,
        "preset_rows": preset.total_rows,
        "requested_rows": experiment.mixture_rows,
        "sampled_rows": int(order.total_rows),
        "rows_bound_to_real_examples": bound,
        "split_manifest_fingerprint": result.fingerprint,
        "scale": (
            "synthetic declaration (not real evidence)"
            if mode == "synthetic" else "real corpus"
        ),
    }
    return data


def _tiny_model_config() -> RapportConfig:
    """A one-layer trunk keeps the CPU smoke run short (synthetic only)."""
    from src.model.core import SpecConstants

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


def _model_config(ctx: RunContext) -> RapportConfig:
    """Real runs use the YAML variant; only synthetic smoke uses the tiny one."""
    if ctx.mode == "synthetic":
        return _tiny_model_config()
    variant = str(ctx.experiment.value("variant", "compact")).strip().lower()
    factories: dict[str, Callable[[], RapportConfig]] = {
        "ultra": RapportConfig.ultra,
        "compact": RapportConfig.compact,
        "large": RapportConfig.large,
    }
    if variant not in factories:
        raise E2EError(f"unknown model variant {variant!r}; expected {sorted(factories)}",
                       key="variant")
    config = factories[variant]()
    dropout = float(ctx.experiment.value("dropout", 0.0))
    return replace(config, dropout=dropout)


#: Experiment knobs the runner HONORS today, with the value it implements.
#: A config that declares anything else is refused with a precise message
#: instead of being silently ignored (an unexecuted knob is a lie).
_HONORED_EXPERIMENT_KNOBS: dict[str, Any] = {
    "init": "scratch",
    "readout": "cosine",
    "head_only": False,
    "kd": False,
    "label_smoothing": 0.0,
    "optimizer": "adamw",
    "schedule": "cosine",
    "weight_floor": 0.05,
    "weight_cap": 4.0,
}
_VARIANT_DIMENSIONS = {"ultra": (6, 320), "compact": (12, 384), "large": (16, 448)}
_ORDERING_IMPLEMENTED = ("uniform", "fingerprint")


def _unsupported_experiment_knobs(experiment: Experiment) -> list[str]:
    """Return the declared knobs this runner cannot honor (never ignore one)."""
    problems: list[str] = []
    document = experiment.document
    for key, implemented in _HONORED_EXPERIMENT_KNOBS.items():
        if key not in document:
            continue
        value = document[key]
        if isinstance(value, str) and isinstance(implemented, str):
            equal = value.strip().lower() == implemented
        elif isinstance(value, bool) or isinstance(implemented, bool):
            equal = bool(value) is bool(implemented)
        elif isinstance(value, (int, float)) and isinstance(implemented, (int, float)):
            equal = float(value) == float(implemented)
        else:
            equal = value == implemented
        if not equal:
            problems.append(f"{key}={value!r} (this runner implements {implemented!r})")
    variant = str(document.get("variant", "compact")).strip().lower()
    if variant in _VARIANT_DIMENSIONS:
        layers, hidden = _VARIANT_DIMENSIONS[variant]
        if "layers" in document and int(document["layers"]) != layers:
            problems.append(f"layers={document['layers']} (variant {variant} is {layers})")
        if "hidden" in document and int(document["hidden"]) != hidden:
            problems.append(f"hidden={document['hidden']} (variant {variant} is {hidden})")
    else:
        problems.append(f"variant={variant!r} (expected one of {sorted(_VARIANT_DIMENSIONS)})")
    ordering = document.get("ordering")
    if ordering is not None and str(ordering).strip().lower() not in _ORDERING_IMPLEMENTED:
        problems.append(
            f"ordering={ordering!r} (this runner implements "
            f"{list(_ORDERING_IMPLEMENTED)})"
        )
    context = document.get("context")
    if context not in (None, ""):
        # "512-train-1024-infer" is a summary of the two implemented budgets;
        # it is honored only when it agrees with them.
        match = re.fullmatch(r"(\d+)-train-(\d+)-infer", str(context))
        if match is None:
            problems.append(f"context={context!r} (expected '<train>-train-<infer>-infer')")
        else:
            train_len, infer_len = int(match.group(1)), int(match.group(2))
            declared_train = int(document.get("max_length", train_len))
            declared_infer = int(document.get("infer_max_length", infer_len))
            if train_len != declared_train or infer_len != declared_infer:
                problems.append(
                    f"context={context!r} disagrees with max_length/infer_max_length "
                    f"({declared_train}/{declared_infer})"
                )
    return problems


def _declared_alternates(experiment: Experiment) -> list[str]:
    """Arm lists the config declares but this invocation does not execute."""
    document = experiment.document
    names: list[str] = []
    variants = document.get("variants")
    if isinstance(variants, (list, tuple)):
        for variant in variants:
            if isinstance(variant, Mapping) and variant.get("name"):
                names.append(str(variant["name"]))
    for key in ("init_alternates", "readout_alternates", "head_only_alternates",
                "ordering_alternates"):
        values = document.get(key)
        if isinstance(values, (list, tuple)):
            names.extend(f"{key}:{value}" for value in values)
    return names


def _file_fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise E2EError(f"cannot read evidence input {path}: {exc}") from exc


def build_context(
    args: Args,
    experiment: Experiment,
    mode: str,
    data: PipelineData,
) -> RunContext:
    """Create the run directory, manifest, state file and JSONL loggers."""
    # Precedence: configured first seed < RAPPORT_SEED < explicit --seed.
    seed = args.seed if args.seed is not None else int(experiment.to_resolved().seed)
    split_result = data.split_result
    if split_result is None:  # pragma: no cover - prepare_pipeline_data always sets it
        raise StageFailureError("split result is missing", key="split")
    split_fingerprint = split_result.fingerprint
    fingerprint = _rows_fingerprint(split_result.examples)
    code_fingerprint = _code_fingerprint()
    device, device_name, compute_capability = _device_info()
    requested_fp16 = bool(experiment.value("fp16", True))
    overrides = {
        "seed": seed,
        "device": device,
        "data_dir": str(args.data_dir) if args.data_dir else "data",
        "runs_dir": str(args.runs_dir),
        "fp16": bool(requested_fp16 and device == "cuda"),
        "data_fingerprint": fingerprint,
    }
    resolved = experiment.to_resolved(overrides=overrides)
    config_hash = compute_config_hash(resolved)
    run_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "experiment": experiment.fingerprint,
                "code": code_fingerprint,
                "training_rows": fingerprint,
                "mode": mode,
                "rows_per_family": args.rows_per_family,
                "steps": args.steps,
                "seed": seed,
                "starter": args.starter,
                "report_mode": (
                    "starter"
                    if (args.starter or (mode == "synthetic" and not args.acceptance))
                    else "acceptance"
                ),
                "evidence_file": _file_fingerprint(args.evidence_file),
                "reference_losses": _file_fingerprint(args.reference_losses),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    arm = _validated_arm(experiment)
    run_id = f"{arm}-{mode}-{run_fingerprint[:12]}"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    git_sha, git_dirty = read_git_state()
    if git_sha == "unknown":
        git_sha = _git_sha()
    manifest = RunManifest(
        run_id=run_id,
        git_sha=git_sha,
        git_dirty=git_dirty,
        dependency_lock_sha256=read_dependency_lock_digest(),
        config_hash=config_hash,
        data_fingerprint=fingerprint,
        seed=seed,
        device=device_name,
        compute_capability=compute_capability,
        determinism=resolved.determinism,
        versions=read_versions(),
    )
    write_run_manifest(manifest, args.runs_dir)
    state_path = run_dir / "state.json"
    state = {
        "experiment_fingerprint": experiment.fingerprint,
        "code_fingerprint": code_fingerprint,
        "git_sha": git_sha,
        "run_fingerprint": run_fingerprint,
        "config_hash": config_hash,
        "data_fingerprint": fingerprint,
        "split_manifest_fingerprint": split_fingerprint,
        "arm": arm,
        "mode": mode,
        "starter": args.starter,
        "mixture": experiment.mixture,
        "mixture_rows": experiment.mixture_rows,
        "seeds_configured": list(experiment.seeds),
        "seed_used": seed,
        "multi_seed_evidence": False,
        "multi_seed_note": (
            f"config declares {len(experiment.seeds)} seed(s); this run executes "
            f"only seed {seed}; no multi-seed accuracy claim is made"
        ),
        "planned_steps": _planned_steps(experiment),
        "steps": args.steps,
        "device": device,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "determinism": resolved.determinism,
        "fp16": resolved.fp16,
        "ood_source": str(experiment.value("ood_source", e2e_fixtures.DEFAULT_OOD_SOURCE)),
        "stage": args.stage,
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
        data=data,
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


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically so a kill cannot leave truncated run metadata."""
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_state(ctx: RunContext) -> None:
    _atomic_json_write(ctx.state_path, ctx.state)


def mark_stage(ctx: RunContext, name: str) -> None:
    stages = ctx.state.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise StageFailureError("state stages block is not an object", key="stages")
    stages[name] = stage_fingerprint(ctx, name)
    write_state(ctx)


def write_artifact(ctx: RunContext, name: str, payload: Mapping[str, Any]) -> None:
    _atomic_json_write(ctx.run_dir / name, payload)


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
    # A cache hit must not repeat side effects. A marker whose artifact was
    # deleted/corrupted is not a hit: recompute and heal it instead.
    if done and artifact is not None and name not in {"train", "report"}:
        if (ctx.run_dir / artifact).is_file():
            ctx.logger.log(
                event="stage_skipped",
                step=0,
                split="eval",
                metrics={"stage": name, "fingerprint": stage_fingerprint(ctx, name)},
            )
            return read_artifact(ctx, artifact)
        done = False
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

    A checkpoint with a different experiment/config/data fingerprint is refused
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
        checkpoints = [path for path in run_dir.rglob("checkpoint_step*.pt") if path.is_file()]
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
            and state.get("code_fingerprint") == ctx.state.get("code_fingerprint")
            and state.get("config_hash") == ctx.config_hash
            and state.get("run_fingerprint") == ctx.run_fingerprint
            and state.get("data_fingerprint") == ctx.data_fingerprint
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
            "checkpoint(s) found only under a different config/data/code: "
            + ", ".join(mismatches),
            key="config_hash",
        )
    return None


# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------


def stage_acquire(ctx: RunContext) -> dict[str, Any]:
    rows = ctx.data.raw_rows
    families = sorted({str(row.get("task_family", "")) for row in rows})
    sources = sorted({str(row.get("source", "")) for row in rows})
    return {
        "mode": ctx.mode,
        "rows": len(rows),
        "families": families,
        "sources": sources,
        "ood_source": ctx.ood_source,
        "ood_rows": sum(1 for row in rows if row.get("source") == ctx.ood_source),
        "fingerprint": ctx.data_fingerprint,
        "mixture_rows_requested": ctx.experiment.mixture_rows,
    }


def stage_clean(ctx: RunContext) -> dict[str, Any]:
    return dict(ctx.data.clean)


def stage_adapt(ctx: RunContext) -> dict[str, Any]:
    return dict(ctx.data.adapt)


def stage_split(ctx: RunContext) -> dict[str, Any]:
    return dict(ctx.data.split)


def stage_sample(ctx: RunContext) -> dict[str, Any]:
    return dict(ctx.data.sample)


def _training_config(ctx: RunContext) -> TrainingConfig:
    experiment = ctx.experiment
    weight_mode = str(experiment.value("weight_mode", "fixed"))
    return TrainingConfig(
        steps=ctx.args.steps,
        epochs=int(experiment.value("epochs", 8)),
        batch_size=int(experiment.value("batch_size", 64)),
        seed=ctx.resolved.seed,
        device=ctx.resolved.device,
        learning_rate=ctx.resolved.learning_rate,
        head_lr=(
            float(experiment.value("head_lr", None))
            if experiment.value("head_lr", None) is not None else None
        ),
        min_learning_rate=1.0e-5,
        weight_decay=float(experiment.value("weight_decay", 0.01)),
        warmup_ratio=float(experiment.value("warmup_ratio", 0.05)),
        grad_clip_norm=float(experiment.value("grad_clip", 1.0)),
        fp16=ctx.resolved.fp16,
        # A run-wide cadence would save only at the final step, so a Colab
        # session drop would lose the entire run. Default to the trainer's
        # 500-step cadence and let the experiment override it.
        checkpoint_every=int(experiment.value("checkpoint_every", 500)),
        checkpoint_dir=str(ctx.checkpoint_dir),
        # Retention is a disk-budget control: keep the three most recent
        # checkpoints unless the config says otherwise.
        keep_last_n=int(experiment.value("keep_last_n", 3)),
        determinism=ctx.resolved.determinism,
        weight_mode=weight_mode,
    )


def _load_trajectory(ctx: RunContext) -> dict[str, float]:
    """Rebuild the loss trajectory from the fsynced append-only step log."""
    path = ctx.run_dir / "train.jsonl"
    trajectory: dict[str, float] = {}
    if not path.is_file():
        return trajectory
    try:
        for record in iter_records(path):
            step = record.get("step")
            metrics = record.get("metrics")
            if not isinstance(step, int) or isinstance(step, bool) or not isinstance(metrics, dict):
                continue
            loss = metrics.get("loss")
            if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                continue
            if np.isfinite(float(loss)):
                trajectory[str(step)] = float(loss)
    except Exception:
        return {}
    return dict(sorted(trajectory.items(), key=lambda item: int(item[0])))


def _save_trajectory(ctx: RunContext, trajectory: Mapping[str, float]) -> None:
    """Write the derived artifact once and atomically (never per step)."""
    path = ctx.run_dir / "loss_trajectory.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _real_batches(ctx: RunContext) -> list[dict[str, torch.Tensor]]:
    """Tokenize, bucket and build trainer batches from the sampled real rows."""
    from src.tokenization import TASK_ID, bucket_examples

    order = ctx.data.mixture_order
    by_id = availability_rows_by_id(ctx.data.clean_rows)
    sampled = [by_id[slot.example_id] for slot in order.slots]
    if not sampled or len(sampled) != len(order.slots):
        raise StageFailureError("mixture sampling bound to no real rows", key="sample")
    if any(row.split != "train" for row in sampled):
        raise StageFailureError(
            "dev/test/OOD row reached the optimizer; refusing to train on held-out data",
            key="sample",
        )
    budget = int(ctx.experiment.value("max_length", 512))
    bucketed = bucket_examples(
        sampled,
        budget=budget,
        batch_size=int(ctx.experiment.value("batch_size", 64)),
        manifest=ctx.state,
    )
    task_id_by_family = {
        "choice": 0, "preference": 1, "scoring": 2,
        "probability": 3, "abstention": 4, "risk": 5,
    }
    batches: list[dict[str, torch.Tensor]] = []
    for bucket in bucketed:
        width = bucket.padded_length
        size = len(bucket.examples)
        option_count = max(len(row.options) for row in bucket.examples)
        option_mask = torch.zeros(size, option_count, dtype=torch.long)
        task_ids = torch.zeros(size, dtype=torch.long)
        chosen = torch.full((size,), -1, dtype=torch.long)
        scores = torch.zeros(size, option_count)
        probabilities = torch.zeros(size)
        answerable = torch.zeros(size)
        risk = torch.zeros(size, 8)
        score_mask = torch.zeros(size, dtype=torch.bool)
        probability_mask = torch.zeros(size, dtype=torch.bool)
        answerable_mask = torch.zeros(size, dtype=torch.bool)
        risk_mask = torch.zeros(size, dtype=torch.bool)
        preference_pairs: list[tuple[int, int, int]] = []
        for index, row in enumerate(bucket.examples):
            option_mask[index, :len(row.options)] = 1
            task_ids[index] = task_id_by_family.get(row.task_family, 0)
            if row.chosen_index is not None and row.chosen_index < option_count:
                chosen[index] = row.chosen_index
            if row.task_family == "preference" and row.chosen_index is not None:
                winner = int(row.chosen_index)
                loser = 1 - winner  # preference rows always carry two options
                preference_pairs.append((index, winner, loser))
            if row.task_family == "scoring":
                score_mask[index] = True
                if row.scores is not None:
                    scores[index, :len(row.scores)] = torch.tensor(row.scores[:option_count])
            if row.task_family == "probability":
                probability_mask[index] = True
                if row.answerable is not None:
                    probabilities[index] = float(row.answerable)
            if row.task_family == "abstention":
                answerable_mask[index] = True
                if row.answerable is not None:
                    answerable[index] = float(row.answerable)
            if row.task_family == "risk":
                risk_mask[index] = True
                if row.risk_labels is not None:
                    for label in row.risk_labels:
                        match = re.search(r"(\d+)", str(label))
                        if match and int(match.group(1)) < 8:
                            risk[index, int(match.group(1))] = 1.0
        batch: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(bucket.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(bucket.attention_mask, dtype=torch.long),
            "option_mask": option_mask,
            "task_ids": task_ids,
            "chosen_index": chosen,
            "score_targets": scores,
            "score_mask": score_mask,
            "probability_target": probabilities,
            "probability_mask": probability_mask,
            "answerable_target": answerable,
            "answerable_mask": answerable_mask,
            "risk_targets": risk,
            "risk_mask": risk_mask,
            "preference_pairs": torch.tensor(
                preference_pairs, dtype=torch.long
            ).reshape(-1, 3),
        }
        # A deterministic reference for the task marker is sufficient for the
        # trainer's positional readout; width is the bucket's own padded width.
        assert width == batch["input_ids"].shape[1]
        assert TASK_ID is not None
        batches.append(batch)
    return batches


def _reproducibility_gate(ctx: RunContext) -> dict[str, Any]:
    """Bounded, network-free check that the harness is deterministic."""
    from src.ops import run_reference_workload

    recorded = run_reference_workload(ctx.resolved.seed, REPLAY_STEPS)
    result = verify_resume(
        seed=ctx.resolved.seed,
        replay_steps=REPLAY_STEPS,
        recorded_loss=recorded,
    )
    payload = result.to_dict()
    payload["scope"] = "harness determinism replay (not a trained-network claim)"
    return payload


def stage_train(ctx: RunContext) -> TrainOutcome:
    """Train with TASK-12's Trainer, resuming a verified checkpoint if asked."""
    train_path = ctx.run_dir / "train.json"
    if stage_done(ctx, "train") and train_path.is_file():
        artifact = read_artifact(ctx, "train.json")
        print(f"[e2e] train stage already complete at step {artifact.get('step')}", flush=True)
        return TrainOutcome(
            model=RapportModel(_model_config(ctx)),
            step=int(artifact.get("step", ctx.args.steps)),
            resumed_step=int(artifact.get("resumed_step", 0)),
            trajectory=_load_trajectory(ctx),
            final_loss=artifact.get("final_loss"),
            completed=True,
        )

    config = _training_config(ctx)
    model_config = _model_config(ctx)
    # Seed once at the train stage, before model init. The trainer itself never
    # reseeds (including on resume); on resume the checkpoint's RNG state is
    # restored over this seed by ``trainer.load`` below, while deterministic
    # algorithm settings stay enabled in both processes.
    set_seed(ctx.resolved.seed, ctx.resolved.determinism)
    checkpoint = discover_resume(ctx) if ctx.args.resume == "auto" else None
    resumed_step = 0
    if checkpoint is None:
        reason = (
            "--resume auto found no matching checkpoint"
            if ctx.args.resume == "auto" else "--resume auto not requested"
        )
        print(f"[e2e] training path: FRESH ({reason})", flush=True)
    else:
        # A deliberately different init makes a successful load meaningful.
        torch.manual_seed(ctx.resolved.seed + 999)

    model = RapportModel(model_config)
    trainer = Trainer(config, model, logger=ctx.train_logger)
    if checkpoint is not None:
        resumed_step = trainer.load(checkpoint)
        ctx.resumed_from = checkpoint
        print(
            f"[e2e] training path: RESUMED from {checkpoint} at step {resumed_step}",
            flush=True,
        )

    if not ctx.resolved.fp16:
        fp16_reason = (
            "no CUDA device"
            if ctx.resolved.device != "cuda"
            else "fp16 disabled by configuration"
        )
        print(
            f"[e2e] FP16 disabled: {fp16_reason}; "
            "training in FP32 (never fabricated)",
            flush=True,
        )

    real_batches = _real_batches(ctx) if ctx.mode == "real" else []
    trajectory = _load_trajectory(ctx)
    base_seed = 10_000 + ctx.resolved.seed
    final_loss: float | None = None
    # The budget counts COMPLETED optimizer steps. A non-finite batch is
    # skipped without advancing ``trainer.step`` (and without changing the
    # scheduler), so iterating a fixed number of *batches* would declare a
    # healthy run failed whenever any skip occurred. Consume batches until the
    # step budget is met, with a generous cap so a pathological stream cannot
    # loop forever; persistent divergence already raises from the trainer.
    # On resume the deterministic batch stream must continue from the restored
    # step, not restart at index 0 — otherwise the first post-resume batch is a
    # different fixed input and the trajectory cannot match the reference run.
    batches_consumed = resumed_step
    remaining = ctx.args.steps - resumed_step
    max_batches = resumed_step + max(4 * remaining, remaining + 64)
    while trainer.step < ctx.args.steps:
        if batches_consumed >= max_batches:
            raise StageFailureError(
                f"training did not reach {ctx.args.steps} optimizer steps after "
                f"{batches_consumed} batches consumed",
                key="train",
            )
        if real_batches:
            batch = real_batches[batches_consumed % len(real_batches)]
        else:
            batch = e2e_fixtures.make_batch(base_seed + batches_consumed)
        result = trainer.train_batch(batch)
        batches_consumed += 1
        if result.loss is not None:
            trajectory[str(result.step)] = result.loss
            final_loss = result.loss
        if ctx.args.kill_after_step is not None and result.step == ctx.args.kill_after_step:
            trainer.save()  # guaranteed checkpoint at the preemption step
            _save_trajectory(ctx, trajectory)
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
    _save_trajectory(ctx, trajectory)
    gate = _reproducibility_gate(ctx)
    write_artifact(ctx, "resume_gate.json", gate)
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
            "reproducibility_gate_passed": bool(gate.get("passed")),
        },
    )
    if not outcome.completed:
        raise StageFailureError(
            f"training stopped at step {outcome.step} of {ctx.args.steps}", key="train"
        )
    return outcome


def stage_calibrate(ctx: RunContext, outcome: TrainOutcome) -> dict[str, Any]:
    """Calibration is evidence-driven; with no dev records it stays not-measured."""
    supplied = _supplied_evidence(ctx)
    calibration_map = supplied.get("calibration_map") if supplied else None
    if isinstance(calibration_map, Mapping):
        thresholds = calibration_map.get("risk_thresholds")
        return {
            "status": "measured",
            "source": "supplied frozen-dev calibration map",
            "risk_thresholds": thresholds,
            "risk_thresholds_fingerprint": _fingerprint_payload(thresholds),
            "steps": outcome.step,
        }
    return {
        "status": "not-measured",
        "reason": "no frozen-dev calibration map; per-bucket thresholds need recorded dev rows",
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


def _fingerprint_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reference_abs_diff(ctx: RunContext, outcome: TrainOutcome) -> float | None:
    """Compare only a REAL resumed next step with its uninterrupted reference.

    A fresh run, a partial overlap, or a missing next step is not resume
    evidence and must remain not-measured rather than producing a Y13 pass.
    """
    if (
        ctx.args.reference_losses is None
        or ctx.resumed_from is None
        or outcome.resumed_step < 1
    ):
        return None
    try:
        reference = json.loads(ctx.args.reference_losses.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E2EError(f"cannot read reference losses: {exc}", key="reference_losses") from exc
    if not isinstance(reference, dict):
        raise E2EError("reference losses must be a JSON object", key="reference_losses")
    next_step = str(outcome.resumed_step + 1)
    if next_step not in reference or next_step not in outcome.trajectory:
        return None
    return abs(float(reference[next_step]) - float(outcome.trajectory[next_step]))


def _supplied_evidence(ctx: RunContext) -> dict[str, Any]:
    if ctx.args.evidence_file is None:
        return {}
    if ctx.mode == "synthetic" and ctx.args.acceptance:
        print(
            "[e2e] refusing operator evidence for a synthetic --acceptance run; "
            "no real measurements may back an acceptance verdict",
            flush=True,
        )
        return {}
    try:
        supplied = json.loads(ctx.args.evidence_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E2EError(f"cannot read evidence file: {exc}", key="evidence_file") from exc
    if not isinstance(supplied, dict):
        raise E2EError("evidence file must be a JSON object", key="evidence_file")
    return supplied


def _bind_fingerprints(value: Any, fingerprint: str, where: str) -> None:
    """Bind supplied evidence to this run, rejecting any conflicting claim."""
    if not isinstance(value, dict):
        return
    for key in ("split_manifest_fingerprint", "fingerprint"):
        given = value.get(key)
        if given is None:
            value[key] = fingerprint
        elif given != fingerprint:
            raise E2EError(
                f"{where}.{key} disagrees with the run split-manifest fingerprint; "
                "refusing to mix evidence across datasets",
                key="evidence_file",
            )
    for name, child in value.items():
        if name in {"split_manifest_fingerprint", "fingerprint", "measurements", "slots"}:
            continue
        if isinstance(child, dict):
            _bind_fingerprints(child, fingerprint, f"{where}.{name}")


def _risk_evidence(ctx: RunContext, supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Compute frozen-dev-threshold risk evidence from supplied records.

    ``compute_metrics(..., risk_thresholds=calibration_map.risk_thresholds)`` is
    the only route to real Y7/Y8 recall/over-refusal numbers; absent records the
    criteria remain not-measured.
    """
    from src.calibration.map import CalibrationMap
    from src.evaluation.metrics import MetricRecord, compute_metrics, keyword_risk_baseline

    calibration = supplied.get("calibration_map")
    aegis = supplied.get("aegis_records")
    xstest = supplied.get("xstest_records")
    if not (isinstance(calibration, Mapping) and aegis and xstest):
        return {}

    def records(raw: Any) -> list[MetricRecord]:
        out: list[MetricRecord] = []
        for item in raw:
            out.append(MetricRecord(
                chosen_index=int(item["chosen_index"]),
                predicted_index=int(item["predicted_index"]),
                scores=tuple(float(v) for v in item["scores"]),
                gold_scores=tuple(float(v) for v in item["gold_scores"]),
                probabilities=tuple(float(v) for v in item["probabilities"]),
                risk_targets=tuple(int(v) for v in item["risk_targets"]),
                risk_probabilities=tuple(float(v) for v in item["risk_probabilities"]),
            ))
        return out

    risk_thresholds = calibration.get("risk_thresholds")
    if not isinstance(risk_thresholds, Mapping):
        return {}
    CalibrationMap.from_dict(calibration)  # validate the supplied map
    tfp = _fingerprint_payload(risk_thresholds)
    aegis_result = compute_metrics(records(aegis), risk_thresholds=risk_thresholds)
    xstest_result = compute_metrics(records(xstest), risk_thresholds=risk_thresholds)
    keyword = keyword_risk_baseline(
        [str(item.get("text", "")) for item in aegis],
        [[int(v) for v in item["risk_targets"]] for item in aegis],
    )
    measurements = {
        category: dict(values)
        for category, values in aegis_result.risk_measurement.items()
    }
    provenance = {
        "cohort": "aegis",
        "split_manifest_fingerprint": ctx.split_manifest_fingerprint,
        "thresholds_from": "dev",
        "risk_thresholds_fingerprint": tfp,
        "risk_thresholds": dict(risk_thresholds),
        "fpr_threshold": 0.10,
        "keyword_risk_baseline": {"recall": dict(keyword.recall)},
        "measurements": measurements,
    }
    return {
        "y7": provenance,
        "y8": {
            "cohort": "xstest",
            "split_manifest_fingerprint": ctx.split_manifest_fingerprint,
            "thresholds_from": "dev",
            "risk_thresholds_fingerprint": tfp,
            "risk_thresholds": dict(risk_thresholds),
            "xstest_over_refusal_rate": xstest_result.over_refusal_rate,
            "n_safe": xstest_result.over_refusal_n_safe,
            "n_flagged": xstest_result.over_refusal_n_flagged,
        },
    }


def _baselines(ctx: RunContext) -> dict[str, Any]:
    """Real baselines carry the run's split-manifest fingerprint; synthetic none."""
    if ctx.mode != "real" or ctx.data.split_result is None:
        return {}
    from src.evaluation.baselines import run_baselines

    result = ctx.data.split_result
    metrics = run_baselines(
        list(result.train), list(result.dev), list(result.test), result.fingerprint
    )
    return {entry.baseline: entry.to_dict() for entry in metrics}


def _run_manifest(ctx: RunContext, outcome: TrainOutcome,
                  split_artifact: Mapping[str, Any]) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in outcome.model.parameters())
    evidence: dict[str, Any] = {}
    supplied = _supplied_evidence(ctx)
    abs_diff = _reference_abs_diff(ctx, outcome)
    if abs_diff is not None:
        evidence["resume"] = {
            "abs_diff": abs_diff,
            "passed": abs_diff <= RESUME_TOLERANCE,
            "resumed_from": str(ctx.resumed_from),
            "resumed_step": outcome.resumed_step,
            "next_step": outcome.resumed_step + 1,
        }
    supplied_evidence = supplied.get("evidence") or {}
    if isinstance(supplied_evidence, Mapping):
        _deep_merge(evidence, dict(supplied_evidence))
    if ctx.state.get("reproducibility_delta_points") is not None:
        evidence["reproducibility_delta_points"] = ctx.state["reproducibility_delta_points"]
    _deep_merge(evidence, _risk_evidence(ctx, supplied))
    # Bind every supplied provenance field to THIS run; conflicting claims are
    # rejected instead of being silently blended into the report.
    _bind_fingerprints(evidence, ctx.split_manifest_fingerprint, "evidence")

    ood_rows = (
        [row for row in ctx.data.split_result.examples if row.source == ctx.ood_source]
        if ctx.data.split_result is not None
        else []
    )
    robustness: dict[str, Any] = {
        "ood": {
            "source": ctx.ood_source,
            "sources": sorted({row.source for row in ood_rows}),
            "n_rows": len(ood_rows),
            "n_train_rows": sum(row.split == "train" for row in ood_rows),
            "n_dev_rows": sum(row.split == "dev" for row in ood_rows),
            "split_manifest_fingerprint": ctx.split_manifest_fingerprint,
            "empty": not ood_rows,
        }
    }
    supplied_robustness = supplied.get("robustness")
    if isinstance(supplied_robustness, Mapping):
        _deep_merge(robustness, dict(supplied_robustness))
        _bind_fingerprints(robustness, ctx.split_manifest_fingerprint, "robustness")

    benchmark: dict[str, Any] = {
        "parameters": parameters,
        "fp32_size_bytes": parameters * 4,
        "device": ctx.resolved.device,
        "device_name": ctx.state.get("device_name"),
        "compute_capability": ctx.state.get("compute_capability"),
        "precision": "fp16" if ctx.resolved.fp16 else "fp32",
        "latency_method": "not-measured: synchronized T4 FP16 timing requires CUDA",
        "batch": [],
        "p50_batch1_ms": None,
        "throughput_batch32": None,
        "reason": (
            "T4 FP16 latency was not measured on this CPU host; no timing is fabricated"
        ),
    }
    supplied_benchmark = supplied.get("benchmark")
    if isinstance(supplied_benchmark, Mapping):
        _deep_merge(benchmark, dict(supplied_benchmark))
        _bind_fingerprints(benchmark, ctx.split_manifest_fingerprint, "benchmark")

    baselines = _baselines(ctx)
    supplied_baselines = supplied.get("baselines") or {}
    if isinstance(supplied_baselines, Mapping):
        for key, value in supplied_baselines.items():
            if key not in baselines:
                baselines[key] = dict(value) if isinstance(value, Mapping) else value
                _bind_fingerprints(
                    baselines[key], ctx.split_manifest_fingerprint, f"baselines.{key}"
                )

    return {
        "run_id": ctx.run_id,
        "config_hash": ctx.config_hash,
        "data_fingerprint": ctx.data_fingerprint,
        "seed": ctx.resolved.seed,
        "arm": ctx.experiment.arm,
        "device": ctx.state.get("device_name") or ctx.resolved.device,
        "versions": read_versions(),
        "mode": "starter" if ctx.starter else "acceptance",
        "evidence": evidence,
        "robustness": robustness,
        "benchmark": benchmark,
        "baselines": baselines,
        "metrics": supplied.get("metrics") or {},
    }


def _log_eval(ctx: RunContext, result: Mapping[str, Any], mode: str) -> None:
    """Emit an honest evaluation record (never a fabricated metric)."""
    rows = [
        row.to_dict() if hasattr(row, "to_dict") else dict(row)
        for row in result.get("verdicts", ())
    ]
    measured = [row["id"] for row in rows if row.get("status") == "pass"]
    not_measured = [row["id"] for row in rows if row.get("status") == "not-measured"]
    ctx.logger.log_eval(
        step=0,
        metrics={
            "mode": mode,
            "exit_code": int(result.get("exit_code", 0)),
            "n_verdicts": len(rows),
            "n_pass": len(measured),
            "n_not_measured": len(not_measured),
            "status": "recorded" if rows else "not-measured",
        },
        split="eval",
    )


def stage_report(ctx: RunContext, outcome: TrainOutcome,
                 split_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Render the single report; its exit code is the CLI's exit code."""
    report_json = ctx.run_dir / REPORT_JSON_NAME
    expected_mode = "starter" if ctx.starter else "acceptance"
    if stage_done(ctx, "report") and report_json.is_file():
        try:
            stored = json.loads(report_json.read_text(encoding="utf-8"))
            code = int(stored.get("exit_code", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise StageFailureError(f"cannot read stored report: {exc}", key="report") from exc
        if stored.get("mode") == expected_mode:
            return {"exit_code": code, "reused": True, "mode": stored.get("mode")}
        # Never inherit an exit code across report modes.

    run = _run_manifest(ctx, outcome, split_artifact)
    result = build_report(
        runs=[run],
        baselines=run["baselines"],
        ablations={},
        out_dir=ctx.run_dir,
        negative_results=None,
        starter=ctx.starter,
    )
    write_artifact(
        ctx,
        "report_summary.json",
        {
            "exit_code": result.exit_code,
            "mode": "starter" if ctx.starter else "acceptance",
            "verdicts": [row.to_dict() for row in result.verdicts],
            "negative_results": list(result.negative_results),
            "multi_seed_evidence": False,
        },
    )
    _log_eval(ctx, {"verdicts": [row.to_dict() for row in result.verdicts],
                    "exit_code": result.exit_code},
              "starter" if ctx.starter else "acceptance")
    for row in result.verdicts:
        print(f"[e2e] {row.id}: {row.status} — {row.detail}", flush=True)
    print(
        f"[e2e] report: {result.report_md_path} "
        f"({len(result.verdicts)}/{len(Y_CRITERIA)} verdict rows) — "
        f"mode={'starter' if ctx.starter else 'acceptance'}",
        flush=True,
    )
    return {"exit_code": result.exit_code, "reused": False,
            "mode": "starter" if ctx.starter else "acceptance"}


def stage_export(ctx: RunContext, outcome: TrainOutcome) -> dict[str, Any]:
    """Report the real INT8 gate when supplied; otherwise not-measured."""
    supplied = _supplied_evidence(ctx)
    supplied_export = supplied.get("int8_export") if supplied else None
    if isinstance(supplied_export, Mapping):
        payload = dict(supplied_export)
        _bind_fingerprints(payload, ctx.split_manifest_fingerprint, "int8_export")
        return payload
    return {
        "status": "not-measured",
        "reason": "no quantized artifact was produced or validated on this host",
        "parameters": sum(parameter.numel() for parameter in outcome.model.parameters()),
    }


def stage_bench(ctx: RunContext, outcome: TrainOutcome) -> dict[str, Any]:
    """Emit an honest benchmark record; T4 FP16 timings stay not-measured on CPU."""
    device = ctx.resolved.device
    payload = {
        "device": ctx.state.get("device_name") or device,
        "compute_capability": ctx.state.get("compute_capability"),
        "latency_measured": False,
        "p50_batch1_ms": None,
        "throughput_batch32": None,
        "reason": "no T4/CUDA device present; T4 FP16 latency is deliberately not measured",
        "parameters": sum(parameter.numel() for parameter in outcome.model.parameters()),
    }
    ctx.logger.log_benchmark(
        step=0,
        metrics={"latency_measured": 0, "device": str(payload["device"]),
                 "status": "not-measured"},
        split="benchmark",
    )
    return payload


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _stage_indices(stage: str) -> list[str]:
    """Stages to execute, in order, for a --stage selector."""
    if stage == "e2e":
        return list(STAGES)
    if stage == "data":
        return ["acquire", "clean", "adapt"]
    return list(STAGES[: STAGES.index(stage) + 1])


def execute(args: Args) -> int:
    """Run the selected stages; return the report's exit code (0 for starter)."""
    experiment = _load_experiment(args.config_path)
    unsupported = _unsupported_experiment_knobs(experiment)
    if unsupported:
        raise E2EError(
            "the experiment declares knobs this runner does not implement: "
            + "; ".join(unsupported)
            + " — implement them or narrow the config instead of silently ignoring them",
            key="experiment",
        )
    alternates = _declared_alternates(experiment)
    if alternates:
        print(
            "[e2e] this invocation executes the config's top-level arm only; declared "
            f"alternates are NOT executed: {', '.join(alternates)}",
            flush=True,
        )
    seed = args.seed if args.seed is not None else int(experiment.to_resolved().seed)
    mode, raw_rows = prepare_data(args, experiment)
    if mode == "real" and not args.steps_explicit:
        # Real runs execute the declared budget, never the synthetic default.
        args = replace(args, steps=_planned_steps(experiment))
    data = prepare_pipeline_data(args, experiment, mode, raw_rows, seed)
    ctx = build_context(args, experiment, mode, data)
    print(
        f"[e2e] run_id={ctx.run_id} arm={experiment.arm} mode={mode} "
        f"mixture={experiment.mixture} mixture_rows={experiment.mixture_rows} "
        f"planned_steps={_planned_steps(experiment)} executed_steps={args.steps} "
        f"seed={seed} stage={args.stage}",
        flush=True,
    )
    if len(experiment.seeds) > 1:
        print(
            f"[e2e] config declares seeds {list(experiment.seeds)}; executing ONLY "
            f"seed {seed} (no multi-seed evidence is claimed)",
            flush=True,
        )
    if ctx.starter:
        print(
            "[e2e] starter/synthetic path: no acceptance verdict "
            "(report mode=starter, not real evidence)",
            flush=True,
        )
    if args.resume == "auto":
        checkpoint = discover_resume(ctx)
        if checkpoint is None:
            print("[e2e] resume mode: FRESH (no matching checkpoint found)", flush=True)
        else:
            print(f"[e2e] resume mode: RESUME ({checkpoint})", flush=True)

    selected = set(_stage_indices(args.stage))
    split_artifact: dict[str, Any] = {}
    for name, fn, artifact in (
        ("acquire", stage_acquire, "acquire.json"),
        ("clean", stage_clean, "clean.json"),
        ("adapt", stage_adapt, "adapt.json"),
        ("split", stage_split, "split.json"),
        ("sample", stage_sample, "sample.json"),
    ):
        if name not in selected:
            continue
        result = run_stage(ctx, name, fn, artifact)
        if name == "split":
            split_artifact = result

    outcome: TrainOutcome | None = None
    if "train" in selected:
        outcome = run_stage(ctx, "train", stage_train, "train.json")
    if "calibrate" in selected:
        live: TrainOutcome = outcome if outcome is not None else _replay_outcome(ctx)
        run_stage(ctx, "calibrate", lambda _: stage_calibrate(_, live), "calibrate.json")

    if ctx.starter:
        write_artifact(
            ctx,
            "smoke.json",
            {
                "status": "ok",
                "starter": True,
                "synthetic": ctx.mode == "synthetic",
                "verdict": False,
                "mode": "starter",
                "steps": outcome.step if outcome else 0,
                "mixture_rows": experiment.mixture_rows,
                "evidence": False,
            },
        )
        print("[e2e] starter/synthetic smoke complete; no acceptance verdict", flush=True)

    report: dict[str, Any] = {"exit_code": 0}
    if "report" in selected:
        live = outcome if outcome is not None else _replay_outcome(ctx)
        report = run_stage(
            ctx,
            "report",
            lambda _: stage_report(_, live, split_artifact),
            "report_result.json",
        )
    if "export" in selected:
        live = outcome if outcome is not None else _replay_outcome(ctx)
        run_stage(ctx, "export", lambda _: stage_export(_, live), "export.json")
    if "bench" in selected:
        live = outcome if outcome is not None else _replay_outcome(ctx)
        run_stage(ctx, "bench", lambda _: stage_bench(_, live), "bench.json")
    exit_code = int(report.get("exit_code", 0))
    print(f"[e2e] done; exit_code={exit_code} (from report.exit_code)", flush=True)
    return exit_code


def _replay_outcome(ctx: RunContext) -> TrainOutcome:
    """Reconstruct a train outcome from artifacts for downstream-only stages."""
    artifact = read_artifact(ctx, "train.json")
    return TrainOutcome(
        model=RapportModel(_model_config(ctx)),
        step=int(artifact.get("step", ctx.args.steps)),
        resumed_step=int(artifact.get("resumed_step", 0)),
        trajectory=_load_trajectory(ctx),
        final_loss=artifact.get("final_loss"),
        completed=bool(artifact.get("completed", True)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI boundary: typed errors print a stable code and exit non-zero."""
    try:
        return execute(parse_args(argv))
    except (E2EError, ConfigError, CheckpointError, RunRegistryError, OSError, ValueError) as exc:
        code = getattr(exc, "code", "E2E_UNEXPECTED")
        print(f"ERROR {code}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
