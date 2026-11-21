"""Inference benchmarks: latency, throughput, memory and size (TASK-24).

Covers FR-INFER-03 (p50/p95 batch-1 latency, batch-32 throughput, peak VRAM,
peak RSS, serialized sizes) and the FR-CONFIG-02 half of FR-INFER-03's report
boundary: the report records the GPU name, compute capability, driver, CUDA
and torch versions, and can flag any difference against an earlier run.

Design (the contract left the following open; all decisions are recorded here)
------------------------------------------------------------------------------

* **One entry point.** :func:`run_benchmark` warms up ``warmup`` passes
  (default 50), runs ``iters`` timed iterations (default 1000) for every
  requested ``batch_size`` × ``precision`` and returns a single JSON-serialisable
  :class:`BenchReport`. The per-precision, per-batch breakdown lives under
  ``measurements``; the verbatim contract fields (``batch``, ``warmup``,
  ``iters``, ``p50_ms``, ``p95_ms``, ``p99_ms``, ``throughput_decisions_per_s``,
  ``peak_rss_mb``, ``peak_vram_mb``, ``environment``, ``precision_note``) are
  always present. ``p50_ms``/``p95_ms``/``p99_ms`` are keyed by batch size
  (``"1"``, ``"32"``) for the primary precision (FP32 when requested, else the
  first requested precision); ``throughput_decisions_per_s`` is the primary
  precision's batch-32 throughput (or the largest requested batch when 32 is
  not benchmarked). Values are ``None`` when a precision could not be
  measured (e.g. INT8 with no artifact and no temporary export available).
* **Determinism.** Inputs come from :func:`make_reference_batch` with the
  fixed seed ``BATCH_SEED = 20261119`` (identical to the export reference
  batch seed). The batch never leaves the process. Warmup iterations are
  discarded. ``iters``/``warmup`` are validated (positive/non-negative ints,
  bools refused) and every timed pass uses ``time.perf_counter`` — except the
  torch path on CUDA, where ``torch.cuda.Event`` with an explicit
  ``torch.cuda.synchronize`` is used (documented in ``precision_note`` and
  ``latency_method``). The ONNX Runtime path always uses ``perf_counter``.
* **One synchronized latency array.** p50/p95/p99 all use the nearest-rank
  rule on ONE synchronized timed array per (precision, batch) — CUDA events
  with an explicit ``torch.cuda.synchronize`` for torch on CUDA, perf_counter
  elsewhere. Throughput is derived from the same pass. A separate unsynchronized
  p95 pass (previously reused from ``export.benchmark_p95``) is deliberately
  removed: mixing timing methods made p95 disagree with its neighbours.
* **INT8 gate baseline.** The 2× speedup condition compares the INT8 ONNX
  Runtime session against an FP32 ONNX Runtime CPU session under the identical
  input/timing protocol — never against the torch FP32 path (a different
  runtime and kernel stack). When no FP32 ONNX artifact is available the gate
  baseline is ``None`` (not evaluated), never silently substituted.
* **CPU degradation (documented in ``precision_note``).** No CUDA-specific
  measurement is faked:

  - FP32 runs natively (CUDA kernels when present, else CPU).
  - FP16 on CUDA runs real half-precision kernels. On CPU, ``Half`` kernels
    are missing/limited, so FP16 is **emulated**: parameters are rounded to
    the fp16 grid (``.to(float16).to(float32)``) and executed in fp32.
    Latencies are indicative only and numerically equal to FP32 modulo
    rounding; activation rounding is not applied.
  - INT8 uses an ONNX INT8 artifact when one is present (explicit
    ``int8_artifact=``, ``RAPPORT_INT8_ARTIFACT``/``RAPPORT_ARTIFACTS_DIR``
    environment variables, a ``*int8*.onnx`` in the artifacts directory, or a
    temporary export — see below). The artifact is loaded with
    ``onnxruntime`` and timed; the cosine score path stays FP32 per the
    TASK-23 quantization contract. When no artifact is available, INT8
    latencies are reported as ``None`` and :func:`quantization_gate` is
    exercised as a **stand-in**: a report containing only the measurable
    keys (``fp32_p95_ms``, ``int8_p95_ms``/``int8_size_bytes`` when known)
    is fed to the gate, which refuses it for the missing choice-accuracy /
    bucket-macro-ECE keys (those are not measurable in this harness). The
    typed :class:`QuantizationReportError` is caught and recorded under
    ``quantization_gate`` with ``status="not_evaluated"`` and the exact
    missing keys — no accuracy numbers are invented. Pass ``gate_report=``
    with choice-accuracy/ECE measurements to evaluate the real five-condition
    gate; the outcome is then recorded under ``quantization_gate`` with
    ``status="passed"``/``"failed"``.
* **Serialized sizes** come from :func:`serialized_sizes`. When an artifacts
  directory is found (``artifacts_dir=`` kwarg, ``RAPPORT_ARTIFACTS_DIR``, or
  ``./artifacts``) the three sizes are read from it. Otherwise a temporary
  export is attempted (FP32 export → FP16 conversion → INT8 dynamic
  quantization into a ``tempfile.mkdtemp`` directory, deleted in a
  ``finally``); if that fails (e.g. no ``onnx`` installed), sizes are
  reported as ``None`` and the failure is recorded in ``precision_note``. The
  temporary export is how the acceptance criteria (serialized sizes for
  FP32, FP16 and INT8) are met on a CPU-only host without pre-existing
  artifacts; pass ``artifacts_dir=`` to reuse existing artifacts instead.
* **RSS / VRAM.** ``peak_rss_mb`` reports the max of the ``psutil`` current
  RSS sample (``psutil`` 7.x is present in the repo venv) and
  ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` converted to MiB (bytes on
  macOS/BSD, KiB on Linux — true process peak). On hosts without ``psutil``
  or ``resource`` the remaining source is used; on hosts with neither, 0.0 is
  reported. ``peak_vram_mb`` is ``torch.cuda.max_memory_allocated`` in MiB
  with peaks reset at the start; on CPU (or if the reading fails) it is
  ``None`` — never a fabricated number.
* **Environment block.** :func:`environment_block` fills exactly the six
  declared keys — ``gpu``, ``compute_capability``, ``driver``,
  ``cuda_version``, ``torch_version``, ``python_version`` — each in its own
  ``try/except``; any failure yields ``None`` for that key and nothing is
  raised. The driver string is read from ``pynvml`` when importable, else
  from ``nvidia-smi`` (5 s timeout), else ``None``. ``cuda_version`` records
  ``torch.version.cuda`` (``None`` on CPU-only torch builds).
* **Flagging differences.** :meth:`BenchReport.flags_against` compares two
  environment blocks (a previous report or a bare mapping) key by key and
  returns human-readable differences (``"gpu changed: None -> 'NVIDIA T4'"``);
  an empty list means the environments are identical. Passing
  ``previous_report=`` to :func:`run_benchmark` stores the flags in
  ``flags_against_previous``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.inference.export import (
    REFERENCE_SEQ_LEN,
    QuantizationReportError,
    make_reference_batch,
    quantization_gate,
    serialized_sizes,
)
from src.model.head import MAX_OPTIONS, RapportModel

BATCH_SEED = 20261119
ENVIRONMENT_KEYS = (
    "gpu",
    "compute_capability",
    "driver",
    "cuda_version",
    "torch_version",
    "python_version",
)
PRECISIONS = ("fp32", "fp16", "int8")


class BenchError(Exception):
    """Base error for the benchmark boundary."""


# -- environment block (FR-CONFIG-02) -----------------------------------------


def _driver_version() -> str | None:
    """Driver version from pynvml when present, else from nvidia-smi; None otherwise."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                version = pynvml.nvmlDeviceGetDriverVersion(handle)
            except AttributeError:  # older pynvml exposes only the system-level query
                version = pynvml.nvmlSystemGetDriverVersion()
            return str(version)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-driver-version=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


def environment_block() -> dict[str, Any]:
    """Fill the six-key environment block defensively; never raises.

    Each key is produced by its own ``try/except``: a failure yields ``None``
    for that key and is not propagated. CUDA-absent hosts therefore report
    ``gpu``/``compute_capability``/``cuda_version`` as ``None`` and the
    ``driver`` key whatever ``pynvml``/``nvidia-smi`` can see (or ``None``).
    """
    block: dict[str, Any] = {key: None for key in ENVIRONMENT_KEYS}

    def _gpu() -> str | None:
        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))

    def _compute_capability() -> str | None:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"{major}.{minor}"

    def _cuda_version() -> str | None:
        version = getattr(torch.version, "cuda", None)
        return str(version) if version else None

    fillers: dict[str, Any] = {
        "gpu": _gpu,
        "compute_capability": _compute_capability,
        "driver": _driver_version,
        "cuda_version": _cuda_version,
        "torch_version": lambda: str(torch.__version__),
        "python_version": lambda: platform.python_version(),
    }
    for key in ENVIRONMENT_KEYS:
        try:
            block[key] = fillers[key]()
        except Exception:
            block[key] = None
    return block


# -- memory probes ------------------------------------------------------------


def _peak_rss_mb() -> float:
    """Max of the psutil RSS sample and the ru_maxrss peak, in MiB.

    ``psutil`` reports the *current* RSS (sampled before returning);
    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` reports the *true process
    peak* (bytes on macOS/BSD, KiB on Linux). The maximum is reported, so the
    figure is at least the true peak whenever ``resource`` is available.
    """
    candidates: list[float] = []
    try:
        import psutil

        candidates.append(float(psutil.Process().memory_info().rss) / (1024 * 1024))
    except Exception:
        pass
    try:
        import resource
        import sys

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        candidates.append(raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024)
    except Exception:
        pass
    return max(candidates) if candidates else 0.0


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def _peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    try:
        return float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)
    except Exception:
        return None


# -- timing helpers -----------------------------------------------------------


def _nearest_rank(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile: sorted[ceil(q * n) - 1] (same rule as TASK-23)."""
    rank = math.ceil(q * len(sorted_values)) - 1
    return float(sorted_values[rank])


def _to_numpy_feed(batch: Mapping[str, Any]) -> dict[str, Any]:
    feed: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            feed[name] = value.detach().cpu().contiguous().numpy()
        else:
            feed[name] = value
    return feed


def _as_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    import numpy as np

    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    array = np.ascontiguousarray(value)
    return torch.from_numpy(array).to(device=device, dtype=torch.long)


class _TorchSession:
    """ORT-like adapter (``run(None, feed)``) so a torch model can be timed.

    ONNX Runtime accepts a numpy feed; the torch model needs tensors, so this
    adapter converts back. It lets the shared timed pass run unmodified on
    either backend.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device

    def run(self, output_names: Any, feed: Mapping[str, Any]) -> tuple[torch.Tensor]:
        with torch.no_grad():
            output = self._model(
                _as_long_tensor(feed["input_ids"], self._device),
                _as_long_tensor(feed["attention_mask"], self._device),
                _as_long_tensor(feed["option_mask"], self._device),
                _as_long_tensor(feed["task_ids"], self._device),
            )
        return (output.choice_logits,)


def _time_session(
    session: Any,
    feed: Mapping[str, Any],
    *,
    warmup: int,
    iters: int,
    use_cuda_events: bool,
) -> tuple[list[float], float]:
    """Warm up ``warmup`` discarded passes, then time ``iters`` passes.

    Returns the sorted latencies in milliseconds and the total wall-clock
    seconds of the timed pass (warmup excluded).
    """
    for _ in range(warmup):
        session.run(None, feed)

    latencies: list[float] = []
    start_all = time.perf_counter()
    if use_cuda_events:
        for _ in range(iters):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            session.run(None, feed)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(float(start_event.elapsed_time(end_event)))
    else:
        for _ in range(iters):
            start = time.perf_counter()
            session.run(None, feed)
            latencies.append((time.perf_counter() - start) * 1000.0)
    total_seconds = time.perf_counter() - start_all
    latencies.sort()
    return latencies, total_seconds


# -- artifact resolution (FR-INFER-03 serialized sizes + INT8 timing) ---------


def _artifacts_dir(artifacts_dir: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if artifacts_dir is not None:
        candidates.append(Path(artifacts_dir))
    env_value = os.environ.get("RAPPORT_ARTIFACTS_DIR")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(Path.cwd() / "artifacts")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _first_int8_onnx(directory: Path) -> Path | None:
    matches = sorted(p for p in directory.glob("*.onnx") if "int8" in p.name.lower())
    return matches[0] if matches else None


def _explicit_int8_artifact(int8_artifact: str | Path | None) -> Path | None:
    if int8_artifact is not None:
        candidate = Path(int8_artifact)
        return candidate if candidate.is_file() else None
    env_value = os.environ.get("RAPPORT_INT8_ARTIFACT")
    if env_value:
        candidate = Path(env_value)
        return candidate if candidate.is_file() else None
    return None


def _tokenizer_spec(tokenizer: Any, model: nn.Module) -> dict[str, Any]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": int(vocab_size) if vocab_size is not None else None,
    }


def _fp16_emulated_copy(model: nn.Module) -> nn.Module:
    """Rounded-to-fp16 parameters executed in fp32 (CPU FP16 emulation)."""
    clone = copy.deepcopy(model)
    clone.eval()
    with torch.no_grad():
        for parameter in clone.parameters():
            parameter.copy_(parameter.detach().to(torch.float16).to(torch.float32))
    return clone


def _measure(
    session: Any,
    batch: Mapping[str, Any],
    *,
    rows: int,
    warmup: int,
    iters: int,
    device_type: str,
) -> dict[str, Any]:
    feed = _to_numpy_feed(batch)
    use_cuda_events = device_type == "cuda" and torch.cuda.is_available()
    latencies, total_seconds = _time_session(
        session, feed, warmup=warmup, iters=iters, use_cuda_events=use_cuda_events
    )
    # p50/p95/p99 come from ONE synchronized latency array (the CUDA path uses
    # cuda events + explicit synchronize; the CPU path uses perf_counter). A
    # second, unsynchronized p95 pass mixed methods and could disagree with its
    # neighbours — that is deliberately removed.
    p50 = _nearest_rank(latencies, 0.50)
    p95 = _nearest_rank(latencies, 0.95)
    p99 = _nearest_rank(latencies, 0.99)
    throughput = float(rows * iters / total_seconds) if total_seconds > 0 else None
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_decisions_per_s": throughput,
        "iterations": int(iters),
        "warmup": int(warmup),
    }


def _skipped_measurement(warmup: int, iters: int, reason: str) -> dict[str, Any]:
    return {
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "throughput_decisions_per_s": None,
        "iterations": int(iters),
        "warmup": int(warmup),
        "skipped": reason,
    }


# -- report -------------------------------------------------------------------


@dataclass
class BenchReport:
    """JSON-serializable benchmark report (FR-INFER-03, FR-CONFIG-02).

    ``p50_ms``/``p95_ms``/``p99_ms`` are keyed by batch size (``"1"``,
    ``"32"``) for the primary precision (FP32 when requested, otherwise the
    first requested precision). Per-precision, per-batch detail lives under
    ``measurements[precision][batch]`` with the same keys plus
    ``throughput_decisions_per_s`` and ``iterations``/``warmup``.
    """

    batch: list[int]
    warmup: int
    iters: int
    p50_ms: dict[str, float | None]
    p95_ms: dict[str, float | None]
    p99_ms: dict[str, float | None]
    throughput_decisions_per_s: float | None
    peak_rss_mb: float
    peak_vram_mb: float | None
    environment: dict[str, Any]
    precision_note: str
    precision: list[str]
    serialized_sizes: dict[str, int | None]
    measurements: dict[str, dict[str, dict[str, Any]]]
    seq_len: int
    latency_method: str
    quantization_gate: dict[str, Any]
    flags_against_previous: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def flags_against(self, previous_report: "BenchReport | Mapping[str, Any]") -> list[str]:
        """Human-readable differences between two environment blocks.

        Accepts a previous :class:`BenchReport` or a bare environment mapping.
        Returns one flag per changed (or missing/added) key; an empty list
        means the environments are identical.
        """
        if isinstance(previous_report, BenchReport):
            previous = previous_report.environment
        elif isinstance(previous_report, Mapping):
            previous = dict(previous_report)
        else:
            raise TypeError(
                "previous_report must be a BenchReport or a mapping of environment keys, "
                f"got {type(previous_report).__name__}"
            )
        flags: list[str] = []
        for key in sorted(set(self.environment) | set(previous)):
            old = previous.get(key)
            new = self.environment.get(key)
            if old != new:
                flags.append(f"{key} changed: {old!r} -> {new!r}")
        return flags


# -- public entry point --------------------------------------------------------


def run_benchmark(
    model: nn.Module,
    tokenizer: Any,
    batch_sizes: tuple[int, ...] = (1, 32),
    warmup: int = 50,
    iters: int = 1000,
    precision: tuple[str, ...] = ("fp32", "fp16", "int8"),
    *,
    seq_len: int | None = None,
    artifacts_dir: str | Path | None = None,
    int8_artifact: str | Path | None = None,
    gate_report: Mapping[str, Any] | None = None,
    previous_report: "BenchReport | Mapping[str, Any] | None" = None,
) -> BenchReport:
    """Benchmark ``model`` on the fixed synthetic batch (seed ``20261119``).

    ``batch_sizes``/``precision`` are validated; unknown tokens raise
    ``ValueError``. See the module docstring for the CPU degradation rules,
    the artifact resolution order (explicit kwarg → environment variables →
    ``./artifacts`` → temporary export), and the RSS/VRAM methodology.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}")
    if tokenizer is None:
        raise TypeError("tokenizer must not be None")
    if isinstance(batch_sizes, (str, bytes)) or not batch_sizes:
        raise ValueError("batch_sizes must be a non-empty sequence of integers")
    normalized_batches: list[int] = []
    for entry in batch_sizes:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(f"batch_sizes entries must be integers, got {entry!r}")
        if entry < 1:
            raise ValueError(f"batch_sizes entries must be >= 1, got {entry}")
        if entry not in normalized_batches:
            normalized_batches.append(entry)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative integer, got {warmup!r}")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError(f"iters must be a positive integer, got {iters!r}")
    if isinstance(precision, (str, bytes)) or not precision:
        raise ValueError("precision must be a non-empty sequence of precision tokens")
    normalized_precisions: list[str] = []
    for token in precision:
        if token not in PRECISIONS:
            raise ValueError(f"unknown precision {token!r}; expected one of {PRECISIONS}")
        if token not in normalized_precisions:
            normalized_precisions.append(token)

    device = _model_device(model)
    resolved_seq_len = int(seq_len) if seq_len is not None else REFERENCE_SEQ_LEN
    if resolved_seq_len < 16:
        raise ValueError(f"seq_len must be >= 16, got {resolved_seq_len}")

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    vocab_size = int(vocab_size) if vocab_size is not None else 30_522

    model.eval()
    _reset_peak_vram(device)

    batches = {
        size: make_reference_batch(
            seed=BATCH_SEED,
            rows=size,
            width=MAX_OPTIONS,
            seq_len=resolved_seq_len,
            vocab_size=vocab_size,
        )
        for size in normalized_batches
    }

    # -- artifact resolution (sizes + INT8 session) -------------------------
    temp_dir: Path | None = None
    try:
        sizes, int8_path, sizes_note, temp_dir, fp32_ort_path = _resolve_artifacts(
            model, tokenizer, artifacts_dir, int8_artifact
        )
    # pragma: no cover - defensive; _resolve_artifacts handles its own errors
    except Exception as exc:
        sizes = {"fp32": None, "fp16": None, "int8": None}
        int8_path = None
        sizes_note = f"artifact resolution failed unexpectedly ({type(exc).__name__}: {exc})"
        temp_dir = None
        fp32_ort_path = None

    int8_session: Any = None
    if int8_path is not None:
        int8_session, int8_error = _load_ort_session(int8_path)
        if int8_error is not None:
            sizes_note = f"{sizes_note}; INT8 session failed to load ({int8_error})"

    # FP32 ORT CPU session: the INT8 speedup gate must compare INT8 against an
    # FP32 ONNX Runtime baseline under an IDENTICAL input/timing protocol —
    # not against the torch FP32 path (different runtime and kernel stack).
    fp32_ort_session: Any = None
    if fp32_ort_path is not None:
        fp32_ort_session, fp32_ort_error = _load_ort_session(fp32_ort_path)
        if fp32_ort_error is not None:
            sizes_note = (
                f"{sizes_note}; FP32-ORT baseline failed to load ({fp32_ort_error})"
            )

    notes: list[str] = []
    notes.append(
        "fp32: native torch FP32 execution (CUDA kernels when available, else CPU)."
    )

    # -- prepare sessions per precision ---------------------------------------
    sessions: dict[str, Any] = {}
    precision_skip: dict[str, str | None] = {}
    if "fp32" in normalized_precisions:
        sessions["fp32"] = _TorchSession(model, device)
        precision_skip["fp32"] = None
    if "fp16" in normalized_precisions:
        if device.type == "cuda":
            clone = copy.deepcopy(model).half().eval()
            precision_skip["fp16"] = None
            notes.append("fp16: torch half-precision kernels (CUDA).")
        else:
            clone = _fp16_emulated_copy(model)
            precision_skip["fp16"] = None
            notes.append(
                "fp16 (CPU-emulated): CPU Half kernels are limited, so parameters were "
                "rounded to the fp16 grid and executed in fp32; latencies are indicative "
                "only and numerically equal to fp32 modulo rounding (activation rounding "
                "not applied)."
            )
        sessions["fp16"] = _TorchSession(clone, device)
    if "int8" in normalized_precisions:
        if int8_session is not None:
            sessions["int8"] = int8_session
            precision_skip["int8"] = None
            if temp_dir is not None and int8_path is not None:
                int8_label = (
                    f"{int8_path.name} produced by a temporary export (removed after the run)"
                )
            else:
                int8_label = str(int8_path)
            notes.append(
                f"int8: ONNX Runtime dynamic INT8 artifact ({int8_label}); cosine score "
                "path kept FP32 per the TASK-23 quantization contract."
            )
        else:
            reason = (
                "no ONNX INT8 artifact found (looked at int8_artifact=, RAPPORT_INT8_ARTIFACT, "
                "the artifacts directory and a temporary export); INT8 metrics unavailable"
            )
            precision_skip["int8"] = reason
            notes.append(
                "int8: no ONNX INT8 artifact available; INT8 latencies are None and the "
                "quantization gate was exercised as a stand-in (it refuses report dicts "
                "without choice-accuracy/bucket-macro-ECE measurements)."
            )

    # -- measure ---------------------------------------------------------------
    device_type = device.type
    measurements: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for token in normalized_precisions:
            per_batch: dict[str, dict[str, Any]] = {}
            for size in normalized_batches:
                session = sessions.get(token)
                if session is None:
                    per_batch[str(size)] = _skipped_measurement(
                        warmup, iters, precision_skip.get(token) or "precision not measurable"
                    )
                    continue
                per_batch[str(size)] = _measure(
                    session,
                    batches[size],
                    rows=size,
                    warmup=warmup,
                    iters=iters,
                    device_type=device_type if token != "int8" else "cpu",
                )
            measurements[token] = per_batch
        # Time the FP32-ORT baseline with the SAME protocol as INT8 (CPU,
        # perf_counter, identical warmup/iters/batch).
        if fp32_ort_session is not None and 1 in normalized_batches:
            fp32_ort_measure = _measure(
                fp32_ort_session, batches[1], rows=1,
                warmup=warmup, iters=iters, device_type="cpu",
            )
            measurements.setdefault("fp32_ort", {})["1"] = fp32_ort_measure
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    primary = "fp32" if "fp32" in normalized_precisions else normalized_precisions[0]
    throughput_batch = 32 if 32 in normalized_batches else max(normalized_batches)
    primary_measurements = measurements.get(primary, {})

    def _metric(name: str) -> dict[str, float | None]:
        return {
            str(size): primary_measurements.get(str(size), {}).get(name)
            for size in normalized_batches
        }

    throughput = primary_measurements.get(str(throughput_batch), {}).get(
        "throughput_decisions_per_s"
    )

    latency_method = (
        "single synchronized latency array per (precision, batch): cuda-events with "
        "explicit synchronize for torch on CUDA, perf_counter elsewhere; warmup "
        "discarded; p50/p95/p99/throughput all derived from that same array"
    )

    notes.append(sizes_note)
    notes.append(
        "latency: one synchronized array per (precision,batch) — CUDA events with "
        "synchronize on CUDA, perf_counter elsewhere; p50/p95/p99 from that same array. "
        "The INT8 speedup gate uses the FP32 ONNX Runtime session as baseline under "
        "the identical protocol, never the torch FP32 path."
    )
    notes.append(
        "RSS: max of psutil current RSS and resource.getrusage(RUSAGE_SELF).ru_maxrss "
        "(bytes on macOS/BSD, KiB on Linux); VRAM: torch.cuda.max_memory_allocated, "
        "None on CPU."
    )
    notes.append(
        f"inputs: fixed synthetic batch via make_reference_batch(seed={BATCH_SEED}); "
        f"seq_len={resolved_seq_len}; the batch never leaves the process."
    )

    # Gate baseline: FP32 ONNX Runtime (same runtime as INT8), falling back to
    # None — never to the torch FP32 p95 — when no FP32 ONNX artifact exists.
    fp32_p95 = measurements.get("fp32_ort", {}).get("1", {}).get("p95_ms")
    int8_p95 = measurements.get("int8", {}).get("1", {}).get("p95_ms")
    gate_outcome = _evaluate_gate(gate_report, fp32_p95, int8_p95, sizes.get("int8"))

    report = BenchReport(
        batch=list(normalized_batches),
        warmup=int(warmup),
        iters=int(iters),
        p50_ms=_metric("p50_ms"),
        p95_ms=_metric("p95_ms"),
        p99_ms=_metric("p99_ms"),
        throughput_decisions_per_s=throughput,  # None for a skipped precision
        peak_rss_mb=_peak_rss_mb(),
        peak_vram_mb=_peak_vram_mb(device),
        environment=environment_block(),
        precision_note=" ".join(notes),
        precision=list(normalized_precisions),
        serialized_sizes=sizes,
        measurements=measurements,
        seq_len=resolved_seq_len,
        latency_method=latency_method,
        quantization_gate=gate_outcome,
        flags_against_previous=[],
    )
    if previous_report is not None:
        report.flags_against_previous = report.flags_against(previous_report)
    return report


# -- internals: devices, sessions, gate ----------------------------------------


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_artifacts(
    model: nn.Module,
    tokenizer: Any,
    artifacts_dir: str | Path | None,
    int8_artifact: str | Path | None,
) -> tuple[dict[str, int | None], Path | None, str, Path | None, Path | None]:
    """Return (sizes, int8_path, note, temp_dir, fp32_ort_path).

    Resolution order: explicit ``int8_artifact`` → artifacts directory (kwarg,
    ``RAPPORT_ARTIFACTS_DIR`` or ``./artifacts``) → temporary export. When a
    directory is found, all three sizes and any INT8 artifact are read from
    it; ``int8_artifact`` overrides only the INT8 path. Otherwise a temporary
    export (FP32 → FP16 → INT8) is attempted; the temp directory is returned
    so the caller can remove it in a ``finally``.
    """
    explicit_int8 = _explicit_int8_artifact(int8_artifact)
    directory = _artifacts_dir(artifacts_dir)
    if directory is not None:
        sizes = serialized_sizes(directory)
        int8_path = explicit_int8 or _first_int8_onnx(directory)
        from src.inference.export import _classify_variant

        fp32_candidates = sorted(
            p for p in directory.glob("*.onnx") if _classify_variant(p.name) == "fp32"
        )
        fp32_path = fp32_candidates[0] if fp32_candidates else None
        note = (
            f"serialized sizes read from artifacts directory {directory}; "
            f"INT8 artifact {'at ' + str(int8_path) if int8_path else 'not found'}"
        )
        return dict(sizes), int8_path, note, None, fp32_path

    # Temporary-export fallback: keeps FR-INFER-03 (sizes for all three
    # variants) satisfiable on hosts without pre-existing artifacts. Only the
    # declared RapportModel can go through the ONNX exporter.
    if not isinstance(model, RapportModel):
        return (
            {"fp32": None, "fp16": None, "int8": None},
            explicit_int8,
            "no artifacts directory found and the model is not a RapportModel; "
            "temporary export is only defined for RapportModel — serialized sizes unavailable",
            None, None,
        )
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="rapport-bench-"))
    except OSError as exc:
        return (
            {"fp32": None, "fp16": None, "int8": None},
            explicit_int8,
            f"no artifacts directory found and a temporary directory could not be created "
            f"({exc}); serialized sizes are unavailable",
            None, None,
        )
    notes: list[str] = []
    try:
        from src.inference import export as export_module

        spec = _tokenizer_spec(tokenizer, model)
        fp32_path = temp_dir / "rapport-fp32.onnx"
        export_module.export_onnx(model, spec, fp32_path)
        # FP16 conversion may refuse an invalid graph; INT8 timing and the
        # FP32-ORT gate baseline do not depend on it, so keep going and report
        # fp16 size as None in that case.
        fp16_path = temp_dir / "rapport-fp16.onnx"
        try:
            export_module.convert_fp32_to_fp16(fp32_path, fp16_path)
        except Exception as exc:  # pragma: no cover - depends on onnxruntime build
            notes.append(f"FP16 conversion refused ({type(exc).__name__}: {exc})")
        int8_path = temp_dir / "rapport-int8.onnx"
        export_module.quantize_dynamic_int8(fp32_path, int8_path)
        sizes = dict(serialized_sizes(temp_dir))
        note = (
            f"no artifacts directory found; FP32 and INT8 artifacts were temporarily "
            f"exported to {temp_dir} to measure serialized sizes and time INT8 "
            f"(removed after the run)"
        )
        if notes:
            note += "; " + " ".join(notes)
        return sizes, explicit_int8 or int8_path, note, temp_dir, fp32_path
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return (
            {"fp32": None, "fp16": None, "int8": None},
            explicit_int8,
            f"no artifacts directory found and the temporary export failed "
            f"({type(exc).__name__}: {exc}); serialized sizes and temporary INT8 timings "
            f"are unavailable",
            None, None,
        )


def _load_ort_session(int8_path: Path) -> tuple[Any, str | None]:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        return session, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _evaluate_gate(
    gate_report: Mapping[str, Any] | None,
    fp32_p95_ms: float | None,
    int8_p95_ms: float | None,
    int8_size_bytes: int | None,
) -> dict[str, Any]:
    if gate_report is not None:
        if fp32_p95_ms is None or int8_p95_ms is None:
            return {
                "status": "not_evaluated",
                "reason": "synchronized FP32-ORT and INT8 latency measurements are required",
            }
        # Caller evidence supplies accuracy/ECE/cosine fields only. Latency and
        # size are always overridden by THIS run's synchronized measurements;
        # a stale caller report can never bypass the measured baseline.
        measured_report = dict(gate_report)
        measured_report["fp32_p95_ms"] = float(fp32_p95_ms)
        measured_report["int8_p95_ms"] = float(int8_p95_ms)
        if int8_size_bytes is not None:
            measured_report["int8_size_bytes"] = int(int8_size_bytes)
        try:
            outcome = quantization_gate(measured_report)
        except QuantizationReportError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        payload = outcome.to_dict()
        payload["status"] = "passed" if outcome.passed else "failed"
        return payload
    stand_in: dict[str, Any] = {}
    if fp32_p95_ms is not None:
        stand_in["fp32_p95_ms"] = float(fp32_p95_ms)
    if int8_p95_ms is not None:
        stand_in["int8_p95_ms"] = float(int8_p95_ms)
    if int8_size_bytes is not None:
        stand_in["int8_size_bytes"] = int(int8_size_bytes)
    try:
        quantization_gate(stand_in)
    except QuantizationReportError as exc:
        return {
            "status": "not_evaluated",
            "reason": (
                f"stand-in report rejected by the gate ({exc}); choice-accuracy and "
                "bucket-macro-ECE measurements are not producible in this harness — "
                "pass gate_report= to evaluate the five-condition INT8 gate"
            ),
            "stand_in_report": stand_in,
        }
    return {
        "status": "not_evaluated",
        "reason": (
            "stand-in report unexpectedly satisfied the gate; refusing to record an "
            "unverified result — pass gate_report= to evaluate the five-condition gate"
        ),
        "stand_in_report": stand_in,
    }
