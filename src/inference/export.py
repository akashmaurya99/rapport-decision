"""ONNX export and INT8 dynamic quantization (TASK-23, FR-INFER-02, E9).

This module is the artifact-producing boundary for the RAPPORT inference
stack. It exports the :class:`RapportModel` forward signature to ONNX, verifies
numerical parity against PyTorch, converts FP32 to FP16, applies *dynamic*
INT8 quantization while keeping the cosine score path in FP32, and gates the
INT8 artifact behind the five declared acceptance conditions.

Contract summary
----------------
* :func:`export_onnx` — export with dynamic axes on batch and option count,
  verify choice logits against PyTorch on a fixed 64-row synthetic reference
  batch (seed ``20261119``), assert ``|Δlogit| <= 1e-3`` and record ``opset``,
  ``sample_count=64`` and ``tolerance=1e-3`` in the returned export report.
  The report is JSON-serializable and is what the caller records in the run
  manifest.
* :func:`quantize_dynamic_int8` — ONNX Runtime dynamic quantization. The
  cosine score projection (``readout.query`` = ``W_q``, ``readout.option`` =
  ``W_o``) nodes are excluded from quantization and stay FP32. Returns a
  report with the per-head (per-output-tensor) drift measured on the
  reference batch plus the serialized sizes.
* :func:`quantization_gate` — pure decision function. Takes a report dict
  with the five-condition measurements and returns a JSON-serializable
  :class:`GateOutcome` with per-condition booleans. ``artifact`` is ``"int8"``
  only when every condition passes, otherwise ``"fp16"``.
* :func:`validate_and_select` — applies :func:`quantization_gate`, records
  the decision into a *copy* of the report (``report["decision"]``) and
  returns an :class:`ExportSelection`. With ``strict=True`` a failing gate
  raises :class:`QuantizationRejectedError` instead of silently retaining
  FP16.
* :func:`serialized_sizes` — byte sizes for the FP32/FP16/INT8 variants,
  either for one file or by scanning a directory.
* :func:`benchmark_p95` — nearest-rank p95 latency (ms) for any ORT-like
  session (``session.run(None, feed)``); used to compare INT8 p95 against
  the FP32 ONNX baseline (the 2× condition).

Acceptance bounds (``.archgen/rapport/docs/t4-optimization.md`` §8, E9)
----------------------------------------------------------------------
* choice accuracy drops by at most **1.0 point** (accuracy values are
  fractions in ``[0, 1]`` here; 1 point = ``0.01``),
* bucket-macro ECE rises by at most **0.02**,
* the cosine score path stays **FP32** (``readout.query``/``readout.option``
  MatMuls are excluded from dynamic quantization),
* serialized INT8 size is at most **35 MB** (``35 * 1024 * 1024`` bytes),
* p95 CPU latency is at least **2× faster** than the FP32 ONNX baseline
  (``fp32_p95_ms / int8_p95_ms >= 2.0``).

If any bound fails the INT8 artifact is refused and the FP16 artifact is
retained by :func:`validate_and_select` (or raises with ``strict=True``).

Export mechanics and why they look like this
--------------------------------------------
* The dynamo-based ``torch.onnx.export`` path needs the optional
  ``onnxscript`` package, which is deliberately absent (dependency budget).
  The legacy TorchScript exporter (``dynamo=False``) is used instead; this
  is recorded in the report under ``exporter``.
* The legacy exporter emits *identity* ``Cast`` nodes (same input/output
  dtype, e.g. ``float32 -> float32`` from the unconditional ``.float()``
  calls in RMSNorm and RoPE). onnxruntime's Level-1 rule-based optimizer
  mishandles these and refuses the model ("Node input … is not a graph
  input, initializer, or output of a previous node"). :func:`_strip_identity_casts`
  removes them with recursive rewiring (including ``If`` subgraph input
  captures) after ``onnx.shape_inference.infer_shapes``. The step computes
  nothing, and the post-strip 1e-3 verification is the hard guarantee that
  the rewrite is sound — any corruption fails the export loudly.
* The model's own ``forward`` locates the ``<goal>``/``<opt>`` marker
  positions with a Python loop over rows and ``nonzero``, which neither
  traces nor exports to dynamic axes. :class:`_OnnxExportModule` re-implements
  the forward with the same submodules and weights, computing marker
  positions in a traceable way: ``argmax`` for the single goal marker and a
  cumsum-rank match for the option slots (padding slots match nothing and
  resolve to index 0, exactly like the reference implementation). The
  wrapper is verified to emit exactly the same tensors as ``RapportModel``
  before export.
* Dynamic axes are declared on batch dim 0 of every input/output and on
  option-count dim 1 of ``option_mask``, ``choice_logits``, ``scores`` and
  ``score_vector``.
* Marker-position safety does not rely on the wrapper alone: the export
  report records the reference-batch fingerprint (sha256 over the four
  input tensors) so a mismatch can always be reproduced.

Everything here is CPU-only. ``onnx``/``onnxruntime`` are imported lazily
inside the functions that need them, so the pure gate/benchmark/size logic
and its tests work without them; callers get :class:`OnnxNotInstalledError`
with a clear message otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from src.model.head import MAX_OPTIONS, MIN_OPTIONS, RapportModel
from src.model.head.heads import confidence_features
from src.tokenization.tokenizer import (
    GOAL_ID,
    OPT_CLOSE_ID,
    OPT_ID,
    PAD_ID,
    TASK_ID,
)

# -- declared constants (referenced by tests and by the manifest recorder) ----

REFERENCE_BATCH_SEED = 20261119
REFERENCE_BATCH_ROWS = 64
REFERENCE_SEQ_LEN = 64
EXPORT_OPSET = 17
EXPORT_TOLERANCE = 1e-3

# Slack for float noise on the fraction comparisons so that exactly-at-bound
# measurements pass (e.g. 0.80 - 0.79 != 0.01 in binary floating point).
_BOUNDARY_EPS = 1e-9

# Gate bounds (E9 / t4-optimization.md §8). Accuracy/ECE values are fractions.
MAX_ACCURACY_DROP = 0.01
MAX_BUCKET_ECE_RISE = 0.02
MAX_INT8_BYTES = 35 * 1024 * 1024
MIN_P95_SPEEDUP = 2.0

OUTPUT_NAMES = (
    "choice_logits",
    "scores",
    "probability",
    "confidence",
    "answerable",
    "risk",
    "score_vector",
)
INPUT_NAMES = ("input_ids", "attention_mask", "option_mask", "task_ids")

_COSINE_PATH_TOKENS = ("readout.query", "readout.option", "w_q", "w_o")

__all__ = [
    "EXPORT_OPSET",
    "EXPORT_TOLERANCE",
    "INPUT_NAMES",
    "MAX_ACCURACY_DROP",
    "MAX_BUCKET_ECE_RISE",
    "MAX_INT8_BYTES",
    "MIN_P95_SPEEDUP",
    "OUTPUT_NAMES",
    "REFERENCE_BATCH_ROWS",
    "REFERENCE_BATCH_SEED",
    "REFERENCE_SEQ_LEN",
    "ExportError",
    "ExportSelection",
    "GateOutcome",
    "OnnxExportMismatchError",
    "OnnxNotInstalledError",
    "QuantizationRejectedError",
    "QuantizationReportError",
    "benchmark_p95",
    "convert_fp32_to_fp16",
    "export_onnx",
    "make_reference_batch",
    "quantization_gate",
    "quantize_dynamic_int8",
    "serialized_sizes",
    "tensor_fingerprint",
    "validate_and_select",
]


# -- errors -------------------------------------------------------------------


class ExportError(Exception):
    """Base error for the export/quantization boundary."""


class OnnxNotInstalledError(ExportError):
    """``onnx``/``onnxruntime`` (or ``onnxruntime.quantization``) is missing."""


class OnnxExportMismatchError(ExportError):
    """Exported logits differ from PyTorch by more than the tolerance."""

    def __init__(self, message: str, *, max_abs_logit_diff: float, tolerance: float) -> None:
        super().__init__(message)
        self.max_abs_logit_diff = max_abs_logit_diff
        self.tolerance = tolerance


class QuantizationReportError(ExportError):
    """The gate input report is missing keys or carries out-of-contract values."""


class QuantizationRejectedError(ExportError):
    """Strict mode: the INT8 artifact was refused by the five-condition gate."""

    def __init__(self, message: str, *, gate: "GateOutcome") -> None:
        super().__init__(message)
        self.gate = gate


# -- lazy onnx access ----------------------------------------------------------


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - exercised only without onnx
        raise OnnxNotInstalledError(
            "onnx is not installed; install onnx and onnxruntime to run ONNX export"
        ) from exc
    return onnx


def _require_ort() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only without onnxruntime
        raise OnnxNotInstalledError(
            "onnxruntime is not installed; install onnxruntime to run ONNX verification"
        ) from exc
    return onnxruntime


# -- JSON helpers ---------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Recursively coerce ``value`` into JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True)


# -- reference batch --------------------------------------------------------------


def tensor_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Deterministic sha256 over a mapping of tensor-like inputs.

    Torch tensors are detached to CPU and serialized as little-endian
    C-contiguous bytes together with their name, dtype and shape, so the
    fingerprint changes with *any* bit difference in the reference batch.
    """
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if isinstance(tensor, Tensor):
            arr = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(arr.shape)).encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(arr.numpy().astype("<i8", copy=False).tobytes())
        else:  # numpy array passed directly
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.tobytes())
    return digest.hexdigest()


def make_reference_batch(
    seed: int = REFERENCE_BATCH_SEED,
    rows: int = REFERENCE_BATCH_ROWS,
    width: int = MAX_OPTIONS,
    seq_len: int = REFERENCE_SEQ_LEN,
    vocab_size: int = 30_522,
    num_tasks: int = 6,
) -> dict[str, Tensor]:
    """Build the fixed synthetic reference batch (seeded, deterministic).

    Layout per row: ``<task>`` at position 0, then ``<goal>``, ``state`` filler
    tokens, then per real option ``<opt> ... </opt>``; trailing padding uses
    ``PAD_ID`` with ``attention_mask == 0``. Per-row option counts cycle
    through ``2..width`` so the option-count contract is exercised. The
    returned mapping carries ``input_ids``/``attention_mask``/``option_mask``/
    ``task_ids`` only — the batch fingerprint is recorded in the export
    report, the batch itself never leaves the process.
    """
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    if not MIN_OPTIONS <= width <= MAX_OPTIONS:
        raise ValueError(f"width must lie in [{MIN_OPTIONS}, {MAX_OPTIONS}], got {width}")
    if seq_len < 16:
        raise ValueError(f"seq_len must be >= 16, got {seq_len}")
    if vocab_size <= OPT_CLOSE_ID + 1:
        raise ValueError(f"vocab_size must be > {OPT_CLOSE_ID + 1}, got {vocab_size}")
    if num_tasks < 1:
        raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")

    generator = torch.Generator().manual_seed(seed)
    ids = torch.full((rows, seq_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((rows, seq_len), dtype=torch.long)
    option_mask = torch.zeros((rows, width), dtype=torch.long)
    task_ids = torch.arange(rows, dtype=torch.long) % num_tasks

    for row in range(rows):
        count = MIN_OPTIONS + (row % (width - MIN_OPTIONS + 1))
        option_mask[row, :count] = 1
        pos = 0
        ids[row, pos] = TASK_ID
        pos += 1
        tokens: list[int] = [GOAL_ID]
        state_len = 4 + row % 5
        tokens += torch.randint(7, vocab_size, (state_len,), generator=generator).tolist()
        for _ in range(count):
            tokens.append(OPT_ID)
            option_len = 2 + int(torch.randint(0, 3, (1,), generator=generator))
            tokens += torch.randint(7, vocab_size, (option_len,), generator=generator).tolist()
            tokens.append(OPT_CLOSE_ID)
        tokens = tokens[: seq_len - 1]
        for token in tokens:
            ids[row, pos] = token
            pos += 1
        attention_mask[row, :pos] = 1

    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "option_mask": option_mask,
        "task_ids": task_ids,
    }


# -- ONNX-exportable wrapper ------------------------------------------------------


def _marker_positions_traceable(
    ids: Tensor, option_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Traceable replacement for ``RapportModel._marker_positions``.

    Goal: exactly one ``<goal>`` marker per row (validated upstream), so
    ``argmax`` over the match indicators locates it. Options: the *j*-th
    ``<opt>`` marker is the position ``t`` whose running count of ``<opt>``
    markers equals ``j + 1``; a cumsum-rank equality mask finds it in one
    shot. Padding slots match nothing and resolve to index 0, exactly as in
    the reference implementation (padding slots point at index 0).
    """
    is_goal = ids == GOAL_ID
    goal_positions = is_goal.to(torch.int64).argmax(dim=1)

    is_opt = ids == OPT_ID
    cumulative = is_opt.to(torch.int64).cumsum(dim=1)
    seq_len = ids.shape[1]
    width = option_mask.shape[1]
    slots = torch.arange(1, width + 1, device=ids.device).view(1, width, 1)
    match = is_opt.unsqueeze(1) & (cumulative.unsqueeze(1) == slots)
    steps = torch.arange(seq_len, device=ids.device).view(1, 1, seq_len)
    option_positions = (match.to(torch.int64) * steps).sum(dim=-1)
    return goal_positions, option_positions


class _OnnxExportModule(nn.Module):
    """Export-facing view of :class:`RapportModel` with identical weights.

    Returns the seven head outputs as a plain tuple (ONNX graphs need flat
    tensor outputs; ``RapportOutput`` is a dataclass the exporter cannot
    serialize). ``answerable`` re-emits the probability head output, exactly
    like the model's own absorption.
    """

    def __init__(self, model: RapportModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        option_mask: Tensor,
        task_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        model = self.model
        ids = input_ids.to(torch.long)
        tasks = task_ids.to(torch.long)
        keep = option_mask.to(torch.bool)

        hidden = model.trunk(ids, attention_mask, tasks)
        goal_positions, option_positions = _marker_positions_traceable(ids, option_mask)
        readout = model.readout(hidden, goal_positions, option_positions, tasks, keep)

        cls_vector = hidden[:, 0]
        probability = model.probability_head(cls_vector)  # [B, 1]
        confidence_input = confidence_features(cls_vector, readout.probabilities)
        confidence = model.confidence_head(confidence_input)  # [B, 1]
        risk = model.risk_head(cls_vector)  # [B, 8]
        return (
            readout.choice_logits,
            readout.scores,
            probability,
            confidence,
            probability,
            risk,
            readout.score_vector,
        )


# -- identity Cast elimination ----------------------------------------------------


def _graph_attr_tensors(attr: Any) -> list[Any]:
    onnx = _require_onnx()
    if attr.type == onnx.AttributeProto.GRAPH:
        return [attr.g]
    if attr.type == onnx.AttributeProto.GRAPHS:
        return list(attr.graphs)
    return []


def _cast_target_dtype(node: Any) -> int | None:
    for attr in node.attribute:
        if attr.name == "to" and attr.HasField("i"):
            return int(attr.i)
    return None


def _strip_identity_casts(model: Any) -> int:
    """Remove Cast nodes whose input/output dtypes are identical.

    Rewrites node inputs and ``If``/``Loop``/``Scan`` subgraph input captures
    recursively. Identity casts carry no semantics; removing them works
    around an onnxruntime Level-1 optimizer defect. Returns the number of
    removed nodes. Callers must verify numerics afterwards (``export_onnx``
    does), so a bad rewrite fails the export instead of shipping silently.
    """
    initializers = {init.name: init.data_type for init in model.graph.initializer}
    removed = 0

    def process(graph: Any) -> None:
        nonlocal removed
        for node in graph.node:
            for attr in node.attribute:
                for sub in _graph_attr_tensors(attr):
                    process(sub)
        typemap: dict[str, int] = dict(initializers)
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            tensor_type = vi.type.tensor_type
            if vi.type.HasField("tensor_type"):
                typemap[vi.name] = tensor_type.elem_type

        rewire: dict[str, str] = {}
        for node in list(graph.node):
            if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
                continue
            target = _cast_target_dtype(node)
            if target is None:
                continue
            source = typemap.get(node.input[0])
            if source is not None and source == target:
                rewire[node.output[0]] = node.input[0]
                graph.node.remove(node)

        if not rewire:
            return

        def resolve(name: str) -> str:
            # rewire may chain (/trunk/Cast_output_0 -> /Cast_1_output_0 -> task_ids)
            seen = set()
            while name in rewire and name not in seen:
                seen.add(name)
                name = rewire[name]
            return name

        def propagate(sub: Any) -> None:
            # an outer identity cast may be captured by a subgraph at any depth
            for index, entry in enumerate(sub.input):
                sub.input[index].name = resolve(entry.name)
            for sub_node in sub.node:
                sub_node.input[:] = [resolve(name) for name in sub_node.input]
                for sub_attr in sub_node.attribute:
                    for deeper in _graph_attr_tensors(sub_attr):
                        propagate(deeper)

        for node in graph.node:
            node.input[:] = [resolve(name) for name in node.input]
            for attr in node.attribute:
                for sub in _graph_attr_tensors(attr):
                    propagate(sub)
        removed += len(rewire)

    process(model.graph)
    return removed


def _dynamic_axes() -> dict[str, dict[int, str]]:
    return {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "option_mask": {0: "batch", 1: "options"},
        "task_ids": {0: "batch"},
        "choice_logits": {0: "batch", 1: "options"},
        "scores": {0: "batch", 1: "options"},
        "probability": {0: "batch"},
        "confidence": {0: "batch"},
        "answerable": {0: "batch"},
        "risk": {0: "batch"},
        "score_vector": {0: "batch", 1: "options"},
    }


def _reference_fingerprint(batch: Mapping[str, Tensor]) -> str:
    return tensor_fingerprint({name: batch[name] for name in INPUT_NAMES})


def _numpy_feed(batch: Mapping[str, Any]) -> dict[str, Any]:
    feed: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, Tensor):
            feed[name] = value.detach().cpu().contiguous().numpy()
        else:
            feed[name] = value
    return feed


def _torch_reference_outputs(
    module: _OnnxExportModule, batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    with torch.inference_mode():
        outputs = module(
            batch["input_ids"],
            batch["attention_mask"],
            batch["option_mask"],
            batch["task_ids"],
        )
    return dict(zip(OUTPUT_NAMES, outputs, strict=True))


def _onnx_session(path: Path) -> Any:
    ort = _require_ort()
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _compare_against_pytorch(
    module: _OnnxExportModule, path: Path, batch: Mapping[str, Tensor]
) -> dict[str, float]:
    import numpy as np

    session = _onnx_session(path)
    outputs = session.run(list(OUTPUT_NAMES), _numpy_feed(batch))
    reference = _torch_reference_outputs(module, batch)
    diffs: dict[str, float] = {}
    for name, onnx_out in zip(OUTPUT_NAMES, outputs, strict=True):
        torch_out = reference[name].detach().cpu().numpy()
        diff = np.abs(onnx_out.astype(np.float64) - torch_out.astype(np.float64)).max()
        diffs[name] = float(diff)
    return diffs


# -- public API: export ------------------------------------------------------------


def export_onnx(
    model: RapportModel,
    tokenizer_spec: Mapping[str, Any],
    path: str | Path,
    opset: int = EXPORT_OPSET,
) -> dict[str, Any]:
    """Export ``model`` to ONNX and verify choice logits against PyTorch.

    Writes the FP32 ONNX artifact to ``path`` with dynamic axes on batch and
    option count, then runs a reference-batch comparison. ``tokenizer_spec``
    is recorded verbatim (JSON-sanitized) in the returned report; it is
    informational only — the export itself needs no vocabulary.

    The returned report is JSON-serializable and records ``opset``,
    ``sample_count`` (always 64), ``tolerance`` (1e-3), the reference-batch
    fingerprint, per-output max-abs diffs, the dynamic-axes declaration and
    the serialized size. Raises :class:`OnnxExportMismatchError` when
    ``|Δlogit|`` exceeds the tolerance; the artifact is left on disk for
    inspection but must not be shipped.
    """
    if not isinstance(model, RapportModel):
        raise TypeError(f"model must be a RapportModel, got {type(model).__name__}")
    onnx = _require_onnx()
    ort = _require_ort()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = _OnnxExportModule(model).eval()
    batch = make_reference_batch(
        seed=REFERENCE_BATCH_SEED,
        rows=REFERENCE_BATCH_ROWS,
        width=MAX_OPTIONS,
        seq_len=REFERENCE_SEQ_LEN,
        vocab_size=int(model.config.vocab_size),
    )
    fingerprint = _reference_fingerprint(batch)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            wrapper,
            (
                batch["input_ids"],
                batch["attention_mask"],
                batch["option_mask"],
                batch["task_ids"],
            ),
            str(out_path),
            opset_version=opset,
            dynamo=False,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=_dynamic_axes(),
        )

    exported = onnx.load(str(out_path))
    inferred = onnx.shape_inference.infer_shapes(exported)
    identity_casts_removed = _strip_identity_casts(inferred)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, str(out_path))

    diffs = _compare_against_pytorch(wrapper, out_path, batch)
    max_abs_logit_diff = diffs["choice_logits"]

    # Dynamic-axes smoke check: a different shape must run and match too.
    smoke_batch = make_reference_batch(
        seed=REFERENCE_BATCH_SEED + 1,
        rows=3,
        width=4,
        seq_len=REFERENCE_SEQ_LEN,
        vocab_size=int(model.config.vocab_size),
    )
    smoke_diffs = _compare_against_pytorch(wrapper, out_path, smoke_batch)

    if max_abs_logit_diff > EXPORT_TOLERANCE:
        raise OnnxExportMismatchError(
            f"ONNX choice logits differ from PyTorch by {max_abs_logit_diff:.3e}, "
            f"exceeding the {EXPORT_TOLERANCE:.1e} tolerance",
            max_abs_logit_diff=max_abs_logit_diff,
            tolerance=EXPORT_TOLERANCE,
        )

    report: dict[str, Any] = {
        "artifact": "fp32",
        "path": str(out_path),
        "opset": int(opset),
        "sample_count": REFERENCE_BATCH_ROWS,
        "tolerance": EXPORT_TOLERANCE,
        "reference_batch": {
            "seed": REFERENCE_BATCH_SEED,
            "rows": REFERENCE_BATCH_ROWS,
            "seq_len": REFERENCE_SEQ_LEN,
            "width": MAX_OPTIONS,
            "vocab_size": int(model.config.vocab_size),
            "sha256": fingerprint,
        },
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "dynamic_axes": _dynamic_axes(),
        "max_abs_logit_diff": float(max_abs_logit_diff),
        "max_abs_diffs": {name: float(value) for name, value in diffs.items()},
        "dynamic_axes_check": {
            "rows": 3,
            "width": 4,
            "max_abs_logit_diff": float(smoke_diffs["choice_logits"]),
        },
        "identity_casts_removed": int(identity_casts_removed),
        "serialized_size_bytes": out_path.stat().st_size,
        "tokenizer_spec": _jsonable(tokenizer_spec),
        "exporter": "torch.onnx.export(dynamo=False, TorchScript tracer)",
        "onnx_version": str(getattr(onnx, "__version__", "unknown")),
        "onnxruntime_version": str(getattr(ort, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
    }
    return report


# -- public API: FP16 conversion ------------------------------------------------------


def convert_fp32_to_fp16(fp32_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Convert an FP32 ONNX artifact to FP16 weights (keeps I/O dtypes).

    Returns a JSON-serializable report with the FP16 path and both file
    sizes. Raises :class:`OnnxNotInstalledError` when onnxruntime's
    float16 transformer is unavailable.
    """
    _require_ort()
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:  # pragma: no cover - depends on onnxruntime build
        raise OnnxNotInstalledError(
            "onnxruntime.transformers.float16 is unavailable in this onnxruntime build"
        ) from exc

    onnx = _require_onnx()
    source = Path(fp32_path)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    converted = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(target))
    return _jsonable(
        {
            "artifact": "fp16",
            "path": str(target),
            "source_fp32_bytes": source.stat().st_size,
            "fp16_size_bytes": target.stat().st_size,
            "keep_io_types": True,
        }
    )


# -- public API: INT8 dynamic quantization -------------------------------------------


def _is_cosine_path_name(name: str) -> bool:
    normalized = name.replace("/", ".").lower()
    return any(token in normalized for token in _COSINE_PATH_TOKENS)


def _cosine_path_node_names(model: Any) -> list[str]:
    return [node.name for node in model.graph.node if _is_cosine_path_name(node.name or "")]


def _infer_vocab_size(model: Any) -> int | None:
    for init in model.graph.initializer:
        name = init.name.lower()
        if "embedding" in name and name.endswith("weight"):
            dims = list(init.dims)
            if dims:
                return int(dims[0])
    return None


def quantize_dynamic_int8(
    onnx_path: str | Path,
    out_path: str | Path,
    *,
    reference_batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dynamically quantize an FP32 ONNX artifact to INT8 (per-channel QInt8).

    The cosine score projection (``readout.query`` = ``W_q`` and
    ``readout.option`` = ``W_o``) MatMul nodes are excluded from
    quantization and stay in FP32 — quantizing them is the documented
    failure mode behind ranking damage (R13). Everything else is quantized
    dynamically.

    The returned JSON-serializable report records which nodes were excluded,
    whether the cosine path stayed FP32, per-head (per-output-tensor) max-abs
    drift on the reference batch, and both serialized sizes. Pass
    ``reference_batch`` to override the seeded synthetic batch (its vocab
    size is then the caller's responsibility).
    """
    _require_onnx()
    _require_ort()
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover - depends on onnxruntime build
        raise OnnxNotInstalledError(
            "onnxruntime.quantization is unavailable in this onnxruntime build"
        ) from exc

    onnx = _require_onnx()
    source = Path(onnx_path)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source))
    cosine_nodes = _cosine_path_node_names(model)
    cosine_score_path_fp32 = bool(cosine_nodes)

    quantize_dynamic(
        str(source),
        str(target),
        per_channel=True,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=cosine_nodes,
    )

    if reference_batch is None:
        vocab_size = _infer_vocab_size(model)
        if vocab_size is None:
            raise QuantizationReportError(
                "could not infer vocab_size from the ONNX graph; pass reference_batch="
            )
        reference_batch = make_reference_batch(
            seed=REFERENCE_BATCH_SEED,
            rows=REFERENCE_BATCH_ROWS,
            width=MAX_OPTIONS,
            seq_len=REFERENCE_SEQ_LEN,
            vocab_size=vocab_size,
        )
    fingerprint = _reference_fingerprint(reference_batch)

    per_head_drift: dict[str, Any] = {}
    try:
        import numpy as np

        fp32_session = _onnx_session(source)
        int8_session = _onnx_session(target)
        feed = _numpy_feed(reference_batch)
        fp32_outputs = fp32_session.run(list(OUTPUT_NAMES), feed)
        int8_outputs = int8_session.run(list(OUTPUT_NAMES), feed)
        for name, fp32_out, int8_out in zip(OUTPUT_NAMES, fp32_outputs, int8_outputs, strict=True):
            per_head_drift[name] = float(
                np.abs(int8_out.astype(np.float64) - fp32_out.astype(np.float64)).max()
            )
    except Exception as exc:  # pragma: no cover - drift is best-effort metadata
        per_head_drift = {"error": str(exc)}

    return _jsonable(
        {
            "artifact": "int8",
            "source_path": str(source),
            "path": str(target),
            "weight_type": "QInt8",
            "per_channel": True,
            "nodes_excluded_from_quantization": list(cosine_nodes),
            "cosine_score_path_fp32": cosine_score_path_fp32,
            "reference_batch": {
                "seed": REFERENCE_BATCH_SEED,
                "rows": REFERENCE_BATCH_ROWS,
                "seq_len": REFERENCE_SEQ_LEN,
                "width": MAX_OPTIONS,
                "sha256": fingerprint,
            },
            "per_head_drift": per_head_drift,
            "fp32_size_bytes": source.stat().st_size,
            "int8_size_bytes": target.stat().st_size,
        }
    )


# -- public API: the five-condition gate ---------------------------------------------


def _require_fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuantizationReportError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise QuantizationReportError(f"{name} must lie in [0, 1], got {number}")
    return number


def _require_size_bytes(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuantizationReportError(f"{name} must be a byte count, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise QuantizationReportError(f"{name} must be a non-negative finite number, got {number}")
    return int(number)


def _require_latency_ms(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuantizationReportError(f"{name} must be milliseconds, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise QuantizationReportError(f"{name} must be a positive finite duration, got {number}")
    return number


@dataclass(frozen=True)
class GateOutcome:
    """JSON-serializable result of the five-condition INT8 gate."""

    passed: bool
    artifact: str  # "int8" when every condition passes, else "fp16"
    conditions: dict[str, bool]
    margins: dict[str, float]
    limits: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact not in ("int8", "fp16"):
            raise QuantizationReportError(
                f"artifact must be 'int8' or 'fp16', got {self.artifact!r}"
            )
        expected = all(self.conditions.values())
        if self.passed != expected:
            raise QuantizationReportError(
                f"passed={self.passed} is inconsistent with conditions={self.conditions}"
            )
        if self.artifact != ("int8" if self.passed else "fp16"):
            raise QuantizationReportError(
                f"artifact={self.artifact!r} is inconsistent with passed={self.passed}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def quantization_gate(report: Mapping[str, Any]) -> GateOutcome:
    """Evaluate the five INT8 acceptance conditions against a report mapping.

    Required keys (fraction accuracies/ECE, byte counts, millisecond
    latencies; ``cosine_score_path_fp32`` must be a real bool):

    * ``choice_accuracy_fp32``, ``choice_accuracy_int8`` — dev choice accuracy
      as fractions; the **drop** must be ``<= 0.01`` (one point),
    * ``bucket_macro_ece_fp32``, ``bucket_macro_ece_int8`` — bucket-macro ECE
      fractions; the **rise** must be ``<= 0.02``,
    * ``cosine_score_path_fp32`` — true iff the cosine projection stayed in
      FP32 (i.e. ``quantize_dynamic_int8`` excluded the nodes),
    * ``int8_size_bytes`` — serialized INT8 bytes; must be ``<= 35 MiB``,
    * ``fp32_p95_ms``, ``int8_p95_ms`` — p95 CPU latencies; INT8 must be at
      least ``2×`` faster (``fp32_p95_ms / int8_p95_ms >= 2.0``).

    Missing/invalid values raise :class:`QuantizationReportError`. The
    outcome is JSON-serializable via :meth:`GateOutcome.to_dict`/``to_json``.
    """
    required = (
        "choice_accuracy_fp32",
        "choice_accuracy_int8",
        "bucket_macro_ece_fp32",
        "bucket_macro_ece_int8",
        "cosine_score_path_fp32",
        "int8_size_bytes",
        "fp32_p95_ms",
        "int8_p95_ms",
    )
    missing = [name for name in required if name not in report]
    if missing:
        raise QuantizationReportError(
            f"quantization gate report is missing required keys: {', '.join(sorted(missing))}"
        )

    acc_fp32 = _require_fraction("choice_accuracy_fp32", report["choice_accuracy_fp32"])
    acc_int8 = _require_fraction("choice_accuracy_int8", report["choice_accuracy_int8"])
    ece_fp32 = _require_fraction("bucket_macro_ece_fp32", report["bucket_macro_ece_fp32"])
    ece_int8 = _require_fraction("bucket_macro_ece_int8", report["bucket_macro_ece_int8"])
    cosine_fp32 = report["cosine_score_path_fp32"]
    if not isinstance(cosine_fp32, bool):
        raise QuantizationReportError(
            f"cosine_score_path_fp32 must be a bool, got {type(cosine_fp32).__name__}"
        )
    int8_size = _require_size_bytes("int8_size_bytes", report["int8_size_bytes"])
    fp32_p95 = _require_latency_ms("fp32_p95_ms", report["fp32_p95_ms"])
    int8_p95 = _require_latency_ms("int8_p95_ms", report["int8_p95_ms"])

    accuracy_drop = acc_fp32 - acc_int8
    ece_rise = ece_int8 - ece_fp32
    speedup = fp32_p95 / int8_p95

    conditions = {
        "choice_accuracy_drop": accuracy_drop <= MAX_ACCURACY_DROP + _BOUNDARY_EPS,
        "bucket_macro_ece_rise": ece_rise <= MAX_BUCKET_ECE_RISE + _BOUNDARY_EPS,
        "cosine_score_path_fp32": cosine_fp32,
        "serialized_size": int8_size <= MAX_INT8_BYTES,
        "p95_latency_speedup": speedup >= MIN_P95_SPEEDUP - _BOUNDARY_EPS,
    }
    passed = all(conditions.values())
    return GateOutcome(
        passed=passed,
        artifact="int8" if passed else "fp16",
        conditions=conditions,
        margins={
            "choice_accuracy_drop": float(accuracy_drop),
            "bucket_macro_ece_rise": float(ece_rise),
            "cosine_score_path_fp32": 1.0 if cosine_fp32 else 0.0,
            "serialized_size_bytes": float(int8_size),
            "p95_latency_speedup": float(speedup),
        },
        limits={
            "max_accuracy_drop": MAX_ACCURACY_DROP,
            "max_ece_rise": MAX_BUCKET_ECE_RISE,
            "max_int8_bytes": float(MAX_INT8_BYTES),
            "min_speedup": MIN_P95_SPEEDUP,
        },
    )


@dataclass(frozen=True)
class ExportSelection:
    """Kept artifact plus the per-condition breakdown and the recorded decision."""

    artifact: str  # "int8" (accepted) or "fp16" (INT8 refused)
    accepted: bool
    gate: GateOutcome
    report: dict[str, Any]

    @property
    def kept_artifact(self) -> str:
        """Alias kept for callers that prefer the serialized name."""
        return self.artifact

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.report)
        payload["selection"] = {
            "artifact": self.artifact,
            "accepted": self.accepted,
            "gate": self.gate.to_dict(),
        }
        return _jsonable(payload)

    def to_json(self) -> str:
        return _pretty_json(self.to_dict())


def validate_and_select(report: Mapping[str, Any], *, strict: bool = False) -> ExportSelection:
    """Apply the five-condition gate and select the artifact to ship.

    Returns an :class:`ExportSelection` whose ``artifact`` is ``"int8"`` when
    every condition passes, otherwise ``"fp16"`` — the FP16 artifact is
    retained and the INT8 artifact is refused. The returned ``report`` is a
    *copy* of the input with a ``decision`` section recording the artifact,
    the per-condition booleans and a human-readable reason; the caller's
    mapping is never mutated. With ``strict=True`` a failing gate raises
    :class:`QuantizationRejectedError` (carrying the :class:`GateOutcome`)
    instead of returning the FP16 selection.
    """
    gate = quantization_gate(report)
    selected_report = dict(report)
    failed = sorted(name for name, ok in gate.conditions.items() if not ok)
    reason = (
        "all five INT8 gate conditions satisfied; INT8 artifact accepted"
        if gate.passed
        else f"INT8 artifact refused (failed: {', '.join(failed)}); FP16 artifact retained"
    )
    selected_report["decision"] = {
        "artifact": gate.artifact,
        "passed": gate.passed,
        "conditions": dict(gate.conditions),
        "margins": dict(gate.margins),
        "limits": dict(gate.limits),
        "reason": reason,
    }
    if not gate.passed and strict:
        raise QuantizationRejectedError(reason, gate=gate)
    return ExportSelection(
        artifact=gate.artifact,
        accepted=gate.passed,
        gate=gate,
        report=_jsonable(selected_report),
    )


# -- public API: sizes and latency ----------------------------------------------------


def _classify_variant(filename: str) -> str:
    lowered = filename.lower()
    if "int8" in lowered:
        return "int8"
    if any(token in lowered for token in ("fp16", "f16", "float16")):
        return "fp16"
    return "fp32"


def serialized_sizes(path: str | Path) -> dict[str, Any]:
    """Report serialized bytes for the FP32/FP16/INT8 variants.

    * ``path`` is a **file**: returns ``{variant: bytes}`` for the variant
      implied by the filename (``int8``/``fp16`` tokens, otherwise ``fp32``).
    * ``path`` is a **directory**: scans ``*.onnx`` and returns all three
      keys; a key is the byte size of the first matching file in sorted
      order, or ``None`` when that variant is absent.
    """
    target = Path(path)
    if target.is_dir():
        sizes: dict[str, int | None] = {"fp32": None, "fp16": None, "int8": None}
        for candidate in sorted(target.glob("*.onnx")):
            variant = _classify_variant(candidate.name)
            if sizes[variant] is None:
                sizes[variant] = candidate.stat().st_size
        return sizes
    if not target.is_file():
        raise FileNotFoundError(f"serialized_sizes: no such file or directory: {target}")
    return {_classify_variant(target.name): target.stat().st_size}


def benchmark_p95(
    session: Any,
    batch: Mapping[str, Any],
    iters: int = 1000,
    warmup: int = 50,
) -> float:
    """Nearest-rank p95 latency (milliseconds) of ``session.run(None, feed)``.

    ``session`` is any ORT-like object exposing ``run(output_names, feed)``;
    ``batch`` is the input feed mapping (torch tensors are converted to
    numpy). Runs ``warmup`` untimed iterations, then ``iters`` timed ones and
    returns the nearest-rank 95th percentile: ``sorted[ceil(0.95 * iters) - 1]``.
    """
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError(f"iters must be a positive integer, got {iters!r}")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative integer, got {warmup!r}")
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"batch must be a mapping of input names to tensors, got {type(batch).__name__}"
        )

    feed = _numpy_feed(batch)
    for _ in range(warmup):
        session.run(None, feed)

    latencies: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - start) * 1000.0)

    latencies.sort()
    rank = math.ceil(0.95 * len(latencies)) - 1
    return float(latencies[rank])
