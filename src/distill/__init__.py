"""Soft-label distillation for RAPPORT Stage C (TASK-15).

Public surface:

* :func:`build_soft_label_cache` + :class:`DistillOptions` /
  :class:`SoftLabelCacheResult` / :class:`SoftLabelEntry` — the
  content-addressed teacher soft-label cache (FR-DISTILL-01), keyed by
  ``(teacher_id, template_hash, example_id)`` under ``cache/teacher/``.
* :func:`template_hash` / :func:`render_prompt` / :func:`sanitize_teacher_id`
  — the template hash, prompt renderer and path sanitiser.
* :func:`distillation_loss` — ``T^2 * KL(softmax(t/T) || softmax(s/T))``
  (FR-DISTILL-02 support), teacher detached, softmax in FP32.
* :func:`distillation_gate` + :class:`GateOutcome` — the pure Stage C
  usefulness gate (FR-DISTILL-02, risk R9, E8). ``abort`` is a recorded
  negative result, never an error; the gate never reaches through to
  training.
* :data:`ENTROPY_EXCLUSION_PERCENTILE` + :func:`entropy_from_logprobs` /
  :func:`percentile_linear` / :func:`entropy_exclusion_mask` — the
  95th-percentile entropy filter (documented constant).
* :class:`Teacher`, :func:`load_local_teacher`, :class:`LocalCausalLMTeacher`,
  :data:`DEFAULT_TEMPLATE` — the local-teacher protocol and live path
  (never exercised by the unit tests; a one-time download, then offline).
"""

from .cache import (
    CACHE_VERSION,
    DEFAULT_CACHE_ROOT,
    INDEX_VERSION,
    DistillOptions,
    SoftLabelCacheResult,
    SoftLabelEntry,
    build_soft_label_cache,
    entry_filename,
    index_key,
    sanitize_teacher_id,
    template_hash,
)
from .entropy import (
    ENTROPY_EXCLUSION_PERCENTILE,
    entropy_exclusion_mask,
    entropy_from_logprobs,
    percentile_linear,
)
from .errors import (
    DistillError,
    DistillInputError,
    GateInputError,
    TeacherUnavailableError,
)
from .gate import (
    OUTCOME_ABORT,
    OUTCOME_PASS,
    REASON_KD_GAIN,
    REASON_OK,
    REASON_TEACHER_MARGIN,
    REASON_TEACHER_UNAVAILABLE,
    GateOutcome,
    distillation_gate,
)
from .loss import DEFAULT_TEMPERATURE, distillation_loss
from .teacher import (
    DEFAULT_TEMPLATE,
    LocalCausalLMTeacher,
    Teacher,
    load_local_teacher,
    render_prompt,
)

__all__ = [
    "CACHE_VERSION",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_TEMPLATE",
    "DEFAULT_TEMPERATURE",
    "ENTROPY_EXCLUSION_PERCENTILE",
    "INDEX_VERSION",
    "LocalCausalLMTeacher",
    "OUTCOME_ABORT",
    "OUTCOME_PASS",
    "REASON_KD_GAIN",
    "REASON_OK",
    "REASON_TEACHER_MARGIN",
    "REASON_TEACHER_UNAVAILABLE",
    "DistillError",
    "DistillInputError",
    "DistillOptions",
    "GateInputError",
    "GateOutcome",
    "SoftLabelCacheResult",
    "SoftLabelEntry",
    "Teacher",
    "TeacherUnavailableError",
    "build_soft_label_cache",
    "distillation_gate",
    "distillation_loss",
    "entropy_exclusion_mask",
    "entropy_from_logprobs",
    "entry_filename",
    "index_key",
    "load_local_teacher",
    "percentile_linear",
    "render_prompt",
    "sanitize_teacher_id",
    "template_hash",
]
