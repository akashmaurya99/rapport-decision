"""The DecisionResult prediction API (TASK-22, FR-INFER-01).

This module is the inference boundary for RAPPORT. A :class:`Predictor` loads
the model exactly once — either from a ``RapportModel`` instance or from a
YAML config path — and answers :meth:`Predictor.predict` with a frozen
:class:`DecisionResult`. The result carries numbers only: there is
deliberately no string field anywhere on the schema, so no generated answer
can leak into it (task identity stays implicit to the caller).

Contract summary
----------------
* ``chosen_index`` — argmax of the calibrated probabilities (ties resolve to
  the lowest index); ``None`` when ``abstain`` is set. The option texts are
  never emitted.
* ``scores`` — the model's sigmoid utility per option, in option order,
  uncalibrated (calibration applies to the softmax distribution, not to the
  per-option utility).
* ``probabilities`` — softmax over the listwise choice logits at the
  calibration temperature; sums to 1.0.
* ``confidence`` — the calibrated confidence, i.e. ``max(probabilities)``.
  This is the same quantity :meth:`CalibrationMap.apply` reports and the same
  quantity the abstention policy consumes; the raw confidence head remains on
  ``RapportOutput`` for training/evaluation use.
* ``abstain`` — ``True`` iff ``confidence < abstention_policy.threshold``
  (strictly below; the default policy derives the declared 0.75 threshold
  from the utility triple ``(1, -3, 0)``). Scores and probabilities are still
  returned when abstaining; ``chosen_index`` is ``None``.
* ``risk`` — the eight sigmoid raw values from the risk head, in slot order.
  No threshold is applied here; the ad-hoc veto lives in the orchestration
  layer, not in the predictor.
* ``temperature`` — the calibration temperature that produced
  ``probabilities``/``confidence``. ``1.0`` when no calibration map is
  supplied (default behaviour; the uncalibrated distribution is reported and
  ``temperature`` records the identity mapping). For Platt-scaled buckets the
  map contract stores ``1.0`` as the temperature placeholder, so ``1.0`` is
  recorded and the Platt parameters do the calibration.

Design decisions (see ``.archgen/rapport/.agent/notes/TASK-22.md``)
-------------------------------------------------------------------
* Temperature comes from the bucket ``task_family × option_count_bucket``
  (``"<family>|<2|3-5|6-8>"``) of the supplied :class:`CalibrationMap`.
  Buckets absent from the map fall back to the pooled entry (same as
  ``CalibrationMap.apply(..., strict=False)``); a map that was never fitted
  for the requested bucket therefore cannot crash prediction.
* The abstain decision always uses the predictor's own
  :class:`AbstentionPolicy` instance (default: the declared 0.75 utility
  triple), never the policy frozen into the calibration map — one source of
  truth for the abstain boundary, injectable for evaluation.
* ``config_path`` — a YAML mapping with optional ``variant`` (one of
  ``"ultra"``, ``"compact"``, ``"large"``; default ``"compact"``) selecting the
  declared ``RapportConfig``, and optional ``weights_path`` (a ``state_dict``
  checkpoint loaded with ``map_location=device``). An unreadable/invalid
  config raises :class:`PredictorConfigError`; nothing is half-loaded.
* Validation failures on the request raise :class:`InvalidRequestError` with a
  stable machine-readable ``code`` and no traceback semantics — callers catch
  one typed error. E1 codes: ``INFER_EMPTY_OPTIONS``, ``INFER_MISSING_STATE``.
  Option counts stay inside the 2–8 contract; 1 and 9 are refused with
  ``INFER_INVALID_OPTION_COUNT`` (the model's ``InvalidOptionCountError`` is
  mapped onto the same code as a safety net). Duplicate/blank options are the
  caller's schema concern (``normalize_example``) and are refused here only
  for blank/non-string values (``INFER_INVALID_OPTION``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from src.calibration import AbstentionPolicy, CalibrationMap, make_bucket_key_for_count
from src.calibration.map import load_calibration_map
from src.calibration.metrics import softmax_probs
from src.data.core.schema import DecisionExample, Option, compute_example_id
from src.model.core import RapportConfig
from src.model.head import MAX_OPTIONS, MIN_OPTIONS, InvalidOptionCountError, RapportModel
from src.tokenization import RapportTokenizer, encode_example

# Schema order of the six task families; matches the task-conditioning ids
# documented in src/model/head/model.py (ids 0-5, 6-7 reserved).
TASK_FAMILY_ORDER: tuple[str, ...] = (
    "choice",
    "preference",
    "scoring",
    "probability",
    "abstention",
    "risk",
)

CODE_EMPTY_OPTIONS = "INFER_EMPTY_OPTIONS"
CODE_MISSING_STATE = "INFER_MISSING_STATE"
CODE_MISSING_GOAL = "INFER_MISSING_GOAL"
CODE_INVALID_OPTION_COUNT = "INFER_INVALID_OPTION_COUNT"
CODE_INVALID_OPTION = "INFER_INVALID_OPTION"
CODE_UNKNOWN_TASK_FAMILY = "INFER_UNKNOWN_TASK_FAMILY"
CODE_INVALID_REQUEST = "INFER_INVALID_REQUEST"
CODE_BAD_CONFIG = "INFER_BAD_CONFIG"


class PredictorError(Exception):
    """Base error for the inference package.

    ``code`` is a stable machine-readable token; ``message`` is human-facing.
    """

    code = "INFER_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class InvalidRequestError(PredictorError):
    """The decision request violates the inference contract (E1 and friends)."""

    code = CODE_INVALID_REQUEST


class PredictorConfigError(PredictorError):
    """The model could not be loaded from the given config path."""

    code = CODE_BAD_CONFIG


@dataclass(frozen=True)
class DecisionRequest:
    """One prediction request.

    ``options`` is coerced to a tuple at validation time; pass any sequence of
    option strings. ``state`` may be ``None`` — a missing state is reported
    with :data:`CODE_MISSING_STATE`, never with a traceback.
    """

    goal: str
    state: str | None
    options: Sequence[str]
    task_family: str


@dataclass(frozen=True)
class DecisionResult:
    """The prediction payload. Numbers only — no string field, by design.

    ``chosen_index`` is ``None`` exactly when ``abstain`` is ``True``. The
    annotation set is checked by the unit tests to contain no free text.
    """

    chosen_index: int | None
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    confidence: float
    abstain: bool
    risk: tuple[float, ...]
    temperature: float


_CONFIG_VARIANTS: dict[str, RapportConfig] = {
    "ultra": RapportConfig.ultra(),
    "compact": RapportConfig.compact(),
    "large": RapportConfig.large(),
}


class Predictor:
    """Loads the model once and answers decision requests.

    Parameters
    ----------
    model_or_config:
        A ready :class:`RapportModel`, or a path to a YAML config (see the
        module docstring for the schema: optional ``variant`` and
        ``weights_path``).
    tokenizer:
        The tokenizer used to build joint sequences (any ``RapportTokenizer``).
    calibration_map:
        A fitted :class:`CalibrationMap`, a path to a ``calibration.json``
        artifact, or ``None`` for the uncalibrated default (temperature 1.0).
    abstention_policy:
        The policy for the abstain boundary; defaults to
        :class:`AbstentionPolicy` with the declared utilities (θ = 0.75).
    device:
        Torch device for inference; defaults to CPU.
    """

    def __init__(
        self,
        model_or_config: RapportModel | str | Path,
        tokenizer: RapportTokenizer,
        calibration_map: CalibrationMap | str | Path | None = None,
        *,
        abstention_policy: AbstentionPolicy | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.abstention_policy = (
            abstention_policy if abstention_policy is not None else AbstentionPolicy()
        )
        self.calibration_map = self._resolve_calibration_map(calibration_map)
        self.model = self._resolve_model(model_or_config)

    # ------------------------------------------------------------------
    # Loading (once, at construction)
    # ------------------------------------------------------------------
    def _resolve_calibration_map(
        self, calibration_map: CalibrationMap | str | Path | None
    ) -> CalibrationMap | None:
        if calibration_map is None:
            return None
        if isinstance(calibration_map, CalibrationMap):
            return calibration_map
        if isinstance(calibration_map, (str, Path)):
            return load_calibration_map(calibration_map)
        raise PredictorConfigError(
            f"calibration_map must be a CalibrationMap, a path or None, "
            f"got {type(calibration_map).__name__}"
        )

    def _resolve_model(self, model_or_config: RapportModel | str | Path) -> RapportModel:
        if isinstance(model_or_config, RapportModel):
            model = model_or_config
        elif isinstance(model_or_config, (str, Path)):
            model = self._model_from_config(Path(model_or_config))
        else:
            raise PredictorConfigError(
                f"model_or_config must be a RapportModel or a config path, "
                f"got {type(model_or_config).__name__}"
            )
        model.to(self.device)
        model.eval()
        return model

    def _model_from_config(self, path: Path) -> RapportModel:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PredictorConfigError(f"could not read model config at {path}: {exc}") from exc
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise PredictorConfigError(f"model config at {path} is not valid YAML: {exc}") from exc
        if not isinstance(payload, dict):
            raise PredictorConfigError(
                f"model config at {path} must be a mapping, got {type(payload).__name__}"
            )
        unknown = sorted(set(payload) - {"variant", "weights_path"})
        if unknown:
            raise PredictorConfigError(
                f"model config at {path} has unknown keys {unknown} "
                f"(supported: variant, weights_path)"
            )
        variant = payload.get("variant", "compact")
        if not isinstance(variant, str) or variant not in _CONFIG_VARIANTS:
            raise PredictorConfigError(
                f"model config at {path}: variant must be one of "
                f"{sorted(_CONFIG_VARIANTS)}, got {variant!r}"
            )
        model = RapportModel(_CONFIG_VARIANTS[variant])
        weights_path = payload.get("weights_path")
        if weights_path is not None:
            if not isinstance(weights_path, str) or not weights_path.strip():
                raise PredictorConfigError(
                    f"model config at {path}: weights_path must be a non-empty string, "
                    f"got {weights_path!r}"
                )
            try:
                state = torch.load(
                    weights_path, map_location=self.device, weights_only=True
                )
            except Exception as exc:
                raise PredictorConfigError(
                    f"could not load weights from {weights_path}: {exc}"
                ) from exc
            try:
                model.load_state_dict(state)
            except (RuntimeError, KeyError) as exc:
                raise PredictorConfigError(
                    f"weights at {weights_path} do not match the {variant} architecture: {exc}"
                ) from exc
        return model

    # ------------------------------------------------------------------
    # Request validation (E1)
    # ------------------------------------------------------------------
    @staticmethod
    def _validated_options(raw: Any) -> tuple[str, ...]:
        if raw is None:
            return ()
        if isinstance(raw, str):
            raise InvalidRequestError(
                "options must be a sequence of option strings, not a single string",
                code=CODE_INVALID_OPTION,
            )
        try:
            options = tuple(raw)
        except TypeError as exc:
            raise InvalidRequestError(
                f"options must be a sequence of strings, got {type(raw).__name__}",
                code=CODE_INVALID_OPTION,
            ) from exc
        if not options:
            return ()
        for index, option in enumerate(options):
            if not isinstance(option, str) or not option.strip():
                raise InvalidRequestError(
                    f"option at index {index} must be a non-blank string, "
                    f"got {type(option).__name__}",
                    code=CODE_INVALID_OPTION,
                )
        return options

    def _validate_request(self, request: DecisionRequest) -> tuple[str, str, tuple[str, ...], str]:
        if not isinstance(request, DecisionRequest):
            raise InvalidRequestError(
                f"predict expects a DecisionRequest, got {type(request).__name__}",
                code=CODE_INVALID_REQUEST,
            )
        task_family = request.task_family
        if not isinstance(task_family, str) or task_family not in TASK_FAMILY_ORDER:
            raise InvalidRequestError(
                f"task_family must be one of {list(TASK_FAMILY_ORDER)}, got {task_family!r}",
                code=CODE_UNKNOWN_TASK_FAMILY,
            )
        goal = request.goal
        if not isinstance(goal, str) or not goal.strip():
            raise InvalidRequestError(
                "goal must be a non-empty string", code=CODE_MISSING_GOAL
            )
        state = request.state
        if not isinstance(state, str) or not state.strip():
            raise InvalidRequestError(
                "state must be provided as a non-empty string; a missing or "
                "blank state is refused before any compute",
                code=CODE_MISSING_STATE,
            )
        options = self._validated_options(request.options)
        if not options:
            raise InvalidRequestError(
                "options must contain at least one option; got an empty sequence",
                code=CODE_EMPTY_OPTIONS,
            )
        if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
            raise InvalidRequestError(
                f"option count must be between {MIN_OPTIONS} and {MAX_OPTIONS}, "
                f"got {len(options)}",
                code=CODE_INVALID_OPTION_COUNT,
            )
        return goal, state, options, task_family

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _joint_batch(
        self, goal: str, state: str, options: tuple[str, ...], task_family: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        option_records = tuple(Option(text=text) for text in options)
        texts = [opt.text for opt in option_records]
        example_id = compute_example_id(task_family, goal, state, texts)
        example = DecisionExample(
            example_id=example_id,
            task_family=task_family,
            goal=goal,
            state=state,
            options=option_records,
            source="inference",
            license="unknown",
            group_id="inference",
            split="test",
        )
        encoded = encode_example(example, tokenizer=self.tokenizer)
        ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long, device=self.device)
        attention = torch.tensor(
            [list(encoded.attention_mask)], dtype=torch.long, device=self.device
        )
        option_mask = torch.ones(
            (1, len(options)), dtype=torch.long, device=self.device
        )
        task_ids = torch.tensor(
            [TASK_FAMILY_ORDER.index(task_family)], dtype=torch.long, device=self.device
        )
        return ids, attention, option_mask, task_ids

    def predict(self, request: DecisionRequest) -> DecisionResult:
        """Predict a decision for ``request``; never emits a generated answer."""
        goal, state, options, task_family = self._validate_request(request)
        ids, attention, option_mask, task_ids = self._joint_batch(
            goal, state, options, task_family
        )
        try:
            with torch.no_grad():
                output = self.model(ids, attention, option_mask, task_ids)
        except InvalidOptionCountError as exc:
            # Safety net: request-side validation already enforces 2..8.
            raise InvalidRequestError(
                f"option count refused by the model: {exc}",
                code=CODE_INVALID_OPTION_COUNT,
            ) from exc

        logits = [float(v) for v in output.choice_logits[0].tolist()]
        if self.calibration_map is not None:
            bucket_key = make_bucket_key_for_count(task_family, len(options))
            calibrated = self.calibration_map.apply(logits, bucket_key, strict=False)
            probabilities = tuple(float(p) for p in calibrated.probabilities)
            confidence = float(calibrated.confidence)
            temperature = float(calibrated.temperature)
        else:
            probabilities = tuple(softmax_probs(logits, 1.0))
            confidence = max(probabilities)
            temperature = 1.0

        abstain = self.abstention_policy.should_abstain(confidence)
        chosen_index: int | None = (
            None if abstain else max(range(len(probabilities)), key=probabilities.__getitem__)
        )
        scores = tuple(float(v) for v in output.scores[0].tolist())
        risk = tuple(float(v) for v in output.risk[0].tolist())
        return DecisionResult(
            chosen_index=chosen_index,
            scores=scores,
            probabilities=probabilities,
            confidence=confidence,
            abstain=abstain,
            risk=risk,
            temperature=temperature,
        )


__all__ = [
    "CODE_EMPTY_OPTIONS",
    "CODE_MISSING_STATE",
    "CODE_MISSING_GOAL",
    "CODE_INVALID_OPTION",
    "CODE_INVALID_OPTION_COUNT",
    "CODE_UNKNOWN_TASK_FAMILY",
    "CODE_INVALID_REQUEST",
    "CODE_BAD_CONFIG",
    "DecisionRequest",
    "DecisionResult",
    "InvalidRequestError",
    "Predictor",
    "PredictorConfigError",
    "PredictorError",
    "TASK_FAMILY_ORDER",
]
