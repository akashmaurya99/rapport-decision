"""Tests for src/inference/predictor.py — the DecisionResult prediction API.

Covers TASK-22 acceptance criteria (E1, FR-INFER-01): typed validation errors
with stable codes and no traceback, the frozen numbers-only output schema,
calibration-temperature selection from the per-bucket map, and the 2..8
option-count contract mapped onto the model's InvalidOptionCountError path.
"""

from __future__ import annotations

import dataclasses
import sys
import typing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from src.calibration import (  # noqa: E402
    AbstentionPolicy,
    CalibrationEntry,
    CalibrationMap,
    save_calibration_map,
)
from src.calibration.map import SCHEMA_VERSION  # noqa: E402
from src.inference.predictor import (  # noqa: E402
    CODE_EMPTY_OPTIONS,
    CODE_INVALID_OPTION,
    CODE_INVALID_OPTION_COUNT,
    CODE_MISSING_GOAL,
    CODE_MISSING_STATE,
    CODE_UNKNOWN_TASK_FAMILY,
    DecisionRequest,
    DecisionResult,
    InvalidRequestError,
    Predictor,
    PredictorConfigError,
)
from src.model.core import RapportConfig, SpecConstants  # noqa: E402
from src.model.head import InvalidOptionCountError, RapportModel  # noqa: E402
from src.tokenization import RapportTokenizer  # noqa: E402


def _tiny_config() -> RapportConfig:
    spec = SpecConstants(
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
        variant="tiny-test",
        layers=1,
        d_model=32,
        heads=2,
        head_dim=16,
        ffn=64,
        vocab_size=30_522,
        embedding_dim=32,
        spec=spec,
    )


@pytest.fixture(scope="module")
def tokenizer() -> RapportTokenizer:
    return RapportTokenizer()


@pytest.fixture(scope="module")
def model() -> RapportModel:
    torch.manual_seed(20261119)
    return RapportModel(_tiny_config()).eval()


@pytest.fixture()
def predictor(model, tokenizer) -> Predictor:
    return Predictor(model, tokenizer)


def _request(**overrides) -> DecisionRequest:
    payload = {
        "goal": "pick the safest route",
        "state": "two routes are available; route A is shorter, route B avoids traffic",
        "options": ("route A", "route B"),
        "task_family": "choice",
    }
    payload.update(overrides)
    return DecisionRequest(**payload)


def _entry(bucket_key: str, temperature: float) -> CalibrationEntry:
    return CalibrationEntry(
        bucket_key=bucket_key,
        method="temperature",
        temperature=temperature,
        n_items=10,
        ece_before=0.1,
        ece_after=0.05,
        folds=5,
    )


def _map(pooled_temperature: float, bucket: tuple[str, float] | None = None) -> CalibrationMap:
    buckets = {}
    if bucket is not None:
        buckets[bucket[0]] = _entry(bucket[0], bucket[1])
    return CalibrationMap(
        schema_version=SCHEMA_VERSION,
        folds=5,
        pooled=_entry("pooled", pooled_temperature),
        buckets=buckets,
    )


# --- Happy path: field contract ---------------------------------------------


def _never_abstain() -> AbstentionPolicy:
    # threshold == 0.0: with random weights the distribution is near-uniform,
    # so abstention is disabled to exercise the chosen_index path.
    return AbstentionPolicy(u_correct=1.0, u_wrong=-3.0, u_abstain=-3.0)


def test_predict_returns_decision_result_with_contract_fields(model, tokenizer):
    predictor = Predictor(model, tokenizer, abstention_policy=_never_abstain())
    result = predictor.predict(_request())
    assert isinstance(result, DecisionResult)
    assert isinstance(result.chosen_index, int)
    assert result.chosen_index in (0, 1)
    assert isinstance(result.scores, tuple) and len(result.scores) == 2
    assert isinstance(result.probabilities, tuple) and len(result.probabilities) == 2
    assert isinstance(result.confidence, float)
    assert isinstance(result.abstain, bool)
    assert isinstance(result.risk, tuple) and len(result.risk) == 8
    assert isinstance(result.temperature, float)


def test_probabilities_are_a_distribution(predictor):
    result = predictor.predict(_request(options=("a", "b", "c", "d")))
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in result.probabilities)


def test_chosen_index_is_argmax_of_probabilities(model, tokenizer):
    predictor = Predictor(model, tokenizer, abstention_policy=_never_abstain())
    result = predictor.predict(_request(options=("a", "b", "c")))
    assert result.chosen_index == max(
        range(len(result.probabilities)), key=result.probabilities.__getitem__
    )


def test_scores_and_risk_stay_in_sigmoid_range(predictor):
    result = predictor.predict(_request())
    assert all(0.0 <= s <= 1.0 for s in result.scores)
    assert all(0.0 <= r <= 1.0 for r in result.risk)


def test_confidence_equals_max_probability(predictor):
    result = predictor.predict(_request())
    assert result.confidence == pytest.approx(max(result.probabilities))


def test_platt_does_not_flip_the_choice(model, tokenizer):
    """Platt correctness must never re-rank the options (a=0, b=-3, logits [4,0]).

    The calibrated correctness is ``sigmoid(-3) ~= 0.047`` — far below the
    raw max probability — yet the chosen index must remain the raw argmax 0.
    """
    import math

    from src.calibration.platt import apply_platt_to_distribution

    # (1) Direct helper: probabilities keep the raw argmax; confidence is the
    #     separate Platt correctness scalar.
    raw = [1.0 / (1.0 + math.exp(-4.0)), 1.0 / (1.0 + math.exp(0.0))]
    probs, cal = apply_platt_to_distribution(raw, 0.0, -3.0)
    assert max(range(len(probs)), key=probs.__getitem__) == 0
    assert sum(probs) == pytest.approx(1.0)
    assert cal == pytest.approx(1.0 / (1.0 + math.exp(3.0)))
    assert cal != pytest.approx(max(probs))

    # (2) Calibration map apply: same guarantees.
    platt_map = CalibrationMap(
        schema_version=1,
        folds=0,
        pooled=CalibrationEntry(
            bucket_key="pooled",
            method="platt",
            temperature=1.0,
            n_items=60,
            ece_before=0.1,
            ece_after=0.05,
            folds=0,
            platt={"a": 0.0, "b": -3.0},
        ),
        buckets={},
    )
    calibrated = platt_map.apply([4.0, 0.0])
    assert max(range(len(calibrated.probabilities)),
               key=calibrated.probabilities.__getitem__) == 0
    assert calibrated.confidence == pytest.approx(1.0 / (1.0 + math.exp(3.0)))
    assert calibrated.abstain is True  # 0.047 < 0.75

    # (3) Predictor end-to-end: force the model to emit logits [4, 0].
    predictor = Predictor(model, tokenizer, calibration_map=platt_map,
                          abstention_policy=_never_abstain())

    class _Out:
        choice_logits = torch.tensor([[4.0, 0.0]])
        scores = torch.tensor([[0.5, 0.5]])
        risk = torch.zeros(1, 8)

    original_forward = predictor.model.forward
    predictor.model.forward = lambda *args, **kwargs: _Out()  # type: ignore[method-assign]
    try:
        result = predictor.predict(_request())
    finally:
        predictor.model.forward = original_forward  # type: ignore[method-assign]
    assert result.chosen_index == 0
    assert result.probabilities[0] > result.probabilities[1]
    assert result.confidence == pytest.approx(1.0 / (1.0 + math.exp(3.0)))
    assert result.confidence != pytest.approx(max(result.probabilities))


@pytest.mark.parametrize("n_options", [2, 3, 5, 6, 8])
def test_every_supported_option_count_predicts(predictor, n_options):
    options = tuple(f"option {i}" for i in range(n_options))
    result = predictor.predict(_request(options=options))
    assert len(result.scores) == n_options
    assert len(result.probabilities) == n_options
    if result.chosen_index is not None:
        assert 0 <= result.chosen_index < n_options


# --- E1: empty options / missing state — typed errors, no traceback ---------


def test_empty_options_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options=()))
    assert excinfo.value.code == CODE_EMPTY_OPTIONS


def test_none_options_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options=None))
    assert excinfo.value.code == CODE_EMPTY_OPTIONS


def test_missing_state_none_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(state=None))
    assert excinfo.value.code == CODE_MISSING_STATE


def test_missing_state_blank_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(state="   \n\t  "))
    assert excinfo.value.code == CODE_MISSING_STATE


def test_missing_goal_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(goal=""))
    assert excinfo.value.code == CODE_MISSING_GOAL


def test_one_option_is_refused_with_invalid_option_count(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options=("only one",)))
    assert excinfo.value.code == CODE_INVALID_OPTION_COUNT


def test_nine_options_are_refused_with_invalid_option_count(predictor):
    options = tuple(f"option {i}" for i in range(9))
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options=options))
    assert excinfo.value.code == CODE_INVALID_OPTION_COUNT


def test_model_invalid_option_count_is_mapped_to_typed_error(predictor, monkeypatch):
    def _explode(*args, **kwargs):
        raise InvalidOptionCountError("slot width out of contract", field="option_mask")

    monkeypatch.setattr(predictor.model, "forward", _explode)
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request())
    assert excinfo.value.code == CODE_INVALID_OPTION_COUNT


def test_unknown_task_family_raise_typed_error(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(task_family="not-a-family"))
    assert excinfo.value.code == CODE_UNKNOWN_TASK_FAMILY


def test_blank_option_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options=("route A", "   ")))
    assert excinfo.value.code == CODE_INVALID_OPTION


def test_non_string_option_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options=("route A", 7)))
    assert excinfo.value.code == CODE_INVALID_OPTION


def test_bare_string_options_raise_typed_error_with_stable_code(predictor):
    with pytest.raises(InvalidRequestError) as excinfo:
        predictor.predict(_request(options="route A"))
    assert excinfo.value.code == CODE_INVALID_OPTION


# --- FR-INFER-01: the output schema carries no free text ---------------------


def _contains_str(hint) -> bool:
    if hint is str:
        return True
    return any(_contains_str(arg) for arg in typing.get_args(hint))


def test_decision_result_schema_has_no_string_field():
    hints = typing.get_type_hints(DecisionResult)
    offenders = [name for name, hint in hints.items() if _contains_str(hint)]
    assert offenders == []


def test_decision_result_field_set_is_exactly_the_contract():
    assert [f.name for f in dataclasses.fields(DecisionResult)] == [
        "chosen_index",
        "scores",
        "probabilities",
        "confidence",
        "abstain",
        "risk",
        "temperature",
    ]


def test_decision_result_is_frozen():
    result = DecisionResult(
        chosen_index=0,
        scores=(0.5, 0.5),
        probabilities=(0.5, 0.5),
        confidence=0.5,
        abstain=False,
        risk=(0.1,) * 8,
        temperature=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.confidence = 0.9  # type: ignore[misc]


# --- Calibration temperature selection ----------------------------------------


def test_without_calibration_map_temperature_defaults_to_one(predictor):
    result = predictor.predict(_request())
    assert result.temperature == 1.0


def test_temperature_comes_from_the_matching_bucket(model, tokenizer):
    calibration = _map(pooled_temperature=3.5, bucket=("choice|2", 2.5))
    result = Predictor(model, tokenizer, calibration).predict(_request())
    assert result.temperature == pytest.approx(2.5)


def test_unknown_bucket_falls_back_to_the_pooled_entry(model, tokenizer):
    calibration = _map(pooled_temperature=3.5, bucket=("choice|2", 2.5))
    # 4 options land in the "3-5" bucket, which the map does not carry.
    result = Predictor(model, tokenizer, calibration).predict(
        _request(options=("a", "b", "c", "d"))
    )
    assert result.temperature == pytest.approx(3.5)


def test_calibration_map_can_be_loaded_from_its_json_path(model, tokenizer, tmp_path):
    calibration = _map(pooled_temperature=1.5, bucket=("choice|2", 4.0))
    path = tmp_path / "calibration.json"
    save_calibration_map(calibration, path)
    result = Predictor(model, tokenizer, path).predict(_request())
    assert result.temperature == pytest.approx(4.0)


def test_calibration_map_rejects_unknown_argument_types(model, tokenizer):
    with pytest.raises(PredictorConfigError):
        Predictor(model, tokenizer, calibration_map=123)


# --- Abstention ---------------------------------------------------------------


def test_default_abstention_policy_threshold_is_075():
    assert AbstentionPolicy().threshold == pytest.approx(0.75)


def test_abstain_when_confidence_is_below_the_threshold(model, tokenizer):
    # A huge temperature flattens the distribution: confidence ~= 1 / n < 0.75.
    calibration = _map(pooled_temperature=1000.0)
    result = Predictor(model, tokenizer, calibration).predict(_request())
    assert result.abstain is True
    assert result.chosen_index is None
    # Scores and probabilities are still returned when abstaining.
    assert len(result.scores) == 2
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-6)


def test_no_abstain_when_the_policy_threshold_is_zero(model, tokenizer):
    policy = AbstentionPolicy(u_correct=1.0, u_wrong=-3.0, u_abstain=-3.0)
    assert policy.threshold == pytest.approx(0.0)
    result = Predictor(model, tokenizer, abstention_policy=policy).predict(_request())
    assert result.abstain is False
    assert result.chosen_index is not None


def test_abstain_decision_uses_the_predictors_policy_not_the_maps(model, tokenizer):
    # Map's frozen policy would be irrelevant: the predictor's policy governs.
    policy = AbstentionPolicy(u_correct=1.0, u_wrong=-3.0, u_abstain=10.0)
    assert policy.threshold > 1.0  # abstains on every distribution
    result = Predictor(model, tokenizer, abstention_policy=policy).predict(_request())
    assert result.abstain is True
    assert result.chosen_index is None


# --- Construction --------------------------------------------------------------


def test_predictor_reuses_the_single_model_instance(model, tokenizer):
    predictor = Predictor(model, tokenizer)
    assert predictor.model is model


def test_config_path_builds_a_model_and_predicts(tokenizer, tmp_path):
    weights = tmp_path / "weights.pt"
    torch.save(RapportModel(RapportConfig.ultra()).state_dict(), weights)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(f"variant: ultra\nweights_path: {weights!s}\n", encoding="utf-8")
    predictor = Predictor(config_path, tokenizer)
    result = predictor.predict(_request())
    assert result.temperature == 1.0
    assert len(result.probabilities) == 2


def test_config_path_requires_trained_weights(tokenizer, tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text("variant: ultra\n", encoding="utf-8")
    with pytest.raises(PredictorConfigError, match="weights_path is required"):
        Predictor(config_path, tokenizer)


def test_config_path_rejects_an_unknown_variant(tokenizer, tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text("variant: enormous\n", encoding="utf-8")
    with pytest.raises(PredictorConfigError) as excinfo:
        Predictor(config_path, tokenizer)
    assert excinfo.value.code == "INFER_BAD_CONFIG"


def test_config_path_rejects_unknown_keys(tokenizer, tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text("variant: ultra\nsurprise: true\n", encoding="utf-8")
    with pytest.raises(PredictorConfigError) as excinfo:
        Predictor(config_path, tokenizer)
    assert excinfo.value.code == "INFER_BAD_CONFIG"


def test_config_path_missing_file_raises_typed_error(tokenizer, tmp_path):
    with pytest.raises(PredictorConfigError) as excinfo:
        Predictor(tmp_path / "does-not-exist.yaml", tokenizer)
    assert excinfo.value.code == "INFER_BAD_CONFIG"


def test_model_or_config_rejects_other_types(tokenizer):
    with pytest.raises(PredictorConfigError):
        Predictor(123, tokenizer)
