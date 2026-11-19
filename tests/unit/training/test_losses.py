"""Unit tests for the multi-task loss functions (TASK-11).

Every expected value below is hand-computed from the fixture tensors (see the
comments per test) and asserted within 1e-5. The fixture is 2 examples x 3
options; example 1 has one padding slot (slot 2, ``MASK_NEG``) so the masked
paths are exercised everywhere.

Fixture (module-level constants below):

* ``choice_logits = [[1.0, 2.0, 0.5], [0.5, 1.5, -1e9]]``
* ``scores        = [[0.9, 0.8, 0.2], [0.4, 0.6, 0.0]]``
* ``probability   = [[0.7], [0.3]]``, ``confidence = [[0.6], [0.35]]``,
  ``answerable   = [[0.8], [0.2]]``
* ``risk          = [[0.9, 0.1, 0.8, 0.05, 0.5, 0.2, 0.3, 0.7],
                     [0.2, 0.8, 0.1, 0.6, 0.7, 0.4, 0.6, 0.1]]``
* ``option_mask   = [[1, 1, 1], [1, 1, 0]]``
* ``chosen_index  = [2, 1]``
* ``score_targets = [[0.65, 0.75, 0.25], [0.45, 0.65, 0.99]]`` (0.99 is the
  padding slot and must be ignored)
* ``probability_target = [0.9, 0.1]``
* ``answerable_target  = [1.0, 0.0]`` with ``answerable_mask = [True, False]``
* ``risk_targets  = [[1, 0, 1, 0, 0, 0, 0, 0], [0] * 8]`` with
  ``risk_mask = [True, False]``
* ``risk_pos_weight = [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]``
* ``preference_pairs = [[0, 1, 2], [0, 1, 0], [1, 1, 2]]`` — the third pair
  references the padding slot of example 1 and must be dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.head import RapportOutput  # noqa: E402
from src.training.losses import (  # noqa: E402
    DEFAULT_WEIGHTS,
    LossNotFiniteError,
    compute_losses,
)

MASK_NEG = -1e9
TOL = 1e-5

CHOICE_LOGITS = [[1.0, 2.0, 0.5], [0.5, 1.5, MASK_NEG]]
SCORES = [[0.9, 0.8, 0.2], [0.4, 0.6, 0.0]]
PROBABILITY = [[0.7], [0.3]]
CONFIDENCE = [[0.6], [0.35]]
ANSWERABLE = [[0.8], [0.2]]
RISK = [
    [0.9, 0.1, 0.8, 0.05, 0.5, 0.2, 0.3, 0.7],
    [0.2, 0.8, 0.1, 0.6, 0.7, 0.4, 0.6, 0.1],
]
OPTION_MASK = [[1, 1, 1], [1, 1, 0]]
CHOSEN_INDEX = [2, 1]
SCORE_TARGETS = [[0.65, 0.75, 0.25], [0.45, 0.65, 0.99]]
PROBABILITY_TARGET = [0.9, 0.1]
ANSWERABLE_TARGET = [1.0, 0.0]
ANSWERABLE_MASK = [True, False]
RISK_TARGETS = [[1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
RISK_MASK = [True, False]
RISK_POS_WEIGHT = [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
PREFERENCE_PAIRS = [[0, 1, 2], [0, 1, 0], [1, 1, 2]]


def _output() -> RapportOutput:
    return RapportOutput(
        choice_logits=torch.tensor(CHOICE_LOGITS, dtype=torch.float32),
        scores=torch.tensor(SCORES, dtype=torch.float32),
        probability=torch.tensor(PROBABILITY, dtype=torch.float32),
        confidence=torch.tensor(CONFIDENCE, dtype=torch.float32),
        answerable=torch.tensor(ANSWERABLE, dtype=torch.float32),
        risk=torch.tensor(RISK, dtype=torch.float32),
        score_vector=torch.zeros(2, 3, dtype=torch.float32),
    )


def _mask() -> torch.Tensor:
    return torch.tensor(OPTION_MASK, dtype=torch.long)


def _kwargs(**overrides):
    kwargs = {
        "chosen_index": torch.tensor(CHOSEN_INDEX, dtype=torch.long),
        "preference_pairs": torch.tensor(PREFERENCE_PAIRS, dtype=torch.long),
        "score_targets": torch.tensor(SCORE_TARGETS, dtype=torch.float32),
        "probability_target": torch.tensor(PROBABILITY_TARGET, dtype=torch.float32),
        "answerable_target": torch.tensor(ANSWERABLE_TARGET, dtype=torch.float32),
        "answerable_mask": torch.tensor(ANSWERABLE_MASK),
        "risk_targets": torch.tensor(RISK_TARGETS, dtype=torch.float32),
        "risk_mask": torch.tensor(RISK_MASK),
        "risk_pos_weight": torch.tensor(RISK_POS_WEIGHT, dtype=torch.float32),
    }
    kwargs.update(overrides)
    return kwargs


def _fixture(**overrides):
    return compute_losses(_output(), _mask(), **_kwargs(**overrides))


# --- hand-computed per-term values (fixture) ---------------------------------


def test_choice_hand_computed():
    # ex0: log(exp(1) + exp(2) + exp(0.5)) - 0.5 = 1.9643687841079447
    # ex1: log(exp(0.5) + exp(1.5)) - 1.5  = 0.3132616875182228 (padding ignored)
    # mean = 1.1388152358130839
    result = _fixture()
    assert result.choice.item() == pytest.approx(1.1388152358130839, abs=TOL)
    assert result.active["choice"] is True


def test_choice_masks_unset_rows():
    # chosen_index = [2, -1]: example 1 is masked out entirely.
    # choice = ex0 CE = 1.9643687841079447; confidence target for ex1 undefined.
    result = _fixture(chosen_index=torch.tensor([2, -1], dtype=torch.long))
    assert result.choice.item() == pytest.approx(1.9643687841079447, abs=TOL)
    assert result.confidence.item() == pytest.approx(0.916290731874155, abs=TOL)  # ex0 only
    assert result.active["choice"] is True


def test_choice_undefined_when_chosen_absent():
    result = _fixture(chosen_index=None)
    assert result.choice.item() == 0.0
    assert result.active["choice"] is False


def test_choice_and_confidence_masked_when_all_chosen_unset():
    result = _fixture(chosen_index=torch.tensor([-1, -1], dtype=torch.long))
    assert result.choice.item() == 0.0
    assert result.confidence.item() == 0.0
    assert result.active["choice"] is False
    assert result.active["confidence"] is False


def test_preference_hand_computed():
    # pair (0, winner 1, loser 2): diff = 2.0 - 0.5 = 1.5
    #   loss = log(1 + e^-1.5) = 0.20141327798275246
    # pair (0, winner 1, loser 0): diff = 2.0 - 1.0 = 1.0
    #   loss = log(1 + e^-1.0) = 0.31326168751822286
    # pair (1, winner 1, loser 2): loser slot 2 is padding -> dropped
    # mean = 0.2573374827504877
    result = _fixture()
    assert result.preference.item() == pytest.approx(0.2573374827504877, abs=TOL)
    assert result.active["preference"] is True


def test_preference_empty_pair_set_contributes_zero():
    result = _fixture(preference_pairs=torch.zeros((0, 3), dtype=torch.long))
    assert result.preference.item() == 0.0
    assert result.active["preference"] is False


def test_preference_all_pairs_on_padding_contributes_zero():
    # every pair references the padding slot (example 1, slot 2) -> dropped
    pairs = torch.tensor([[1, 1, 2], [1, 2, 1]], dtype=torch.long)
    result = _fixture(preference_pairs=pairs)
    assert result.preference.item() == 0.0
    assert result.active["preference"] is False


def test_preference_out_of_range_example_raises():
    pairs = torch.tensor([[7, 0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range example"):
        _fixture(preference_pairs=pairs)


def test_score_hand_computed():
    # entries (padding slot of example 1 excluded):
    #   |0.9-0.65| = 0.25 > 0.1  -> 0.1 * (0.25 - 0.05) = 0.02
    #   |0.8-0.75| = 0.05 <= 0.1 -> 0.5 * 0.05^2       = 0.00125
    #   |0.2-0.25| = 0.05 <= 0.1 ->                      0.00125
    #   |0.4-0.45| = 0.05 <= 0.1 ->                      0.00125
    #   |0.6-0.65| = 0.05 <= 0.1 ->                      0.00125
    # mean over 5 entries = 0.005 ; the padding target 0.99 is ignored
    result = _fixture()
    assert result.score.item() == pytest.approx(0.005, abs=TOL)
    assert result.active["score"] is True


def test_score_fully_masked_contributes_zero():
    result = _fixture(score_mask=torch.tensor([False, False]))
    assert result.score.item() == 0.0
    assert result.active["score"] is False


def test_score_partial_mask_averages_only_labelled_examples():
    # example 0 only: (0.02 + 0.00125 + 0.00125) / 3 = 0.0075
    result = _fixture(score_mask=torch.tensor([True, False]))
    assert result.score.item() == pytest.approx(0.0075, abs=TOL)


def test_probability_hand_computed():
    # ex0: bce = -(0.9*ln0.7 + 0.1*ln0.3) = 0.4414047299774528, brier = 0.2^2 = 0.04
    # ex1: bce = -(0.1*ln0.3 + 0.9*ln0.7) = 0.44140472997745284, brier = 0.2^2 = 0.04
    # each: bce + 0.5*brier = 0.4614047299774528(5) ; mean = 0.46140472997745285
    result = _fixture()
    assert result.probability.item() == pytest.approx(0.46140472997745285, abs=TOL)
    assert result.active["probability"] is True


def test_probability_fully_masked_contributes_zero():
    result = _fixture(probability_mask=torch.tensor([False, False]))
    assert result.probability.item() == 0.0
    assert result.active["probability"] is False


def test_confidence_hand_computed():
    # targets: ex0 argmax(choice_logits)=1 (logit 2.0) != chosen 2 -> 0.0
    #          ex1 argmax over real slots =1 == chosen 1          -> 1.0
    # ce ex0: -ln(1 - 0.6)  = 0.916290731874155
    # ce ex1: -ln(0.35)     = 1.0498221244986778
    # mean = 0.9830564281864165
    result = _fixture()
    assert result.confidence.item() == pytest.approx(0.9830564281864165, abs=TOL)
    assert result.active["confidence"] is True


def test_confidence_unavailable_when_chosen_absent():
    result = _fixture(chosen_index=None)
    assert result.confidence.item() == 0.0
    assert result.active["confidence"] is False


def test_abstain_hand_computed():
    # example 1 is masked out (answerable unset); mean over example 0 only:
    # -ln(0.8) = 0.2231435513142097
    result = _fixture()
    assert result.abstain.item() == pytest.approx(0.2231435513142097, abs=TOL)
    assert result.active["abstain"] is True


def test_abstain_masked_out_when_answerable_unset():
    result = _fixture(answerable_mask=torch.tensor([False, False]))
    assert result.abstain.item() == 0.0
    assert result.active["abstain"] is False


def test_abstain_unset_term_excluded_from_total():
    full = _fixture()
    unset = _fixture(answerable_target=None)
    all_masked = _fixture(answerable_mask=torch.tensor([False, False]))
    # abstain weight is 1.0, so the total drops by exactly the abstain term
    unset_total = unset.total.item()
    all_masked_total = all_masked.total.item()
    assert unset_total == pytest.approx(full.total.item() - full.abstain.item(), abs=TOL)
    assert all_masked_total == pytest.approx(full.total.item() - full.abstain.item(), abs=TOL)
    assert unset.active["abstain"] is False
    assert all_masked.active["abstain"] is False


def test_risk_hand_computed():
    # example 0 only (risk_mask = [True, False]); explicit pos_weight
    # [2, 1, 3, 1, 1, 1, 1, 1], target [1, 0, 1, 0, 0, 0, 0, 0]:
    #   k=0 (y=1, pw=2): 2 * -ln(0.9)          = 0.21072103131565256
    #   k=1 (y=0):       -ln(1 - 0.1)          = 0.10536051565782628
    #   k=2 (y=1, pw=3): 3 * -ln(0.8)          = 0.6694306539426291
    #   k=3 (y=0):       -ln(1 - 0.05)         = 0.05129329438755058
    #   k=4 (y=0):       -ln(1 - 0.5)          = 0.6931471805599453
    #   k=5 (y=0):       -ln(1 - 0.2)          = 0.2231435513142097
    #   k=6 (y=0):       -ln(1 - 0.3)          = 0.35667494393873245
    #   k=7 (y=0):       -ln(1 - 0.7)          = 1.203972804325936
    # mean over categories = 0.4392179969303102
    result = _fixture()
    assert result.risk.item() == pytest.approx(0.4392179969303102, abs=TOL)
    assert result.active["risk"] is True


def test_risk_masked_when_no_risk_labels_present():
    result = _fixture(risk_mask=torch.tensor([False, False]))
    assert result.risk.item() == 0.0
    assert result.active["risk"] is False
    none_supplied = _fixture(risk_targets=None)
    assert none_supplied.risk.item() == 0.0
    assert none_supplied.active["risk"] is False


def test_risk_unset_term_excluded_from_total():
    full = _fixture()
    unset = _fixture(risk_targets=None)
    all_masked = _fixture(risk_mask=torch.tensor([False, False]))
    # risk weight is 0.5, so the total drops by exactly 0.5 * the risk term
    unset_total = unset.total.item()
    all_masked_total = all_masked.total.item()
    assert unset_total == pytest.approx(full.total.item() - 0.5 * full.risk.item(), abs=TOL)
    assert all_masked_total == pytest.approx(full.total.item() - 0.5 * full.risk.item(), abs=TOL)


# Fixture D — uneven category frequency, pos_weight derived from the batch.
# 4 examples, all predictions 0.5, category positives per example:
#   cat0: [1,0,0,0] -> pos 1 neg 3 -> pw 3.0
#   cat1: [1,1,0,0] -> pos 2 neg 2 -> pw 1.0
#   cat2: [0,0,0,0] -> pos 0        -> pw 1.0 (neutral convention)
#   cat3: [1,1,1,1] -> pos 4 neg 0 -> pw 0.0
#   cat4: [0,1,0,1] -> pos 2 neg 2 -> pw 1.0
#   cat5: [0,0,1,0] -> pos 1 neg 3 -> pw 3.0
#   cat6: [1,1,1,0] -> pos 3 neg 1 -> pw 1/3
#   cat7: [0,1,1,1] -> pos 3 neg 1 -> pw 1/3
# with p = 0.5 every per-example term is (pw * y + (1 - y)) * ln 2; the
# category means average to 0.6065037829899521.
_RISK_D_TARGETS = [
    [1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 1],
]


def _output_d() -> RapportOutput:
    return RapportOutput(
        choice_logits=torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0], [1.0, 0.5]]),
        scores=torch.full((4, 2), 0.5),
        probability=torch.full((4, 1), 0.5),
        confidence=torch.full((4, 1), 0.5),
        answerable=torch.full((4, 1), 0.5),
        risk=torch.full((4, 8), 0.5),
        score_vector=torch.zeros(4, 2),
    )


def test_risk_derived_pos_weight_hand_computed():
    result = compute_losses(
        _output_d(),
        torch.ones(4, 2, dtype=torch.long),
        risk_targets=torch.tensor(_RISK_D_TARGETS, dtype=torch.float32),
    )
    assert result.risk.item() == pytest.approx(0.6065037829899521, abs=TOL)


def test_risk_derived_pos_weight_matches_explicit():
    explicit = torch.tensor([3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 1 / 3, 1 / 3])
    derived = compute_losses(
        _output_d(),
        torch.ones(4, 2, dtype=torch.long),
        risk_targets=torch.tensor(_RISK_D_TARGETS, dtype=torch.float32),
    )
    explicit_result = compute_losses(
        _output_d(),
        torch.ones(4, 2, dtype=torch.long),
        risk_targets=torch.tensor(_RISK_D_TARGETS, dtype=torch.float32),
        risk_pos_weight=explicit,
    )
    assert derived.risk.item() == pytest.approx(explicit_result.risk.item(), abs=1e-6)


# --- totals -------------------------------------------------------------------


def test_total_hand_computed():
    # 1.0*choice + 1.0*pref + 0.5*score + 0.5*prob + 0.5*conf + 1.0*abstain + 0.5*risk
    # = 2.563635847424871
    result = _fixture()
    assert result.total.item() == pytest.approx(2.563635847424871, abs=TOL)


def test_total_with_all_ones_weights():
    result = _fixture(weights={name: 1.0 for name in DEFAULT_WEIGHTS})
    assert result.total.item() == pytest.approx(3.507975424971961, abs=TOL)


def test_total_with_zero_weight_excludes_term():
    result = _fixture(weights={"abstain": 0.0})
    full = _fixture()
    assert result.total.item() == pytest.approx(full.total.item() - full.abstain.item(), abs=TOL)


def test_weights_validation():
    with pytest.raises(ValueError, match="unknown terms"):
        _fixture(weights={"nope": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        _fixture(weights={"choice": -1.0})
    with pytest.raises(ValueError, match="finite"):
        _fixture(weights={"choice": float("inf")})


def test_default_weights_match_spec_table():
    assert DEFAULT_WEIGHTS == {
        "choice": 1.0,
        "preference": 1.0,
        "score": 0.5,
        "probability": 0.5,
        "confidence": 0.5,
        "abstain": 1.0,
        "risk": 0.5,
    }


# --- masking of padding / inert slots ----------------------------------------


def test_padding_slot_log_odds_are_inert_for_choice():
    other = RapportOutput(
        choice_logits=torch.tensor([[1.0, 2.0, 0.5], [0.5, 1.5, -500.0]], dtype=torch.float32),
        scores=torch.tensor(SCORES, dtype=torch.float32),
        probability=torch.tensor(PROBABILITY, dtype=torch.float32),
        confidence=torch.tensor(CONFIDENCE, dtype=torch.float32),
        answerable=torch.tensor(ANSWERABLE, dtype=torch.float32),
        risk=torch.tensor(RISK, dtype=torch.float32),
        score_vector=torch.zeros(2, 3),
    )
    base = compute_losses(_output(), _mask(), **_kwargs())
    shifted = compute_losses(other, _mask(), **_kwargs())
    assert shifted.choice.item() == pytest.approx(base.choice.item(), abs=TOL)


def test_expected_padding_slot_utility_target_ignored():
    # covered inside test_score_hand_computed (padding target is 0.99, an
    # impossible utility); assert explicitly that removing it changes nothing
    result = _fixture(
        score_targets=torch.tensor([[0.65, 0.75, 0.25], [0.45, 0.65, 0.0]], dtype=torch.float32)
    )
    assert result.score.item() == pytest.approx(0.005, abs=TOL)


# --- degenerate inputs --------------------------------------------------------


def test_single_real_option_row_raises():
    with pytest.raises(ValueError, match="at least two"):
        compute_losses(_output(), torch.tensor([[1, 0, 0], [1, 1, 0]]), **_kwargs())


def test_chosen_index_pointing_at_padding_raises():
    with pytest.raises(ValueError, match="padding"):
        _fixture(chosen_index=torch.tensor([2, 2], dtype=torch.long))


def test_option_mask_must_be_binary():
    with pytest.raises(ValueError, match="binary"):
        compute_losses(_output(), torch.tensor([[1.0, 0.5, 0.0], [1, 1, 0]]), **_kwargs())


def test_empty_batch_raises():
    output = RapportOutput(
        choice_logits=torch.zeros(0, 3),
        scores=torch.zeros(0, 3),
        probability=torch.zeros(0, 1),
        confidence=torch.zeros(0, 1),
        answerable=torch.zeros(0, 1),
        risk=torch.zeros(0, 8),
        score_vector=torch.zeros(0, 3),
    )
    with pytest.raises(ValueError, match="non-empty"):
        compute_losses(output, torch.zeros(0, 3, dtype=torch.long))


def test_bool_option_mask_accepted():
    result = compute_losses(_output(), torch.tensor(OPTION_MASK, dtype=torch.bool), **_kwargs())
    assert result.choice.item() == pytest.approx(1.1388152358130839, abs=TOL)


# --- non-finite handling (FR-TRAIN-04 / E5) -----------------------------------


def _assert_typed_non_finite(excinfo, component: str):
    err = excinfo.value
    assert isinstance(err, LossNotFiniteError)
    assert err.code == "LOSS_NOT_FINITE"
    assert err.component == component
    assert "LOSS_NOT_FINITE" in str(err)


def test_non_finite_choice_logits_raise():
    output = _output()
    logits = output.choice_logits.clone()
    logits[0, 0] = float("nan")
    broken = RapportOutput(
        choice_logits=logits,
        scores=output.scores,
        probability=output.probability,
        confidence=output.confidence,
        answerable=output.answerable,
        risk=output.risk,
        score_vector=output.score_vector,
    )
    with pytest.raises(LossNotFiniteError) as excinfo:
        compute_losses(broken, _mask(), **_kwargs())
    _assert_typed_non_finite(excinfo, "choice")


def test_non_finite_risk_output_raises():
    output = _output()
    risk = output.risk.clone()
    risk[0, 2] = float("inf")
    broken = RapportOutput(
        choice_logits=output.choice_logits,
        scores=output.scores,
        probability=output.probability,
        confidence=output.confidence,
        answerable=output.answerable,
        risk=risk,
        score_vector=output.score_vector,
    )
    with pytest.raises(LossNotFiniteError) as excinfo:
        compute_losses(broken, _mask(), **_kwargs())
    _assert_typed_non_finite(excinfo, "risk")


def test_confidently_wrong_probability_raises_typed():
    # p = 0.0 with target 1.0 -> BCE = inf; must surface, never be clamped
    output = _output()
    broken = RapportOutput(
        choice_logits=output.choice_logits,
        scores=output.scores,
        probability=torch.tensor([[0.0], [0.3]]),
        confidence=output.confidence,
        answerable=output.answerable,
        risk=output.risk,
        score_vector=output.score_vector,
    )
    with pytest.raises(LossNotFiniteError) as excinfo:
        compute_losses(broken, _mask(), **_kwargs())
    _assert_typed_non_finite(excinfo, "probability")


def test_confidently_wrong_confidence_raises_typed():
    # target 0 (top-1 wrong) but confidence = 1.0 -> BCE = inf
    output = _output()
    broken = RapportOutput(
        choice_logits=output.choice_logits,
        scores=output.scores,
        probability=output.probability,
        confidence=torch.tensor([[1.0], [0.35]]),
        answerable=output.answerable,
        risk=output.risk,
        score_vector=output.score_vector,
    )
    with pytest.raises(LossNotFiniteError) as excinfo:
        compute_losses(broken, _mask(), **_kwargs())
    _assert_typed_non_finite(excinfo, "confidence")


def test_confidently_wrong_abstain_raises_typed():
    # answerable = 0.0 with target 1.0 -> BCE = inf
    output = _output()
    broken = RapportOutput(
        choice_logits=output.choice_logits,
        scores=output.scores,
        probability=output.probability,
        confidence=output.confidence,
        answerable=torch.tensor([[0.0], [0.2]]),
        risk=output.risk,
        score_vector=output.score_vector,
    )
    with pytest.raises(LossNotFiniteError) as excinfo:
        compute_losses(broken, _mask(), **_kwargs())
    _assert_typed_non_finite(excinfo, "abstain")


def test_non_finite_loss_never_reaches_total():
    # sanity: the exception carries the component, so a caller (TASK-12) can
    # count and skip; the total is never produced on this path.
    output = _output()
    broken = RapportOutput(
        choice_logits=output.choice_logits,
        scores=output.scores,
        probability=output.probability,
        confidence=output.confidence,
        answerable=output.answerable,
        risk=torch.full((2, 8), float("nan")),
        score_vector=output.score_vector,
    )
    with pytest.raises(LossNotFiniteError) as excinfo:
        compute_losses(broken, _mask(), **_kwargs())
    assert excinfo.value.component == "risk"


# --- gradient flows / interface shape ------------------------------------------


def test_total_is_differentiable_and_backward_runs():
    logits = torch.tensor(CHOICE_LOGITS, dtype=torch.float32, requires_grad=True)
    output = RapportOutput(
        choice_logits=logits,
        scores=torch.tensor(SCORES, dtype=torch.float32, requires_grad=True),
        probability=torch.tensor(PROBABILITY, dtype=torch.float32, requires_grad=True),
        confidence=torch.tensor(CONFIDENCE, dtype=torch.float32, requires_grad=True),
        answerable=torch.tensor(ANSWERABLE, dtype=torch.float32, requires_grad=True),
        risk=torch.tensor(RISK, dtype=torch.float32, requires_grad=True),
        score_vector=torch.zeros(2, 3),
    )
    result = compute_losses(output, _mask(), **_kwargs())
    result.total.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_loss_breakdown_fields():
    result = _fixture()
    for field_name in (
        "choice",
        "preference",
        "score",
        "probability",
        "confidence",
        "abstain",
        "risk",
        "total",
    ):
        assert getattr(result, field_name).dim() == 0
    assert set(result.active) == set(DEFAULT_WEIGHTS)
    assert set(result.weights) == set(DEFAULT_WEIGHTS)
