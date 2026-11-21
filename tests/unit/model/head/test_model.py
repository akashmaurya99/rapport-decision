"""Tests for the RAPPORT model assembly: shapes, validation and budget.

Covers TASK-06 acceptance criteria 1, 3 and 4 (E2, FR-MODEL-03/04). Criterion 2
(permutation equivariance) lives in ``test_permutation.py``.
"""

from __future__ import annotations

import pytest
import torch
from head_helpers import example_specs, make_batch

from src.model.core import RapportConfig, count_parameters
from src.model.head import (
    NUM_TASKS,
    RISK_CATEGORIES,
    AttentionMaskError,
    GoalMarkerError,
    InvalidOptionCountError,
    InvalidTaskIdError,
    OptionMaskError,
    RapportModel,
    RapportOutput,
    mask_neg_for,
)
from src.tokenization.tokenizer import GOAL_ID, OPT_ID, PAD_ID


def _model() -> RapportModel:
    torch.manual_seed(20261119)
    return RapportModel(RapportConfig.compact()).eval()


# --- Criterion 1 / 3: shapes for batch 1 and 32, n in 2..8 -------------------

@pytest.mark.parametrize("n_options", range(2, 9))
@pytest.mark.parametrize("batch_size", [1, 32])
def test_forward_returns_exactly_n_scores_per_example(n_options, batch_size):
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(n_options, batch_size))
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    assert isinstance(out, RapportOutput)
    assert out.choice_logits.shape == (batch_size, n_options)
    assert out.scores.shape == (batch_size, n_options)
    assert out.score_vector.shape == (batch_size, n_options)
    assert out.probability.shape == (batch_size, 1)
    assert out.confidence.shape == (batch_size, 1)
    assert out.answerable.shape == (batch_size, 1)
    assert out.risk.shape == (batch_size, RISK_CATEGORIES)


@pytest.mark.parametrize("n_options", range(2, 9))
def test_all_six_heads_return_declared_shapes_and_ranges(n_options):
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(n_options, 2))
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    for one in (out.probability, out.confidence, out.answerable, out.risk, out.scores):
        assert bool((one >= 0).all()) and bool((one <= 1).all())
    assert torch.isfinite(out.choice_logits).all()
    assert torch.isfinite(out.score_vector).all()


# --- E2: n of 1 or 9 refused by input validation ----------------------------

def test_slot_width_1_is_refused_before_compute():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 2))
    with torch.no_grad(), pytest.raises(InvalidOptionCountError) as excinfo:
        model(ids, attention, option_mask[:, :1], task_ids)
    assert excinfo.value.code == "MODEL_INVALID_OPTION_COUNT"


def test_slot_width_9_is_refused_before_compute():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 2))
    wide = torch.zeros(2, 9, dtype=torch.int64)
    wide[:, :2] = option_mask[:, :2]
    with torch.no_grad(), pytest.raises(InvalidOptionCountError) as excinfo:
        model(ids, attention, wide, task_ids)
    assert excinfo.value.code == "MODEL_INVALID_OPTION_COUNT"


def test_row_with_single_real_option_is_refused():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 2))
    option_mask[0] = torch.tensor([1, 0])  # one real option in row 0
    with torch.no_grad(), pytest.raises(InvalidOptionCountError):
        model(ids, attention, option_mask, task_ids)


def test_validation_happens_before_any_trunk_compute():
    model = _model()
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("trunk must not run when input validation fails")

    model.trunk.forward = spy  # instance-level patch; nn.Module allows plain attrs
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 2))
    with torch.no_grad(), pytest.raises(InvalidOptionCountError):
        model(ids, attention, option_mask[:, :1], task_ids)
    assert calls == []


# --- Input-contract failures are typed ----------------------------------------

def test_non_prefix_option_mask_is_rejected():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(3, 1))
    option_mask[0] = torch.tensor([1, 0, 1])  # padding slot before an option
    with torch.no_grad(), pytest.raises(OptionMaskError):
        model(ids, attention, option_mask, task_ids)


def test_marker_count_mismatch_is_rejected():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    option_mask = torch.tensor([[1, 1, 1]])  # declares three, ids carry two markers
    ids = torch.nn.functional.pad(ids, (0, 4), value=PAD_ID)
    attention = torch.nn.functional.pad(attention, (0, 4))
    with torch.no_grad(), pytest.raises(OptionMaskError) as excinfo:
        model(ids, attention, option_mask, task_ids)
    assert "marker" in excinfo.value.message


def test_missing_goal_marker_is_rejected():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    ids = ids.clone()
    ids[ids == GOAL_ID] = PAD_ID  # remove the only <goal>
    with torch.no_grad(), pytest.raises(GoalMarkerError):
        model(ids, attention, option_mask, task_ids)


@pytest.mark.parametrize("bad", [-1, NUM_TASKS, 99])
def test_out_of_range_task_id_is_rejected(bad):
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    task_ids = task_ids.clone()
    task_ids[0] = bad
    with torch.no_grad(), pytest.raises(InvalidTaskIdError):
        model(ids, attention, option_mask, task_ids)


def test_float_task_ids_are_rejected():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    with torch.no_grad(), pytest.raises(InvalidTaskIdError):
        model(ids, attention, option_mask, task_ids.float())


def test_attention_mask_shape_and_values_are_checked():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    with torch.no_grad(), pytest.raises(AttentionMaskError):
        model(ids, attention[:, :-1], option_mask, task_ids)
    bad = attention.clone()
    bad[0, 0] = 2
    with torch.no_grad(), pytest.raises(AttentionMaskError):
        model(ids, bad, option_mask, task_ids)


def test_option_mask_marker_positions_align_with_gather():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(3, 1))
    positions = model._marker_positions(ids, option_mask)[1]
    for slot, position in enumerate(positions[0].tolist()):
        assert int(ids[0, position]) == OPT_ID


# --- Padding slots are inert --------------------------------------------------

def test_padding_slots_are_inert_in_mixed_batch():
    model = _model()
    specs = example_specs(2, 1) + example_specs(4, 1)
    ids, attention, option_mask, task_ids = make_batch(specs)
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    assert out.choice_logits[0, 2:].eq(mask_neg_for(out.choice_logits.dtype)).all()
    assert out.scores[0, 2:].eq(0.0).all()
    assert out.score_vector[0, 2:].eq(0.0).all()
    first_row_probs = torch.softmax(out.choice_logits[0], dim=-1)
    assert torch.isclose(first_row_probs.sum(), torch.tensor(1.0))
    assert first_row_probs[2:].eq(0.0).all()


# --- Criterion 4: one joint forward pass, no per-option pass -----------------

def test_single_forward_pass_encodes_state_and_all_options_jointly():
    model = _model()
    trunk_calls, readout_calls = [], []

    trunk_forward = model.trunk.forward
    readout_forward = model.readout.forward

    def trunk_spy(*args, **kwargs):
        trunk_calls.append((args, kwargs))
        return trunk_forward(*args, **kwargs)

    def readout_spy(*args, **kwargs):
        readout_calls.append((args, kwargs))
        return readout_forward(*args, **kwargs)

    model.trunk.forward = trunk_spy
    model.readout.forward = readout_spy
    ids, attention, option_mask, task_ids = make_batch(example_specs(4, 2))
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    assert len(trunk_calls) == 1
    assert len(readout_calls) == 1
    # All four options were scored from the one joint encoding.
    assert out.scores.shape == (2, 4)


def test_answerable_is_absorbed_into_probability_head():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(3, 2))
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    # Documented contract: one shared 384→192→1 stack emits both (heads.py).
    assert torch.equal(out.answerable, out.probability)


def test_eval_forward_is_deterministic():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 2))
    with torch.no_grad():
        first = model(ids, attention, option_mask, task_ids)
        second = model(ids, attention, option_mask, task_ids)
    for a, b in zip(
        (first.choice_logits, first.scores, first.probability, first.confidence,
         first.answerable, first.risk, first.score_vector),
        (second.choice_logits, second.scores, second.probability, second.confidence,
         second.answerable, second.risk, second.score_vector),
    ):
        assert torch.equal(a, b)


# --- Parameter budget (heads ≤ 0.6M; trunk/embedding match TASK-05's spec) ----

def test_heads_add_at_most_0_6m_parameters_and_trunk_matches_spec():
    config = RapportConfig.compact()
    model = _model()
    with torch.no_grad():
        report = count_parameters(model, config)  # raises on trunk/embedding drift
    spec = config.spec
    heads = report.total - spec.embedding - spec.trunk
    assert heads <= 600_000
    # The spec's heads_total (497,370) reconciles exactly without the per-task
    # temperature; the implementation carries 8 extra log_tau parameters.
    assert abs(report.total - spec.total) <= 16
    assert report.trunk == spec.trunk


@pytest.mark.parametrize("variant", ["ultra", "large"])
def test_ultra_and_large_heads_reconcile_exactly_with_spec(variant):
    """U/L spec literals were wrong; the full model now reconciles exactly.

    Mirrors the compact reconciliation above, but the corrected U/L literals
    are exact (+8 log_tau included), so no tolerance is needed.
    """
    config = getattr(RapportConfig, variant)()
    model = RapportModel(config)
    report = count_parameters(model)
    spec = config.spec
    heads = report.total - spec.embedding - spec.trunk
    assert report.trunk == spec.trunk
    assert report.components["embedding"] == spec.embedding
    assert heads == spec.heads_total
    assert report.total == spec.total


def test_choice_logits_are_listwise_softmax_compatible():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(3, 1))
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    probs = torch.softmax(out.choice_logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-6)
    # Listwise CE for a target in range is finite (TASK-11 consumes this).
    target = torch.tensor([1])
    loss = torch.nn.functional.cross_entropy(out.choice_logits, target)
    assert torch.isfinite(loss)
