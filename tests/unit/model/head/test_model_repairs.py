"""Regression tests for the GPU/fp16/validation repairs in the model package.

Covers the C1 MPS crash (CPU marker tensors gathered against MPS hidden states),
the fp16 mask-fill overflow, task-marker derivation, masked-tail markers,
device agreement, task-vector init scale and the dropout capability.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from head_helpers import example_specs, make_batch

from src.model.core import RapportConfig, SpecConstants
from src.model.head import (
    DeviceMismatchError,
    MarkerReadout,
    ModelError,
    OptionMaskError,
    RapportModel,
    TaskMarkerError,
    mask_neg_for,
)
from src.tokenization.tokenizer import OPT_ID, PAD_ID

_OUTPUT_FIELDS = (
    "choice_logits",
    "scores",
    "score_vector",
    "probability",
    "confidence",
    "answerable",
    "risk",
)


def _tiny_config(dropout: float = 0.0) -> RapportConfig:
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
        dropout=dropout,
    )


def _tiny_model(dropout: float = 0.0) -> RapportModel:
    torch.manual_seed(7)
    return RapportModel(_tiny_config(dropout)).eval()


def test_fp16_readout_with_zero_vectors_stays_finite():
    """FP16 eps underflow used to produce 0/0 NaNs for zero hidden states."""
    readout = MarkerReadout(d_model=32, num_tasks=6).half()
    hidden = torch.zeros((1, 4, 32), dtype=torch.float16)
    goal = torch.zeros((1, 1), dtype=torch.long)
    options = torch.tensor([[0, 1]], dtype=torch.long)
    mask = torch.ones((1, 2), dtype=torch.long)
    out = readout(
        hidden, goal, options, torch.zeros((1,), dtype=torch.long), mask
    )
    for value in out.choice_logits.flatten().tolist():
        assert math.isfinite(value)
    assert torch.isfinite(out.probabilities).all()


@pytest.mark.skipif(not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
                    reason="MPS backend unavailable")
def test_positive_dropout_training_works_on_mps():
    """MPS SDPA has no dropout kernel; the explicit path must still train."""
    model = RapportModel(_tiny_config(dropout=0.1)).to("mps").train()
    ids, attention, option_mask, task_ids = make_batch(
        [("alpha beta", "state text", ["one option", "another option"])]
    )
    out = model(
        ids.to("mps"), attention.to("mps"), option_mask.to("mps"), task_ids.to("mps")
    )
    loss = out.choice_logits.float().sum()
    loss.backward()
    assert torch.isfinite(out.choice_logits).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())



def _devices() -> list[pytest.param]:
    devices = [pytest.param(torch.device("cpu"), id="cpu")]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(pytest.param(torch.device("mps"), id="mps"))
    else:
        devices.append(
            pytest.param(
                torch.device("mps"),
                marks=pytest.mark.skip(reason="MPS backend unavailable"),
                id="mps",
            )
        )
    return devices


# --- C1: forward on CPU and MPS, batch 1/3, n in 2..8 --------------------------

@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("n_options", range(2, 9))
def test_forward_grid_is_finite_on_cpu_and_mps(n_options, batch_size, device):
    model = _tiny_model().to(device)
    ids, attention, option_mask, task_ids = make_batch(example_specs(n_options, batch_size))
    tensors = [t.to(device) for t in (ids, attention, option_mask, task_ids)]
    with torch.no_grad():
        out = model(*tensors)
    assert out.choice_logits.shape == (batch_size, n_options)
    assert out.choice_logits.device.type == device.type
    for field in _OUTPUT_FIELDS:
        value = getattr(out, field)
        assert torch.isfinite(value).all(), f"{field} not finite on {device}"


# --- fp16: dtype-aware finite mask fill ----------------------------------------

def test_readout_fp16_padded_rows_and_fully_padded_row_stay_finite():
    torch.manual_seed(0)
    readout = MarkerReadout(8, num_tasks=2).eval().half()
    hidden = torch.randn(2, 12, 8, dtype=torch.float16)
    goal = torch.tensor([0, 3])
    options = torch.tensor([[1, 2, 4, 5], [6, 7, 8, 9]])
    task_ids = torch.tensor([0, 1])
    option_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long)

    out = readout(hidden, goal, options, task_ids, option_mask)
    assert out.choice_logits.dtype == torch.float16
    assert torch.isfinite(out.choice_logits).all()
    assert torch.isfinite(out.scores).all()
    assert torch.isfinite(out.probabilities).all()
    assert out.choice_logits[0, 2:].eq(mask_neg_for(torch.float16)).all()
    probs = torch.softmax(out.choice_logits[0], dim=-1)
    assert torch.isclose(probs.sum().float(), torch.tensor(1.0), atol=1e-3)
    assert probs[2:].eq(0).all()

    # Fully padded row: finite fill must not turn into -inf/NaN.
    all_pad = torch.zeros(1, 4, dtype=torch.long)
    padded = readout(hidden[:1], goal[:1], options[:1], task_ids[:1], all_pad)
    assert torch.isfinite(padded.choice_logits).all()
    assert torch.isfinite(torch.softmax(padded.choice_logits, dim=-1)).all()


def test_full_tiny_model_fp16_forward_with_padded_rows_is_finite():
    torch.manual_seed(0)
    model = _tiny_model().half()
    # n=2 padded to width 4: exercises the padded-slot mask fill in fp16.
    ids, attention, option_mask, task_ids = make_batch(
        example_specs(2, 1) + example_specs(4, 1)
    )
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    assert out.choice_logits.dtype == torch.float16
    for field in _OUTPUT_FIELDS:
        value = getattr(out, field)
        assert torch.isfinite(value).all(), f"{field} not finite in fp16 forward"
    assert out.choice_logits[0, 2:].eq(mask_neg_for(torch.float16)).all()
    assert out.scores[0, 2:].eq(0).all()


# --- marker / device validation ------------------------------------------------

def test_left_padded_batch_raises_typed_error():
    model = _tiny_model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    ids = F.pad(ids, (2, 0), value=PAD_ID)
    attention = F.pad(attention, (2, 0), value=0)
    with pytest.raises(TaskMarkerError) as excinfo:
        model(ids, attention, option_mask, task_ids)
    assert isinstance(excinfo.value, ModelError)
    assert excinfo.value.code == "MODEL_TASK_MARKER_INVALID"


def test_opt_marker_in_masked_tail_is_not_counted_and_raises():
    """A marker only in the masked tail must not satisfy the declared count."""
    model = _tiny_model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    ids = F.pad(ids, (0, 1), value=PAD_ID)
    attention = F.pad(attention, (0, 1), value=0)
    ids[0, -1] = OPT_ID  # real <opt> is gone; this one is attention-masked
    option_mask = torch.tensor([[1, 1, 1]])  # declares three markers

    with pytest.raises(OptionMaskError) as excinfo:
        model(ids, attention, option_mask, task_ids)
    assert "unmasked" in excinfo.value.message


def test_marker_positions_ignore_masked_tail_positions():
    model = _tiny_model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(3, 1))
    ids = F.pad(ids, (0, 1), value=PAD_ID)
    attention = F.pad(attention, (0, 1), value=0)
    ids[0, -1] = OPT_ID  # extra marker, but masked out
    goal_positions, option_positions = model._marker_positions(ids, option_mask, attention)
    assert int(ids[0, goal_positions[0]]) != OPT_ID
    for position in option_positions[0].tolist():
        assert int(ids[0, position]) == OPT_ID  # never the masked tail index


@pytest.mark.parametrize("field", ["attention_mask", "option_mask", "task_ids"])
def test_mixed_device_inputs_raise_typed_model_error(field):
    model = _tiny_model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(2, 1))
    tensors = {
        "input_ids": ids,
        "attention_mask": attention,
        "option_mask": option_mask,
        "task_ids": task_ids,
    }
    tensors[field] = tensors[field].to("meta")  # always present, never CPU
    with pytest.raises(DeviceMismatchError) as excinfo:
        model(**tensors)
    assert isinstance(excinfo.value, ModelError)
    assert excinfo.value.code == "MODEL_DEVICE_MISMATCH"


# --- task-vector init scale ----------------------------------------------------

def test_task_vector_init_std_matches_token_embedding_scale():
    config = _tiny_config()
    model = RapportModel(config, num_tasks=64)
    weight = model.trunk.task_vector.weight
    assert weight.shape == (64, config.d_model)
    std = weight.detach().std().item()
    assert math.isclose(std, math.sqrt(config.d_model), rel_tol=0.1)


# --- dropout capability --------------------------------------------------------

def test_dropout_zero_is_bitwise_identical_to_default():
    ids, attention, option_mask, task_ids = make_batch(example_specs(4, 2))
    torch.manual_seed(11)
    baseline = RapportModel(_tiny_config()).eval()
    torch.manual_seed(11)
    explicit_zero = RapportModel(_tiny_config(dropout=0.0)).eval()

    for (base_name, base_param), (zero_name, zero_param) in zip(
        baseline.named_parameters(), explicit_zero.named_parameters()
    ):
        assert base_name == zero_name
        assert torch.equal(base_param, zero_param)

    with torch.no_grad():
        default_out = baseline(ids, attention, option_mask, task_ids)
        zero_out = explicit_zero(ids, attention, option_mask, task_ids)
    for field in _OUTPUT_FIELDS:
        assert torch.equal(getattr(default_out, field), getattr(zero_out, field))

    # train() with p=0 must also be bitwise identical to eval().
    explicit_zero.train()
    with torch.no_grad():
        train_out = explicit_zero(ids, attention, option_mask, task_ids)
    for field in _OUTPUT_FIELDS:
        assert torch.equal(getattr(train_out, field), getattr(zero_out, field))


def test_positive_dropout_changes_training_outputs_but_not_eval():
    model = _tiny_model(dropout=0.1)
    block = model.trunk.blocks[0]
    assert block.attn.dropout_p == pytest.approx(0.1)
    assert block.attn_dropout.p == pytest.approx(0.1)
    assert block.ffn_dropout.p == pytest.approx(0.1)

    ids, attention, option_mask, task_ids = make_batch(example_specs(4, 2))
    model.train()
    with torch.no_grad():
        train_out = model(ids, attention, option_mask, task_ids)
    model.eval()
    with torch.no_grad():
        eval_out = model(ids, attention, option_mask, task_ids)
    assert any(
        not torch.equal(getattr(train_out, field), getattr(eval_out, field))
        for field in _OUTPUT_FIELDS
    )
