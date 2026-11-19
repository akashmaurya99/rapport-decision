"""Permutation equivariance of the readout and heads (TASK-06, FR-MODEL-02).

The 1e-4 guarantee is stated — and tested here — at the level of the readout
and everything above it, because token positions are inherent to the sequence:
RoPE is the trunk's only position mechanism (spec §4), so re-encoding a
sequence whose option blocks are swapped moves marker token positions and
perturbs the hidden states the trunk emits (spec §6: "positions are inherent
to the sequence"). What FR-MODEL-02 forbids — and these tests pin down — is an
order-dependent pathway *above the trunk*: there is no ordinal embedding, no
slot index anywhere in the readout or heads, so permuting the option slots
permutes the scores by exactly the same permutation.

End-to-end behaviour under re-encoded permutation is measured by TASK-19's
robustness harness; here we also run one full forward pass with a stub trunk
(standing in for position-free hidden states) to prove everything above the
trunk — marker gather, W_q/W_o, cosine, affine, softmax, all six heads — is
exactly slot-permutation-equivariant in the real ``RapportModel.forward`` path.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from head_helpers import example_specs, make_batch

from src.model.core import RapportConfig
from src.model.head import MASK_NEG, RapportModel
from src.model.head.heads import confidence_features


def _model() -> RapportModel:
    torch.manual_seed(20261119)
    return RapportModel(RapportConfig.compact()).eval()


class _ContentStubTrunk(torch.nn.Module):
    """Content-aware, position-free stub trunk.

    Stands in for the trunk so the *entire* head path (validation, marker
    gather, readout, heads, output assembly) is exercised end-to-end while
    isolating it from RoPE's inherent position signal. The hidden state at
    each position is a fixed function of the token window there
    (``h[i] = e[i] + 0.5·e[i+1] + 0.25·e[i+2]``), never of the absolute or
    relative position, so swapping whole option blocks in the input permutes
    the marker hidden states — and hence every score — by exactly the same
    permutation. That is the strongest end-to-end reading of FR-MODEL-02 this
    architecture admits (re-encoding with the real RoPE trunk is inherently
    position-dependent; spec §6).
    """

    def __init__(self, vocab_size: int, d_model: int, seed: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, d_model)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(seed)
            self.emb.weight.copy_(
                torch.randn(vocab_size, d_model, generator=generator)
            )

    def forward(self, input_ids, attention_mask=None, task_ids=None):
        ids = input_ids.to(torch.long)
        e = self.emb(ids)
        e1 = torch.nn.functional.pad(e, (0, 0, 0, 1))[:, 1:]
        e2 = torch.nn.functional.pad(e, (0, 0, 0, 2))[:, 2:]
        return e + 0.5 * e1 + 0.25 * e2


def _fixed_hidden_model(seed: int = 7) -> RapportModel:
    model = _model()
    hidden_args = (model.config.vocab_size, model.config.d_model, seed)
    model.trunk = _ContentStubTrunk(*hidden_args)
    return model


def _permuted_options(options: list[str], permutation: tuple[int, ...]) -> list[str]:
    return [options[i] for i in permutation]


# --- Exact slot-permutation equivariance through the full forward path ------

def _distinct_option_texts(n_options: int) -> list[str]:
    """Option texts whose leading token windows differ, one per slot.

    The generic fixture's options (``"option 0 text"`` …) share every token
    up to the digit, which would make a content-based test vacuous; distinct
    leading words keep the permutation signal non-degenerate.
    """
    base = [
        "alpha bravo",
        "charlie delta",
        "echo foxtrot",
        "golf hotel",
        "india juliet",
        "kilo lima",
        "mike november",
        "oscar papa",
    ]
    return base[:n_options]


@pytest.mark.parametrize("n_options", [2, 3, 4])
@pytest.mark.parametrize("seed", [11, 23])
def test_full_forward_scores_permute_exactly(n_options, seed):
    options = _distinct_option_texts(n_options)
    specs_a = [("goal", "state", options)]
    ids_a, attn_a, mask_a, tasks_a = make_batch(specs_a)

    model = _fixed_hidden_model(seed)
    with torch.no_grad():
        out_a = model(ids_a, attn_a, mask_a, tasks_a)
    # The test is non-vacuous: distinct option content yields distinct scores.
    assert out_a.choice_logits.std() > 0

    for permutation in itertools.permutations(range(n_options)):
        specs_b = [("goal", "state", _permuted_options(options, permutation))]
        ids_b, attn_b, mask_b, tasks_b = make_batch(specs_b)
        with torch.no_grad():
            out_b = model(ids_b, attn_b, mask_b, tasks_b)
        # Scores/logits of the permuted example appear in permuted slot order.
        assert torch.allclose(
            out_b.choice_logits[0], out_a.choice_logits[0][list(permutation)], atol=1e-4
        )
        assert torch.allclose(
            out_b.scores[0], out_a.scores[0][list(permutation)], atol=1e-4
        )
        assert torch.allclose(
            out_b.score_vector[0], out_a.score_vector[0][list(permutation)], atol=1e-4
        )
        # The CLS head is slot-independent: probability/confidence/risk/answerable do not move.
        assert torch.allclose(out_a.probability, out_b.probability, atol=1e-6)
        assert torch.allclose(out_a.confidence, out_b.confidence, atol=1e-6)
        assert torch.equal(out_a.risk, out_b.risk)
        # choice distribution permutes with the same permutation.
        probs_a = torch.softmax(out_a.choice_logits[0], dim=-1)
        probs_b = torch.softmax(out_b.choice_logits[0], dim=-1)
        assert torch.allclose(probs_b, probs_a[list(permutation)], atol=1e-6)


def test_partial_permutation_with_padding_slots():
    model = _fixed_hidden_model()
    specs = [("goal", "state", ["one", "two", "three", "four"]),
             ("goal", "state", ["one", "two"])]
    ids, attention, option_mask, task_ids = make_batch(specs)
    with torch.no_grad():
        out = model(ids, attention, option_mask, task_ids)
    # Row 1 (2 real options + 2 padding): padding is inert.
    assert out.choice_logits[1, 2:].eq(MASK_NEG).all()
    assert out.scores[1, 2:].eq(0.0).all()
    assert torch.softmax(out.choice_logits[1], dim=-1)[2:].eq(0.0).all()


# --- Readout-level exact equivariance against the real trunk output ---------

@pytest.mark.parametrize("n_options", [2, 3, 5, 8])
def test_readout_is_exactly_permutation_equivariant(n_options):
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(n_options, 2))
    with torch.no_grad():
        hidden = model.trunk(ids, attention, task_ids)
        goal_positions, option_positions = model._marker_positions(ids, option_mask)
        base = model.readout(hidden, goal_positions, option_positions, task_ids, option_mask)
        for permutation in itertools.permutations(range(n_options)):
            permuted_positions = option_positions[:, list(permutation)]
            permuted_mask = option_mask[:, list(permutation)]
            permuted = model.readout(
                hidden, goal_positions, permuted_positions, task_ids, permuted_mask
            )
            assert torch.allclose(
                permuted.choice_logits, base.choice_logits[:, list(permutation)], atol=1e-6
            )
            assert torch.allclose(permuted.scores, base.scores[:, list(permutation)], atol=1e-6)
            assert torch.allclose(
                permuted.score_vector, base.score_vector[:, list(permutation)], atol=1e-6
            )
            assert torch.allclose(
                permuted.probabilities, base.probabilities[:, list(permutation)], atol=1e-6
            )


def test_confidence_features_are_permutation_invariant():
    model = _model()
    ids, attention, option_mask, task_ids = make_batch(example_specs(4, 1))
    with torch.no_grad():
        readout = model.readout(hidden=model.trunk(ids, attention, task_ids),
                                goal_positions=model._marker_positions(ids, option_mask)[0],
                                option_positions=model._marker_positions(ids, option_mask)[1],
                                task_ids=task_ids,
                                option_mask=option_mask)
        generator = torch.Generator().manual_seed(3)
        cls_vector = torch.randn(1, model.config.d_model, generator=generator)
        base = confidence_features(cls_vector, readout.probabilities)
        for permutation in itertools.permutations(range(4)):
            permuted = confidence_features(
                cls_vector, readout.probabilities[:, list(permutation)]
            )
            assert torch.allclose(base, permuted, atol=1e-6)


# --- Structural guard: no ordinal embedding / order-dependent path -----------

_BANNED = ("ordinal", "slot_emb", "opt_index", "option_pos", "order_emb", "position")


def test_no_ordinal_or_order_dependent_parameter_names():
    model = _model()
    for name, _ in model.named_parameters():
        lowered = name.lower()
        for banned in _BANNED:
            assert banned not in lowered, f"order-dependent parameter found: {name}"


def test_readout_parameters_carry_no_slot_axis():
    """No parameter in the readout is indexed by option slot.

    The readout may only carry task-axis (num_tasks) or projection parameters.
    """
    model = _model()
    task_axis = (model.readout.log_tau, model.readout.task_scale, model.readout.task_bias)
    for parameter in model.readout.parameters():
        assert parameter.dim() <= 2
    for parameter in task_axis:
        assert parameter.shape == (model.num_tasks,)
    # Projections are bias-free d×d per spec §4 (147K each).
    assert model.readout.query.bias is None
    assert model.readout.option.bias is None
    assert model.readout.query.weight.shape == (model.config.d_model, model.config.d_model)
    assert model.readout.option.weight.shape == (model.config.d_model, model.config.d_model)


def test_re_encoded_permutation_is_finite_and_same_shape():
    """Documented: re-encoding is NOT exactly equivariant (RoPE positions).

    This pins down the honest boundary of the guarantee: with the real trunk, a
    permuted option order re-embeds at different token positions; outputs stay
    finite and correctly shaped but are not claimed to match within 1e-4.
    """
    options = ["alpha beta", "gamma delta", "epsilon zeta"]
    ids_a, attn_a, mask_a, tasks_a = make_batch([("goal", "state", options)])
    ids_b, attn_b, mask_b, tasks_b = make_batch([("goal", "state", list(reversed(options)))])
    model = _model()
    with torch.no_grad():
        out_a = model(ids_a, attn_a, mask_a, tasks_a)
        out_b = model(ids_b, attn_b, mask_b, tasks_b)
    assert out_a.choice_logits.shape == out_b.choice_logits.shape
    assert torch.isfinite(out_b.choice_logits).all()
    assert torch.isfinite(out_b.scores).all()
