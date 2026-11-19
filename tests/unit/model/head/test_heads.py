"""Unit tests for the micro-heads and the confidence-feature builder.

Hand-computed expectations for the confidence-features helper; shape and range
checks for the heads (spec §4/§6). See ``heads.py`` for the documented choice
of absorbing the answerable output into the probability head.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.head import (
    RISK_CATEGORIES,
    ConfidenceHead,
    ProbabilityHead,
    RiskHead,
    confidence_features,
)


def _cls(batch: int = 2, d: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4242)
    return torch.randn(batch, d, generator=generator)


# --- Probability head ----------------------------------------------------------

def test_probability_head_shape_and_range():
    head = ProbabilityHead(16)
    out = head(_cls(4, 16))
    assert out.shape == (4, 1)
    assert bool((out >= 0).all()) and bool((out <= 1).all())


def test_probability_head_rejects_non_2d_input():
    with pytest.raises(ValueError):
        ProbabilityHead(4)(torch.randn(2, 2, 4))


# --- Confidence head -----------------------------------------------------------

def test_confidence_head_shape_and_range():
    head = ConfidenceHead(16)
    out = head(torch.randn(3, 19))
    assert out.shape == (3, 1)
    assert bool((out >= 0).all()) and bool((out <= 1).all())


def test_confidence_head_rejects_non_2d_input():
    with pytest.raises(ValueError):
        ConfidenceHead(4)(torch.randn(1, 1, 9))


# --- Risk head -----------------------------------------------------------------

def test_risk_head_shape_and_range():
    head = RiskHead(16, RISK_CATEGORIES)
    out = head(_cls(5, 16))
    assert out.shape == (5, RISK_CATEGORIES)
    assert bool((out >= 0).all()) and bool((out <= 1).all())


# --- confidence_features: hand-computed fixture --------------------------------

def test_confidence_features_match_hand_computed_values():
    cls_vector = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # One padding slot (exactly 0) and three real options.
    probabilities = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    features = confidence_features(cls_vector, probabilities)

    p_max = 0.5
    margin = 0.5 - 0.25
    entropy = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, p_max, margin, entropy]])
    assert torch.allclose(features, expected, atol=1e-6)


def test_confidence_features_ignore_padding_slots():
    cls_vector = torch.zeros(1, 4)
    with_padding = confidence_features(cls_vector, torch.tensor([[0.6, 0.4, 0.0, 0.0]]))
    without_padding = confidence_features(cls_vector, torch.tensor([[0.6, 0.4]]))
    assert torch.allclose(with_padding, without_padding, atol=1e-6)


def test_confidence_features_uniform_distribution():
    cls_vector = torch.zeros(1, 2)
    probabilities = torch.full((1, 4), 0.25)
    features = confidence_features(cls_vector, probabilities)
    assert torch.isclose(features[0, -3], torch.tensor(0.25))  # p_max
    assert torch.isclose(features[0, -2], torch.tensor(0.0))  # margin
    assert torch.isclose(features[0, -1], torch.tensor(math.log(4)), atol=1e-6)  # entropy


def test_confidence_features_rejects_non_2d_probabilities():
    with pytest.raises(ValueError):
        confidence_features(torch.zeros(1, 4), torch.zeros(4))


# --- Readout softmax contract ---------------------------------------------------

def test_choice_distribution_is_a_probability_distribution():
    from src.model.head import MarkerReadout

    torch.manual_seed(0)
    readout = MarkerReadout(8, num_tasks=2).eval()
    hidden = torch.randn(2, 12, 8)
    goal = torch.tensor([0, 3])
    options = torch.tensor([[1, 2], [4, 5]])
    task_ids = torch.tensor([0, 1])
    option_mask = torch.ones(2, 2, dtype=torch.long)
    with torch.no_grad():
        out = readout(hidden, goal, options, task_ids, option_mask)
    assert torch.allclose(
        out.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6
    )
    assert out.probabilities.shape == (2, 2)
    assert out.choice_logits.shape == (2, 2)
    assert out.score_vector.shape == (2, 2)
