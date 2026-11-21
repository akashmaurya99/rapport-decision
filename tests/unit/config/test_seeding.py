"""Tests for src/config/seeding.py — seeding and determinism modes."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.config import ConfigValueError, set_seed


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1)[0])


def test_set_seed_makes_random_numpy_and_torch_reproducible():
    set_seed(13, determinism="relaxed")
    first = _draw()
    set_seed(13, determinism="relaxed")
    second = _draw()
    assert first == second


def test_different_seeds_diverge():
    set_seed(1, determinism="relaxed")
    first = _draw()
    set_seed(2, determinism="relaxed")
    second = _draw()
    assert first != second


def test_cuda_seed_is_set_when_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in this environment")
    set_seed(13, determinism="relaxed")
    first = torch.rand(4, device="cuda")
    set_seed(13, determinism="relaxed")
    second = torch.rand(4, device="cuda")
    torch.testing.assert_close(first, second)


def test_strict_determinism_enables_deterministic_torch():
    set_seed(13, determinism="strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    set_seed(13, determinism="relaxed")  # restore default for other tests
    torch.use_deterministic_algorithms(False)


def test_strict_to_relaxed_restores_relaxed_flags():
    set_seed(13, determinism="strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_seed(13, determinism="relaxed")
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_relaxed_to_strict_reapplies_strict_flags():
    set_seed(13, determinism="relaxed")
    assert not torch.are_deterministic_algorithms_enabled()
    try:
        set_seed(13, determinism="strict")
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        set_seed(13, determinism="relaxed")


def test_out_of_range_seed_fails_typed():
    with pytest.raises(ConfigValueError) as excinfo:
        set_seed(-1)
    assert "seed" in str(excinfo.value)
    with pytest.raises(ConfigValueError):
        set_seed(2**32)


def test_unknown_determinism_mode_fails_typed():
    with pytest.raises(ConfigValueError) as excinfo:
        set_seed(13, determinism="paranoid")
    assert "determinism" in str(excinfo.value)
