"""RAPPORT encoder core: configuration, RoPE trunk and parameter accounting."""

from .attention import MultiHeadSelfAttention
from .block import EncoderBlock
from .config import LayerRow, RapportConfig, SpecConstants
from .counting import (
    ParameterCountMismatchError,
    ParameterReport,
    assert_within_spec,
    count_parameters,
    expected_parameter_counts,
)
from .embedding import TrunkEmbedding
from .ffn import SwiGLUFeedForward
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rope
from .trunk import RapportTrunk

__all__ = [
    "EncoderBlock",
    "LayerRow",
    "MultiHeadSelfAttention",
    "ParameterCountMismatchError",
    "ParameterReport",
    "RapportConfig",
    "RapportTrunk",
    "RMSNorm",
    "RotaryEmbedding",
    "SpecConstants",
    "SwiGLUFeedForward",
    "TrunkEmbedding",
    "apply_rope",
    "assert_within_spec",
    "count_parameters",
    "expected_parameter_counts",
]
