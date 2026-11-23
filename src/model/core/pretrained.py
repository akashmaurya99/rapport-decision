"""Audited pretrained-trunk initialization (Stage A).

BGE-small-en-v1.5 is a BERT-style encoder (LayerNorm with bias, GELU FFN),
while the RAPPORT trunk uses pre-RMSNorm (gain only) and bias-free SwiGLU.
A literal "copy every tensor by name" is therefore impossible. This loader
copies the tensors that are architecturally compatible (token embeddings and
the four attention projections), converts nothing silently, and records every
skipped source tensor in a manifest. Any source key outside the declared
allow-list raises instead of being ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .config import RapportConfig


class PretrainedInitError(Exception):
    code = "PRETRAINED_INIT_ERROR"

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.key = key


@dataclass
class InitReport:
    model_id: str
    copied: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    initialized_random: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "copied": dict(self.copied),
            "skipped": dict(self.skipped),
            "initialized_random": list(self.initialized_random),
            "n_copied": len(self.copied),
            "n_skipped": len(self.skipped),
        }


# Source keys that are deliberately not copied, with the reason.
_SKIP_REASONS = {
    "embeddings.position_embeddings.weight": "learned position table dropped in favour of RoPE",
    "embeddings.token_type_embeddings.weight": "RAPPORT has no token-type embedding",
    "embeddings.LayerNorm.weight": "no embedding LayerNorm in RAPPORT trunk",
    "embeddings.LayerNorm.bias": "no embedding LayerNorm in RAPPORT trunk",
    "pooler.dense.weight": "no pooler in RAPPORT trunk",
    "pooler.dense.bias": "no pooler in RAPPORT trunk",
}


def _skip_reason(key: str) -> str | None:
    if key in _SKIP_REASONS:
        return _SKIP_REASONS[key]
    if key.endswith(".bias"):
        return "RAPPORT projections/norms are bias-free"
    if ".intermediate.dense." in key or ".output.dense." in key and ".attention." not in key:
        return "BERT GELU FFN is not SwiGLU; initialized randomly"
    if ".attention.output.LayerNorm." in key or ".output.LayerNorm." in key:
        return "BERT post-LayerNorm is not pre-RMSNorm; gain initialized randomly"
    return None


def load_bge_state_dict(model_id: str = "BAAI/bge-small-en-v1.5") -> dict[str, Tensor]:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise PretrainedInitError("transformers is required for pretrained init", key="transformers") from exc
    try:
        model = AutoModel.from_pretrained(model_id)
    except Exception as exc:
        raise PretrainedInitError(f"could not load {model_id!r}: {exc}", key="model_id") from exc
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def init_trunk_from_bge(
    trunk: torch.nn.Module,
    config: RapportConfig,
    state: dict[str, Tensor] | None = None,
    model_id: str = "BAAI/bge-small-en-v1.5",
) -> InitReport:
    """Copy compatible BGE tensors into ``trunk``; return an audited report."""
    if config.layers != 12 or config.d_model != 384:
        raise PretrainedInitError(
            f"BGE-small init requires the compact 12x384 trunk, got {config.layers}x{config.d_model}",
            key="config",
        )
    if state is None:
        state = load_bge_state_dict(model_id)
    report = InitReport(model_id=model_id)
    target = trunk.state_dict()

    def copy(src: str, dst: str) -> None:
        if src not in state:
            raise PretrainedInitError(f"expected source tensor {src!r} is missing", key=src)
        if dst not in target:
            raise PretrainedInitError(f"expected target tensor {dst!r} is missing", key=dst)
        if state[src].shape != target[dst].shape:
            raise PretrainedInitError(
                f"shape mismatch {src} {tuple(state[src].shape)} -> {dst} {tuple(target[dst].shape)}",
                key=src,
            )
        target[dst].copy_(state[src])
        report.copied[src] = dst

    # token embeddings (compact trunk is non-factorized)
    copy("embeddings.word_embeddings.weight", "embedding.weight.weight")
    for i in range(config.layers):
        prefix = f"encoder.layer.{i}"
        copy(f"{prefix}.attention.self.query.weight", f"blocks.{i}.attn.q_proj.weight")
        copy(f"{prefix}.attention.self.key.weight", f"blocks.{i}.attn.k_proj.weight")
        copy(f"{prefix}.attention.self.value.weight", f"blocks.{i}.attn.v_proj.weight")
        copy(f"{prefix}.attention.output.dense.weight", f"blocks.{i}.attn.o_proj.weight")

    # audit every remaining source key
    for key in state:
        if key in report.copied:
            continue
        reason = _skip_reason(key)
        if reason is None:
            raise PretrainedInitError(f"unexpected source tensor {key!r}", key=key)
        report.skipped[key] = reason

    trunk.load_state_dict(target)
    report.initialized_random = [
        k for k in target
        if k not in set(report.copied.values())
    ]
    return report
