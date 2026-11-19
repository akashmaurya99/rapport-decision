"""Per-component parameter counting against the spec (spec §3 and §4).

``count_parameters`` walks a module's parameters, buckets them by component
(embedding / attention / swiglu / norms / other) and derives the non-embedding
trunk. When a spec is passed it also *fails the build*: any component whose
count drifts more than 1% from the spec's stated value raises
``ParameterCountMismatchError`` instead of returning quietly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import nn

from .config import RapportConfig, SpecConstants

_ATTENTION_MARKERS = ("attn", "attention")
_SWIGLU_MARKERS = ("ffn", "swiglu", "gate_proj", "up_proj", "down_proj")
_EMBEDDING_MARKERS = ("embed",)
_NORM_MARKERS = ("norm",)

# Components the counter checks against the spec; "other" is reported, not checked,
# so a head accidentally reaching the trunk surfaces in the report instead of raising.
_CHECKED = ("embedding", "attention", "swiglu", "norms", "trunk")


class ParameterCountMismatchError(ValueError):
    """Raised when implementation and spec disagree by more than the tolerance."""


@dataclass(frozen=True)
class ParameterReport:
    components: Mapping[str, int]
    trunk: int  # attention + swiglu + norms (non-embedding)
    total: int  # everything the walked module owns

    def rows(self) -> dict[str, int]:
        return {**self.components, "trunk": self.trunk, "total": self.total}


def _classify(parameter_name: str) -> str:
    name = parameter_name.lower()
    if any(marker in name for marker in _EMBEDDING_MARKERS):
        return "embedding"
    if any(marker in name for marker in _ATTENTION_MARKERS):
        return "attention"
    if any(marker in name for marker in _SWIGLU_MARKERS):
        return "swiglu"
    if any(marker in name for marker in _NORM_MARKERS):
        return "norms"
    return "other"


def expected_parameter_counts(config: RapportConfig) -> dict[str, int]:
    """Formula-derived counts (spec §3): the cross-check for the spec literals."""
    d = config.d_model
    attn = config.layers * 4 * d * d
    swiglu = config.layers * 3 * d * config.ffn
    norms = config.layers * 2 * d
    if config.embedding_dim != d:
        embedding = config.vocab_size * config.embedding_dim + config.embedding_dim * d
    else:
        embedding = config.vocab_size * d
    return {
        "attention": attn,
        "swiglu": swiglu,
        "norms": norms,
        "trunk": attn + swiglu + norms,
        "embedding": embedding,
        "total": attn + swiglu + norms + embedding,
    }


def count_parameters(
    module: nn.Module,
    spec: RapportConfig | SpecConstants | None = None,
    *,
    tolerance: float = 0.01,
) -> ParameterReport:
    """Walk ``module`` and count parameters per component.

    With ``spec`` given, raises ``ParameterCountMismatchError`` when any checked
    component (embedding, attention, swiglu, norms, trunk) differs from the spec
    by more than ``tolerance`` (relative). Failing loudly here is what makes the
    spec table a build-time contract rather than documentation.
    """
    components: dict[str, int] = {}
    for name, param in module.named_parameters():
        bucket = _classify(name)
        components[bucket] = components.get(bucket, 0) + param.numel()
    trunk = (
        components.get("attention", 0)
        + components.get("swiglu", 0)
        + components.get("norms", 0)
    )
    report = ParameterReport(components=components, trunk=trunk, total=sum(components.values()))

    if spec is not None:
        constants = spec.spec if isinstance(spec, RapportConfig) else spec
        assert_within_spec(report, constants, tolerance=tolerance)
    return report


def assert_within_spec(
    report: ParameterReport,
    spec: SpecConstants,
    tolerance: float = 0.01,
) -> None:
    """Raise ``ParameterCountMismatchError`` listing every component out of tolerance."""
    actual = {
        "embedding": report.components.get("embedding", 0),
        "attention": report.components.get("attention", 0),
        "swiglu": report.components.get("swiglu", 0),
        "norms": report.components.get("norms", 0),
        "trunk": report.trunk,
    }
    expected = {
        "embedding": spec.embedding,
        "attention": spec.attention_total,
        "swiglu": spec.swiglu_total,
        "norms": spec.norms_total,
        "trunk": spec.trunk,
    }
    offenders: list[str] = []
    for component in _CHECKED:
        want, got = expected[component], actual[component]
        if want == 0 or abs(got - want) > tolerance * want:
            delta = abs(got - want) / max(want, 1)
            offenders.append(f"{component}: spec {want:,} vs actual {got:,} (Δ {delta:.2%})")
    if offenders:
        raise ParameterCountMismatchError(
            "implementation disagrees with the architecture spec by more than "
            f"{tolerance:.0%}: " + "; ".join(offenders)
        )
