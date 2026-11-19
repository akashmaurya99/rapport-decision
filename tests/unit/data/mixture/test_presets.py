"""Acceptance tests: the packaged starter (55k) and full (115.6k) mixtures.

Row counts are the ones declared in docs/dataset-strategy.md §6; each is
asserted within 1% of the document (in practice they are exact).
"""

from __future__ import annotations

import pytest

from src.data.mixture import load_preset, preset_path, sample_mixture

FP = "ef" * 32

# docs/dataset-strategy.md §6 — starter subset (55,000).
STARTER_COUNTS = {
    "choice": 12_000,
    "preference": 20_000,
    "scoring": 5_000,
    "probability": 4_600,
    "abstention": 6_600,
    "risk": 6_800,
}
STARTER_SOURCES = {
    ("preference", "Anthropic/hh-rlhf"): 16_000,
    ("preference", "nvidia/HelpSteer2"): 4_000,
    ("scoring", "nvidia/HelpSteer2"): 5_000,
    ("choice", "cais/mmlu#auxiliary_train"): 12_000,
    ("probability", "truthfulqa/truthful_qa"): 1_600,
    ("probability", "cais/mmlu#auxiliary_train-derived-binary"): 3_000,
    ("abstention", "AbstentionBench-derived"): 6_600,
    ("risk", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"): 6_800,
}

# docs/dataset-strategy.md §6 — final recommended mixture (115,600).
FULL_COUNTS = {
    "preference": 33_000,
    "choice": 30_000,
    "risk": 18_000,
    "abstention": 12_000,
    "probability": 11_600,
    "scoring": 11_000,
}
FULL_SOURCES = {
    ("preference", "Anthropic/hh-rlhf"): 23_000,
    ("preference", "nvidia/HelpSteer2"): 10_000,
    ("choice", "cais/mmlu#auxiliary_train"): 30_000,
    ("risk", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"): 18_000,
    ("abstention", "AbstentionBench-derived"): 12_000,
    ("probability", "truthfulqa/truthful_qa"): 1_600,
    ("probability", "cais/mmlu#auxiliary_train-derived-binary"): 10_000,
    ("scoring", "nvidia/HelpSteer2"): 11_000,
}

DOC_STARTED_TOTAL = sum(STARTER_COUNTS.values())
DOC_FULL_TOTAL = sum(FULL_COUNTS.values())


def within_one_percent(actual: int, expected: int) -> bool:
    return abs(actual - expected) <= max(1.0, 0.01 * expected)


def assert_counts_within_1pct(actual: dict[str, int], expected: dict[str, int]) -> None:
    assert set(actual) == set(expected), f"family sets differ: {set(actual)} vs {set(expected)}"
    for family, rows in expected.items():
        assert within_one_percent(actual[family], rows), (
            f"{family}: {actual[family]} is not within 1% of the documented {rows}"
        )


class TestStarterPreset:
    def test_preset_file_exists(self):
        assert preset_path("starter").is_file()

    def test_family_counts_match_the_document(self):
        declaration = load_preset("starter")
        assert declaration.name == "starter"
        assert declaration.fingerprint == ""  # placeholder; resolved at acquisition time
        assert_counts_within_1pct(declaration.family_rows(), STARTER_COUNTS)
        assert declaration.total_rows == DOC_STARTED_TOTAL == 55_000

    def test_source_counts_match_the_document(self):
        declaration = load_preset("starter")
        flat: dict[tuple[str, str], int] = {}
        for family, sources in declaration.counts().items():
            for source, rows in sources.items():
                flat[(family, source)] = rows
        for key, rows in STARTER_SOURCES.items():
            assert key in flat, f"missing starter allocation {key}"
            assert within_one_percent(flat[key], rows), f"{key}: {flat[key]} vs documented {rows}"
        # HelpSteer2 starter usage: 4,000 preference + 5,000 scoring = 9,000 ≤ 21,362
        assert declaration.counts()["preference"]["nvidia/HelpSteer2"] == 4_000
        assert declaration.counts()["scoring"]["nvidia/HelpSteer2"] == 5_000

    def test_sampling_reproduces_an_identical_order_for_a_fixed_seed(self):
        declaration = load_preset("starter")
        first = sample_mixture(declaration, None, seed=13, fingerprint=FP)
        second = sample_mixture(declaration, None, seed=13, fingerprint=FP)
        assert first.example_ids() == second.example_ids()
        assert first.total_rows == 55_000

    def test_starter_contains_no_ood_source_rows(self):
        declaration = load_preset("starter")
        flat_sources = {source for sources in declaration.counts().values() for source in sources}
        assert "UltraFeedback-binarized" not in flat_sources
        assert not any("ultrafeedback" in source.lower() for source in flat_sources)


class TestFullPreset:
    def test_preset_file_exists(self):
        assert preset_path("full").is_file()

    def test_family_counts_match_the_document(self):
        declaration = load_preset("full")
        assert declaration.name == "full"
        assert declaration.fingerprint == ""
        assert_counts_within_1pct(declaration.family_rows(), FULL_COUNTS)
        assert declaration.total_rows == DOC_FULL_TOTAL == 115_600

    def test_source_counts_match_the_document(self):
        declaration = load_preset("full")
        flat: dict[tuple[str, str], int] = {}
        for family, sources in declaration.counts().items():
            for source, rows in sources.items():
                flat[(family, source)] = rows
        for key, rows in FULL_SOURCES.items():
            assert key in flat, f"missing full allocation {key}"
            assert within_one_percent(flat[key], rows), f"{key}: {flat[key]} vs documented {rows}"
        # HelpSteer2 full usage: 10,000 preference + 11,000 scoring = 21,000 ≤ 21,362
        assert (
            declaration.counts()["preference"]["nvidia/HelpSteer2"]
            + declaration.counts()["scoring"]["nvidia/HelpSteer2"]
            == 21_000
        )
        # HH-RLHF 23,000 of 169,352 ✓
        assert declaration.counts()["preference"]["Anthropic/hh-rlhf"] == 23_000
        # TruthfulQA 1,600 of 1,634 ✓
        assert declaration.counts()["probability"]["truthfulqa/truthful_qa"] == 1_600

    def test_sampling_reproduces_an_identical_order_for_a_fixed_seed(self):
        declaration = load_preset("full")
        first = sample_mixture(declaration, None, seed=47, fingerprint=FP)
        second = sample_mixture(declaration, None, seed=47, fingerprint=FP)
        assert first.example_ids() == second.example_ids()
        assert first.total_rows == 115_600

    def test_full_contains_no_ood_source_rows(self):
        declaration = load_preset("full")
        flat_sources = {source for sources in declaration.counts().values() for source in sources}
        assert not any("ultrafeedback" in source.lower() for source in flat_sources)

    def test_full_mixture_family_shares_match_the_document(self):
        # §6 shares: 28.5 / 25.9 / 15.6 / 10.4 / 10.0 / 9.5 percent.
        declaration = load_preset("full")
        shares = {
            family: rows / declaration.total_rows
            for family, rows in declaration.family_rows().items()
        }
        expected = {
            "preference": 0.285,
            "choice": 0.259,
            "risk": 0.156,
            "abstention": 0.104,
            "probability": 0.100,
            "scoring": 0.095,
        }
        for family, share in expected.items():
            assert shares[family] == pytest.approx(share, abs=0.005), (
                f"{family} share {shares[family]:.4f} differs from the documented {share}"
            )
