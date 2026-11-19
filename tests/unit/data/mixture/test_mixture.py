"""Unit tests for the deterministic mixture sampler (FR-DATA-07)."""

from __future__ import annotations

import pytest

from src.data.core.schema import TASK_FAMILIES
from src.data.mixture import (
    MixtureDeclaration,
    MixtureFamilyError,
    MixtureFingerprintError,
    MixtureFingerprintMismatchError,
    MixtureInputError,
    MixtureOrder,
    MixtureSource,
    MixtureSourceError,
    load_mixture,
    sample_mixture,
)

FP = "ab" * 32
FP_OTHER = "cd" * 32


def small_sources() -> dict[str, object]:
    return {
        "choice": {"source": "fixture/choice", "rows": 24},
        "preference": [
            {"source": "fixture/pref-a", "rows": 16},
            {"source": "fixture/pref-b", "rows": 8},
        ],
        "risk": {"source": "fixture/risk", "rows": 12},
    }


class TestDeterministicOrder:
    def test_two_calls_produce_identical_ordered_ids(self):
        first = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        second = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        assert isinstance(first, MixtureOrder)
        assert first.example_ids() == second.example_ids()
        assert [slot.key for slot in first.slots] == [slot.key for slot in second.slots]

    def test_order_is_permutation_invariant_in_the_declaration(self):
        sources = small_sources()
        reordered = dict(reversed(list(sources.items())))
        a = sample_mixture(sources, None, seed=13, fingerprint=FP)
        b = sample_mixture(reordered, None, seed=13, fingerprint=FP)
        assert a.example_ids() == b.example_ids()

    def test_seed_changes_the_order(self):
        a = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        b = sample_mixture(small_sources(), None, seed=47, fingerprint=FP)
        assert a.example_ids() != b.example_ids()

    def test_fingerprint_changes_the_order(self):
        a = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        b = sample_mixture(small_sources(), None, seed=13, fingerprint=FP_OTHER)
        assert a.example_ids() != b.example_ids()

    def test_slots_are_ordered_and_carry_the_fingerprint(self):
        order = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        assert [slot.order for slot in order.slots] == list(range(order.total_rows))
        assert order.fingerprint == FP
        assert order.seed == 13
        assert order.total_rows == 60
        keys = [slot.key for slot in order.slots]
        assert keys == sorted(keys)
        for slot in order.slots:
            assert slot.example_id.startswith("mixture:")

    def test_example_ids_are_unique(self):
        order = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        assert len(set(order.example_ids())) == order.total_rows


class TestFamilyRowsAndWeights:
    def test_family_rows_and_totals(self):
        order = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        assert order.family_rows == {"choice": 24, "preference": 24, "risk": 12}
        assert order.total_rows == 60
        assert order.counts == {
            "choice": {"fixture/choice": 24},
            "preference": {"fixture/pref-a": 16, "fixture/pref-b": 8},
            "risk": {"fixture/risk": 12},
        }

    def test_default_weights_are_row_proportional(self):
        order = sample_mixture(small_sources(), None, seed=13, fingerprint=FP)
        assert order.family_shares == pytest.approx({"choice": 0.4, "preference": 0.4, "risk": 0.2})
        assert order.expected_shares == pytest.approx(order.family_shares)
        assert all(error == pytest.approx(0.0) for error in order.share_errors.values())

    def test_weights_argument_overrides_row_default(self):
        order = sample_mixture(
            small_sources(),
            {"choice": 1.0, "preference": 1.0, "risk": 3.0},
            seed=13,
            fingerprint=FP,
        )
        assert order.weights["risk"] == 3.0
        assert order.expected_shares["risk"] == pytest.approx(0.6)

    def test_weights_must_not_name_unknown_families(self):
        with pytest.raises(MixtureInputError):
            sample_mixture(
                small_sources(),
                {"choice": 1.0, "not-a-family": 2.0},
                seed=13,
                fingerprint=FP,
            )

    def test_weights_must_not_name_absent_families(self):
        with pytest.raises(MixtureInputError):
            sample_mixture(
                small_sources(),
                {"choice": 1.0, "scoring": 1.0},
                seed=13,
                fingerprint=FP,
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), "1.0", True, None])
    def test_weights_reject_non_positive_or_non_numeric(self, bad):
        with pytest.raises(MixtureSourceError):
            sample_mixture(small_sources(), {"choice": bad}, seed=13, fingerprint=FP)

    def test_conflicting_entry_weights_are_rejected(self):
        sources = {
            "preference": [
                {"source": "fixture/a", "rows": 4, "weight": 1.0},
                {"source": "fixture/b", "rows": 4, "weight": 2.0},
            ]
        }
        with pytest.raises(MixtureInputError):
            sample_mixture(sources, None, seed=13, fingerprint=FP)

    def test_common_entry_weight_becomes_the_family_weight(self):
        sources = {
            "preference": [
                {"source": "fixture/a", "rows": 4, "weight": 2.0},
                {"source": "fixture/b", "rows": 4, "weight": 2.0},
            ]
        }
        order = sample_mixture(sources, None, seed=13, fingerprint=FP)
        assert order.weights["preference"] == 2.0


class TestInputValidation:
    def test_unknown_family_is_rejected(self):
        with pytest.raises(MixtureFamilyError):
            sample_mixture({"not-a-family": {"source": "x", "rows": 1}}, None, 13, FP)

    def test_all_six_families_are_accepted(self):
        sources = {family: {"source": f"fixture/{family}", "rows": 2} for family in TASK_FAMILIES}
        order = sample_mixture(sources, None, seed=13, fingerprint=FP)
        assert order.total_rows == 12

    def test_sources_must_be_a_non_empty_mapping(self):
        with pytest.raises(MixtureInputError):
            sample_mixture({}, None, 13, FP)
        with pytest.raises(MixtureInputError):
            sample_mixture(["not", "a", "mapping"], None, 13, FP)  # type: ignore[arg-type]

    def test_zero_total_rows_is_rejected(self):
        with pytest.raises(MixtureInputError):
            sample_mixture({"choice": {"source": "fixture/x", "rows": 0}}, None, 13, FP)

    @pytest.mark.parametrize("bad", [True, "13", 1.5, -1, 2**32, None])
    def test_seed_must_be_a_bounded_integer(self, bad):
        with pytest.raises(MixtureInputError):
            sample_mixture(small_sources(), None, bad, FP)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad", ["", "   ", "a" * 63, "A" * 64, "g" * 64, 123, None, "0X" + "ab" * 31]
    )
    def test_fingerprint_must_be_a_lowercase_sha256_hex(self, bad):
        with pytest.raises(MixtureFingerprintError):
            sample_mixture(small_sources(), None, 13, bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "10", None])
    def test_rows_must_be_a_non_negative_int(self, bad):
        with pytest.raises(MixtureSourceError):
            sample_mixture({"choice": {"source": "fixture/x", "rows": bad}}, None, 13, FP)

    def test_missing_source_key_is_rejected(self):
        with pytest.raises(MixtureSourceError):
            sample_mixture({"choice": {"rows": 3}}, None, 13, FP)

    def test_missing_rows_key_is_rejected(self):
        with pytest.raises(MixtureSourceError):
            sample_mixture({"choice": {"source": "fixture/x"}}, None, 13, FP)

    def test_unknown_entry_key_is_rejected(self):
        with pytest.raises(MixtureSourceError):
            sample_mixture(
                {"choice": {"source": "fixture/x", "rows": 3, "licence": "MIT"}},
                None,
                13,
                FP,
            )

    def test_empty_source_name_is_rejected(self):
        with pytest.raises(MixtureSourceError):
            sample_mixture({"choice": {"source": "   ", "rows": 3}}, None, 13, FP)

    def test_duplicate_source_within_a_family_is_rejected(self):
        with pytest.raises(MixtureSourceError):
            sample_mixture(
                {
                    "choice": [
                        {"source": "fixture/dup", "rows": 3},
                        {"source": "fixture/dup", "rows": 4},
                    ]
                },
                None,
                13,
                FP,
            )

    def test_same_source_may_serve_two_families(self):
        sources = {
            "preference": {"source": "nvidia/HelpSteer2", "rows": 4},
            "scoring": {"source": "nvidia/HelpSteer2", "rows": 4},
        }
        order = sample_mixture(sources, None, seed=13, fingerprint=FP)
        assert order.counts["preference"]["nvidia/HelpSteer2"] == 4
        assert order.counts["scoring"]["nvidia/HelpSteer2"] == 4

    def test_mixture_source_instances_are_accepted(self):
        sources = {"choice": MixtureSource(source="fixture/x", rows=3)}
        order = sample_mixture(sources, None, seed=13, fingerprint=FP)
        assert order.total_rows == 3


class TestFingerprintPinning:
    def test_declaration_mismatch_raises_typed_error(self, tmp_path):
        pinned = tmp_path / "mixture.yaml"
        pinned.write_text(
            "name: pinned\n"
            f'fingerprint: "{FP}"\n'
            "families:\n"
            "  choice: {source: fixture/x, rows: 3}\n",
        )
        declaration = load_mixture(pinned)
        with pytest.raises(MixtureFingerprintMismatchError) as excinfo:
            sample_mixture(declaration, None, 13, FP_OTHER)
        assert excinfo.value.expected == FP
        assert excinfo.value.actual == FP_OTHER
        # The pinned fingerprint itself is accepted.
        order = sample_mixture(declaration, None, 13, FP)
        assert order.mixture == "pinned"

    def test_unpinned_declaration_accepts_any_valid_fingerprint(self, tmp_path):
        document = tmp_path / "mixture.yaml"
        document.write_text("name: open\nfamilies:\n  choice: {source: fixture/x, rows: 3}\n")
        declaration = load_mixture(document)
        order = sample_mixture(declaration, None, 13, FP_OTHER)
        assert order.fingerprint == FP_OTHER

    def test_declaration_pinning_invalid_fingerprint_is_rejected(self):
        declaration = MixtureDeclaration(
            name="bad",
            fingerprint="not-a-digest",
            sources={"choice": (MixtureSource(source="fixture/x", rows=3),)},
        )
        with pytest.raises(MixtureFingerprintError):
            sample_mixture(declaration, None, 13, FP)
