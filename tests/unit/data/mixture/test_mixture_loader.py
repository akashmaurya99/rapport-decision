"""Tests for the mixture YAML loader, the format and the packaged presets."""

from __future__ import annotations

import pytest

from src.data.mixture import (
    PRESET_NAMES,
    PRESETS_DIR,
    MixtureConfigError,
    MixtureConfigFieldError,
    MixtureFamilyError,
    MixtureFingerprintMismatchError,
    MixtureSourceError,
    load_mixture,
    load_preset,
    parse_mixture_document,
    preset_path,
)

FP = "12" * 32


class TestLoadMixture:
    def test_round_trip_single_source_family(self, tmp_path):
        document = tmp_path / "mix.yaml"
        document.write_text(
            "name: demo\n"
            "families:\n"
            "  choice: {source: fixture/choice, rows: 10}\n"
        )
        declaration = load_mixture(document)
        assert declaration.name == "demo"
        assert declaration.fingerprint == ""
        assert declaration.counts() == {"choice": {"fixture/choice": 10}}
        assert declaration.total_rows == 10

    def test_multi_source_family_list_form(self, tmp_path):
        document = tmp_path / "mix.yaml"
        document.write_text(
            "families:\n"
            "  preference:\n"
            "    - {source: fixture/a, rows: 4}\n"
            "    - {source: fixture/b, rows: 6, weight: 2.5}\n"
        )
        declaration = load_mixture(document)
        assert [entry.source for entry in declaration.sources["preference"]] == [
            "fixture/a",
            "fixture/b",
        ]
        assert declaration.sources["preference"][1].weight == 2.5
        # name defaults to the file stem
        assert declaration.name == "mix"

    def test_missing_file_is_a_typed_error(self, tmp_path):
        with pytest.raises(MixtureConfigError):
            load_mixture(tmp_path / "does-not-exist.yaml")

    def test_unparsable_yaml_is_a_typed_error(self, tmp_path):
        document = tmp_path / "broken.yaml"
        document.write_text("families: [unclosed\n")
        with pytest.raises(MixtureConfigError):
            load_mixture(document)

    def test_non_mapping_document_is_a_typed_error(self, tmp_path):
        document = tmp_path / "list.yaml"
        document.write_text("- just\n- a\n- list\n")
        with pytest.raises(MixtureConfigError):
            load_mixture(document)

    def test_unknown_top_level_key_is_rejected(self, tmp_path):
        document = tmp_path / "extra.yaml"
        document.write_text(
            "families:\n  choice: {source: fixture/x, rows: 1}\nnote: surprise\n"
        )
        with pytest.raises(MixtureConfigFieldError) as excinfo:
            load_mixture(document)
        assert excinfo.value.field == "note"

    def test_missing_families_key_is_rejected(self, tmp_path):
        document = tmp_path / "nofamilies.yaml"
        document.write_text("name: empty\n")
        with pytest.raises(MixtureConfigFieldError) as excinfo:
            load_mixture(document)
        assert excinfo.value.field == "families"

    def test_empty_families_mapping_is_rejected(self, tmp_path):
        document = tmp_path / "empty.yaml"
        document.write_text("families: {}\n")
        with pytest.raises(MixtureConfigFieldError):
            load_mixture(document)

    def test_family_value_must_be_mapping_or_list(self, tmp_path):
        document = tmp_path / "badfamily.yaml"
        document.write_text("families:\n  choice: 42\n")
        with pytest.raises(MixtureSourceError):
            load_mixture(document)

    def test_empty_family_list_is_rejected(self, tmp_path):
        document = tmp_path / "emptylist.yaml"
        document.write_text("families:\n  choice: []\n")
        with pytest.raises(MixtureSourceError):
            load_mixture(document)

    def test_unknown_family_is_rejected(self, tmp_path):
        document = tmp_path / "badname.yaml"
        document.write_text("families:\n  trilean: {source: fixture/x, rows: 1}\n")
        with pytest.raises(MixtureFamilyError):
            load_mixture(document)

    def test_pinned_fingerprint_must_be_a_sha256_digest(self, tmp_path):
        document = tmp_path / "badfingerprint.yaml"
        document.write_text(
            'fingerprint: "xyz"\nfamilies:\n  choice: {source: fixture/x, rows: 1}\n'
        )
        with pytest.raises(MixtureConfigFieldError):
            load_mixture(document)

    def test_pinned_fingerprint_round_trips(self, tmp_path):
        document = tmp_path / "pinned.yaml"
        document.write_text(
            f'fingerprint: "{FP}"\nfamilies:\n  choice: {{source: fixture/x, rows: 1}}\n'
        )
        declaration = load_mixture(document)
        assert declaration.fingerprint == FP

    def test_pinned_declaration_rejects_a_different_fingerprint(self, tmp_path):
        document = tmp_path / "pinned.yaml"
        document.write_text(
            f'fingerprint: "{FP}"\nfamilies:\n  choice: {{source: fixture/x, rows: 1}}\n'
        )
        declaration = load_mixture(document)
        with pytest.raises(MixtureFingerprintMismatchError):
            declaration.sample(seed=13, fingerprint="ff" * 32)

    def test_unpinned_declaration_sample_requires_a_fingerprint(self, tmp_path):
        document = tmp_path / "open.yaml"
        document.write_text("families:\n  choice: {source: fixture/x, rows: 1}\n")
        declaration = load_mixture(document)
        with pytest.raises(MixtureConfigError):
            declaration.sample(seed=13)
        order = declaration.sample(seed=13, fingerprint=FP)
        assert order.fingerprint == FP


class TestParseMixtureDocument:
    def test_scalar_fingerprint_rejected(self):
        with pytest.raises(MixtureConfigFieldError):
            parse_mixture_document(
                {"fingerprint": 123, "families": {"choice": {"source": "x", "rows": 1}}}
            )

    def test_non_string_name_rejected(self):
        with pytest.raises(MixtureConfigFieldError):
            parse_mixture_document(
                {"name": 7, "families": {"choice": {"source": "x", "rows": 1}}}
            )


class TestPresets:
    def test_preset_names_are_starter_and_full(self):
        assert PRESET_NAMES == ("starter", "full")

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_packaged_presets_load(self, name):
        path = preset_path(name)
        assert path.parent == PRESETS_DIR
        assert path.is_file()
        declaration = load_preset(name)
        assert declaration.name == name
        assert declaration.total_rows > 0

    def test_unknown_preset_is_a_typed_error(self):
        with pytest.raises(MixtureConfigError):
            preset_path("everything")
        with pytest.raises(MixtureConfigError):
            load_preset("everything")
