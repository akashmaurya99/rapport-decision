"""Guardrails: quarantine refusal and real-row binding for mixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.acquire.manifest import (
    STATUS_ADMITTED,
    STATUS_QUARANTINE,
    SourceManifest,
    compute_shard_set_sha256,
    write_manifest,
)
from src.data.core import normalize_example
from src.data.mixture import (
    MixtureAvailabilityError,
    MixtureQuarantinedSourceError,
    load_source_statuses,
    sample_mixture,
)

FP = "ab" * 32


def _example(family: str, source: str, index: int):
    return normalize_example(
        {
            "task_family": family,
            "goal": f"choose the safest plan for case {family}-{index}",
            "state": f"the rollout window for case {index} is four hours",
            "options": [
                {"text": "roll out to ten percent of traffic first"},
                {"text": "roll out to all of the traffic immediately"},
            ],
            "chosen_index": 0,
            "source": source,
            "license": "synthetic",
            "group_id": f"group-{family}-{index}",
            "split": "train",
        }
    )


class TestQuarantineRefusal:
    def test_mapping_marks_quarantine_as_fatal(self):
        sources = {"choice": {"source": "bad/source", "rows": 2}}
        with pytest.raises(MixtureQuarantinedSourceError) as excinfo:
            sample_mixture(sources, None, 13, FP, source_status={"bad/source": "quarantine"})
        assert excinfo.value.code == "MIXTURE_SOURCE_QUARANTINED"
        assert excinfo.value.source == "bad/source"

    def test_admitted_source_is_allowed(self):
        sources = {"choice": {"source": "good/source", "rows": 2}}
        order = sample_mixture(sources, None, 13, FP, source_status={"good/source": "admitted"})
        assert order.total_rows == 2

    def test_manifest_directory_reports_statuses(self, tmp_path):
        manifest = SourceManifest(
            source="bad/source",
            license=None,
            declared_rows=None,
            observed_rows=0,
            sha256=compute_shard_set_sha256([]),
            acquired_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status=STATUS_QUARANTINE,
            quarantine_reason="no license",
        )
        write_manifest(manifest, tmp_path / "bad")
        assert load_source_statuses(tmp_path) == {"bad/source": STATUS_QUARANTINE}
        sources = {"choice": {"source": "bad/source", "rows": 2}}
        with pytest.raises(MixtureQuarantinedSourceError):
            sample_mixture(sources, None, 13, FP, source_status=tmp_path)

    def test_admitted_manifest_directory_is_accepted(self, tmp_path):
        manifest = SourceManifest(
            source="good/source",
            license="MIT",
            declared_rows=5,
            observed_rows=5,
            sha256=compute_shard_set_sha256([]),
            acquired_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status=STATUS_ADMITTED,
        )
        write_manifest(manifest, tmp_path / "good")
        sources = {"choice": {"source": "good/source", "rows": 2}}
        order = sample_mixture(sources, None, 13, FP, source_status=tmp_path)
        assert order.total_rows == 2


class TestRealRowBinding:
    def test_slots_bind_distinct_real_ids_deterministically(self):
        choice = [_example("choice", "fixture/choice", i) for i in range(6)]
        risk = [_example("risk", "fixture/risk", i) for i in range(5)]
        sources = {
            "choice": {"source": "fixture/choice", "rows": 3},
            "risk": {"source": "fixture/risk", "rows": 2},
        }
        admitted = {"fixture/choice": "admitted", "fixture/risk": "admitted"}
        first = sample_mixture(
            sources, None, 13, FP, available_rows=choice + risk, source_status=admitted
        )
        second = sample_mixture(
            sources, None, 13, FP, available_rows=choice + risk, source_status=admitted
        )

        ids = first.example_ids()
        assert first.slots[0].key == sample_mixture(sources, None, 13, FP).slots[0].key
        assert ids == second.example_ids()
        assert len(set(ids)) == 5
        assert set(ids) <= {row.example_id for row in choice + risk}
        for slot in first.slots:
            family, source = slot.family, slot.source
            rows = choice if source == "fixture/choice" else risk
            assert any(row.task_family == family for row in rows)
        assert {slot.source for slot in first.slots} == {"fixture/choice", "fixture/risk"}

    def test_request_more_rows_than_available_is_fatal(self):
        rows = [_example("choice", "fixture/choice", i) for i in range(2)]
        with pytest.raises(MixtureAvailabilityError) as excinfo:
            sample_mixture(
                {"choice": {"source": "fixture/choice", "rows": 5}},
                None,
                13,
                FP,
                available_rows=rows,
                source_status={"fixture/choice": "admitted"},
            )
        assert excinfo.value.requested == 5
        assert excinfo.value.available == 2

    def test_unavailable_source_family_combination_is_fatal(self):
        rows = [_example("preference", "Anthropic/hh-rlhf", i) for i in range(6)]
        sources = {
            "preference": {"source": "Anthropic/hh-rlhf", "rows": 4},
            "scoring": {"source": "Anthropic/hh-rlhf", "rows": 2},
        }
        with pytest.raises(MixtureAvailabilityError) as excinfo:
            sample_mixture(
                sources,
                None,
                13,
                FP,
                available_rows=rows,
                source_status={
                    "Anthropic/hh-rlhf": "admitted",
                    "nvidia/HelpSteer2": "admitted",
                },
            )
        assert (excinfo.value.family, excinfo.value.source) == (
            "scoring",
            "Anthropic/hh-rlhf",
        )

    def test_declaration_sample_forwards_guardrails(self, tmp_path):
        document = tmp_path / "mixture.yaml"
        document.write_text(
            "families:\n  choice: {source: bad/source, rows: 2}\n", encoding="utf-8"
        )
        from src.data.mixture import load_mixture

        declaration = load_mixture(document)
        with pytest.raises(MixtureQuarantinedSourceError):
            declaration.sample(13, FP, source_status={"bad/source": "quarantine"})
