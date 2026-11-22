"""Acceptance tests for the end-to-end acceptance runner (TASK-25).

The runner now fails closed: synthetic/starter runs report ``mode: starter``
with no acceptance verdict, while acceptance-mode runs without real evidence
exit non-zero with ``not-measured`` verdicts. These tests pin both behaviours.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from e2e_test_support import (
    ACCEPTANCE,
    REPO_ROOT,
    SCRIPT,
    STARTER,
    only_run_dir,
    read_json,
    run_e2e,
)

from scripts import e2e_fixtures

TOLERANCE = 1e-4
SMALL = ("--rows-per-family", "4", "--steps", "12")


def _checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_step*.pt"))
    assert checkpoints
    return checkpoints[-1]


def _copy_config(tmp_path: Path, **changes: object) -> Path:
    """Write a temp experiment YAML with selected top-level keys changed."""
    config = dict(read_json_yaml(ACCEPTANCE))
    config.update(changes)
    target = tmp_path / "experiment.yaml"
    target.write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in config.items()) + "\n",
        encoding="utf-8",
    )
    return target


def read_json_yaml(path: Path) -> dict[str, object]:
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


class TestFixtureContract:
    def test_rows_span_all_six_families_and_are_deterministic(self) -> None:
        first = e2e_fixtures.make_rows(3)
        second = e2e_fixtures.make_rows(3)
        assert {row.task_family for row in first} == set(e2e_fixtures.FAMILIES)
        assert len(first) == 18
        assert [row.example_id for row in first] == [row.example_id for row in second]
        assert e2e_fixtures.data_fingerprint(first) == e2e_fixtures.data_fingerprint(second)
        assert e2e_fixtures.data_fingerprint(first) != e2e_fixtures.data_fingerprint(
            e2e_fixtures.make_rows(4)
        )

    def test_batch_is_deterministic_and_carries_multi_task_labels(self) -> None:
        first = e2e_fixtures.make_batch(123)
        second = e2e_fixtures.make_batch(123)
        assert first.keys() == second.keys()
        for key in first:
            assert torch_equal(first[key], second[key])
        assert tuple(first["input_ids"].shape) == (2, e2e_fixtures.SEQ_LEN)
        assert tuple(first["option_mask"].shape) == (2, e2e_fixtures.N_OPTIONS)
        assert tuple(first["risk_targets"].shape) == (2, 8)

    def test_raw_rows_finalize_to_the_same_schema_rows(self) -> None:
        from src.data.core.normalize import normalize_example

        raw = e2e_fixtures.make_raw_rows(3)
        assert len(raw) == 18
        assert e2e_fixtures.make_rows(3) == [normalize_example(row) for row in raw]


def torch_equal(left: object, right: object) -> bool:
    import torch

    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
    return bool(torch.equal(left, right))


class TestE10KillResume:
    def test_resumed_loss_matches_uninterrupted_within_tolerance(self, tmp_path: Path) -> None:
        reference_runs = tmp_path / "reference"
        killed_runs = tmp_path / "killed"

        reference = run_e2e(reference_runs, *SMALL)
        assert reference.returncode == 0, reference.stderr

        killed = run_e2e(
            killed_runs, *SMALL, "--kill-after-step", "5", "--resume", "auto"
        )
        assert killed.returncode == 1, killed.stderr
        assert "simulated preemption after step 5" in killed.stdout

        resumed = run_e2e(killed_runs, *SMALL, "--resume", "auto")
        assert resumed.returncode == 0, resumed.stderr
        assert "training path: RESUMED" in resumed.stdout

        expected = read_json(only_run_dir(reference_runs) / "loss_trajectory.json")
        actual = read_json(only_run_dir(killed_runs) / "loss_trajectory.json")
        assert set(expected) == set(actual) == {str(step) for step in range(1, 13)}
        differences = [abs(expected[step] - actual[step]) for step in expected]
        assert max(differences) <= TOLERANCE

        gate = read_json(only_run_dir(killed_runs) / "resume_gate.json")
        assert gate["passed"] is True
        assert "harness determinism" in gate["scope"]

    def test_resume_prints_fresh_path_without_the_flag(self, tmp_path: Path) -> None:
        result = run_e2e(tmp_path, "--rows-per-family", "4", "--steps", "4")
        assert result.returncode == 0, result.stderr
        assert "training path: FRESH" in result.stdout


class TestAcceptanceReport:
    def test_full_mixture_declaration_and_all_fourteen_verdicts(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        result = run_e2e(runs_dir, "--rows-per-family", "40", "--steps", "4")
        assert result.returncode == 0, result.stderr
        run_dir = only_run_dir(runs_dir)

        state = read_json(run_dir / "state.json")
        assert state["mixture_rows"] == 115600
        assert state["mode"] == "synthetic"
        assert state["seeds_configured"] == [13, 29, 47]
        assert state["seed_used"] == 13
        assert state["multi_seed_evidence"] is False
        assert len(state["split_manifest_fingerprint"]) == 64
        assert len(state["data_fingerprint"]) == 64
        sample = read_json(run_dir / "sample.json")
        assert sample["requested_rows"] == 115600
        assert sample["preset_rows"] == 115600
        assert sample["split_manifest_fingerprint"] == state["split_manifest_fingerprint"]

        split = read_json(run_dir / "split.json")
        assert sum(split["counts"].values()) > 0
        assert split["manifest"]["fingerprint"] == state["split_manifest_fingerprint"]

        report_md = (run_dir / "report.md").read_text(encoding="utf-8")
        for index in range(1, 15):
            assert f"| Y{index} " in report_md, f"Y{index} missing from report.md"
        report = read_json(run_dir / "report.json")
        assert report["mode"] == "starter"
        assert [row["id"] for row in report["verdicts"]] == [
            f"Y{index}" for index in range(1, 15)
        ]
        # Y11 (parameter count) is a genuine structural measurement; every
        # evidence-driven criterion must remain not-measured for a starter run.
        evidence_rows = [row for row in report["verdicts"] if row["id"] != "Y11"]
        assert all(row["status"] == "not-measured" for row in evidence_rows)
        ood = report["robustness"]["ood"]
        assert ood["source"] == "UltraFeedback-binarized"
        assert ood["n_rows"] > 0
        assert ood["empty"] is False

    def test_blocking_failure_exits_nonzero(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps({
                "evidence": {
                    "resume": {
                        "abs_diff": 0.5,
                        "passed": False,
                        "resumed_from": "/runs/killed/checkpoint_step5.pt",
                        "resumed_step": 5,
                        "next_step": 6,
                    }
                }
            }),
            encoding="utf-8",
        )
        runs_dir = tmp_path / "runs"
        result = run_e2e(
            runs_dir,
            "--rows-per-family",
            "4",
            "--steps",
            "4",
            "--evidence-file",
            str(evidence),
        )
        assert result.returncode == 1, result.stdout + result.stderr
        report = read_json(only_run_dir(runs_dir) / "report.json")
        y13 = next(row for row in report["verdicts"] if row["id"] == "Y13")
        assert y13["status"] == "fail"
        assert y13["blocking"] is True

    def test_acceptance_without_evidence_exits_nonzero_honestly(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        result = run_e2e(
            runs_dir, "--rows-per-family", "4", "--steps", "4", "--acceptance"
        )
        assert result.returncode != 0, result.stdout
        report = read_json(only_run_dir(runs_dir) / "report.json")
        assert report["mode"] == "acceptance"
        evidence_rows = [row for row in report["verdicts"] if row["id"] != "Y11"]
        assert all(row["status"] == "not-measured" for row in evidence_rows)
        assert any(row["status"] == "not-measured" for row in report["verdicts"])
        summary = read_json(only_run_dir(runs_dir) / "report_summary.json")
        assert summary["mode"] == "acceptance"
        assert summary["multi_seed_evidence"] is False

    def test_synthetic_evidence_never_passes_acceptance(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "metrics": {
                        "main": {
                            "choice": {
                                "test": {
                                    "accuracy": {
                                        "mean": 0.99,
                                        "n_seeds": 1,
                                        "per_seed": [0.99],
                                    }
                                }
                            }
                        }
                    },
                    "evidence": {"reproducibility_delta_points": 0.0},
                }
            ),
            encoding="utf-8",
        )
        runs_dir = tmp_path / "runs"
        result = run_e2e(
            runs_dir,
            "--rows-per-family",
            "4",
            "--steps",
            "4",
            "--acceptance",
            "--evidence-file",
            str(evidence),
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert "refusing operator evidence" in result.stdout
        report = read_json(only_run_dir(runs_dir) / "report.json")
        evidence_rows = [row for row in report["verdicts"] if row["id"] != "Y11"]
        assert all(row["status"] == "not-measured" for row in evidence_rows)

    def test_starter_result_is_never_reused_by_an_acceptance_run(
        self, tmp_path: Path
    ) -> None:
        """A cached starter report must never satisfy an acceptance exit code."""
        runs_dir = tmp_path / "runs"
        starter = run_e2e(runs_dir, "--rows-per-family", "4", "--steps", "4")
        assert starter.returncode == 0, starter.stderr
        acceptance = run_e2e(
            runs_dir, "--rows-per-family", "4", "--steps", "4", "--acceptance"
        )
        assert acceptance.returncode != 0, acceptance.stdout + acceptance.stderr
        reports = list(runs_dir.glob("*/report.json"))
        modes = {read_json(path)["mode"] for path in reports}
        assert modes == {"starter", "acceptance"}

    def test_environment_seed_overrides_the_configured_first_seed(
        self, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        result = run_e2e(
            runs_dir,
            "--rows-per-family",
            "4",
            "--steps",
            "4",
            env={"RAPPORT_SEED": "77"},
        )
        assert result.returncode == 0, result.stderr
        state = read_json(only_run_dir(runs_dir) / "state.json")
        assert state["seed_used"] == 77

    def test_unsupported_experiment_knob_is_refused(self, tmp_path: Path) -> None:
        """A knob the runner cannot honor must fail loudly, never be ignored."""
        config = tmp_path / "bad.yaml"
        config.write_text(
            "arm: bad\nmixture: starter\nmixture_rows: 10\nseeds: [13]\n"
            "init: bge-small\n",
            encoding="utf-8",
        )
        result = run_e2e(
            tmp_path / "runs", "--rows-per-family", "2", "--steps", "2", config=config
        )
        assert result.returncode != 0
        assert "does not implement" in (result.stdout + result.stderr)

    def test_second_invocation_reuses_completed_stages(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        first = run_e2e(runs_dir, "--rows-per-family", "4", "--steps", "4")
        assert first.returncode == 0, first.stderr
        again = run_e2e(runs_dir, "--rows-per-family", "4", "--steps", "4")
        assert again.returncode == 0, again.stderr
        log = (only_run_dir(runs_dir) / "e2e.jsonl").read_text(encoding="utf-8")
        assert log.count('"event":"stage_skipped"') >= 10

    def test_eval_and_benchmark_records_are_honest(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        result = run_e2e(runs_dir, "--rows-per-family", "4", "--steps", "4")
        assert result.returncode == 0, result.stderr
        log = (only_run_dir(runs_dir) / "e2e.jsonl").read_text(encoding="utf-8")
        assert '"event":"eval"' in log
        assert '"event":"benchmark"' in log
        bench = read_json(only_run_dir(runs_dir) / "bench.json")
        assert bench["latency_measured"] is False
        assert bench["p50_batch1_ms"] is None
        assert bench["throughput_batch32"] is None


class TestStarterSmoke:
    def test_starter_has_no_acceptance_verdict(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        result = run_e2e(
            runs_dir,
            "--starter",
            "--config",
            str(STARTER),
            "--rows-per-family",
            "4",
            "--steps",
            "4",
        )
        assert result.returncode == 0, result.stderr
        assert "no acceptance verdict" in result.stdout
        run_dir = only_run_dir(runs_dir)
        smoke = read_json(run_dir / "smoke.json")
        assert smoke["starter"] is True
        assert smoke["verdict"] is False
        assert smoke["evidence"] is False
        assert smoke["mixture_rows"] == 55000
        report = read_json(run_dir / "report.json")
        assert report["mode"] == "starter"
        evidence_rows = [row for row in report["verdicts"] if row["id"] != "Y11"]
        assert all(row["status"] == "not-measured" for row in evidence_rows)


class TestStageSelection:
    @pytest.mark.parametrize(
        "stage,expected,absent",
        [
            ("data", ("acquire.json", "clean.json", "adapt.json"), ("split.json", "train.json")),
            ("split", ("split.json",), ("train.json", "report.json")),
            ("train", ("train.json", "loss_trajectory.json"), ("report.json",)),
            ("report", ("report.json",), ("bench.json",)),
            ("e2e", ("report.json", "bench.json", "export.json"), ()),
        ],
    )
    def test_stage_selects_execution_depth(
        self, tmp_path: Path, stage: str, expected: tuple[str, ...], absent: tuple[str, ...]
    ) -> None:
        runs_dir = tmp_path / "runs"
        result = run_e2e(
            runs_dir, "--rows-per-family", "4", "--steps", "4", "--stage", stage
        )
        assert result.returncode == 0, result.stdout + result.stderr
        run_dir = only_run_dir(runs_dir)
        for name in expected:
            assert (run_dir / name).is_file(), f"{name} missing for --stage {stage}"
        for name in absent:
            assert not (run_dir / name).exists(), f"{name} present for --stage {stage}"
        assert read_json(run_dir / "state.json")["stage"] == stage

    def test_unknown_stage_is_a_cli_error(self, tmp_path: Path) -> None:
        result = run_e2e(tmp_path, "--rows-per-family", "4", "--stage", "bogus")
        assert result.returncode != 0
        assert "invalid choice" in result.stderr


class TestConfigBoundary:
    def test_unknown_config_key_is_rejected(self, tmp_path: Path) -> None:
        config = _copy_config(tmp_path, bogus_key=1)
        result = run_e2e(
            tmp_path / "runs", "--config", str(config),
            "--rows-per-family", "4", "--steps", "2",
        )
        assert result.returncode != 0
        assert "unknown config key" in result.stderr

    def test_rapport_env_override_reaches_resolved_context(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        env = dict(os.environ, RAPPORT_DETERMINISM="relaxed")
        command = [
            sys.executable, str(SCRIPT), "--config", str(ACCEPTANCE),
            "--synthetic", "--runs-dir", str(runs_dir),
            "--rows-per-family", "4", "--steps", "2",
        ]
        result = subprocess.run(
            command, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=120, check=False,
        )
        assert result.returncode == 0, result.stderr
        state = read_json(only_run_dir(runs_dir) / "state.json")
        assert state["determinism"] == "relaxed"

    @pytest.mark.parametrize("arm", ["../escape", "a/b", "a..b", "", "."])
    def test_path_traversal_arm_is_rejected(self, tmp_path: Path, arm: str) -> None:
        config = _copy_config(tmp_path, arm=arm)
        result = run_e2e(
            tmp_path / "runs", "--config", str(config),
            "--rows-per-family", "4", "--steps", "2",
        )
        assert result.returncode != 0
        # The config wave now validates the arm name itself; either the typed
        # config error or e2e's own path-component guard is an honest refusal.
        assert (
            "unsafe experiment arm" in result.stderr
            or "CONFIG_BAD_VALUE" in result.stderr
            or "config key 'arm'" in result.stderr
        )


class TestTypedErrors:
    @pytest.mark.parametrize("corrupt", ["mismatch", "missing"])
    def test_bad_checkpoint_sidecar_is_a_typed_error(
        self, tmp_path: Path, corrupt: str
    ) -> None:
        runs_dir = tmp_path / "runs"
        killed = run_e2e(
            runs_dir,
            "--rows-per-family",
            "4",
            "--steps",
            "10",
            "--kill-after-step",
            "3",
            "--resume",
            "auto",
        )
        assert killed.returncode == 1, killed.stderr
        checkpoint = _checkpoint(only_run_dir(runs_dir))
        sidecar = Path(str(checkpoint) + ".sha256")
        if corrupt == "mismatch":
            sidecar.write_text("00" * 32 + "  " + checkpoint.name + "\n", encoding="utf-8")
        else:
            sidecar.unlink()

        resumed = run_e2e(
            runs_dir,
            "--rows-per-family",
            "4",
            "--steps",
            "10",
            "--resume",
            "auto",
        )
        assert resumed.returncode != 0
        expected = (
            "CHECKPOINT_HASH_MISMATCH"
            if corrupt == "mismatch"
            else "CHECKPOINT_SIDECAR_MISSING"
        )
        assert expected in resumed.stderr

    def test_resume_from_another_config_is_refused(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        killed = run_e2e(
            runs_dir,
            "--rows-per-family",
            "4",
            "--steps",
            "10",
            "--kill-after-step",
            "3",
            "--resume",
            "auto",
        )
        assert killed.returncode == 1, killed.stderr
        wrong = run_e2e(
            runs_dir,
            "--config",
            str(STARTER),
            "--starter",
            "--rows-per-family",
            "4",
            "--steps",
            "10",
            "--resume",
            "auto",
        )
        assert wrong.returncode != 0
        assert "E2E_CONFIG_FINGERPRINT_MISMATCH" in wrong.stderr


class TestSyntheticFallback:
    def test_no_data_directory_announces_the_fallback(self, tmp_path: Path) -> None:
        command = [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ACCEPTANCE),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--rows-per-family",
            "4",
            "--steps",
            "4",
        ]
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "synthetic fallback" in result.stdout
        assert "NOT real corpus evidence" in result.stdout
