"""Acceptance tests for the end-to-end acceptance runner (TASK-25)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import (  # noqa: I001
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
        sample = read_json(run_dir / "sample.json")
        assert sample["requested_rows"] == 115600
        assert sample["preset_rows"] == 115600

        report_md = (run_dir / "report.md").read_text(encoding="utf-8")
        for index in range(1, 15):
            assert f"| Y{index} " in report_md, f"Y{index} missing from report.md"
        report = read_json(run_dir / "report.json")
        assert [row["id"] for row in report["verdicts"]] == [
            f"Y{index}" for index in range(1, 15)
        ]
        ood = report["robustness"]["ood"]
        assert ood["source"] == "UltraFeedback-binarized"
        assert ood["n_rows"] > 0
        assert ood["empty"] is False

    def test_blocking_failure_exits_nonzero(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps({"evidence": {"resume": {"abs_diff": 0.5, "passed": False}}}),
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

    def test_second_invocation_reuses_completed_stages(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        first = run_e2e(runs_dir, "--rows-per-family", "4", "--steps", "4")
        assert first.returncode == 0, first.stderr
        again = run_e2e(runs_dir, "--rows-per-family", "4", "--steps", "4")
        assert again.returncode == 0, again.stderr
        log = (only_run_dir(runs_dir) / "e2e.jsonl").read_text(encoding="utf-8")
        assert log.count('"event":"stage_skipped"') >= 10


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
        assert smoke["mixture_rows"] == 55000
        assert not (run_dir / "report.md").exists()
        assert not (run_dir / "report.json").exists()


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
