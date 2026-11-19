"""Acceptance tests for configs/experiments (one YAML per ablation arm).

Criterion 3: configs/experiments contains one YAML per ablation arm A1–A8
declared in docs/evaluation.md §5, plus starter.yaml (the 55k run) and
acceptance.yaml (the 115.6k run, whose config records the withheld OOD source
UltraFeedback-binarized and the Y2 comparator). Every config names its arm,
init, seed set and data fingerprint, model variant, epochs, batch_size and lr.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = REPO_ROOT / "configs" / "experiments"

ARMS = {
    "a1-init.yaml": "A1",
    "a2-readout.yaml": "A2",
    "a3-depth.yaml": "A3",
    "a4-weights.yaml": "A4",
    "a5-head-only.yaml": "A5",
    "a6-data.yaml": "A6",
    "a7-ordering.yaml": "A7",
    "a8-context.yaml": "A8",
}
ALL_CONFIGS = [*ARMS, "starter.yaml", "acceptance.yaml"]
SEED_DOMAIN = {13, 29, 47}
REQUIRED_KEYS = {
    "arm",
    "init",
    "seeds",
    "data_fingerprint",
    "variant",
    "epochs",
    "batch_size",
    "lr",
}


def load(name: str) -> dict:
    path = EXPERIMENTS / name
    assert path.is_file(), f"missing experiment config {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{name} must parse to a mapping"
    return data


class TestCommonContract:
    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_required_keys_present(self, name):
        config = load(name)
        missing = REQUIRED_KEYS - set(config)
        assert not missing, f"{name} is missing required keys: {sorted(missing)}"

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_data_fingerprint_placeholder_key_exists(self, name):
        config = load(name)
        assert "data_fingerprint" in config
        assert config["data_fingerprint"] == ""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_seeds_are_subsets_of_the_registered_set(self, name):
        config = load(name)
        seeds = config["seeds"]
        assert isinstance(seeds, list) and seeds, f"{name}: seeds must be a non-empty list"
        assert set(seeds) <= SEED_DOMAIN, f"{name}: unexpected seeds {seeds}"

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_epochs_batch_size_lr_are_well_formed(self, name):
        config = load(name)
        assert isinstance(config["epochs"], int) and config["epochs"] >= 1
        assert isinstance(config["batch_size"], int) and config["batch_size"] >= 1
        assert isinstance(config["lr"], (int, float)) and config["lr"] > 0

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_variant_is_a_registered_model_variant(self, name):
        config = load(name)
        assert config["variant"] in {"ultra", "compact", "large"}

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_configs_do_not_compute_verdicts(self, name):
        config = load(name)
        assert "verdicts" not in config
        assert not any(isinstance(key, str) and key.startswith("y1") for key in config)


class TestAblationArms:
    @pytest.mark.parametrize(("name", "arm"), ARMS.items())
    def test_arm_label_matches_the_filename(self, name, arm):
        assert load(name)["arm"] == arm

    def test_all_eight_ablation_arms_are_present(self):
        assert sorted(ARMS.values()) == [f"A{i}" for i in range(1, 9)]

    def test_a1_init_alternates(self):
        config = load("a1-init.yaml")
        assert config["init"] == "bge-small"
        alternates = {variant["init"] for variant in config["variants"]}
        assert alternates == {"bge-small", "random", "all-MiniLM-L6-v2"}

    def test_a2_readout_alternates(self):
        config = load("a2-readout.yaml")
        alternates = {variant["readout"] for variant in config["variants"]}
        assert alternates == {"cosine", "cross-attn", "cls-pooled"}

    def test_a3_depth_alternates(self):
        config = load("a3-depth.yaml")
        alternates = {variant["variant"] for variant in config["variants"]}
        assert alternates == {"ultra", "compact", "large"}

    def test_a4_weight_mode_alternates(self):
        config = load("a4-weights.yaml")
        alternates = {variant["weight_mode"] for variant in config["variants"]}
        assert alternates == {"learned", "fixed", "uniform"}

    def test_a5_head_only_alternates(self):
        config = load("a5-head-only.yaml")
        alternates = {variant["head_only"] for variant in config["variants"]}
        assert alternates == {False, True}

    def test_a6_data_ladder_and_seed_subset(self):
        config = load("a6-data.yaml")
        assert set(config["seeds"]) <= SEED_DOMAIN
        assert len(config["seeds"]) < 3, "A6 is the arm allowed a seed subset"
        names = {variant["name"] for variant in config["variants"]}
        assert names == {
            "starter-55k",
            "full-115.6k",
            "full-115.6k-kd",
            "full-115.6k-uniform-order",
        }

    def test_a7_ordering_alternates(self):
        config = load("a7-ordering.yaml")
        alternates = {variant["ordering"] for variant in config["variants"]}
        assert alternates == {"curriculum", "uniform"}

    def test_a8_context_alternates(self):
        config = load("a8-context.yaml")
        alternates = {variant["context"] for variant in config["variants"]}
        assert alternates == {"512-train-1024-infer", "512-512"}
        by_name = {variant["name"]: variant for variant in config["variants"]}
        assert by_name["512-train-1024-infer"]["infer_max_length"] == 1024
        assert by_name["512-512"]["infer_max_length"] == 512


class TestStarterAndAcceptance:
    def test_starter_is_the_55k_run(self):
        config = load("starter.yaml")
        assert config["arm"] == "starter"
        assert config["mixture"] == "starter"
        assert config["mixture_rows"] == 55_000
        assert config["variant"] == "compact"
        assert config["init"] == "bge-small"

    def test_acceptance_is_the_115_6k_run(self):
        config = load("acceptance.yaml")
        assert config["arm"] == "acceptance"
        assert config["mixture"] == "full"
        assert config["mixture_rows"] == 115_600
        assert config["variant"] == "compact"
        assert config["init"] == "bge-small"

    def test_acceptance_records_the_withheld_ood_source(self):
        config = load("acceptance.yaml")
        assert config["ood_source"] == "UltraFeedback-binarized"

    def test_acceptance_records_the_y2_comparator(self):
        config = load("acceptance.yaml")
        assert config["y2_comparator"] == "B3"

    def test_starter_has_no_ood_keys(self):
        config = load("starter.yaml")
        assert "ood_source" not in config
        assert "y2_comparator" not in config

    def test_no_config_trains_on_the_ood_source(self):
        for name in ALL_CONFIGS:
            config = load(name)
            assert config.get("mixture") in {"starter", "full"}
            # The packaged presets carry zero UltraFeedback rows; nothing to re-check here.
