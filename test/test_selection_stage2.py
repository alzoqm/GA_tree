import numpy as np
import pandas as pd
import pytest


from selection.loader import SelectionPaths
from selection.pipeline import SelectionPipeline


def _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir) -> SelectionPaths:
    return SelectionPaths(
        base_dir=tmp_path,
        final_features_pkl=None,
        final_features_csv=data_path,
        model_config_path=model_path,
        feature_active_path=active_path,
        artifact_dir=artifact_dir,
    )


def test_stage2_stability_selection_and_diversity(tmp_path):
    rng = np.random.default_rng(2025)
    n = 720
    base = rng.standard_normal(n)
    beta_source = 0.6 * base + 0.4 * rng.standard_normal(n)
    noise = rng.standard_normal((n, 4))
    target = 0.5 * base + 0.35 * beta_source + 0.15 * rng.standard_normal(n)

    frame = pd.DataFrame(
        {
            "target": target,
            "alpha_signal_main": base + 0.05 * rng.standard_normal(n),
            "alpha_signal_aux": base * 0.9 + 0.12 * rng.standard_normal(n),
            "beta_signal_main": beta_source + 0.05 * rng.standard_normal(n),
            "beta_signal_aux": beta_source * 0.85 + 0.1 * rng.standard_normal(n),
            "gamma_noise": noise[:, 0],
            "delta_noise": noise[:, 1],
            "epsilon_noise": noise[:, 2],
            "flag_event": rng.integers(0, 2, n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)
    model_path = tmp_path / "model.yaml"
    model_path.write_text("feature_num: {}\nfeature_bool: []\n", encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 7, "device": "cpu"},
        "cpcv": {"folds": 5, "test_folds": 1, "min_trades_per_split": 96, "shuffle_seed": 3},
        "training_strategy": {"max_hold_period_bars": 30},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 120,
        "stage1_screening": {"combine": {"method": "simes", "alpha": 0.15, "fdr": "bh"}},
        "stage2_selection": {
            "corr_cap": 0.9,
            "max_features": 4,
            "inner_ratio": 0.5,
            "stability_selection": {
                "enabled": True,
                "subsamples": 24,
                "sample_fraction": 0.6,
                "cutoff": 0.55,
                "max_features": 4,
                "alpha_grid": [0.8, 0.3, 0.1],
                "random_state": 99,
                "max_iter": 150,
                "tolerance": 1e-4,
            },
        },
        "stage3_diversity": {
            "enabled": True,
            "method": "facility_location",
            "top_k": 3,
            "family_cap": {"max_per_family": 1},
        },
        "stage4_calibration": {
            "vif_threshold": 8.0,
            "robust_quantiles": [0.01, 0.99],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.stage2 is not None
    assert artifacts.stage2.method == "stability_selection"
    main_freq = artifacts.stage2.frequencies.get("alpha_signal_main", 0.0)
    noise_freq = artifacts.stage2.frequencies.get("gamma_noise", 0.0)
    assert main_freq > noise_freq
    assert "alpha_signal_main" in artifacts.stage2.selected_features
    assert len(artifacts.stage2.selected_features) <= selection_cfg["stage2_selection"]["stability_selection"]["max_features"]

    assert artifacts.diversity is not None
    family_labels = {name.split("_", 1)[0] for name in artifacts.diversity.selected}
    assert len(family_labels) == len(artifacts.diversity.selected)

    assert "alpha_signal_main" in artifacts.selected_features
    assert any(name.startswith("beta_signal") for name in artifacts.selected_features)


def test_stage2_disabled_falls_back_to_diversity(tmp_path):
    rng = np.random.default_rng(3310)
    n = 360
    base = rng.standard_normal(n)
    aux = 0.5 * base + 0.5 * rng.standard_normal(n)
    frame = pd.DataFrame(
        {
            "target": base + 0.2 * rng.standard_normal(n),
            "sig_primary": base + 0.05 * rng.standard_normal(n),
            "sig_aux": aux + 0.05 * rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)
    model_path = tmp_path / "model.yaml"
    model_path.write_text("feature_num: {}\nfeature_bool: []\n", encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 11, "device": "cpu"},
        "cpcv": {"folds": 4, "test_folds": 1, "min_trades_per_split": 64, "shuffle_seed": 5},
        "training_strategy": {"max_hold_period_bars": 24},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 80,
        "stage1_screening": {"combine": {"method": "simes", "alpha": 0.1, "fdr": "bh"}},
        "stage2_selection": {
            "corr_cap": 0.85,
            "max_features": 3,
            "inner_ratio": 0.5,
            "stability_selection": {
                "enabled": False,
            },
        },
        "stage3_diversity": {
            "enabled": True,
            "method": "facility_location",
            "top_k": 2,
            "family_cap": {"max_per_family": 1},
        },
        "stage4_calibration": {
            "vif_threshold": 6.0,
            "robust_quantiles": [0.02, 0.98],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.stage2 is None
    assert artifacts.diversity is not None
    assert len(artifacts.diversity.selected) <= selection_cfg["stage3_diversity"]["top_k"]
    assert "sig_primary" in artifacts.selected_features


def test_stage2_high_cutoff_triggers_fallback(tmp_path):
    rng = np.random.default_rng(8842)
    n = 640
    base = rng.standard_normal(n)
    gamma = 0.7 * base + 0.3 * rng.standard_normal(n)
    frame = pd.DataFrame(
        {
            "target": base + 0.1 * rng.standard_normal(n),
            "feat_main": base + 0.05 * rng.standard_normal(n),
            "feat_gamma": gamma + 0.05 * rng.standard_normal(n),
            "feat_aux": 0.4 * base + 0.6 * rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)
    model_path = tmp_path / "model.yaml"
    model_path.write_text("feature_num: {}\nfeature_bool: []\n", encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 19, "device": "cpu"},
        "cpcv": {"folds": 5, "test_folds": 1, "min_trades_per_split": 96, "shuffle_seed": 2},
        "training_strategy": {"max_hold_period_bars": 28},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 120,
        "stage1_screening": {"combine": {"method": "simes", "alpha": 0.05, "fdr": "bh"}},
        "stage2_selection": {
            "corr_cap": 0.98,
            "max_features": 2,
            "inner_ratio": 0.6,
            "stability_selection": {
                "enabled": True,
                "subsamples": 20,
                "sample_fraction": 0.6,
                "cutoff": 1.1,  # Force fallback since frequencies <= 1
                "max_features": 2,
                "alpha_grid": [0.9, 0.5, 0.2],
                "random_state": 13,
                "max_iter": 120,
                "tolerance": 1e-4,
            },
        },
        "stage3_diversity": {
            "enabled": False,
        },
        "stage4_calibration": {
            "vif_threshold": 20.0,
            "robust_quantiles": [0.01, 0.99],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.stage2 is not None
    assert artifacts.stage2.models_evaluated > 0
    # Fallback should have selected features despite cutoff > 1
    assert len(artifacts.stage2.selected_features) == selection_cfg["stage2_selection"]["max_features"]
    assert "feat_main" in artifacts.selected_features
    assert "feat_gamma" in artifacts.selected_features


def test_knockoff_chain_records_attempts(tmp_path):
    rng = np.random.default_rng(7721)
    n = 720
    latent = rng.standard_normal(n)
    target = latent + 0.1 * rng.standard_normal(n)
    strong = latent + 0.05 * rng.standard_normal(n)
    aux = 0.8 * latent + 0.2 * rng.standard_normal(n)
    noise = rng.standard_normal((n, 2))

    frame = pd.DataFrame(
        {
            "target": target,
            "feat_primary": strong,
            "feat_aux": aux,
            "noise_a": noise[:, 0],
            "noise_b": noise[:, 1],
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)
    model_path = tmp_path / "model.yaml"
    model_path.write_text("feature_num: {}\nfeature_bool: []\n", encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 27, "device": "cpu"},
        "cpcv": {"folds": 4, "test_folds": 1, "min_trades_per_split": 96, "shuffle_seed": 8},
        "training_strategy": {"max_hold_period_bars": 32},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 120,
        "stage1_screening": {
            "combine": {"method": "simes", "alpha": 0.2, "fdr": "bh"},
        },
        "stage2_selection": {
            "corr_cap": 0.9,
            "max_features": 3,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
            "knockoffs": {
                "enabled": True,
                "fdr": 0.25,
                "statistic": "correlation",
                "shrinkage": 1e-3,
                "random_state": 101,
                "generator": "stability_selection",
                "fallback": ["robust", "modelx"],
            },
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 10.0,
            "robust_quantiles": [0.02, 0.98],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.stage2 is not None
    assert artifacts.stage2.knockoff_attempts[0] == "stability_selection"
    assert "robust" in artifacts.stage2.knockoff_attempts
    assert artifacts.stage2.knockoff_generator in {"robust", "modelx"}
    assert "feat_primary" in artifacts.stage2.knockoff_statistics
