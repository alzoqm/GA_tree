from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from selection.integrity import apply_integrity_gates
from selection.loader import SelectionPaths
from selection.pipeline import SelectionPipeline


def _make_paths(tmp_path: Path, data_path: Path, model_path: Path, active_path: Path, artifact_dir: Path) -> SelectionPaths:
    return SelectionPaths(
        base_dir=tmp_path,
        final_features_pkl=None,
        final_features_csv=data_path,
        model_config_path=model_path,
        feature_active_path=active_path,
        artifact_dir=artifact_dir,
    )


def test_integrity_gates_drop_problem_columns():
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5],
            "missing_col": [1, None, None, None, None],
            "constant_col": [3, 3, 3, 3, 3],
            "sparse_flag": [0, 0, 0, 0, 1],
            "duplicate_a": [1, 2, 3, 4, 5],
            "duplicate_b": [1, 2, 3, 4, 5],
            "correlated": [2, 4, 6, 8, 10],
            "useful": [1, 4, 9, 16, 25],
        }
    )

    cleaned, report = apply_integrity_gates(
        df,
        feature_columns=[c for c in df.columns if c != "target"],
        max_missing_ratio=0.6,
        min_unique_fraction=0.2,
        max_zero_fraction=0.99,
        correlation_threshold=0.995,
    )

    assert "missing_col" not in cleaned.columns
    assert "constant_col" not in cleaned.columns
    assert "duplicate_b" not in cleaned.columns
    remaining = {"duplicate_a", "correlated"}.intersection(cleaned.columns)
    assert len(remaining) == 1
    assert "useful" in cleaned.columns
    assert report.total_dropped >= 4


def test_pipeline_selects_signal_features(tmp_path):
    rng = np.random.default_rng(123)
    n = 480
    signal = np.zeros(n)
    for idx in range(1, n):
        signal[idx] = 0.85 * signal[idx - 1] + rng.standard_normal()
    feat_shift = np.roll(signal, 1)
    noise = rng.standard_normal(n)

    frame = pd.DataFrame(
        {
            "feat_signal": signal + rng.standard_normal(n) * 0.05,
            "feat_signal_shift": feat_shift + rng.standard_normal(n) * 0.05,
            "feat_noise": noise,
            "flag_event": rng.integers(0, 2, n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")

    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 42, "device": "cpu"},
        "cpcv": {"folds": 5, "test_folds": 1, "min_trades_per_split": 48, "shuffle_seed": 0},
        "training_strategy": {"max_hold_period_bars": 10},
    }
    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "supervision": "unsupervised",
        "target_column": None,
        "min_bars_per_split": 96,
        "stage1_screening": {
            "methods": {"auto_spearman": {"enabled": True, "lags": [1]}},
            "combine": {"method": "simes", "alpha": 0.1, "fdr": "bh"},
        },
        "stage2_selection": {
            "corr_cap": 0.85,
            "max_features": 5,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 8.0,
            "robust_quantiles": [0.01, 0.99],
            "mad_margin": {"factor": 0.0},
            "boolean_stats": {"enabled": False},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)

    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert "feat_signal" in artifacts.selected_features
    assert artifacts.screening.effects["feat_signal"] >= artifacts.screening.effects["feat_noise"]
    assert len(artifacts.selected_features) <= selection_cfg["stage2_selection"]["max_features"]


def test_pipeline_persists_outputs(tmp_path, caplog):
    rng = np.random.default_rng(7)
    n = 420
    signal = np.zeros(n)
    for idx in range(1, n):
        signal[idx] = 0.88 * signal[idx - 1] + rng.standard_normal() * 0.1
    frame = pd.DataFrame(
        {
            "feat_alpha": signal + rng.standard_normal(n) * 0.05,
            "feat_beta": rng.standard_normal(n),
            "flag_breakout": rng.integers(0, 2, n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    baseline_model = {"feature_num": {"feat_beta": [-1.0, 1.0]}, "feature_bool": ["flag_breakout"]}
    model_path.write_text(yaml.safe_dump(baseline_model), encoding="utf-8")

    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 21, "device": "cpu"},
        "cpcv": {"folds": 4, "test_folds": 1, "min_trades_per_split": 64, "shuffle_seed": 1},
        "training_strategy": {"max_hold_period_bars": 20},
    }
    selection_cfg = {
        "enabled": True,
        "dry_run": False,
        "supervision": "unsupervised",
        "target_column": None,
        "min_bars_per_split": 80,
        "artifact_dir": str(artifact_dir),
        "stage1_screening": {
            "methods": {"auto_spearman": {"enabled": True, "lags": [1]}},
            "combine": {"method": "simes", "alpha": 0.05, "fdr": "by"},
            "enforce_significance": False,
        },
        "stage2_selection": {
            "corr_cap": 0.9,
            "max_features": 4,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 10.0,
            "robust_quantiles": [0.05, 0.95],
            "mad_margin": {"factor": 0.0},
            "boolean_stats": {"enabled": False},
        },
        "feature_whitelist": ["flag_breakout"],
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)

    caplog.set_level("WARNING", logger="selection.pipeline")
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=False)
    artifacts = pipeline.run(frame, baseline_model)

    # Model config updated on disk
    written_model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    assert "feat_alpha" in written_model["feature_num"]
    low, high = written_model["feature_num"]["feat_alpha"]
    assert low < high
    assert "flag_breakout" not in written_model.get("feature_bool", [])

    # Snapshot file created with selection_snapshot
    snapshot_data = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    assert "selection_snapshot" in snapshot_data
    snapshot_whitelist = snapshot_data["selection_snapshot"]["active_feature_whitelist"]
    assert "feat_alpha" in snapshot_whitelist
    assert "flag_breakout" not in snapshot_whitelist


def test_stage1_supports_rff_and_sampled_backends(tmp_path):
    rng = np.random.default_rng(5150)
    n = 360
    latent = rng.uniform(-np.pi, np.pi, size=n)
    target = np.sin(latent)
    frame = pd.DataFrame(
        {
            "target": target,
            "feature_latent": latent,
            "feature_latent_abs": np.abs(latent) + 0.1 * rng.standard_normal(n),
            "noise_feature": rng.standard_normal(n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 19, "device": "cpu"},
        "cpcv": {"folds": 4, "test_folds": 1, "min_trades_per_split": 96, "shuffle_seed": 3},
        "training_strategy": {"max_hold_period_bars": 24},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 120,
        "stage1_screening": {
            "random_state": 321,
            "heavy_switch_n": 32,
            "hsic_rff_dim": 32,
            "dcor_sample_size": 64,
            "methods": {
                "hsic": {
                    "enabled": True,
                    "backend": "rff",
                    "rff_dim": 32,
                    "bootstrap": {"samples": 10, "block_size": 24, "method": "block"},
                },
                "dcor": {
                    "enabled": True,
                    "backend": "sampled",
                    "sample_size": 64,
                    "permutations": 12,
                    "block_size": 24,
                },
            },
            "combine": {"method": "simes", "alpha": 0.3, "fdr": "bh"},
        },
        "stage2_selection": {
            "corr_cap": 0.9,
            "max_features": 6,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
            "knockoffs": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 10.0,
            "robust_quantiles": [0.05, 0.95],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert set(artifacts.screening.methods) == {"dcor", "hsic"}
    assert "feature_latent" in artifacts.screening.passing_features
    assert "noise_feature" not in artifacts.screening.passing_features
    assert artifacts.screening.details  # ensure details recorded when enabled


def test_stage1_record_details_toggle(tmp_path):
    rng = np.random.default_rng(9090)
    n = 300
    signal = rng.standard_normal(n)
    target = signal
    frame = pd.DataFrame(
        {
            "target": target,
            "feature_signal": signal + 0.05 * rng.standard_normal(n),
            "feature_noise": rng.standard_normal(n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 23, "device": "cpu"},
        "cpcv": {"folds": 4, "test_folds": 1, "min_trades_per_split": 96, "shuffle_seed": 4},
        "training_strategy": {"max_hold_period_bars": 24},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 96,
        "stage1_screening": {
            "record_details": False,
            "methods": {"spearman": {"enabled": True}},
            "combine": {"method": "simes", "alpha": 0.5, "fdr": "bh"},
        },
        "stage2_selection": {
            "corr_cap": 0.9,
            "max_features": 5,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
            "knockoffs": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 12.0,
            "robust_quantiles": [0.05, 0.95],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert "feature_signal" in artifacts.screening.passing_features
    assert artifacts.screening.details == []


def test_pipeline_whitelist_respected_when_enforcement_disabled(tmp_path):
    rng = np.random.default_rng(11)
    n = 300
    base = np.zeros(n)
    for idx in range(1, n):
        base[idx] = 0.82 * base[idx - 1] + rng.standard_normal() * 0.15
    frame = pd.DataFrame(
        {
            "feat_good": base,
            "flag_signal": rng.integers(0, 2, n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")

    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 3, "device": "cpu"},
        "cpcv": {"folds": 3, "test_folds": 1, "min_trades_per_split": 48, "shuffle_seed": 8},
        "training_strategy": {},
    }
    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "supervision": "unsupervised",
        "target_column": None,
        "min_bars_per_split": 60,
        "stage1_screening": {
            "methods": {"auto_spearman": {"enabled": True, "lags": [1]}},
            "combine": {"method": "simes", "alpha": 0.05, "fdr": "by"},
            "enforce_significance": False,
        },
        "integrity": {"min_unique_fraction": 0.0},
        "feature_whitelist": ["flag_signal"],
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert "flag_signal" in artifacts.selected_features


def test_stage1_effect_threshold_filters_low_effect(tmp_path):
    rng = np.random.default_rng(5150)
    n = 640
    signal = np.zeros(n)
    for idx in range(1, n):
        signal[idx] = 0.9 * signal[idx - 1] + rng.standard_normal() * 0.1
    weak = rng.standard_normal(n) * 0.2

    frame = pd.DataFrame(
        {
            "feat_strong": signal,
            "feat_weak": weak,
            "noise": rng.standard_normal(n),
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 73, "device": "cpu"},
        "cpcv": {
            "folds": 2,
            "test_folds": 1,
            "min_trades_per_split": 40,
            "shuffle_seed": 4,
            "max_combinations": 1,
        },
        "training_strategy": {"max_hold_period_bars": 12},
    }
    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "supervision": "unsupervised",
        "target_column": None,
        "min_bars_per_split": 60,
        "stage1_screening": {
            "methods": {"auto_spearman": {"enabled": True, "lags": [1]}},
            "combine": {"method": "simes", "alpha": 0.99, "fdr": "bh"},
            "effect_min": 0.4,
            "cliffs_delta_min": 0.0,
        },
        "stage2_selection": {
            "corr_cap": 0.95,
            "max_features": 5,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 10.0,
            "robust_quantiles": [0.02, 0.98],
            "mad_margin": {"factor": 0.0},
            "boolean_stats": {"enabled": False},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.screening.passing_features == ["feat_strong"]
    assert artifacts.screening.effects["feat_strong"] >= selection_cfg["stage1_screening"]["effect_min"]
    assert artifacts.screening.cliffs_delta["feat_strong"] >= 0.0
    assert "feat_weak" not in artifacts.selected_features


def test_stage1_kendall_method_records(tmp_path):
    rng = np.random.default_rng(6060)
    n = 400
    time_index = np.arange(n)
    monotonic_feature = time_index + rng.normal(scale=0.5, size=n)
    noise = rng.standard_normal(n)

    frame = pd.DataFrame(
        {
            "feat_trend": monotonic_feature,
            "feat_noise": noise,
        }
    )

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 91, "device": "cpu"},
        "cpcv": {"folds": 3, "test_folds": 1, "min_trades_per_split": 96, "shuffle_seed": 6},
        "training_strategy": {"max_hold_period_bars": 18},
    }
    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "supervision": "unsupervised",
        "target_column": None,
        "min_bars_per_split": 100,
        "stage1_screening": {
            "combine": {"method": "simes", "alpha": 0.1, "fdr": "bh"},
            "methods": {"auto_kendall": {"enabled": True, "lags": [1]}},
        },
        "stage2_selection": {
            "corr_cap": 0.95,
            "max_features": 3,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 12.0,
            "robust_quantiles": [0.05, 0.95],
            "mad_margin": {"factor": 0.0},
            "boolean_stats": {"enabled": False},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.screening.methods == ["auto_kendall"]
    assert "feat_trend" in artifacts.screening.passing_features
    assert all(detail.method == "auto_kendall" for detail in artifacts.screening.details)
    assert "feat_trend" in artifacts.selected_features


def test_pipeline_raises_when_no_significant_features(tmp_path):
    rng = np.random.default_rng(99)
    n = 200
    frame = pd.DataFrame({"pure_noise": rng.standard_normal(n)})

    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)

    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"feature_num": {}, "feature_bool": []}), encoding="utf-8")

    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"

    config = {
        "environment": {"seed": 1, "device": "cpu"},
        "cpcv": {"folds": 3, "test_folds": 1, "min_trades_per_split": 64, "shuffle_seed": 5},
        "training_strategy": {},
    }
    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "supervision": "unsupervised",
        "target_column": None,
        "min_bars_per_split": 64,
        "stage1_screening": {
            "methods": {"auto_spearman": {"enabled": True, "lags": [1]}},
            "combine": {"method": "simes", "alpha": 0.01, "fdr": "by"},
        },
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)

    with pytest.raises(RuntimeError, match="No features passed the FDR threshold"):
        pipeline.run(frame, {"feature_num": {}, "feature_bool": []})
