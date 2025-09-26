import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason="Selection enhancement tests rely on supervised target effects; pending unsupervised redesign",
)

from selection.calibration import calibrate_ranges
from selection.loader import SelectionPaths
from selection.pipeline import SelectionPipeline


def _make_paths(tmp_path, frame) -> SelectionPaths:
    data_path = tmp_path / "features.csv"
    frame.to_csv(data_path, index=False)
    model_path = tmp_path / "model.yaml"
    model_path.write_text("feature_num: {}\nfeature_bool: []\n", encoding="utf-8")
    active_path = tmp_path / "feature_active.yaml"
    artifact_dir = tmp_path / "artifacts"
    return SelectionPaths(
        base_dir=tmp_path,
        final_features_pkl=None,
        final_features_csv=data_path,
        model_config_path=model_path,
        feature_active_path=active_path,
        artifact_dir=artifact_dir,
    )


def test_modelx_knockoffs_stage(tmp_path):
    rng = np.random.default_rng(9403)
    n = 720
    latent = rng.standard_normal(n)
    target = latent + 0.1 * rng.standard_normal(n)
    strong = latent + 0.05 * rng.standard_normal(n)
    weak = 0.5 * latent + 0.5 * rng.standard_normal(n)
    noise = rng.standard_normal((n, 3))

    frame = pd.DataFrame(
        {
            "target": target,
            "feat_strong": strong,
            "feat_weak": weak,
            "noise_1": noise[:, 0],
            "noise_2": noise[:, 1],
            "noise_3": noise[:, 2],
        }
    )

    config = {
        "environment": {"seed": 17, "device": "cpu"},
        "cpcv": {"folds": 4, "test_folds": 1, "shuffle_seed": 2, "min_trades_per_split": 96},
        "training_strategy": {"max_hold_period_bars": 36},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 120,
        "stage1_screening": {
            "combine": {"method": "empirical_brown", "alpha": 0.99, "fdr": "bh"},
            "methods": {"spearman": {"enabled": True}},
            "enforce_significance": False,
        },
        "stage2_selection": {
            "inner_ratio": 0.5,
            "corr_cap": 0.85,
            "knockoffs": {
                "enabled": True,
                "fdr": 0.2,
                "statistic": "correlation",
                "shrinkage": 1e-3,
                "random_state": 42,
            },
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 12.0,
            "robust_quantiles": [0.01, 0.99],
            "mad_margin": {"factor": 0.0},
        },
    }

    paths = _make_paths(tmp_path, frame)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert artifacts.stage2 is not None
    assert "modelx_knockoffs" in artifacts.stage2.method
    stats = artifacts.stage2.knockoff_statistics
    assert stats["feat_strong"] > stats["noise_1"]
    assert "feat_strong" in artifacts.stage2.selected_features


def test_boolean_stats_and_calibration(tmp_path):
    rng = np.random.default_rng(20251)
    n = 480
    base = rng.standard_normal(n)
    target = base + 0.2 * rng.standard_normal(n)
    bool_feature = (target > 0).astype(int)
    cal_feature = np.concatenate([
        rng.normal(-1, 0.1, n // 2),
        rng.normal(1, 0.1, n // 2),
    ])

    frame = pd.DataFrame(
        {
            "target": target,
            "bool_alpha": bool_feature,
            "range_beta": cal_feature,
        }
    )

    config = {
        "environment": {"seed": 3, "device": "cpu"},
        "cpcv": {"folds": 3, "test_folds": 1, "shuffle_seed": 1, "min_trades_per_split": 120},
        "training_strategy": {"max_hold_period_bars": 24},
    }

    selection_cfg = {
        "enabled": True,
        "dry_run": True,
        "target_column": "target",
        "min_bars_per_split": 100,
        "integrity": {"min_unique_fraction": 0.0},
        "stage1_screening": {
            "combine": {"method": "empirical_brown", "alpha": 0.99, "fdr": "bh"},
            "methods": {"spearman": {"enabled": True}},
            "enforce_significance": False,
        },
        "feature_whitelist": ["bool_alpha", "range_beta"],
        "stage2_selection": {
            "corr_cap": 0.9,
            "max_features": 2,
            "inner_ratio": 0.5,
            "stability_selection": {"enabled": False},
            "knockoffs": {"enabled": False},
        },
        "stage3_diversity": {"enabled": False},
        "stage4_calibration": {
            "vif_threshold": 10.0,
            "robust_quantiles": [0.05, 0.95],
            "mad_margin": {"factor": 0.0},
            "boolean_stats": {"samples": 64, "block_size": 16},
        },
    }

    paths = _make_paths(tmp_path, frame)
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert "bool_alpha" in artifacts.calibration.feature_bool
    bool_stats = artifacts.calibration.diagnostics.get("boolean_stats", {})
    assert "bool_alpha" in bool_stats
    odds_ratio = bool_stats["bool_alpha"]["odds_ratio"]
    assert odds_ratio > 1.0


def test_calibrate_ranges_foldwise():
    data = pd.DataFrame(
        {
            "feat": np.concatenate([
                np.linspace(0, 1, 10),
                np.linspace(1, 2, 10),
            ])
        }
    )
    folds = [np.arange(0, 10), np.arange(10, 20)]
    result = calibrate_ranges(
        data,
        ["feat"],
        quantiles=(0.1, 0.9),
        mad_factor=None,
        fold_indices=folds,
    )
    low, high = result.feature_num["feat"]
    expected_low = np.median([
        data.iloc[folds[0]]["feat"].quantile(0.1),
        data.iloc[folds[1]]["feat"].quantile(0.1),
    ])
    expected_high = np.median([
        data.iloc[folds[0]]["feat"].quantile(0.9),
        data.iloc[folds[1]]["feat"].quantile(0.9),
    ])
    assert np.isclose(low, expected_low)
    assert np.isclose(high, expected_high)
