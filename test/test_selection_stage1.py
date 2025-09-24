import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    base_signal = rng.standard_normal(n)
    noise = rng.standard_normal((n, 3))
    target = base_signal + rng.standard_normal(n) * 0.1

    frame = pd.DataFrame(
        {
            "target": target,
            "feat_signal": base_signal + rng.standard_normal(n) * 0.1,
            "feat_signal_shift": np.roll(base_signal, 1) + rng.standard_normal(n) * 0.1,
            "feat_noise": noise[:, 0],
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
        "target_column": "target",
        "min_bars_per_split": 96,
        "stage1_screening": {"combine": {"method": "simes", "alpha": 0.1, "fdr": "bh"}},
        "stage2_selection": {"corr_cap": 0.85, "max_features": 5, "inner_ratio": 0.5},
        "stage4_calibration": {"vif_threshold": 8.0, "robust_quantiles": [0.01, 0.99], "mad_margin": {"factor": 0.0}},
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)

    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=True)
    artifacts = pipeline.run(frame, {"feature_num": {}, "feature_bool": []})

    assert "feat_signal" in artifacts.selected_features
    assert artifacts.screening.effects["feat_signal"] >= artifacts.screening.effects["feat_noise"]
    assert len(artifacts.selected_features) <= selection_cfg["stage2_selection"]["max_features"]


def test_pipeline_persists_outputs(tmp_path):
    rng = np.random.default_rng(7)
    n = 420
    signal = rng.standard_normal(n)
    target = signal + rng.standard_normal(n) * 0.05
    frame = pd.DataFrame(
        {
            "target": target,
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
        "target_column": "target",
        "min_bars_per_split": 80,
        "artifact_dir": str(artifact_dir),
        "stage1_screening": {"combine": {"method": "simes", "alpha": 0.05, "fdr": "by"}},
        "stage2_selection": {"corr_cap": 0.9, "max_features": 4, "inner_ratio": 0.5},
        "stage4_calibration": {"vif_threshold": 10.0, "robust_quantiles": [0.05, 0.95], "mad_margin": {"factor": 0.0}},
        "feature_whitelist": ["flag_breakout"],
    }

    paths = _make_paths(tmp_path, data_path, model_path, active_path, artifact_dir)

    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=False)
    artifacts = pipeline.run(frame, baseline_model)

    # Model config updated on disk
    written_model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    assert "feat_alpha" in written_model["feature_num"]
    low, high = written_model["feature_num"]["feat_alpha"]
    assert low < high
    assert "flag_breakout" in written_model["feature_bool"]

    # Snapshot file created with selection_snapshot
    snapshot_data = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    assert "selection_snapshot" in snapshot_data
    assert "feat_alpha" in snapshot_data["selection_snapshot"]["active_feature_whitelist"]

    # Artifact JSON recorded
    json_files = list(artifact_dir.glob("selection_stage1_*.json"))
    assert json_files, "Expected artifact JSON file"
    artifact_payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert artifact_payload["selected_features"] == artifacts.selected_features
