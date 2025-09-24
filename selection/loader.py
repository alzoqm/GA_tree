# /selection/loader.py
"""Config and data loading helpers for the selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class SelectionPaths:
    """Container with resolved filesystem paths used by the pipeline."""

    base_dir: Path
    final_features_pkl: Optional[Path]
    final_features_csv: Optional[Path]
    model_config_path: Path
    feature_active_path: Path
    artifact_dir: Path

    def final_features_path(self) -> Path:
        if self.final_features_pkl and self.final_features_pkl.exists():
            return self.final_features_pkl
        if self.final_features_csv and self.final_features_csv.exists():
            return self.final_features_csv
        raise FileNotFoundError(
            "No final features file found. Looked for: "
            f"{self.final_features_pkl} and {self.final_features_csv}"
        )


def load_experiment_config(path: str | Path) -> Dict:
    """Load the experiment configuration YAML as a dictionary."""

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_paths(cfg: Dict, config_dir: Path | None = None) -> SelectionPaths:
    """Resolve selection related paths relative to the config location."""

    base_dir = Path(config_dir or ".").resolve()
    data_cfg = cfg.get("data", {})
    selection_cfg = cfg.get("selection_pipeline", {})

    def _resolve(path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        raw = Path(path_value)
        return (base_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()

    final_pkl = _resolve(data_cfg.get("final_features_path_pkl"))
    final_csv = _resolve(data_cfg.get("final_features_path"))
    model_config_path = _resolve(data_cfg.get("model_config_path"))
    feature_active_path = _resolve(
        data_cfg.get("feature_classification_path", "Config/feature_classification_active.yaml")
    )
    artifact_dir = _resolve(selection_cfg.get("artifact_dir", "dataset/selection_artifacts"))

    if model_config_path is None or feature_active_path is None:
        raise ValueError("model_config_path and feature_classification_path must be set in config")

    return SelectionPaths(
        base_dir=base_dir,
        final_features_pkl=final_pkl,
        final_features_csv=final_csv,
        model_config_path=model_config_path,
        feature_active_path=feature_active_path,
        artifact_dir=artifact_dir or base_dir,
    )


def load_final_features(paths: SelectionPaths) -> pd.DataFrame:
    """Load the preprocessed feature dataframe."""

    feature_path = paths.final_features_path()
    if feature_path.suffix == ".pkl":
        df = pd.read_pickle(feature_path)
    else:
        df = pd.read_csv(feature_path)
    return df


def load_model_config(path: str | Path) -> Dict:
    """Load the model config YAML containing feature ranges."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
