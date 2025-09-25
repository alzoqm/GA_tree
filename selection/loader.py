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
        """Return the resolved path to the final features dataset.

        The pipeline now stores final features as pickle by default. To remain
        backward compatible with existing configs that only specify a CSV
        location, we try the following in order:
          1. The explicitly configured pickle path.
          2. A pickle path derived from the CSV path (same stem, ``.pkl``).
          3. The explicitly configured CSV path.
        """

        candidates: list[Path] = []

        if self.final_features_pkl:
            candidates.append(self.final_features_pkl)

        derived_pickle: Optional[Path] = None
        if self.final_features_csv:
            derived_pickle = self.final_features_csv.with_suffix(".pkl")
            # Avoid duplicating the same path if the config already pointed to the
            # pickle variant explicitly.
            if not candidates or derived_pickle != candidates[0]:
                candidates.append(derived_pickle)
            candidates.append(self.final_features_csv)

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate

        searched = ", ".join(str(path) for path in candidates if path)
        raise FileNotFoundError(
            "No final features file found. Looked for: "
            f"{searched or 'no candidates available'}"
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
    project_root = base_dir.parent
    data_cfg = cfg.get("data", {})
    env_cfg = cfg.get("environment", {})
    selection_cfg = cfg.get("selection_pipeline", {})

    data_root: Optional[Path] = None
    raw_data_dir = env_cfg.get("data_dir")
    if raw_data_dir:
        data_root_candidate = Path(raw_data_dir)
        data_root = (
            (base_dir / data_root_candidate).resolve()
            if not data_root_candidate.is_absolute()
            else data_root_candidate.resolve()
        )

    def _resolve(
        path_value: Optional[str],
        *,
        prefer_data_dir: bool = False,
        default_dir: Optional[Path] = None,
        must_exist: bool = False,
    ) -> Optional[Path]:
        if not path_value:
            return None

        raw = Path(path_value)
        if raw.is_absolute():
            candidate = raw.resolve()
            if not must_exist or candidate.exists():
                return candidate
            return candidate

        search_roots: list[Path] = []
        if prefer_data_dir and data_root is not None:
            search_roots.append(data_root)
        if default_dir is not None:
            search_roots.append(default_dir.resolve())
        search_roots.extend([project_root, base_dir])

        for root in search_roots:
            candidate = (root / raw).resolve()
            if not must_exist or candidate.exists():
                return candidate

        fallback_root = search_roots[0] if search_roots else base_dir
        return (fallback_root / raw).resolve()

    final_pkl = _resolve(
        data_cfg.get("final_features_path_pkl"),
        prefer_data_dir=True,
        default_dir=project_root,
    )
    final_csv = _resolve(
        data_cfg.get("final_features_path"),
        prefer_data_dir=True,
        default_dir=project_root,
    )
    model_config_path = _resolve(
        data_cfg.get("model_config_path"),
        prefer_data_dir=True,
        default_dir=project_root,
    )
    feature_active_path = _resolve(
        data_cfg.get(
            "feature_classification_path",
            "Config/feature_classification_active.yaml",
        ),
        default_dir=project_root,
    )
    artifact_dir = _resolve(
        selection_cfg.get("artifact_dir", "dataset/selection_artifacts"),
        default_dir=project_root,
    )

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
