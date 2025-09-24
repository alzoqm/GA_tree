"""Selection pipeline orchestrator implementing Stage 1 of the design."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .calibration import CalibrationResult, calibrate_ranges
from .integrity import IntegrityReport, apply_integrity_gates
from .loader import SelectionPaths
from .screening import ScreeningResult, screening_out_of_fold
from .selector import (
    SelectionDiagnostics,
    enforce_vif_and_partial_r2,
    greedy_representative_selection,
)
from .snapshot import (
    append_selection_snapshot,
    build_snapshot_payload,
    compute_data_meta_hash,
    atomic_write_yaml,
)
from .splits import CrossFitSplit, build_cross_fit_splits, make_cpcv_scheduler


@dataclass
class StageOneConfig:
    target_column: str
    min_bars_per_split: int
    max_missing_ratio: float = 0.2
    min_unique_fraction: float = 0.01
    max_zero_fraction: float = 0.995
    correlation_threshold: float = 0.999
    combine_method: str = "simes"
    combine_alpha: float = 0.1
    fdr_method: str = "by"
    corr_cap: float = 0.9
    max_features: Optional[int] = None
    vif_threshold: float = 10.0
    quantiles: tuple[float, float] = (0.005, 0.995)
    mad_factor: Optional[float] = 1.4826
    whitelist: Sequence[str] = ()


@dataclass
class PipelineArtifacts:
    integrity: IntegrityReport
    screening: ScreeningResult
    selection: SelectionDiagnostics
    calibration: CalibrationResult
    selected_features: List[str]
    updated_model_config: Dict[str, object]


class SelectionPipeline:
    """High-level orchestrator for the Stage 1 pipeline."""

    def __init__(
        self,
        config: Mapping[str, object],
        selection_cfg: Mapping[str, object],
        paths: SelectionPaths,
        *,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.selection_cfg = selection_cfg
        self.paths = paths
        self.dry_run = dry_run or bool(selection_cfg.get("dry_run", False))
        self.stage1 = self._build_stage1_config(selection_cfg)
        self._latest_df_shape: Optional[tuple[int, int]] = None

    @staticmethod
    def _build_stage1_config(cfg: Mapping[str, object]) -> StageOneConfig:
        target_column = cfg.get("target_column", "target")
        min_bars = int(cfg.get("min_bars_per_split", 128))
        screen_cfg = cfg.get("stage1_screening", {})
        combine = screen_cfg.get("combine", {})
        quantiles = tuple(cfg.get("stage4_calibration", {}).get("robust_quantiles", (0.005, 0.995)))
        mad_factor = cfg.get("stage4_calibration", {}).get("mad_margin", {}).get("factor", 1.4826)
        whitelist = tuple(cfg.get("feature_whitelist", []))
        return StageOneConfig(
            target_column=target_column,
            min_bars_per_split=min_bars,
            max_missing_ratio=cfg.get("integrity", {}).get("max_missing_ratio", 0.2),
            min_unique_fraction=cfg.get("integrity", {}).get("min_unique_fraction", 0.01),
            max_zero_fraction=cfg.get("integrity", {}).get("max_zero_fraction", 0.995),
            correlation_threshold=cfg.get("integrity", {}).get("correlation_threshold", 0.999),
            combine_method=combine.get("method", "simes"),
            combine_alpha=float(combine.get("alpha", screen_cfg.get("alpha", 0.1))),
            fdr_method=combine.get("fdr", screen_cfg.get("fdr", "by")),
            corr_cap=cfg.get("stage2_selection", {}).get("corr_cap", 0.9),
            max_features=cfg.get("stage2_selection", {}).get("max_features"),
            vif_threshold=float(cfg.get("stage4_calibration", {}).get("vif_threshold", 10.0)),
            quantiles=(float(quantiles[0]), float(quantiles[1])),
            mad_factor=float(mad_factor) if mad_factor is not None else None,
            whitelist=whitelist,
        )

    def _resolve_feature_columns(self, df: pd.DataFrame) -> List[str]:
        target = self.stage1.target_column
        candidates = [c for c in df.columns if c != target]
        # Keep numeric + boolean columns only
        numeric_mask = df[candidates].apply(lambda col: pd.api.types.is_numeric_dtype(col))
        features = [name for name, keep in numeric_mask.items() if keep]
        return features

    def _make_cross_splits(self, df: pd.DataFrame) -> List[CrossFitSplit]:
        scheduler = make_cpcv_scheduler(
            total_len=len(df),
            cfg=self.selection_cfg.get("cpcv", self.config.get("cpcv", {})),
            train_cfg=self.config.get("training_strategy", {}),
        )
        inner_ratio = float(self.selection_cfg.get("stage2_selection", {}).get("inner_ratio", 0.5))
        return build_cross_fit_splits(
            scheduler=scheduler,
            inner_ratio=inner_ratio,
            require_valid=True,
        )

    def run(self, df: pd.DataFrame, model_config: Mapping[str, object]) -> PipelineArtifacts:
        target = self.stage1.target_column
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe")

        feature_columns = self._resolve_feature_columns(df)
        if not feature_columns:
            raise ValueError("No usable features found after dtype filtering")

        cleaned_df, integrity_report = apply_integrity_gates(
            df,
            feature_columns,
            max_missing_ratio=self.stage1.max_missing_ratio,
            min_unique_fraction=self.stage1.min_unique_fraction,
            max_zero_fraction=self.stage1.max_zero_fraction,
            correlation_threshold=self.stage1.correlation_threshold,
        )
        self._latest_df_shape = cleaned_df.shape
        clean_features = [f for f in feature_columns if f in cleaned_df.columns]

        cross_splits = self._make_cross_splits(cleaned_df)
        if not cross_splits:
            raise RuntimeError("No valid CPCV splits available for screening")

        screening_result = screening_out_of_fold(
            cleaned_df,
            clean_features,
            target,
            cross_splits,
            combine_cfg={
                "method": self.stage1.combine_method,
                "alpha": self.stage1.combine_alpha,
                "fdr": self.stage1.fdr_method,
            },
            min_bars=self.stage1.min_bars_per_split,
        )

        candidate_features = set(screening_result.passing_features) | set(self.stage1.whitelist)
        if not candidate_features:
            candidate_features = set(sorted(clean_features, key=lambda f: screening_result.effects.get(f, 0.0), reverse=True)[:32])

        selected_features, discarded_corr = greedy_representative_selection(
            cleaned_df,
            target,
            list(candidate_features),
            screening_result.effects,
            corr_cap=self.stage1.corr_cap,
            max_features=self.stage1.max_features,
        )

        final_features, selection_diag = enforce_vif_and_partial_r2(
            cleaned_df,
            target,
            selected_features,
            vif_threshold=self.stage1.vif_threshold,
        )
        selection_diag.discarded_high_corr = sorted(
            set(selection_diag.discarded_high_corr).union(discarded_corr)
        )

        calibration_result = calibrate_ranges(
            cleaned_df,
            final_features,
            quantiles=self.stage1.quantiles,
            mad_factor=self.stage1.mad_factor,
        )

        # Merge bool features determined during calibration with whitelist to form final set
        final_feature_set = sorted(set(final_features) | set(calibration_result.feature_bool) | set(self.stage1.whitelist))

        updated_model_cfg = self._update_model_config(
            model_config,
            calibration_result,
            final_feature_set,
        )

        artifacts = PipelineArtifacts(
            integrity=integrity_report,
            screening=screening_result,
            selection=selection_diag,
            calibration=calibration_result,
            selected_features=final_feature_set,
            updated_model_config=updated_model_cfg,
        )

        self._persist_outputs(artifacts)
        return artifacts

    def _update_model_config(
        self,
        model_config: Mapping[str, object],
        calibration: CalibrationResult,
        final_features: Sequence[str],
    ) -> Dict[str, object]:
        updated = copy.deepcopy(model_config)
        existing_num = dict(updated.get("feature_num", {}))
        existing_bool = list(updated.get("feature_bool", []))
        existing_cmp = dict(updated.get("feature_comparison_map", {}))

        final_num: Dict[str, List[float]] = {}
        for feature in final_features:
            if feature in calibration.feature_num:
                bounds = calibration.feature_num[feature]
                final_num[feature] = [float(bounds[0]), float(bounds[1])]
            elif feature in existing_num:
                bounds = existing_num[feature]
                final_num[feature] = [float(bounds[0]), float(bounds[1])]

        final_bool = [feature for feature in calibration.feature_bool if feature in final_features]
        for feature in existing_bool:
            if feature in final_features and feature not in final_bool:
                final_bool.append(feature)

        final_cmp = {
            key: value
            for key, value in existing_cmp.items()
            if key in final_features
        }

        updated["feature_num"] = final_num
        updated["feature_bool"] = final_bool
        updated["feature_comparison_map"] = final_cmp
        return updated

    def _persist_outputs(self, artifacts: PipelineArtifacts) -> None:
        if self.dry_run:
            return
        self.paths.artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = self.paths.artifact_dir / f"selection_stage1_{timestamp}.json"
        payload = {
            "selected_features": artifacts.selected_features,
            "integrity": artifacts.integrity.__dict__,
            "screening": {
                "pvalues": artifacts.screening.pvalues,
                "effects": artifacts.screening.effects,
            },
            "selection": {
                "vif": artifacts.selection.vif,
                "partial_r2": artifacts.selection.partial_r2,
                "discarded_high_vif": artifacts.selection.discarded_high_vif,
                "discarded_high_corr": artifacts.selection.discarded_high_corr,
            },
            "calibration": {
                "feature_num": {k: list(v) for k, v in artifacts.calibration.feature_num.items()},
                "feature_bool": artifacts.calibration.feature_bool,
            },
        }
        with artifact_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        self._write_model_config(artifacts.updated_model_config)
        self._write_snapshot(artifacts)

    def _write_model_config(self, updated_cfg: Mapping[str, object]) -> None:
        if self.dry_run:
            return
        atomic_write_yaml(self.paths.model_config_path, updated_cfg, backup=True)

    def _write_snapshot(self, artifacts: PipelineArtifacts) -> None:
        if self.dry_run:
            return
        git_commit = _git_commit_hash()
        cfg_digest = _config_digest(self.selection_cfg)
        data_meta = compute_data_meta_hash(
            data_path=self.paths.final_features_path(),
            df_shape=self._latest_df_shape or (0, 0),
            config_digest=cfg_digest,
            extra={"combine": self.stage1.combine_method, "fdr": self.stage1.fdr_method},
        )
        env_info = {
            "seed": self.config.get("environment", {}).get("seed"),
            "device": self.config.get("environment", {}).get("device"),
        }
        method_info = {
            "cv": self.selection_cfg.get("cpcv", self.config.get("cpcv", {})),
            "screening": {
                "combine": self.stage1.combine_method,
                "alpha": self.stage1.combine_alpha,
                "fdr": self.stage1.fdr_method,
            },
            "selection": {
                "corr_cap": self.stage1.corr_cap,
                "vif_threshold": self.stage1.vif_threshold,
            },
            "calibration": {
                "quantiles": list(self.stage1.quantiles),
                "mad_factor": self.stage1.mad_factor,
            },
        }
        snapshot_payload = build_snapshot_payload(
            env=env_info,
            method=method_info,
            feature_num=artifacts.calibration.feature_num,
            feature_bool=artifacts.calibration.feature_bool,
            feature_comparison={},
            git_commit=git_commit,
            data_meta_hash=data_meta,
        )
        append_selection_snapshot(self.paths.feature_active_path, snapshot_payload)


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return "unknown"


def _config_digest(cfg: Mapping[str, object]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + _sha256(blob)


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def run_pipeline(
    df: pd.DataFrame,
    *,
    config: Mapping[str, object],
    selection_cfg: Mapping[str, object],
    paths: SelectionPaths,
    model_config: Mapping[str, object],
) -> PipelineArtifacts:
    pipeline = SelectionPipeline(config, selection_cfg, paths, dry_run=selection_cfg.get("dry_run", False))
    return pipeline.run(df, model_config)
