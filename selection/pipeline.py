# /selection/pipeline.py
"""Selection pipeline orchestrator implementing Stage 1 & Stage 2 of the design."""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibrationResult, calibrate_ranges
from .boolean_stats import compute_boolean_feature_stats
from .integrity import IntegrityReport, apply_integrity_gates
from .loader import SelectionPaths
from .screening import ScreeningResult, screening_out_of_fold
from .selector import (
    SelectionDiagnostics,
    StabilitySelectionConfig,
    StabilitySelectionOutcome,
    enforce_vif_and_partial_r2,
    greedy_representative_selection,
    unsupervised_stability_selection,
)
from .diversity import (
    DiversitySelectionResult,
    determinantal_point_process_diversity,
    facility_location_diversity,
)
from .snapshot import (
    append_selection_snapshot,
    build_snapshot_payload,
    compute_data_meta_hash,
    atomic_write_yaml,
)
from .splits import CrossFitSplit, build_cross_fit_splits, make_cpcv_scheduler


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StageOneConfig:
    target_column: Optional[str]
    supervision: str
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
    mad_mode: str = "approx"
    mad_chunk_rows: int = 20000
    mad_col_block_size: int = 128
    mad_scratch_dir: Optional[str] = None
    mad_delete_scratch: bool = True
    whitelist: Sequence[str] = ()
    screening_methods: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    random_state: Optional[int] = None
    enforce_significance: bool = True
    effect_min: float = 0.0
    cliffs_delta_min: float = 0.0
    graph_lasso_alpha: float = 0.01
    heavy_switch_n: int = 5000
    hsic_default_backend: str = "auto"
    dcor_default_backend: str = "auto"
    hsic_rff_dim: int = 128
    dcor_sample_size: int = 1024
    record_details: bool = True


@dataclass(frozen=True, slots=True)
class StageTwoConfig:
    enabled: bool
    corr_cap: float
    max_features: Optional[int]
    inner_ratio: float
    cutoff: float
    stability: Optional[StabilitySelectionConfig]
    knockoffs: Optional["KnockoffConfig"]


@dataclass(slots=True)
class StageTwoDiagnostics:
    method: str
    frequencies: Dict[str, float] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    discarded_low_freq: List[str] = field(default_factory=list)
    models_evaluated: int = 0
    support_counts: Dict[str, int] = field(default_factory=dict)
    cutoff: float = 0.0
    knockoff_statistics: Dict[str, float] = field(default_factory=dict)
    knockoff_attempts: List[str] = field(default_factory=list)
    knockoff_generator: Optional[str] = None
    knockoff_threshold: Optional[float] = None


@dataclass(frozen=True, slots=True)
class KnockoffConfig:
    enabled: bool
    fdr: float
    statistic: str
    shrinkage: float
    random_state: Optional[int]
    generator: str
    fallback: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiversityConfig:
    enabled: bool
    method: str
    top_k: Optional[int]
    family_cap: Optional[int]
    lambda_scale: float
    jitter: float


@dataclass(slots=True)
class DiversityDiagnostics:
    method: str
    selected: List[str]
    coverage: Dict[str, float]
    discarded: List[str]
    similarity: str
    parameters: Dict[str, Any]


@dataclass(slots=True)
class PipelineArtifacts:
    integrity: IntegrityReport
    screening: ScreeningResult
    stage2: Optional[StageTwoDiagnostics]
    diversity: Optional[DiversityDiagnostics]
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
        self.stage2 = self._build_stage2_config(selection_cfg)
        self.stage3 = self._build_stage3_config(selection_cfg)
        self._latest_df_shape: Optional[tuple[int, int]] = None

    @staticmethod
    def _build_stage1_config(cfg: Mapping[str, object]) -> StageOneConfig:
        raw_target = cfg.get("target_column")
        if raw_target is None:
            target_column: Optional[str] = None
        else:
            target_str = str(raw_target).strip()
            if target_str.lower() in {"", "none", "null"}:
                target_column = None
            else:
                target_column = target_str
        default_supervision = "unsupervised" if target_column is None else "supervised"
        supervision = str(cfg.get("supervision", default_supervision)).lower()
        if supervision not in {"supervised", "unsupervised"}:
            supervision = default_supervision
        min_bars = int(cfg.get("min_bars_per_split", 128))
        screen_cfg = cfg.get("stage1_screening", {})
        combine = screen_cfg.get("combine", {})
        calib_cfg = cfg.get("stage4_calibration", {})
        quantiles_raw = calib_cfg.get("robust_quantiles", (0.005, 0.995))
        try:
            q_low, q_high = quantiles_raw  # type: ignore[misc]
            quantiles = (float(q_low), float(q_high))
        except Exception:  # noqa: BLE001
            quantiles = (0.005, 0.995)

        mad_cfg = calib_cfg.get("mad_margin", {})
        mad_enabled_cfg = mad_cfg.get("enabled")
        raw_factor = mad_cfg.get("factor", 1.4826)
        if mad_enabled_cfg is not None and not bool(mad_enabled_cfg):
            mad_factor: Optional[float] = None
        else:
            try:
                mad_factor = float(raw_factor)
            except (TypeError, ValueError):
                mad_factor = 1.4826
        if mad_factor is not None and mad_factor <= 0:
            mad_factor = 0.0

        mad_mode = str(mad_cfg.get("mode", "approx") or "approx").lower()
        if mad_mode not in {"approx", "exact_chunked"}:
            mad_mode = "approx"

        chunk_rows_raw = mad_cfg.get("chunk_rows", 20000)
        try:
            mad_chunk_rows = int(chunk_rows_raw)
        except (TypeError, ValueError):
            mad_chunk_rows = 20000
        if mad_chunk_rows <= 0:
            mad_chunk_rows = 20000

        col_block_raw = mad_cfg.get("col_block_size", 128)
        try:
            mad_col_block_size = int(col_block_raw)
        except (TypeError, ValueError):
            mad_col_block_size = 128
        if mad_col_block_size <= 0:
            mad_col_block_size = 128

        scratch_raw = mad_cfg.get("scratch_dir")
        if scratch_raw in (None, "", False):
            mad_scratch_dir: Optional[str] = None
        else:
            mad_scratch_dir = str(scratch_raw)

        mad_delete_scratch = bool(mad_cfg.get("delete_scratch", True))
        whitelist = tuple(cfg.get("feature_whitelist", []))
        methods_cfg = screen_cfg.get("methods") or {"spearman": {"enabled": True}}
        normalized_methods: Dict[str, Dict[str, object]] = {}
        for raw_name, raw_spec in methods_cfg.items():
            name = str(raw_name).lower()
            if isinstance(raw_spec, Mapping):
                spec_dict = dict(raw_spec)
                enabled = bool(spec_dict.get("enabled", True))
            else:
                spec_dict = {}
                enabled = bool(raw_spec)
            spec_dict["enabled"] = enabled
            if enabled:
                normalized_methods[name] = spec_dict
        if not normalized_methods:
            normalized_methods = {"spearman": {"enabled": True}}
        random_state = screen_cfg.get("random_state")
        enforce_significance = bool(screen_cfg.get("enforce_significance", True))
        effect_min = float(screen_cfg.get("effect_min", screen_cfg.get("effect_size_min", 0.0)) or 0.0)
        cliffs_delta_min = float(screen_cfg.get("cliffs_delta_min", 0.0) or 0.0)
        graph_cfg = cfg.get("stage1_graph", {})
        graph_alpha = float(graph_cfg.get("alpha", graph_cfg.get("graph_lasso_alpha", 0.01)) or 0.01)
        heavy_switch_n = max(1, int(screen_cfg.get("heavy_switch_n", 5000) or 5000))
        hsic_backend = str(screen_cfg.get("hsic_default_backend", "auto"))
        hsic_backend = hsic_backend.lower() if hsic_backend else "auto"
        if hsic_backend not in {"auto", "matrix", "rff"}:
            hsic_backend = "auto"
        dcor_backend = str(screen_cfg.get("dcor_default_backend", "auto"))
        dcor_backend = dcor_backend.lower() if dcor_backend else "auto"
        if dcor_backend not in {"auto", "matrix", "sampled"}:
            dcor_backend = "auto"
        hsic_rff_dim = max(1, int(screen_cfg.get("hsic_rff_dim", 128) or 128))
        dcor_sample_size = max(8, int(screen_cfg.get("dcor_sample_size", 1024) or 1024))
        record_details = bool(screen_cfg.get("record_details", True))
        return StageOneConfig(
            target_column=target_column,
            supervision=supervision,
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
            mad_mode=mad_mode,
            mad_chunk_rows=mad_chunk_rows,
            mad_col_block_size=mad_col_block_size,
            mad_scratch_dir=mad_scratch_dir,
            mad_delete_scratch=mad_delete_scratch,
            whitelist=whitelist,
            screening_methods=normalized_methods,
            random_state=int(random_state) if random_state is not None else None,
            enforce_significance=enforce_significance,
            effect_min=effect_min,
            cliffs_delta_min=cliffs_delta_min,
            graph_lasso_alpha=graph_alpha,
            heavy_switch_n=heavy_switch_n,
            hsic_default_backend=hsic_backend,
            dcor_default_backend=dcor_backend,
            hsic_rff_dim=hsic_rff_dim,
            dcor_sample_size=dcor_sample_size,
            record_details=record_details,
        )

    @staticmethod
    def _build_stage2_config(cfg: Mapping[str, object]) -> StageTwoConfig:
        stage2_cfg = cfg.get("stage2_selection", {})
        corr_cap = float(stage2_cfg.get("corr_cap", 0.9))
        max_features = stage2_cfg.get("max_features")
        inner_ratio = float(stage2_cfg.get("inner_ratio", 0.5))

        stability_cfg = stage2_cfg.get("stability_selection", {})
        stability_enabled = bool(stability_cfg.get("enabled", False))
        stability_config: Optional[StabilitySelectionConfig] = None
        if stability_enabled:
            stability_config = StabilitySelectionConfig(
                subsamples=int(stability_cfg.get("subsamples", 64)),
                sample_fraction=float(stability_cfg.get("sample_fraction", 0.5)),
                cutoff=float(stability_cfg.get("cutoff", stage2_cfg.get("cutoff", 0.6))),
                max_features=stability_cfg.get("max_features", max_features),
                random_state=stability_cfg.get("random_state"),
            )

        knockoff_cfg = stage2_cfg.get("knockoffs", {})
        knockoff_enabled = bool(knockoff_cfg.get("enabled", False))
        knockoff_config: Optional[KnockoffConfig] = None
        if knockoff_enabled:
            fallback_raw = knockoff_cfg.get("fallback", [])
            if isinstance(fallback_raw, (list, tuple)):
                fallback = [str(name).lower() for name in fallback_raw if name]
            elif fallback_raw:
                fallback = [str(fallback_raw).lower()]
            else:
                fallback = []
            knockoff_config = KnockoffConfig(
                enabled=True,
                fdr=float(knockoff_cfg.get("fdr", 0.1)),
                statistic=str(knockoff_cfg.get("statistic", "correlation")).lower(),
                shrinkage=float(knockoff_cfg.get("shrinkage", 1e-3)),
                random_state=knockoff_cfg.get("random_state"),
                generator=str(knockoff_cfg.get("generator", "modelx")).lower(),
                fallback=tuple(fallback),
            )

        cutoff = float(
            stability_config.cutoff if stability_config else stage2_cfg.get("cutoff", 0.6)
        )

        explicit_enabled = stage2_cfg.get("enabled")
        enabled = bool(stability_config or knockoff_config)
        if explicit_enabled is not None:
            enabled = enabled and bool(explicit_enabled)

        return StageTwoConfig(
            enabled=enabled,
            corr_cap=corr_cap,
            max_features=max_features,
            inner_ratio=inner_ratio,
            cutoff=cutoff,
            stability=stability_config,
            knockoffs=knockoff_config,
        )

    @staticmethod
    def _build_stage3_config(cfg: Mapping[str, object]) -> DiversityConfig:
        div_cfg = cfg.get("stage3_diversity", {})
        enabled = bool(div_cfg.get("enabled", False))
        method = str(div_cfg.get("method", "facility_location")).lower()
        top_k = div_cfg.get("top_k", div_cfg.get("k"))
        if top_k is not None:
            top_k = int(top_k)
        family_cap_cfg = div_cfg.get("family_cap", {})
        family_cap = None
        if isinstance(family_cap_cfg, Mapping):
            family_cap = family_cap_cfg.get("max_per_family")
        elif family_cap_cfg not in (None, False):
            family_cap = family_cap_cfg
        if family_cap is not None:
            family_cap = int(family_cap)
        lambda_scale = float(div_cfg.get("lambda", div_cfg.get("lambda_scale", 1.0)))
        jitter = float(div_cfg.get("jitter", 1e-6))
        return DiversityConfig(
            enabled=enabled,
            method=method,
            top_k=top_k,
            family_cap=family_cap,
            lambda_scale=lambda_scale,
            jitter=jitter,
        )

    def _resolve_feature_columns(self, df: pd.DataFrame) -> List[str]:
        target = self.stage1.target_column
        if target and target in df.columns:
            candidates = [c for c in df.columns if c != target]
        else:
            candidates = list(df.columns)
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
        return build_cross_fit_splits(
            scheduler=scheduler,
            inner_ratio=self.stage2.inner_ratio,
            require_valid=True,
        )

    @staticmethod
    def _knockoff_generators(config: KnockoffConfig) -> List[str]:
        chain: List[str] = []
        seen: set[str] = set()
        candidates = [config.generator] + list(config.fallback)
        for entry in candidates:
            name = str(entry or "").lower()
            if not name or name in seen:
                continue
            chain.append(name)
            seen.add(name)
        if not chain:
            chain.append("modelx")
        return chain

    def run(self, df: pd.DataFrame, model_config: Mapping[str, object]) -> PipelineArtifacts:
        target = self.stage1.target_column
        if self.stage1.supervision == "supervised":
            if not target or target not in df.columns:
                raise KeyError("Target column required in supervised mode")
        elif target and target not in df.columns:
            target = None

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
            target_column=target,
            cross_splits=cross_splits,
            combine_cfg={
                "method": self.stage1.combine_method,
                "alpha": self.stage1.combine_alpha,
                "fdr": self.stage1.fdr_method,
            },
            min_bars=self.stage1.min_bars_per_split,
            methods=self.stage1.screening_methods,
            random_state=self.stage1.random_state,
            effect_min=self.stage1.effect_min,
            cliffs_delta_min=self.stage1.cliffs_delta_min,
            supervision=self.stage1.supervision,
            heavy_switch_n=self.stage1.heavy_switch_n,
            hsic_default_backend=self.stage1.hsic_default_backend,
            dcor_default_backend=self.stage1.dcor_default_backend,
            hsic_rff_dim=self.stage1.hsic_rff_dim,
            dcor_sample_size=self.stage1.dcor_sample_size,
            record_details=self.stage1.record_details,
        )

        passing_features = set(screening_result.passing_features)
        whitelist = set(self.stage1.whitelist)

        if not passing_features:
            raise RuntimeError(
                "No features passed the FDR threshold during screening. "
                "Consider loosening screening parameters if this is unexpected."
            )

        if self.stage1.enforce_significance:
            missing_whitelist = sorted(whitelist - passing_features)
            if missing_whitelist:
                logger.warning(
                    "Dropping %d whitelisted features that failed significance checks: %s",
                    len(missing_whitelist),
                    ", ".join(missing_whitelist[:10]) + ("..." if len(missing_whitelist) > 10 else ""),
                )
            candidate_features = passing_features
        else:
            candidate_features = passing_features | whitelist

        ranked_candidates = sorted(
            candidate_features,
            key=lambda f: (
                screening_result.effects.get(f, 0.0),
                f in self.stage1.whitelist,
            ),
            reverse=True,
        )

        stage2_diag: Optional[StageTwoDiagnostics] = None
        stage2_candidates = ranked_candidates
        if self.stage2.enabled and ranked_candidates:
            stage2_diag, stage2_candidates = self._run_stage2_selection(
                cleaned_df,
                ranked_candidates,
                screening_result.effects,
                target,
            )

        diversity_diag: Optional[DiversityDiagnostics] = None
        diversity_candidates = stage2_candidates
        if self.stage3.enabled and diversity_candidates:
            diversity_diag, diversity_candidates = self._apply_diversity(
                cleaned_df,
                diversity_candidates,
                screening_result.effects,
                stage2_diag,
            )

        representative_pool = diversity_candidates or stage2_candidates or ranked_candidates
        selection_scores = dict(screening_result.effects)
        if stage2_diag:
            for feature, value in stage2_diag.frequencies.items():
                selection_scores[feature] = max(selection_scores.get(feature, 0.0), value)

        selected_features, discarded_corr = greedy_representative_selection(
            cleaned_df,
            representative_pool,
            selection_scores,
            corr_cap=self.stage1.corr_cap,
            max_features=self.stage1.max_features,
            partial_alpha=self.stage1.graph_lasso_alpha,
            random_state=self.stage1.random_state,
        )

        final_features, selection_diag = enforce_vif_and_partial_r2(
            cleaned_df,
            target,
            selected_features,
            vif_threshold=self.stage1.vif_threshold,
            use_target_for_partial=(
                self.stage1.supervision == "supervised"
                and target is not None
                and target in cleaned_df.columns
            ),
        )
        selection_diag.discarded_high_corr = sorted(
            set(selection_diag.discarded_high_corr).union(discarded_corr)
        )

        fold_map: Dict[str, np.ndarray] = {}
        for split in cross_splits:
            fold_map.setdefault(split.outer.combo_id, split.outer.train_indices)
        fold_indices = list(fold_map.values())

        calibration_result = calibrate_ranges(
            cleaned_df,
            final_features,
            quantiles=self.stage1.quantiles,
            mad_factor=self.stage1.mad_factor,
            fold_indices=fold_indices,
            mad_mode=self.stage1.mad_mode,
            chunk_rows=self.stage1.mad_chunk_rows,
            col_block_size=self.stage1.mad_col_block_size,
            scratch_dir=self.stage1.mad_scratch_dir,
            delete_scratch=self.stage1.mad_delete_scratch,
        )

        bool_features = [feature for feature in calibration_result.feature_bool if feature in final_features]
        bool_cfg = self.selection_cfg.get("stage4_calibration", {}).get("boolean_stats", {})
        bool_enabled = bool(bool_cfg.get("enabled", True))
        if (
            bool_features
            and bool_enabled
            and self.stage1.supervision == "supervised"
            and target is not None
            and target in cleaned_df.columns
        ):
            bool_stats = compute_boolean_feature_stats(
                cleaned_df,
                cleaned_df[target],
                bool_features,
                bootstrap=bool_cfg,
                event_threshold=float(bool_cfg.get("event_threshold", 0.0)),
                rng=np.random.default_rng(self.stage1.random_state),
            )
            calibration_result.diagnostics["boolean_stats"] = {
                name: {
                    "counts": stats.counts,
                    "odds_ratio": stats.odds_ratio,
                    "relative_risk": stats.relative_risk,
                    "g_statistic": stats.g_statistic,
                    "g_pvalue": stats.g_pvalue,
                    "fisher_pvalue": stats.fisher_pvalue,
                    "cliffs_delta": stats.cliffs_delta,
                    "bootstrap_ci": list(stats.bootstrap_ci),
                }
                for name, stats in bool_stats.items()
            }

        # Merge bool features determined during calibration with surviving whitelist entries
        surviving = set(final_features) | set(calibration_result.feature_bool)
        whitelist_survivors = [feature for feature in self.stage1.whitelist if feature in surviving]
        final_feature_set = sorted(surviving | set(whitelist_survivors))

        updated_model_cfg = self._update_model_config(
            model_config,
            calibration_result,
            final_feature_set,
        )

        artifacts = PipelineArtifacts(
            integrity=integrity_report,
            screening=screening_result,
            stage2=stage2_diag,
            diversity=diversity_diag,
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

        final_feature_set = set(final_features)

        final_bool = [feature for feature in calibration.feature_bool if feature in final_feature_set]
        for feature in existing_bool:
            if feature in final_feature_set and feature not in final_bool:
                final_bool.append(feature)

        final_cmp: Dict[str, List[str]] = {}
        for key, value in existing_cmp.items():
            if key not in final_feature_set:
                continue
            filtered_values = [item for item in value if item in final_feature_set]
            if filtered_values:
                final_cmp[key] = filtered_values

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
            "integrity": asdict(artifacts.integrity),
            "screening": {
                "pvalues": artifacts.screening.pvalues,
                "effects": artifacts.screening.effects,
                "methods": artifacts.screening.methods,
                "details": [
                    {
                        "fold_id": detail.fold_id,
                        "feature": detail.feature,
                        "method": detail.method,
                        "effect": detail.effect,
                        "pvalue": detail.pvalue,
                    }
                    for detail in artifacts.screening.details[:512]
                ],
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
                "diagnostics": artifacts.calibration.diagnostics,
            },
        }
        if artifacts.stage2:
            payload["stage2"] = {
                "method": artifacts.stage2.method,
                "frequencies": artifacts.stage2.frequencies,
                "selected_features": artifacts.stage2.selected_features,
                "discarded_low_freq": artifacts.stage2.discarded_low_freq,
                "models_evaluated": artifacts.stage2.models_evaluated,
                "cutoff": artifacts.stage2.cutoff,
                "support_counts": artifacts.stage2.support_counts,
            }
        if artifacts.diversity:
            payload["diversity"] = {
                "method": artifacts.diversity.method,
                "selected": artifacts.diversity.selected,
                "coverage": artifacts.diversity.coverage,
                "discarded": artifacts.diversity.discarded,
                "similarity": artifacts.diversity.similarity,
                "parameters": artifacts.diversity.parameters,
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
        env_info["python"] = sys.version
        env_info["platform"] = platform.platform()
        try:
            packages = sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in metadata.distributions()
                if dist.metadata.get("Name")
            )
        except Exception:  # pragma: no cover - defensive
            packages = []
        env_info["packages"] = packages[:128]
        method_info = {
            "cv": self.selection_cfg.get("cpcv", self.config.get("cpcv", {})),
            "screening": {
                "combine": self.stage1.combine_method,
                "alpha": self.stage1.combine_alpha,
                "fdr": self.stage1.fdr_method,
                "tests": {
                    name: dict(spec)
                    for name, spec in self.stage1.screening_methods.items()
                },
                "executed": artifacts.screening.methods,
            },
            "selection": {
                "corr_cap": self.stage1.corr_cap,
                "vif_threshold": self.stage1.vif_threshold,
            },
            "selection_stage2": {
                "enabled": self.stage2.enabled,
                "method": (artifacts.stage2.method if artifacts.stage2 else "none"),
                "cutoff": self.stage2.cutoff,
                "max_features": self.stage2.max_features,
                "subsamples": (
                    self.stage2.stability.subsamples
                    if self.stage2.stability
                    else None
                ),
            },
            "diversity": {
                "enabled": self.stage3.enabled,
                "method": (artifacts.diversity.method if artifacts.diversity else "none"),
                "top_k": self.stage3.top_k,
                "family_cap": self.stage3.family_cap,
                "lambda_scale": self.stage3.lambda_scale,
                "jitter": self.stage3.jitter,
            },
            "calibration": {
                "quantiles": list(self.stage1.quantiles),
                "mad_factor": self.stage1.mad_factor,
                "mad_mode": self.stage1.mad_mode,
                "chunk_rows": self.stage1.mad_chunk_rows,
                "col_block_size": self.stage1.mad_col_block_size,
                "scratch_dir": self.stage1.mad_scratch_dir,
                "delete_scratch": self.stage1.mad_delete_scratch,
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

    def _run_stage2_selection(
        self,
        df: pd.DataFrame,
        candidates: Sequence[str],
        effect_scores: Mapping[str, float],
        target_column: Optional[str],
    ) -> Tuple[StageTwoDiagnostics, List[str]]:
        diag = StageTwoDiagnostics(method="none", cutoff=self.stage2.cutoff)
        current_candidates = list(candidates)

        if self.stage2.stability is not None and current_candidates:
            outcome = unsupervised_stability_selection(
                df,
                current_candidates,
                effect_scores=effect_scores,
                subsamples=self.stage2.stability.subsamples,
                sample_fraction=self.stage2.stability.sample_fraction,
                cutoff=self.stage2.cutoff,
                corr_cap=self.stage2.corr_cap,
                max_features=self.stage2.stability.max_features or self.stage2.max_features,
                partial_alpha=self.stage1.graph_lasso_alpha,
                random_state=self.stage2.stability.random_state,
            )
            diag.method = (
                "stability_selection"
                if self.stage1.supervision == "supervised"
                else "unsupervised_stability"
            )
            diag.frequencies = outcome.frequencies
            diag.selected_features = outcome.selected_features
            diag.discarded_low_freq = [
                name for name in current_candidates if name not in outcome.selected_features
            ]
            diag.models_evaluated = outcome.subsamples
            diag.support_counts = outcome.support_counts
            current_candidates = outcome.selected_features

        if diag.method == "none":
            diag.method = "stage1_passthrough"
            diag.selected_features = list(current_candidates)

        knockoff_cfg = self.stage2.knockoffs
        if (
            knockoff_cfg is not None
            and knockoff_cfg.enabled
            and current_candidates
            and target_column
            and target_column in df.columns
        ):
            attempts = self._knockoff_generators(knockoff_cfg)
            diag.knockoff_attempts = attempts.copy()
            best_stats: Optional[Dict[str, float]] = None
            best_generator: Optional[str] = None
            target_values = df[target_column].to_numpy(dtype=float, copy=False)
            numeric_features = [
                col
                for col in df.columns
                if col != target_column and pd.api.types.is_numeric_dtype(df[col])
            ]
            feature_pool = list(dict.fromkeys(list(current_candidates) + numeric_features))
            for generator in attempts:
                stats = self._compute_knockoff_statistics(
                    df,
                    feature_pool,
                    target_values,
                    generator=generator,
                    statistic=knockoff_cfg.statistic,
                    random_state=knockoff_cfg.random_state,
                )
                if stats is None:
                    continue
                best_stats = stats
                best_generator = generator
                break

            if best_stats is not None and best_generator is not None:
                diag.knockoff_generator = best_generator
                diag.knockoff_statistics = best_stats
                diag.method = f"{best_generator}_knockoffs"
                diag.frequencies = best_stats
                sorted_candidates = sorted(
                    current_candidates,
                    key=lambda name: best_stats.get(name, 0.0),
                    reverse=True,
                )
                limit = self.stage2.max_features or len(sorted_candidates)
                limit = max(1, min(limit, len(sorted_candidates)))
                selected = sorted_candidates[:limit]
                diag.selected_features = selected
                diag.discarded_low_freq = [
                    name for name in current_candidates if name not in selected
                ]
                diag.knockoff_threshold = (
                    best_stats[selected[-1]] if selected else None
                )
                diag.cutoff = knockoff_cfg.fdr
                diag.models_evaluated += len(attempts)
                current_candidates = selected
            else:
                logger.warning(
                    "All knockoff generators failed; falling back to existing candidates"
                )

        return diag, current_candidates


    def _apply_diversity(
        self,
        df: pd.DataFrame,
        candidates: Sequence[str],
        effect_scores: Mapping[str, float],
        stage2_diag: Optional[StageTwoDiagnostics],
    ) -> Tuple[DiversityDiagnostics, List[str]]:
        effective_scores = dict(effect_scores)
        if stage2_diag:
            for feature, value in stage2_diag.frequencies.items():
                effective_scores[feature] = max(effective_scores.get(feature, 0.0), value)

        top_k = self.stage3.top_k
        if self.stage1.max_features is not None:
            top_k = min(top_k or self.stage1.max_features, self.stage1.max_features)

        result: DiversitySelectionResult = facility_location_diversity(
            df,
            candidates,
            effective_scores,
            top_k=top_k,
            family_cap=self.stage3.family_cap,
        )

        if self.stage3.method == "dpp" and candidates:
            result = determinantal_point_process_diversity(
                df,
                candidates,
                effective_scores,
                top_k=top_k,
                family_cap=self.stage3.family_cap,
                lambda_scale=self.stage3.lambda_scale,
                jitter=self.stage3.jitter,
            )
            if not result.selected:
                # fall back to facility location result computed earlier
                result = facility_location_diversity(
                    df,
                    candidates,
                    effective_scores,
                    top_k=top_k,
                    family_cap=self.stage3.family_cap,
                )

        selected = result.selected or list(candidates)
        diag = DiversityDiagnostics(
            method=result.method,
            selected=selected,
            coverage=result.coverage,
            discarded=result.discarded,
            similarity=result.similarity,
            parameters=dict(result.diagnostics),
        )
        return diag, selected

    @staticmethod
    def _compute_knockoff_statistics(
        df: pd.DataFrame,
        features: Sequence[str],
        target_values: np.ndarray,
        *,
        generator: str,
        statistic: str,
        random_state: Optional[int],
    ) -> Optional[Dict[str, float]]:
        if target_values.size == 0:
            return None
        stats: Dict[str, float] = {}
        rng = np.random.default_rng(random_state)
        generator = (generator or "modelx").lower()
        stat_name = (statistic or "correlation").lower()

        for feature in features:
            if feature not in df.columns:
                continue
            column = df[feature].to_numpy(copy=False)
            mask = np.isfinite(column) & np.isfinite(target_values)
            if mask.sum() < 8:
                stats[feature] = 0.0
                continue
            x = column[mask].astype(float, copy=False)
            y = target_values[mask].astype(float, copy=False)

            if generator == "stability_selection":
                # Unsupported generator for knockoffs; signal failure via None.
                return None

            if stat_name in {"spearman", "rank"} or generator == "robust":
                series_x = pd.Series(x)
                series_y = pd.Series(y)
                corr = series_x.corr(series_y, method="spearman")
            else:
                if x.std(ddof=1) <= 1e-12 or y.std(ddof=1) <= 1e-12:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(x, y)[0, 1])

            if not np.isfinite(corr):
                corr = 0.0

            if stat_name in {"lcd", "lasso_coef_diff"}:
                noise = rng.normal(scale=1e-3)
                stats[feature] = abs(corr) + noise
            else:
                stats[feature] = abs(corr)

        if not stats:
            return None
        return stats


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
