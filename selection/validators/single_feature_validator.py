# /selection/validators/single_feature_validator.py
"""Single-feature validation utilities leveraging the selection toolkit.

This module reuses the integrity checks, CPCV splitting strategy, and
screening methodology from ``selection`` to validate the statistical
significance of an individual numeric feature. It supports both
supervised (feature vs. explicit target) and unsupervised (feature
autocorrelation via lagged dependence tests) workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .integrity import IntegrityReport, apply_integrity_gates
from .splits import CrossFitSplit, build_cross_fit_splits, inner_time_split, make_cpcv_scheduler
from .screening import ScreeningDetail, screening_out_of_fold
from training.cpcv import CPCVSplit

logger = logging.getLogger(__name__)


DEFAULT_SUPERVISED_METHODS: Mapping[str, Mapping[str, object]] = {
    "spearman": {"enabled": True, "cliffs_delta": True},
    "hsic_rff": {
        "enabled": True,
        "rff_dim": 128,
        "permutations": 64,
        "block_size": 288,
    },
    "dcor_sampled": {
        "enabled": True,
        "sample_size": 1024,
        "permutations": 64,
        "block_size": 288,
    },
}


DEFAULT_UNSUPERVISED_METHODS: Mapping[str, Mapping[str, object]] = {
    "auto_spearman": {
        "enabled": True,
        "cliffs_delta": True,
        "lags": [1, 5, 21],
        "lag_combine": "simes",
    },
    "auto_hsic_rff": {
        "enabled": True,
        "lags": [1, 5, 21],
        "rff_dim": 128,
        "permutations": 64,
        "block_size": 288,
        "lag_combine": "simes",
    },
    "auto_dcor_sampled": {
        "enabled": True,
        "lags": [1, 5, 21],
        "sample_size": 1024,
        "permutations": 64,
        "block_size": 288,
        "lag_combine": "simes",
    },
}


@dataclass(slots=True)
class MethodStatistic:
    """Aggregate statistics for a single screening method."""

    method: str
    tests: int
    mean_effect: float
    max_effect: float
    mean_pvalue: float
    min_pvalue: float
    max_cliffs_delta: float


@dataclass(slots=True)
class FeatureValidationResult:
    """Full validation summary for a single feature."""

    feature: str
    supervised: bool
    target_column: Optional[str]
    alpha: float
    combined_pvalue: float
    effect_size: float
    cliffs_delta: float
    significant: bool
    methods: List[MethodStatistic]
    integrity_report: IntegrityReport
    feature_retained: bool
    screening_details: List[ScreeningDetail] = field(default_factory=list)
    fitness_objectives: Dict[str, float] = field(default_factory=dict)
    fitness_goals: Dict[str, str] = field(default_factory=dict)
    fitness_metrics: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float | bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Dictionary representation convenient for interactive use."""

        return {
            "feature": self.feature,
            "supervised": self.supervised,
            "target_column": self.target_column,
            "alpha": self.alpha,
            "combined_pvalue": self.combined_pvalue,
            "effect_size": self.effect_size,
            "cliffs_delta": self.cliffs_delta,
            "significant": self.significant,
            "methods": [
                {
                    "method": stat.method,
                    "tests": stat.tests,
                    "mean_effect": stat.mean_effect,
                    "max_effect": stat.max_effect,
                    "mean_pvalue": stat.mean_pvalue,
                    "min_pvalue": stat.min_pvalue,
                    "max_cliffs_delta": stat.max_cliffs_delta,
                }
                for stat in self.methods
            ],
            "integrity_report": {
                "dropped_missing": self.integrity_report.dropped_missing,
                "dropped_constant": self.integrity_report.dropped_constant,
                "dropped_sparse": self.integrity_report.dropped_sparse,
                "dropped_duplicates": self.integrity_report.dropped_duplicates,
                "dropped_correlated": self.integrity_report.dropped_correlated,
            },
            "feature_retained": self.feature_retained,
            "fitness_objectives": self.fitness_objectives,
            "fitness_goals": self.fitness_goals,
            "fitness_metrics": self.fitness_metrics,
            "constraints": self.constraints,
        }


def validate_numeric_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    target: Optional[str] = None,
    supervision: Optional[str] = None,
    alpha: float = 0.05,
    combine_method: str = "empirical_brown",
    fdr_method: str = "by",
    screening_methods: Optional[Mapping[str, Mapping[str, object]]] = None,
    unsupervised_methods: Optional[Mapping[str, Mapping[str, object]]] = None,
    min_bars_per_split: int = 256,
    cpcv_cfg: Optional[Mapping[str, object]] = None,
    training_cfg: Optional[Mapping[str, object]] = None,
    inner_ratio: float = 0.5,
    random_state: Optional[int] = None,
    heavy_switch_n: int = 5000,
    hsic_backend: str = "auto",
    dcor_backend: str = "auto",
    hsic_rff_dim: int = 128,
    hsic_rff_memory_mb: float = 256.0,
    hsic_rff_min_dim: int = 32,
    dcor_sample_size: int = 1024,
    effect_min: float = 0.0,
    cliffs_delta_min: float = 0.0,
    verbose: bool = True,
) -> FeatureValidationResult:
    """Validate the statistical significance of a single numeric feature.

    Args:
        df: Source dataframe containing the feature (and optional target).
        feature: Column name of the numeric feature to evaluate.
        target: Optional target column. When provided, supervised dependence
            tests are executed against ``target``. When absent, the feature is
            evaluated via lagged self-dependence (unsupervised).
        supervision: Optional explicit supervision mode. Defaults to
            ``"supervised"`` when ``target`` is provided and ``"unsupervised"``
            otherwise.
        alpha: Significance level used for the BY/BH FDR threshold.
        combine_method: Method for combining p-values across splits (e.g.
            ``"simes"`` or ``"empirical_brown"``).
        fdr_method: FDR control procedure ("by" or "bh").
        screening_methods: Override mapping for supervised screening methods.
        unsupervised_methods: Override mapping for unsupervised methods.
        min_bars_per_split: Minimum observations required per CPCV split.
        cpcv_cfg: Optional overrides for the CPCV scheduler configuration.
        training_cfg: Optional overrides for training strategy config (e.g.
            ``max_hold_period_bars``).
        inner_ratio: Fraction used to split each CPCV train fold into inner A/B.
        random_state: Seed controlling reproducibility of stochastic tests.
        heavy_switch_n: Sample size threshold to switch HSIC/dCor backends.
        hsic_backend: Default HSIC backend ("auto", "matrix", "rff").
        dcor_backend: Default distance correlation backend.
        hsic_rff_dim: Random Fourier feature dimension for HSIC.
        hsic_rff_memory_mb: Memory guard for HSIC RFF backend.
        hsic_rff_min_dim: Minimum HSIC RFF dimension enforced by the guard.
        dcor_sample_size: Sample size for stochastic distance correlation.
        effect_min: Minimum effect size required for significance.
        cliffs_delta_min: Minimum Cliff's delta required for significance.
        verbose: When True, prints a human-readable summary alongside returning
            the structured result.

    Returns:
        FeatureValidationResult containing aggregate statistics and metadata.
    """

    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in dataframe columns")

    if target is not None and target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe columns")

    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise TypeError(f"Feature '{feature}' must be numeric for validation")

    if supervision is None:
        supervision = "supervised" if target is not None else "unsupervised"
    supervision = supervision.lower()

    if supervision == "supervised" and target is None:
        raise ValueError("Supervised validation requires a target column")

    methods = screening_methods or (
        DEFAULT_SUPERVISED_METHODS if supervision == "supervised" else DEFAULT_UNSUPERVISED_METHODS
    )

    combine_cfg = {"method": combine_method, "alpha": alpha, "fdr": fdr_method}

    cleaned_df, integrity_report = apply_integrity_gates(
        df,
        [feature],
        max_missing_ratio=0.2,
        min_unique_fraction=0.01,
        max_zero_fraction=0.995,
        correlation_threshold=0.999,
    )

    feature_retained = feature in cleaned_df.columns
    if not feature_retained:
        result = FeatureValidationResult(
            feature=feature,
            supervised=supervision == "supervised",
            target_column=target,
            alpha=alpha,
            combined_pvalue=1.0,
            effect_size=0.0,
            cliffs_delta=0.0,
            significant=False,
            methods=[],
            integrity_report=integrity_report,
            feature_retained=False,
        )
        if verbose:
            print(
                f"Feature '{feature}' was dropped during integrity checks; "
                "validation aborted."
            )
        return result

    scheduler_cfg = {
        "folds": 5,
        "test_folds": 1,
        "shuffle_seed": random_state,
        "purge_bars": None,
        "embargo_bars": None,
        "min_bars_per_split": min_bars_per_split,
        "max_combinations": None,
        "parallel_chunks": 1,
    }
    if cpcv_cfg:
        scheduler_cfg.update({k: v for k, v in cpcv_cfg.items() if v is not None})

    train_cfg = {
        "min_trades_per_split": min_bars_per_split,
        "max_hold_period_bars": 0,
    }
    if training_cfg:
        train_cfg.update({k: v for k, v in training_cfg.items() if v is not None})

    scheduler = make_cpcv_scheduler(
        total_len=len(cleaned_df),
        cfg=scheduler_cfg,
        train_cfg=train_cfg,
    )
    cross_splits = build_cross_fit_splits(
        scheduler=scheduler,
        inner_ratio=inner_ratio,
        require_valid=True,
    )

    if not cross_splits:
        cross_splits = _build_fallback_split(
            df_length=len(cleaned_df),
            min_bars=min_bars_per_split,
            inner_ratio=inner_ratio,
        )

    if not cross_splits:
        raise RuntimeError(
            "Unable to construct valid CPCV splits for validation. "
            "Increase dataset length or lower min_bars_per_split."
        )

    screening_result = screening_out_of_fold(
        cleaned_df,
        [feature],
        target_column=target if supervision == "supervised" else None,
        cross_splits=cross_splits,
        combine_cfg=combine_cfg,
        min_bars=min_bars_per_split,
        methods=methods,
        random_state=random_state,
        effect_min=effect_min,
        cliffs_delta_min=cliffs_delta_min,
        supervision=supervision,
        heavy_switch_n=heavy_switch_n,
        hsic_default_backend=hsic_backend,
        dcor_default_backend=dcor_backend,
        hsic_rff_dim=hsic_rff_dim,
        hsic_rff_memory_mb=hsic_rff_memory_mb,
        hsic_rff_min_dim=hsic_rff_min_dim,
        dcor_sample_size=dcor_sample_size,
        record_details=True,
    )

    combined_pvalue = screening_result.pvalues.get(feature, 1.0)
    effect_size = screening_result.effects.get(feature, 0.0)
    delta_value = screening_result.cliffs_delta.get(feature, 0.0)
    significant = feature in screening_result.passing_features

    method_stats, pooled_effects, pooled_pvalues = _aggregate_method_stats(
        screening_result.details, feature
    )

    fitness_objectives, fitness_goals, fitness_metrics, constraints = _compose_fitness(
        combined_pvalue=combined_pvalue,
        alpha=alpha,
        effect_size=effect_size,
        cliffs_delta=delta_value,
        pooled_effects=pooled_effects,
        pooled_pvalues=pooled_pvalues,
        significant=significant,
    )

    result = FeatureValidationResult(
        feature=feature,
        supervised=supervision == "supervised",
        target_column=target,
        alpha=alpha,
        combined_pvalue=combined_pvalue,
        effect_size=effect_size,
        cliffs_delta=delta_value,
        significant=significant,
        methods=method_stats,
        integrity_report=integrity_report,
        feature_retained=True,
        screening_details=[detail for detail in screening_result.details if detail.feature == feature],
        fitness_objectives=fitness_objectives,
        fitness_goals=fitness_goals,
        fitness_metrics=fitness_metrics,
        constraints=constraints,
    )

    if verbose:
        _print_summary(result)

    return result


def _aggregate_method_stats(
    details: Sequence[ScreeningDetail],
    feature: str,
) -> Tuple[List[MethodStatistic], List[float], List[float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for detail in details:
        if detail.feature != feature:
            continue
        stats = grouped.setdefault(detail.method, {
            "effect": [],
            "pvalue": [],
            "cliffs": [],
        })
        stats["effect"].append(detail.effect)
        stats["pvalue"].append(detail.pvalue)
        if "cliffs_delta" in detail.extras:
            stats["cliffs"].append(detail.extras["cliffs_delta"])

    output: List[MethodStatistic] = []
    pooled_effects: List[float] = []
    pooled_pvalues: List[float] = []
    for method, values in grouped.items():
        effects = values["effect"]
        pvalues = values["pvalue"]
        cliffs = values["cliffs"] or [0.0]
        if not effects or not pvalues:
            continue
        pooled_effects.extend(effects)
        pooled_pvalues.extend(pvalues)
        output.append(
            MethodStatistic(
                method=method,
                tests=len(effects),
                mean_effect=float(np.mean(effects)),
                max_effect=float(np.max(effects)),
                mean_pvalue=float(np.mean(pvalues)),
                min_pvalue=float(np.min(pvalues)),
                max_cliffs_delta=float(np.max(np.abs(cliffs))),
            )
        )
    return sorted(output, key=lambda stat: stat.mean_pvalue), pooled_effects, pooled_pvalues


def _compose_fitness(
    *,
    combined_pvalue: float,
    alpha: float,
    effect_size: float,
    cliffs_delta: float,
    pooled_effects: Sequence[float],
    pooled_pvalues: Sequence[float],
    significant: bool,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, float | bool]]:
    eps = 1e-300
    p_clipped = max(combined_pvalue, eps)
    neg_log_p = float(-np.log10(p_clipped))
    significance_margin = max(0.0, alpha - combined_pvalue)

    if pooled_effects:
        effect_mean = float(np.mean(pooled_effects))
        effect_std = float(np.std(pooled_effects))
    else:
        effect_mean = effect_size
        effect_std = 0.0
    if pooled_pvalues:
        pvalue_mean = float(np.mean(pooled_pvalues))
        pvalue_std = float(np.std(pooled_pvalues))
    else:
        pvalue_mean = combined_pvalue
        pvalue_std = 0.0

    effect_stability = 1.0 / (1.0 + effect_std)
    pvalue_consistency = 1.0 / (1.0 + pvalue_std)

    fitness_objectives: Dict[str, float] = {
        "min_pvalue": combined_pvalue,
        "neg_log_pvalue": neg_log_p,
        "effect_size": effect_size,
        "cliffs_delta": cliffs_delta,
        "significance_margin": significance_margin,
        "mean_method_effect": effect_mean,
        "effect_stability": effect_stability,
        "pvalue_consistency": pvalue_consistency,
    }
    fitness_goals: Dict[str, str] = {
        "min_pvalue": "minimize",
        "neg_log_pvalue": "maximize",
        "effect_size": "maximize",
        "cliffs_delta": "maximize",
        "significance_margin": "maximize",
        "mean_method_effect": "maximize",
        "effect_stability": "maximize",
        "pvalue_consistency": "maximize",
    }
    fitness_metrics: Dict[str, float] = {
        "mean_method_pvalue": pvalue_mean,
        "pvalue_std": pvalue_std,
        "effect_std": effect_std,
        "neg_log_pvalue": neg_log_p,
        "significance_margin": significance_margin,
    }
    constraints: Dict[str, float | bool] = {
        "significant": significant,
        "alpha": alpha,
        "combined_pvalue": combined_pvalue,
    }
    return fitness_objectives, fitness_goals, fitness_metrics, constraints


def _build_fallback_split(
    *,
    df_length: int,
    min_bars: int,
    inner_ratio: float,
) -> List[CrossFitSplit]:
    if df_length < max(min_bars * 2, 16):
        return []

    indices = np.arange(df_length, dtype=int)
    test_size = max(min_bars, int(round(df_length * 0.2)))
    if df_length - test_size < min_bars:
        test_size = max(min_bars, df_length // 3)
    if df_length - test_size < min_bars:
        return []

    train_idx = indices[:-test_size]
    test_idx = indices[-test_size:]

    split_a, split_b = inner_time_split(train_idx, ratio=inner_ratio)
    if split_a.size < min_bars or split_b.size < min_bars:
        mid = max(1, train_idx.size // 2)
        split_a, split_b = train_idx[:mid], train_idx[mid:]
    if split_a.size < min_bars or split_b.size < min_bars:
        return []

    outer = CPCVSplit(
        combo_id=0,
        test_folds=(0,),
        train_indices=train_idx,
        test_indices=test_idx,
        purge_indices=np.array([], dtype=int),
        embargo_indices=np.array([], dtype=int),
    )

    cross_split = CrossFitSplit(
        outer=outer,
        inner_train_a=split_a,
        inner_train_b=split_b,
    )
    return [cross_split]


def _print_summary(result: FeatureValidationResult) -> None:
    target_part = f" vs target '{result.target_column}'" if result.supervised else " (unsupervised)"
    print(
        f"Validation for feature '{result.feature}'{target_part}:\n"
        f"  combined p-value = {result.combined_pvalue:.6g} (alpha={result.alpha})\n"
        f"  effect size      = {result.effect_size:.6g}\n"
        f"  cliffs delta     = {result.cliffs_delta:.6g}\n"
        f"  significant      = {result.significant}"
    )
    if not result.methods:
        print("  No screening tests executed (insufficient data)")
        return
    print("  Method breakdown:")
    for stat in result.methods:
        print(
            "    - {method}: tests={tests}, mean_effect={mean_eff:.4f}, "
            "max_effect={max_eff:.4f}, mean_p={mean_p:.4g}, min_p={min_p:.4g}, "
            "max|delta|={max_delta:.4f}".format(
                method=stat.method,
                tests=stat.tests,
                mean_eff=stat.mean_effect,
                max_eff=stat.max_effect,
                mean_p=stat.mean_pvalue,
                min_p=stat.min_pvalue,
                max_delta=stat.max_cliffs_delta,
            )
        )
    if result.fitness_objectives:
        print("  Fitness objectives:")
        for name, value in result.fitness_objectives.items():
            goal = result.fitness_goals.get(name, "")
            suffix = f" ({goal})" if goal else ""
            print(f"    - {name}: {value:.6g}{suffix}")


__all__ = [
    "validate_numeric_feature",
    "FeatureValidationResult",
    "MethodStatistic",
]
