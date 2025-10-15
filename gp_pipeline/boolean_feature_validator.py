# /gp_pipeline/boolean_feature_validator.py
"""Boolean-feature validation utilities for GA fitness evaluation.

This module mirrors ``selection.single_feature_validator`` but tailors the
statistical testing pipeline to binary (boolean) features. It evaluates a single
feature under supervised (feature vs. target) or unsupervised (feature
self-dependence) scenarios using CPCV splits, aggregates fold-level statistics,
computes multi-objective fitness metrics, and returns a structured report which
can be consumed by the GA that evolves ``feat_bool`` expressions.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from selection.boolean_stats import BooleanFeatureStats, compute_boolean_feature_stats
from selection.integrity import IntegrityReport, apply_integrity_gates
from selection.screening import ScreeningDetail
from selection.splits import CrossFitSplit, build_cross_fit_splits, inner_time_split, make_cpcv_scheduler
from training.cpcv import CPCVSplit

logger = logging.getLogger(__name__)

try:  # Optional SciPy acceleration for tail probabilities
    from scipy.stats import chi2 as _scipy_chi2  # type: ignore
    from scipy.stats import norm as _scipy_norm  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional
    _scipy_chi2 = None
    _scipy_norm = None

EXPOSURE_RATE_LOWER_BOUND = 0.02
EXPOSURE_RATE_UPPER_BOUND = 0.98


DEFAULT_SUPERVISED_METHODS: Mapping[str, Mapping[str, object]] = {
    "fisher_exact": {"enabled": True, "event_threshold": 0.0},
    "prop_ztest": {"enabled": True, "event_threshold": 0.0},
    "mutual_information": {"enabled": True, "event_threshold": 0.0, "permutations": 256},
}

DEFAULT_UNSUPERVISED_METHODS: Mapping[str, Mapping[str, object]] = {
    "runs_test": {"enabled": True},
    "lag_dependence": {
        "enabled": True,
        "lags": [1, 5, 21],
        "combine": "sidak",
    },
    "imbalance_test": {"enabled": True, "p0": 0.5},
}


@dataclass(slots=True)
class MethodStatistic:
    """Aggregate statistics for a single evaluation method."""

    method: str
    tests: int
    mean_effect: float
    max_effect: float
    mean_pvalue: float
    min_pvalue: float
    max_cliffs_delta: float


@dataclass(slots=True)
class FoldEvaluation:
    """Per-fold evaluation outcome used for aggregation."""

    fold_id: str
    feature: str
    method: str
    effect: float
    pvalue: float
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class BooleanFeatureValidationResult:
    """Full validation summary for a single boolean feature."""

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


def validate_boolean_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    target: Optional[str] = None,
    supervision: Optional[str] = None,
    alpha: float = 0.05,
    combine_method: str = "simes",
    screening_methods: Optional[Mapping[str, Mapping[str, object]]] = None,
    unsupervised_methods: Optional[Mapping[str, Mapping[str, object]]] = None,
    min_bars_per_split: Optional[int] = None,
    cpcv_cfg: Optional[Mapping[str, object]] = None,
    training_cfg: Optional[Mapping[str, object]] = None,
    inner_ratio: float = 0.5,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> BooleanFeatureValidationResult:
    """Validate a single boolean feature for GA fitness evaluation."""

    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in dataframe columns")

    if target is not None and target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe columns")

    if supervision is None:
        supervision = "supervised" if target is not None else "unsupervised"
    supervision = supervision.lower()

    if supervision == "supervised" and target is None:
        raise ValueError("Supervised validation requires a target column")

    if supervision == "supervised":
        methods_cfg = screening_methods or DEFAULT_SUPERVISED_METHODS
    else:
        methods_cfg = screening_methods or unsupervised_methods or DEFAULT_UNSUPERVISED_METHODS

    cleaned_df, integrity_report = apply_integrity_gates(
        df,
        [feature],
        max_missing_ratio=0.2,
        min_unique_fraction=0.0,
        max_zero_fraction=0.98,
        correlation_threshold=0.999,
    )

    feature_retained = feature in cleaned_df.columns
    if not feature_retained:
        result = BooleanFeatureValidationResult(
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
            logger.info(
                "Feature '%s' was dropped during integrity checks; validation aborted.",
                feature,
            )
        return result

    dataset_len = len(cleaned_df)
    adaptive_min_bars = 0
    if dataset_len:
        raw_adaptive = max(1, int(round(dataset_len * 0.1)))
        adaptive_min_bars = min(dataset_len, max(64, raw_adaptive))
    user_min = min_bars_per_split if (min_bars_per_split is not None and min_bars_per_split > 0) else None
    if user_min is None:
        effective_min_bars = adaptive_min_bars
    elif adaptive_min_bars:
        effective_min_bars = min(user_min, adaptive_min_bars)
    else:
        effective_min_bars = user_min
    effective_min_bars = max(16, effective_min_bars or 0)
    if dataset_len:
        effective_min_bars = min(effective_min_bars, dataset_len)

    scheduler_cfg = {
        "folds": 5,
        "test_folds": 1,
        "shuffle_seed": random_state,
        "purge_bars": None,
        "embargo_bars": None,
        "min_bars_per_split": effective_min_bars,
        "max_combinations": None,
        "parallel_chunks": 1,
    }
    if cpcv_cfg:
        scheduler_cfg.update({k: v for k, v in cpcv_cfg.items() if v is not None})

    train_cfg = {
        "min_trades_per_split": effective_min_bars,
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
            min_bars=effective_min_bars,
            inner_ratio=inner_ratio,
        )

    if not cross_splits:
        raise RuntimeError(
            "Unable to construct valid CPCV splits for validation. "
            "Increase dataset length or pass a smaller min_bars_per_split."
        )

    rng = np.random.default_rng(random_state)

    fold_results: List[FoldEvaluation] = []
    for split in cross_splits:
        fold_pairs = [
            (f"{split.outer.combo_id}_A", split.inner_train_a),
            (f"{split.outer.combo_id}_B", split.inner_train_b),
        ]
        for fold_id, indices in fold_pairs:
            if indices.size < effective_min_bars:
                continue
            fold_df = cleaned_df.iloc[indices]
            if fold_df.empty:
                continue
            if supervision == "supervised" and target is not None:
                fold_results.extend(
                    _evaluate_supervised_methods(
                        fold_df,
                        feature,
                        target,
                        methods_cfg,
                        rng,
                        fold_id,
                    )
                )
            else:
                fold_results.extend(
                    _evaluate_unsupervised_methods(
                        fold_df,
                        feature,
                        methods_cfg,
                        rng,
                        fold_id,
                    )
                )

    if not fold_results:
        logger.warning("No fold evaluations were produced for feature '%s'", feature)
        result = BooleanFeatureValidationResult(
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
            feature_retained=True,
        )
        if verbose:
            _print_summary(result)
        return result

    pvalues = [max(min(res.pvalue, 1.0), 1e-320) for res in fold_results]
    combined_pvalue = _combine_pvalues(pvalues, combine_method)
    combined_pvalue = float(np.clip(combined_pvalue, 0.0, 1.0))

    significant = combined_pvalue <= alpha

    method_stats, pooled_effects, pooled_pvalues, extra_summary = _aggregate_method_stats(fold_results)

    if supervision == "supervised":
        primary_effect_candidates = (
            extra_summary.get("prop_diff_abs_mean"),
            extra_summary.get("cliffs_delta_abs_mean"),
        )
    else:
        primary_effect_candidates = (
            extra_summary.get("lag_phi_max_mean"),
            extra_summary.get("imbalance_abs_mean"),
        )
    effect_size = 0.0
    for candidate in primary_effect_candidates:
        if candidate is None:
            continue
        if isinstance(candidate, float) and math.isnan(candidate):
            continue
        effect_size = float(candidate)
        break
    if effect_size == 0.0 and pooled_effects:
        effect_size = float(np.mean(np.abs(pooled_effects)))
    cliffs_delta = float(extra_summary.get("cliffs_delta_abs_mean", 0.0))

    fitness_objectives, fitness_goals, fitness_metrics, constraints = _compose_boolean_fitness(
        combined_pvalue=combined_pvalue,
        alpha=alpha,
        effects=pooled_effects,
        pvalues=pooled_pvalues,
        extra_summary=extra_summary,
        significant=significant,
    )

    screening_details = [
        ScreeningDetail(
            fold_id=res.fold_id,
            feature=res.feature,
            method=res.method,
            effect=res.effect,
            pvalue=res.pvalue,
            extras={k: float(v) for k, v in res.extras.items()},
        )
        for res in fold_results
    ]

    result = BooleanFeatureValidationResult(
        feature=feature,
        supervised=supervision == "supervised",
        target_column=target,
        alpha=alpha,
        combined_pvalue=combined_pvalue,
        effect_size=effect_size,
        cliffs_delta=cliffs_delta,
        significant=significant,
        methods=method_stats,
        integrity_report=integrity_report,
        feature_retained=True,
        screening_details=screening_details,
        fitness_objectives=fitness_objectives,
        fitness_goals=fitness_goals,
        fitness_metrics=fitness_metrics,
        constraints=constraints,
    )

    if verbose:
        _print_summary(result)

    return result


# ---------------------------------------------------------------------------
# Supervised evaluation methods
# ---------------------------------------------------------------------------

def _evaluate_supervised_methods(
    fold_df: pd.DataFrame,
    feature: str,
    target: str,
    methods_cfg: Mapping[str, Mapping[str, object]],
    rng: np.random.Generator,
    fold_id: str,
) -> List[FoldEvaluation]:
    results: List[FoldEvaluation] = []
    if feature not in fold_df.columns:
        return results

    available_methods = {
        name.lower(): cfg
        for name, cfg in methods_cfg.items()
        if bool(cfg.get("enabled", True))
    }
    if not available_methods:
        return results

    event_thresholds = {
        name: float(cfg.get("event_threshold", 0.0))
        for name, cfg in available_methods.items()
    }

    series = fold_df[feature]
    base_target = fold_df[target]

    stats_cache: Dict[float, BooleanFeatureStats | None] = {}
    for method_name, cfg in available_methods.items():
        method = method_name.lower()
        threshold = event_thresholds.get(method_name, 0.0)
        if method in {"fisher_exact"}:
            stats = _lookup_boolean_stats(stats_cache, fold_df, feature, target, threshold)
            res = _run_fisher_test(method, fold_id, feature, stats)
        elif method == "prop_ztest":
            stats = _lookup_boolean_stats(stats_cache, fold_df, feature, target, threshold)
            res = _run_prop_ztest(method, fold_id, feature, stats, cfg)
        elif method == "mutual_information":
            bin_feature, bin_target = _prepare_binary_arrays(series, base_target, threshold)
            res = _run_mutual_information_test(method, fold_id, feature, bin_feature, bin_target, cfg, rng)
        else:
            logger.debug("Unknown supervised method '%s' skipped", method)
            continue
        if res is not None:
            results.append(res)
    return results


def _lookup_boolean_stats(
    cache: Dict[float, BooleanFeatureStats | None],
    fold_df: pd.DataFrame,
    feature: str,
    target: str,
    threshold: float,
) -> BooleanFeatureStats | None:
    if threshold in cache:
        return cache[threshold]
    try:
        stats_map = compute_boolean_feature_stats(
            fold_df[[feature]],
            fold_df[target],
            [feature],
            bootstrap={"samples": 0, "alpha": 0.05, "block_size": 16},
            event_threshold=threshold,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Boolean stats computation failed: %s", exc)
        cache[threshold] = None
        return None
    cache[threshold] = stats_map.get(feature)
    return cache[threshold]


def _run_fisher_test(
    method: str,
    fold_id: str,
    feature: str,
    stats: BooleanFeatureStats | None,
) -> FoldEvaluation | None:
    if stats is None:
        return None
    a = stats.counts.get("a", 0)
    b = stats.counts.get("b", 0)
    c = stats.counts.get("c", 0)
    d = stats.counts.get("d", 0)
    total = max(a + b + c + d, 1)

    exposure = (a + b) / total
    event_rate_pos = a / max(a + b, 1)
    event_rate_neg = c / max(c + d, 1)
    prop_diff = event_rate_pos - event_rate_neg
    effect = abs(prop_diff)

    extras = {
        "odds_ratio": float(stats.odds_ratio),
        "relative_risk": float(stats.relative_risk),
        "prop_diff": float(prop_diff),
        "event_rate_pos": float(event_rate_pos),
        "event_rate_neg": float(event_rate_neg),
        "exposure_rate": float(exposure),
        "cliffs_delta": float(stats.cliffs_delta),
        "g_statistic": float(stats.g_statistic),
        "g_pvalue": float(stats.g_pvalue),
    }
    return FoldEvaluation(
        fold_id=fold_id,
        feature=feature,
        method=method,
        effect=float(effect),
        pvalue=float(stats.fisher_pvalue),
        extras=extras,
    )


def _run_prop_ztest(
    method: str,
    fold_id: str,
    feature: str,
    stats: BooleanFeatureStats | None,
    cfg: Mapping[str, object],
) -> FoldEvaluation | None:
    if stats is None:
        return None
    a = stats.counts.get("a", 0)
    b = stats.counts.get("b", 0)
    c = stats.counts.get("c", 0)
    d = stats.counts.get("d", 0)
    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        return FoldEvaluation(
            fold_id=fold_id,
            feature=feature,
            method=method,
            effect=0.0,
            pvalue=1.0,
            extras={"n1": float(n1), "n0": float(n0)},
        )

    p1 = a / n1
    p0 = c / n0
    diff = p1 - p0
    pooled = (a + c) / (n1 + n0)
    variance = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n0)
    if variance <= 0:
        return FoldEvaluation(
            fold_id=fold_id,
            feature=feature,
            method=method,
            effect=0.0,
            pvalue=1.0,
            extras={"n1": float(n1), "n0": float(n0)},
        )
    z = diff / math.sqrt(variance)
    pvalue = _two_sided_normal_pvalue(z)
    extras = {
        "z_score": float(z),
        "prop_diff": float(diff),
        "p1": float(p1),
        "p0": float(p0),
        "pooled": float(pooled),
        "n1": float(n1),
        "n0": float(n0),
    }
    return FoldEvaluation(
        fold_id=fold_id,
        feature=feature,
        method=method,
        effect=float(abs(diff)),
        pvalue=float(pvalue),
        extras=extras,
    )


def _run_mutual_information_test(
    method: str,
    fold_id: str,
    feature: str,
    binary_feature: np.ndarray,
    binary_target: np.ndarray,
    cfg: Mapping[str, object],
    rng: np.random.Generator,
) -> FoldEvaluation | None:
    n = binary_feature.size
    if n == 0:
        return None
    # Ensure compact uint8 representation for fast bitwise arithmetic.
    binary_feature = binary_feature.astype(np.uint8, copy=False)
    binary_target = binary_target.astype(np.uint8, copy=False)

    total = int(n)
    feature_pos = int(np.sum(binary_feature, dtype=np.int64))
    target_pos = int(np.sum(binary_target, dtype=np.int64))
    joint11 = int(np.sum(binary_feature & binary_target, dtype=np.int64))
    joint10 = feature_pos - joint11
    joint01 = target_pos - joint11
    joint00 = total - joint11 - joint10 - joint01
    mi = _binary_mutual_information_from_counts(joint00, joint01, joint10, joint11)

    permutations = int(cfg.get("permutations", 256))
    if permutations <= 0 or feature_pos == 0 or feature_pos == total or target_pos == 0 or target_pos == total:
        extras = {
            "mi": float(mi),
            "mi_null_mean": float(mi),
            "mi_null_std": 0.0,
            "mi_z": 0.0,
            "permutations": float(max(permutations, 0)),
        }
        return FoldEvaluation(
            fold_id=fold_id,
            feature=feature,
            method=method,
            effect=float(mi),
            pvalue=1.0,
            extras=extras,
        )

    feature_indices = np.flatnonzero(binary_feature)
    permuted_target = binary_target.copy()
    null_samples = np.empty(permutations, dtype=np.float64)

    for i in range(permutations):
        rng.shuffle(permuted_target)
        if feature_indices.size:
            perm_joint11 = int(np.sum(permuted_target[feature_indices], dtype=np.int64))
        else:
            perm_joint11 = 0
        perm_joint10 = feature_pos - perm_joint11
        perm_joint01 = target_pos - perm_joint11
        perm_joint00 = total - perm_joint11 - perm_joint10 - perm_joint01
        null_samples[i] = _binary_mutual_information_from_counts(
            perm_joint00,
            perm_joint01,
            perm_joint10,
            perm_joint11,
        )

    mu = float(np.mean(null_samples))
    sigma = float(np.std(null_samples) + 1e-12)
    z = (mi - mu) / sigma
    greater = int(np.count_nonzero(null_samples >= mi))
    pvalue = (greater + 1.0) / (permutations + 1.0)
    extras = {
        "mi": float(mi),
        "mi_null_mean": mu,
        "mi_null_std": sigma,
        "mi_z": float(z),
        "permutations": float(permutations),
    }
    return FoldEvaluation(
        fold_id=fold_id,
        feature=feature,
        method=method,
        effect=float(mi),
        pvalue=float(pvalue),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Unsupervised evaluation methods
# ---------------------------------------------------------------------------

def _evaluate_unsupervised_methods(
    fold_df: pd.DataFrame,
    feature: str,
    methods_cfg: Mapping[str, Mapping[str, object]],
    rng: np.random.Generator,
    fold_id: str,
) -> List[FoldEvaluation]:
    results: List[FoldEvaluation] = []
    if feature not in fold_df.columns:
        return results

    available_methods = {
        name.lower(): cfg
        for name, cfg in methods_cfg.items()
        if bool(cfg.get("enabled", True))
    }
    if not available_methods:
        return results

    binary = _ensure_binary_vector(fold_df[feature])
    if binary.size == 0:
        return results

    for method, cfg in available_methods.items():
        if method == "runs_test":
            res = _run_runs_test(method, fold_id, feature, binary)
        elif method == "lag_dependence":
            res = _run_lag_dependence(method, fold_id, feature, binary, cfg)
        elif method == "imbalance_test":
            res = _run_imbalance_test(method, fold_id, feature, binary, cfg)
        else:
            logger.debug("Unknown unsupervised method '%s' skipped", method)
            res = None
        if res is not None:
            results.append(res)
    return results


def _run_runs_test(
    method: str,
    fold_id: str,
    feature: str,
    binary: np.ndarray,
) -> FoldEvaluation:
    n = binary.size
    ones = int(binary.sum())
    zeros = n - ones
    if ones == 0 or zeros == 0:
        return FoldEvaluation(
            fold_id=fold_id,
            feature=feature,
            method=method,
            effect=0.0,
            pvalue=1.0,
            extras={"runs": 1.0, "n": float(n), "ones": float(ones)},
        )

    runs = 1
    for i in range(1, n):
        if binary[i] != binary[i - 1]:
            runs += 1
    expected = 1 + (2 * ones * zeros) / (n)
    variance = (
        2 * ones * zeros * (2 * ones * zeros - ones - zeros)
        / (n**2 * (n - 1))
    )
    if variance <= 0:
        z = 0.0
        pvalue = 1.0
    else:
        z = (runs - expected) / math.sqrt(variance)
        pvalue = _two_sided_normal_pvalue(z)
    extras = {
        "runs": float(runs),
        "expected_runs": float(expected),
        "runs_z": float(z),
        "ones": float(ones),
        "zeros": float(zeros),
        "n": float(n),
    }
    return FoldEvaluation(
        fold_id=fold_id,
        feature=feature,
        method=method,
        effect=float(abs(z)),
        pvalue=float(pvalue),
        extras=extras,
    )


def _run_lag_dependence(
    method: str,
    fold_id: str,
    feature: str,
    binary: np.ndarray,
    cfg: Mapping[str, object],
) -> FoldEvaluation | None:
    lags_raw = cfg.get("lags", [1, 5, 21])
    try:
        lags = [int(lag) for lag in lags_raw if int(lag) > 0]
    except Exception:
        lags = [1]
    if not lags:
        lags = [1]

    phi_values: List[float] = []
    pvalues: List[float] = []
    extras: Dict[str, float] = {}

    for lag in lags:
        if binary.size <= lag:
            continue
        current = binary[lag:]
        past = binary[:-lag]
        a = int(np.sum((current == 1) & (past == 1)))
        b = int(np.sum((current == 1) & (past == 0)))
        c = int(np.sum((current == 0) & (past == 1)))
        d = int(np.sum((current == 0) & (past == 0)))
        total = a + b + c + d
        if total == 0:
            continue
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        if denom <= 0:
            phi = 0.0
        else:
            phi = (a * d - b * c) / math.sqrt(denom)
        phi_values.append(phi)
        chi2 = total * (phi ** 2)
        p = _chi2_sf_df1(chi2)
        pvalues.append(p)
        extras[f"lag_phi@{lag}"] = float(phi)
        extras[f"lag_pvalue@{lag}"] = float(p)

    if not phi_values:
        return FoldEvaluation(
            fold_id=fold_id,
            feature=feature,
            method=method,
            effect=0.0,
            pvalue=1.0,
            extras={"lags_considered": float(len(lags))},
        )

    combine_strategy = str(cfg.get("combine", "sidak")).lower()
    combined_p = _combine_pvalues(pvalues, combine_strategy)
    effect = max(abs(phi) for phi in phi_values)
    extras.update(
        {
            "lag_phi_max": float(effect),
            "lag_phi_mean_abs": float(np.mean(np.abs(phi_values))),
            "lag_pvalue_min": float(min(pvalues)),
            "lags_considered": float(len(phi_values)),
        }
    )
    return FoldEvaluation(
        fold_id=fold_id,
        feature=feature,
        method=method,
        effect=float(effect),
        pvalue=float(combined_p),
        extras=extras,
    )


def _run_imbalance_test(
    method: str,
    fold_id: str,
    feature: str,
    binary: np.ndarray,
    cfg: Mapping[str, object],
) -> FoldEvaluation:
    n = binary.size
    ones = int(binary.sum())
    if n == 0:
        return FoldEvaluation(
            fold_id=fold_id,
            feature=feature,
            method=method,
            effect=0.0,
            pvalue=1.0,
            extras={"n": 0.0},
        )
    p_hat = ones / n
    p0 = float(cfg.get("p0", 0.5))
    variance = p0 * (1.0 - p0) / max(n, 1)
    if variance <= 0:
        z = 0.0
        pvalue = 1.0
    else:
        z = (p_hat - p0) / math.sqrt(variance)
        pvalue = _two_sided_normal_pvalue(z)
    effect = abs(p_hat - p0)
    extras = {
        "p_hat": float(p_hat),
        "p0": float(p0),
        "imbalance": float(p_hat - p0),
        "z_score": float(z),
        "n": float(n),
        "ones": float(ones),
    }
    return FoldEvaluation(
        fold_id=fold_id,
        feature=feature,
        method=method,
        effect=float(effect),
        pvalue=float(pvalue),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Aggregation, fitness, and helpers
# ---------------------------------------------------------------------------

def _aggregate_method_stats(
    results: Sequence[FoldEvaluation],
) -> Tuple[List[MethodStatistic], List[float], List[float], Dict[str, float]]:
    grouped: Dict[str, List[FoldEvaluation]] = {}
    extra_values: Dict[str, List[float]] = {}
    pooled_effects: List[float] = []
    pooled_pvalues: List[float] = []

    for res in results:
        pooled_effects.append(float(res.effect))
        pooled_pvalues.append(float(res.pvalue))
        grouped.setdefault(res.method, []).append(res)
        for key, value in res.extras.items():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            try:
                float_value = float(value)
            except TypeError:
                continue
            extra_values.setdefault(key, []).append(float_value)

    method_stats: List[MethodStatistic] = []
    for method, entries in grouped.items():
        effects = [float(entry.effect) for entry in entries]
        pvals = [float(entry.pvalue) for entry in entries]
        cliffs = [
            abs(float(entry.extras.get("cliffs_delta", 0.0)))
            for entry in entries
            if "cliffs_delta" in entry.extras
        ]
        method_stats.append(
            MethodStatistic(
                method=method,
                tests=len(entries),
                mean_effect=float(np.mean(effects)),
                max_effect=float(np.max(effects)),
                mean_pvalue=float(np.mean(pvals)),
                min_pvalue=float(np.min(pvals)),
                max_cliffs_delta=float(max(cliffs) if cliffs else 0.0),
            )
        )

    method_stats.sort(key=lambda stat: stat.mean_pvalue)

    extra_summary: Dict[str, float] = {}
    for key, values in extra_values.items():
        arr = np.asarray(values, dtype=float)
        extra_summary[f"{key}_mean"] = float(np.mean(arr))
        extra_summary[f"{key}_std"] = float(np.std(arr))
        extra_summary[f"{key}_abs_mean"] = float(np.mean(np.abs(arr)))
        extra_summary[f"{key}_abs_std"] = float(np.std(np.abs(arr)))
        extra_summary[f"{key}_max"] = float(np.max(arr))
        extra_summary[f"{key}_min"] = float(np.min(arr))

        # For contingency metrics, compute symmetric magnitude so protection (x<1)
        # and risk (x>1) are treated on the same scale. We also winsorize the
        # magnitude's upper tail mildly to avoid unstable domination in summaries.
        if key in {"odds_ratio", "relative_risk"}:
            clipped = np.clip(arr, 1e-12, 1e12)
            log_abs = np.abs(np.log(clipped))
            mag = np.exp(log_abs)
            # Mild winsorization at 99.5 percentile
            try:
                hi = float(np.quantile(mag, 0.995))
            except Exception:
                hi = float(np.max(mag)) if mag.size else 1.0
            mag_w = np.clip(mag, 1.0, max(1.0, hi))
            # Overload the historical "_abs_*" summary keys to now represent
            # symmetric magnitude statistics to keep downstream compatibility.
            extra_summary[f"{key}_abs_mean"] = float(np.mean(mag_w))
            extra_summary[f"{key}_abs_std"] = float(np.std(mag_w))
            extra_summary[f"{key}_abs_max"] = float(np.max(mag_w))
            extra_summary[f"{key}_abs_min"] = float(np.min(mag_w))
            # Also expose log-scale magnitude as auxiliary metrics
            extra_summary[f"{key}_log_abs_mean"] = float(np.mean(log_abs))
            extra_summary[f"{key}_log_abs_std"] = float(np.std(log_abs))

    return method_stats, pooled_effects, pooled_pvalues, extra_summary


def _compose_boolean_fitness(
    *,
    combined_pvalue: float,
    alpha: float,
    effects: Sequence[float],
    pvalues: Sequence[float],
    extra_summary: Mapping[str, float],
    significant: bool,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, float | bool]]:
    min_pvalue_clip = 1e-16
    p_clipped = float(np.clip(combined_pvalue, min_pvalue_clip, 1.0))
    neg_log_p = float(-math.log10(p_clipped))
    significance_margin = max(0.0, alpha - combined_pvalue)

    effect_arr = np.asarray([abs(val) for val in effects], dtype=float)
    effect_arr = effect_arr[np.isfinite(effect_arr)]
    effect_mean = float(effect_arr.mean()) if effect_arr.size else 0.0
    effect_std = float(effect_arr.std()) if effect_arr.size else 0.0
    pvals_arr = np.asarray(pvalues, dtype=float)
    pvals_arr = pvals_arr[np.isfinite(pvals_arr)]
    pvalue_std = float(pvals_arr.std()) if pvals_arr.size else 0.0

    effect_stability = float(np.clip(1.0 / (1.0 + effect_std), 0.0, 1.0))
    pvalue_consistency = float(np.clip(1.0 / (1.0 + pvalue_std), 0.0, 1.0))

    fitness_objectives: Dict[str, float] = {
        "min_pvalue": p_clipped,
        "neg_log_pvalue": neg_log_p,
        "effect_strength": effect_mean,
        "effect_stability": effect_stability,
        "pvalue_consistency": pvalue_consistency,
        "significance_margin": significance_margin,
    }
    fitness_goals: Dict[str, str] = {
        "min_pvalue": "minimize",
        "neg_log_pvalue": "maximize",
        "effect_strength": "maximize",
        "effect_stability": "maximize",
        "pvalue_consistency": "maximize",
        "significance_margin": "maximize",
    }
    fitness_metrics: Dict[str, float] = {
        "effect_std": effect_std,
        "pvalue_std": pvalue_std,
        "neg_log_pvalue": neg_log_p,
        "significance_margin": significance_margin,
    }

    # Additional metrics from extras
    def _inject_metric(key: str, goal: str, summary_key: str) -> None:
        value = float(extra_summary.get(summary_key, float("nan")))
        if math.isnan(value):
            return
        fitness_objectives[key] = value
        fitness_goals[key] = goal

    def _inject_metric_metric(key: str, summary_key: str) -> None:
        value = float(extra_summary.get(summary_key, float("nan")))
        if math.isnan(value):
            return
        fitness_metrics[key] = value

    _inject_metric("prop_diff_abs", "maximize", "prop_diff_abs_mean")
    _inject_metric_metric("prop_diff_abs_std", "prop_diff_abs_std")
    _inject_metric("odds_ratio_abs", "maximize", "odds_ratio_abs_mean")
    _inject_metric_metric("odds_ratio_abs_std", "odds_ratio_abs_std")
    _inject_metric("relative_risk", "maximize", "relative_risk_mean")
    _inject_metric_metric("relative_risk_std", "relative_risk_std")
    _inject_metric("cliffs_delta_abs", "maximize", "cliffs_delta_abs_mean")
    _inject_metric_metric("cliffs_delta_abs_std", "cliffs_delta_abs_std")
    _inject_metric("mutual_information", "maximize", "mi_mean")
    _inject_metric_metric("mutual_information_std", "mi_std")
    _inject_metric("lag_phi_abs", "maximize", "lag_phi_max_mean")
    _inject_metric_metric("lag_phi_abs_std", "lag_phi_max_std")
    _inject_metric("runs_abs_z", "maximize", "runs_z_abs_mean")
    _inject_metric_metric("runs_abs_z_std", "runs_z_abs_std")
    _inject_metric("imbalance_abs", "maximize", "imbalance_abs_mean")
    _inject_metric_metric("imbalance_abs_std", "imbalance_abs_std")

    constraints: Dict[str, float | bool] = {
        "significant": significant,
        "alpha": alpha,
        "combined_pvalue": combined_pvalue,
    }
    exposure_rate = extra_summary.get("exposure_rate_mean")
    if exposure_rate is not None and not math.isnan(exposure_rate):
        constraints["exposure_rate_mean"] = exposure_rate
        constraints["exposure_rate_within_bounds"] = (
            EXPOSURE_RATE_LOWER_BOUND <= exposure_rate <= EXPOSURE_RATE_UPPER_BOUND
        )

    return fitness_objectives, fitness_goals, fitness_metrics, constraints


def _combine_pvalues(pvalues: Sequence[float], method: str) -> float:
    pvals = [float(p) for p in pvalues if float(p) >= 0.0]
    if not pvals:
        return 1.0
    method = method.lower()
    pvals = [min(max(p, 1e-320), 1.0 - 1e-16) for p in pvals]
    if method in {"simes", "simes_hc"}:
        ordered = np.sort(pvals)
        n = len(ordered)
        adjusted = ordered * (n / np.arange(1, n + 1))
        return float(np.clip(np.minimum(adjusted, 1.0).min(), 0.0, 1.0))
    if method in {"cauchy", "acat"}:
        angles = (0.5 - np.asarray(pvals)) * math.pi
        t_stat = float(np.tan(angles).mean())
        combined = 0.5 - math.atan(t_stat) / math.pi
        return float(np.clip(combined, 0.0, 1.0))
    if method in {"fisher", "chi2"}:
        statistic = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return float(_chi2_sf(statistic, df))
    if method in {"stouffer", "z"}:
        z = np.sum([_normal_isf(p / 2.0) * np.sign(0.5 - p) for p in pvals]) / math.sqrt(len(pvals))
        return _two_sided_normal_pvalue(z)
    if method in {"sidak", "sidak_min"}:
        min_p = min(pvals)
        n = len(pvals)
        return float(1.0 - (1.0 - min_p) ** n)
    # Fallback to Fisher for unsupported methods (including empirical_brown)
    statistic = -2.0 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(_chi2_sf(statistic, df))


def _apply_fdr(pvalues: Sequence[float], alpha: float, method: str) -> List[bool]:
    if not pvalues:
        return []
    method = method.lower()
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[sorted_idx]

    if method == "by":
        harmonic = sum(1.0 / k for k in range(1, n + 1))
        thresholds = alpha * np.arange(1, n + 1) / (n * harmonic)
    else:  # default BH
        thresholds = alpha * np.arange(1, n + 1) / n

    decisions = np.zeros(n, dtype=bool)
    passed = sorted_p <= thresholds
    if np.any(passed):
        max_idx = np.max(np.where(passed)[0])
        decisions[: max_idx + 1] = True
    result = np.zeros(n, dtype=bool)
    result[sorted_idx] = decisions
    return result.tolist()


def _ensure_binary_vector(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=float, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.uint8)


def _prepare_binary_arrays(
    feature_series: pd.Series,
    target_series: pd.Series,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_arr = _ensure_binary_vector(feature_series)
    target_raw = target_series.to_numpy(dtype=float, copy=False)
    target_bin = (target_raw > threshold).astype(np.uint8, copy=False)
    mask = np.ones_like(feature_arr, dtype=bool)
    if target_bin.shape != feature_arr.shape:
        size = min(target_bin.shape[0], feature_arr.shape[0])
        feature_arr = feature_arr[:size]
        target_bin = target_bin[:size]
    return feature_arr, target_bin


def _binary_mutual_information_from_counts(
    joint00: int,
    joint01: int,
    joint10: int,
    joint11: int,
) -> float:
    total = joint00 + joint01 + joint10 + joint11
    if total <= 0:
        return 0.0
    joint = np.array(
        [[joint00, joint01], [joint10, joint11]],
        dtype=np.float64,
    )
    joint /= float(total)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(joint > 0.0, np.log(joint) - np.log(px) - np.log(py), 0.0)
    mi = float(np.sum(joint * log_ratio))
    return max(mi, 0.0)


def _binary_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = x.astype(np.uint8, copy=False)
    y = y.astype(np.uint8, copy=False)
    joint11 = int(np.sum(x & y, dtype=np.int64))
    joint10 = int(np.sum(x, dtype=np.int64)) - joint11
    joint01 = int(np.sum(y, dtype=np.int64)) - joint11
    joint00 = x.size - joint11 - joint10 - joint01
    return _binary_mutual_information_from_counts(joint00, joint01, joint10, joint11)


def _chi2_sf_df1(value: float) -> float:
    if value <= 0:
        return 1.0
    if _scipy_chi2 is not None:
        try:
            return float(_scipy_chi2.sf(value, df=1))
        except Exception:  # pragma: no cover - SciPy edge case fallback
            pass
    return float(math.erfc(math.sqrt(value / 2.0)))


def _chi2_sf(value: float, df: int) -> float:
    if value <= 0:
        return 1.0
    if df <= 0:
        return 1.0
    if _scipy_chi2 is not None:
        try:
            return float(_scipy_chi2.sf(value, df=df))
        except Exception:  # pragma: no cover - SciPy edge case fallback
            pass
    if df == 2:
        return math.exp(-value / 2.0)
    # Wilson-Hilferty approximation for general df
    z = ((value / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return _normal_sf(z)


def _two_sided_normal_pvalue(z: float) -> float:
    return 2.0 * _normal_sf(abs(z))


def _normal_sf(z: float) -> float:
    if _scipy_norm is not None:
        try:
            return float(_scipy_norm.sf(z))
        except Exception:  # pragma: no cover - SciPy edge case fallback
            pass
    return 0.5 * math.erfc(z / math.sqrt(2.0))



def _normal_isf(p: float) -> float:
    # inverse survival function for standard normal via approximation
    if p <= 0.0:
        return math.inf
    if p >= 1.0:
        return -math.inf
    if _scipy_norm is not None:
        try:
            return float(_scipy_norm.isf(p))
        except Exception:  # pragma: no cover - SciPy edge case fallback
            pass
    return math.sqrt(2.0) * _erfc_inv(2.0 * p)


def _erfc_inv(y: float) -> float:
    # numerical approximation of erfc^{-1} (Abramowitz-Stegun 7.1.28 variant)
    if y <= 0.0:
        return math.inf
    if y >= 2.0:
        return -math.inf
    x = 1.0 - y
    w = -math.log((1.0 - x) * (1.0 + x))
    p = 0.810_226_121_24
    q = 1.432_788_828
    r = -3.523_3877
    s = -0.804_853_952
    t = 0.104_993_88
    u = 0.003_808_064
    return x * (p + q * x * x) / (1 + r * x * x + s * x ** 4 + t * x ** 6 + u * x ** 8)


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


def _print_summary(result: BooleanFeatureValidationResult) -> None:
    target_part = f" vs target '{result.target_column}'" if result.supervised else " (unsupervised)"
    logger.info(
        "Validation for feature '%s'%s:\n"
        "  combined p-value = %.6g (alpha=%.3f)\n"
        "  effect strength  = %.6g\n"
        "  cliffs delta     = %.6g\n"
        "  significant      = %s",
        result.feature,
        target_part,
        result.combined_pvalue,
        result.alpha,
        result.effect_size,
        result.cliffs_delta,
        result.significant,
    )
    if not result.methods:
        logger.info("  No evaluation methods executed")
        return
    logger.info("  Method breakdown:")
    for stat in result.methods:
        logger.info(
            "    - %s: tests=%d, mean_effect=%.4f, max_effect=%.4f, "
            "mean_p=%.4g, min_p=%.4g, max|delta|=%.4f",
            stat.method,
            stat.tests,
            stat.mean_effect,
            stat.max_effect,
            stat.mean_pvalue,
            stat.min_pvalue,
            stat.max_cliffs_delta,
        )
    if result.fitness_objectives:
        logger.info("  Fitness objectives:")
        for name, value in result.fitness_objectives.items():
            goal = result.fitness_goals.get(name, "")
            suffix = f" ({goal})" if goal else ""
            logger.info("    - %s: %.6g%s", name, value, suffix)


__all__ = [
    "validate_boolean_feature",
    "BooleanFeatureValidationResult",
    "MethodStatistic",
]
