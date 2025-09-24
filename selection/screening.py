"""Stage 1 screening utilities (rank tests + FDR)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import erf, sqrt
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .splits import CrossFitSplit


@dataclass
class ScreeningDetail:
    fold_id: str
    feature: str
    effect: float
    pvalue: float


@dataclass
class ScreeningResult:
    pvalues: Dict[str, float]
    effects: Dict[str, float]
    details: List[ScreeningDetail]
    passing_features: List[str]


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.size > 1 else 0.0


def _approx_spearman_pvalue(r: float, n: int) -> float:
    if n <= 3:
        return 1.0
    # Normal approximation of Spearman's rho (works for n>10 reasonably well)
    z = r * sqrt(n - 1)
    # two-sided using normal CDF via error function
    return float(max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2)))))))


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return 0.0
    rx = np.sort(x)
    ry = np.sort(y)
    nx = rx.size
    ny = ry.size
    i = j = 0
    greater = lesser = 0
    while i < nx and j < ny:
        if rx[i] > ry[j]:
            greater += nx - i
            j += 1
        elif rx[i] < ry[j]:
            lesser += ny - j
            i += 1
        else:
            i += 1
            j += 1
    return float((greater - lesser) / (nx * ny))


def _combine_pvalues(pvalues: Sequence[float], method: str) -> float:
    valid = [p for p in pvalues if 0.0 <= p <= 1.0]
    if not valid:
        return 1.0
    method = (method or "simes").lower()
    if method == "simes":
        ordered = sorted(valid)
        m = len(ordered)
        combined = min((m / (i + 1)) * p for i, p in enumerate(ordered))
        return float(min(1.0, combined))
    if method == "fisher":
        # Fisher's method without SciPy: -2 sum log p ~ chi2(2m); we approximate via normal
        stat = -2.0 * sum(np.log(p if p > 1e-12 else 1e-12) for p in valid)
        # Wilson-Hilferty normal approximation
        z = ((stat / (2 * len(valid))) ** (1 / 3) - (1 - 2 / (9 * len(valid)))) / sqrt(2 / (9 * len(valid)))
        return float(max(0.0, min(1.0, 1.0 - 0.5 * (1.0 + erf(z / sqrt(2))))))
    if method == "empirical_brown":
        # Stage 1 fallback: reuse Simes as a dependence-aware conservative rule
        ordered = sorted(valid)
        m = len(ordered)
        combined = min((m / (i + 1)) * p for i, p in enumerate(ordered))
        return float(min(1.0, combined))
    raise ValueError(f"Unknown combine method: {method}")


def _apply_fdr(pvalues: Mapping[str, float], alpha: float, method: str) -> List[str]:
    if not pvalues:
        return []
    method = method.lower()
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    selected: List[str] = []
    if method == "bh":
        for rank, (feature, pval) in enumerate(items, start=1):
            threshold = alpha * rank / m
            if pval <= threshold:
                selected.append(feature)
        return selected
    if method == "by":
        harmonic = sum(1.0 / i for i in range(1, m + 1))
        for rank, (feature, pval) in enumerate(items, start=1):
            threshold = alpha * rank / (m * harmonic)
            if pval <= threshold:
                selected.append(feature)
        return selected
    raise ValueError(f"Unsupported FDR method: {method}")


def screening_out_of_fold(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    cross_splits: Sequence[CrossFitSplit],
    combine_cfg: Mapping[str, str | float],
    min_bars: int,
) -> ScreeningResult:
    combine_method = str(combine_cfg.get("method", "simes")).lower()
    fdr_alpha = float(combine_cfg.get("alpha", 0.1))
    fdr_method = str(combine_cfg.get("fdr", "by")).lower()

    per_feature_effects: Dict[str, List[float]] = defaultdict(list)
    per_feature_pvals: Dict[str, List[float]] = defaultdict(list)
    details: List[ScreeningDetail] = []

    for split in cross_splits:
        for fold_label, indices in (("A", split.inner_train_a), ("B", split.inner_train_b)):
            if indices.size < min_bars:
                continue
            fold_df = df.iloc[indices]
            target = fold_df[target_column].to_numpy(copy=False)
            for feature in feature_columns:
                series = fold_df[feature].to_numpy(copy=False)
                mask = ~np.isnan(series) & ~np.isnan(target)
                if mask.sum() < 8:
                    continue
                x = series[mask]
                y = target[mask]
                rho = _spearman_corr(x, y)
                cliff = _cliffs_delta(x, y)
                pval = _approx_spearman_pvalue(rho, mask.sum())
                effect = float(max(abs(rho), abs(cliff)))
                per_feature_effects[feature].append(effect)
                per_feature_pvals[feature].append(pval)
                details.append(
                    ScreeningDetail(
                        fold_id=f"{split.outer.combo_id}_{fold_label}",
                        feature=feature,
                        effect=effect,
                        pvalue=pval,
                    )
                )

    combined_pvalues: Dict[str, float] = {}
    effect_summary: Dict[str, float] = {}
    for feature in feature_columns:
        pvals = per_feature_pvals.get(feature, [])
        combined_pvalues[feature] = _combine_pvalues(pvals, method=combine_method)
        effects = per_feature_effects.get(feature, [])
        effect_summary[feature] = float(np.mean(effects)) if effects else 0.0

    selected = _apply_fdr(combined_pvalues, alpha=fdr_alpha, method=fdr_method)
    # Zero out features not passing FDR to ensure downstream contracts know.
    return ScreeningResult(
        pvalues=combined_pvalues,
        effects=effect_summary,
        details=details,
        passing_features=selected,
    )
