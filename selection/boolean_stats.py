# /selection/boolean_stats.py
"""Boolean event statistics and inference utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .math_utils import chi2_sf
from .utils import block_bootstrap_indices, cliffs_delta


@dataclass
class BooleanFeatureStats:
    feature: str
    counts: Dict[str, int]
    odds_ratio: float
    relative_risk: float
    g_statistic: float
    g_pvalue: float
    fisher_pvalue: float
    cliffs_delta: float
    bootstrap_ci: tuple[float, float]


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    total = a + b + c + d
    if total == 0:
        return 1.0
    row1 = a + b
    col1 = a + c
    min_x = max(0, col1 - (total - row1))
    max_x = min(row1, col1)

    def hypergeom(x: int) -> float:
        return math.comb(row1, x) * math.comb(total - row1, col1 - x) / math.comb(total, col1)

    observed = hypergeom(a)
    p_value = 0.0
    for x in range(min_x, max_x + 1):
        prob = hypergeom(x)
        if prob <= observed + 1e-15:
            p_value += prob
    return min(1.0, p_value)


def _g_test(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    total = a + b + c + d
    if total == 0:
        return 0.0, 1.0
    expected = [
        (a + b) * (a + c) / total,
        (a + b) * (b + d) / total,
        (c + d) * (a + c) / total,
        (c + d) * (b + d) / total,
    ]
    observed = [a, b, c, d]
    g = 0.0
    for obs, exp in zip(observed, expected):
        if obs <= 0 or exp <= 0:
            continue
        g += 2.0 * obs * math.log(obs / exp)
    pval = chi2_sf(g, 1.0)
    return g, min(1.0, pval)


def _odds_ratio(a: int | np.ndarray, b: int | np.ndarray, c: int | np.ndarray, d: int | np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(c, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    numerator = (a_arr + 0.5) * (d_arr + 0.5)
    denominator = (b_arr + 0.5) * (c_arr + 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    return np.nan_to_num(ratio, nan=1.0, posinf=1e9, neginf=1e-9)


def _relative_risk(
    a: int | np.ndarray,
    b: int | np.ndarray,
    c: int | np.ndarray,
    d: int | np.ndarray,
) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(c, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    exposed = a_arr + b_arr
    unexposed = c_arr + d_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (a_arr + 0.5) / (exposed + 1.0)
        denominator = (c_arr + 0.5) / (unexposed + 1.0)
        rr = numerator / denominator
    rr = np.nan_to_num(rr, nan=1.0, posinf=1e9, neginf=1e-9)
    invalid = (exposed <= 0) | (unexposed <= 0)
    return np.where(invalid, 1.0, rr)


def _g_test_vectorized(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vectorized G-test statistic for a batch of 2x2 tables."""

    totals = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d
        expected_a = np.where(totals > 0, row1 * col1 / totals, 0.0)
        expected_b = np.where(totals > 0, row1 * col2 / totals, 0.0)
        expected_c = np.where(totals > 0, row2 * col1 / totals, 0.0)
        expected_d = np.where(totals > 0, row2 * col2 / totals, 0.0)

        def _term(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
            valid = (obs > 0) & (exp > 0)
            ratio = np.where(valid, obs / exp, 1.0)
            return np.where(valid, 2.0 * obs * np.log(ratio), 0.0)

        g_stat = (
            _term(a.astype(float), expected_a)
            + _term(b.astype(float), expected_b)
            + _term(c.astype(float), expected_c)
            + _term(d.astype(float), expected_d)
        )
    return np.nan_to_num(g_stat, nan=0.0, posinf=0.0, neginf=0.0)


def _fisher_exact_vectorized(
    a: Sequence[int],
    b: Sequence[int],
    c: Sequence[int],
    d: Sequence[int],
) -> np.ndarray:
    """Compute Fisher's exact test p-values for many 2x2 tables."""

    return np.array(
        [_fisher_exact(int(ai), int(bi), int(ci), int(di)) for ai, bi, ci, di in zip(a, b, c, d)],
        dtype=float,
    )


def compute_boolean_feature_stats(
    df: pd.DataFrame,
    target: pd.Series,
    features: Iterable[str],
    *,
    bootstrap: Mapping[str, int] | None = None,
    event_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, BooleanFeatureStats]:
    """Compute boolean-event statistics for the provided feature set."""

    feature_list = [name for name in features if name in df.columns]
    if not feature_list:
        return {}

    index = df.index.intersection(target.index)
    if index.empty:
        return {}

    if rng is None:
        rng = np.random.default_rng()
    params = bootstrap or {}
    samples = int(params.get("samples", 200))
    block_size = int(params.get("block_size", 32))
    alpha = float(params.get("alpha", 0.05))

    # Align data once and operate on numpy arrays to avoid per-feature DataFrame joins.
    frame = df.loc[index, feature_list]
    target_aligned = target.loc[index].to_numpy(dtype=float, copy=False)
    indicator = (target_aligned > event_threshold).astype(np.int8, copy=False)

    values = frame.to_numpy(dtype=np.float32, copy=False)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    values_int = values.astype(np.int16, copy=False)
    # Enforce binary representation while matching the original floor-to-int behaviour.
    binary = np.clip(values_int, 0, 1).astype(np.uint8, copy=False)

    target_pos = indicator.astype(np.int32, copy=False)
    ones = binary.sum(axis=0, dtype=np.int64)
    a = binary.T @ target_pos
    b = ones - a
    target_total = target_pos.sum(dtype=np.int64)
    zeros = binary.shape[0] - ones
    c = target_total - a
    d = zeros - c

    g_stats = _g_test_vectorized(a, b, c, d)
    g_pvalues = np.array([chi2_sf(float(stat), 1.0) if stat > 0 else 1.0 for stat in g_stats], dtype=float)
    fisher_pvalues = _fisher_exact_vectorized(a, b, c, d)
    odds_arr = np.asarray(_odds_ratio(a, b, c, d), dtype=float)
    rr_arr = np.asarray(_relative_risk(a, b, c, d), dtype=float)

    # Bootstrap mean-difference confidence intervals computed in a vectorised fashion.
    ci_lower = np.zeros(len(feature_list), dtype=float)
    ci_upper = np.zeros(len(feature_list), dtype=float)
    if samples > 0 and binary.shape[0] > 0:
        diffs = np.zeros((samples, len(feature_list)), dtype=np.float32)
        for draw in range(samples):
            idx = block_bootstrap_indices(binary.shape[0], block_size, rng)
            sample_binary = binary[idx]
            sample_target = target_aligned[idx]
            pos_counts = sample_binary.sum(axis=0, dtype=np.int64)
            pos_sum = np.einsum("ij,i->j", sample_binary, sample_target, dtype=np.float64)
            sample_total = float(np.sum(sample_target))
            sample_size = sample_binary.shape[0]
            neg_counts = sample_size - pos_counts
            neg_sum = sample_total - pos_sum
            pos_mean = np.divide(
                pos_sum,
                pos_counts,
                out=np.zeros_like(pos_sum, dtype=float),
                where=pos_counts > 0,
            )
            neg_mean = np.divide(
                neg_sum,
                neg_counts,
                out=np.zeros_like(neg_sum, dtype=float),
                where=neg_counts > 0,
            )
            diffs[draw] = pos_mean - neg_mean

        sorted_diffs = np.sort(diffs, axis=0)
        low_idx = max(0, int((alpha / 2.0) * samples) - 1)
        high_idx = max(0, min(samples - 1, int((1.0 - alpha / 2.0) * samples) - 1))
        ci_lower = sorted_diffs[low_idx].astype(float)
        ci_upper = sorted_diffs[high_idx].astype(float)

    stats: Dict[str, BooleanFeatureStats] = {}
    for offset, feature in enumerate(feature_list):
        column = binary[:, offset]
        mask_pos = column == 1
        mask_neg = column == 0
        target_pos_values = target_aligned[mask_pos]
        target_neg_values = target_aligned[mask_neg]
        delta = cliffs_delta(target_pos_values, target_neg_values)

        stats[feature] = BooleanFeatureStats(
            feature=feature,
            counts={
                "a": int(a[offset]),
                "b": int(b[offset]),
                "c": int(c[offset]),
                "d": int(d[offset]),
            },
            odds_ratio=float(odds_arr[offset]),
            relative_risk=float(rr_arr[offset]),
            g_statistic=float(g_stats[offset]),
            g_pvalue=float(g_pvalues[offset]),
            fisher_pvalue=float(fisher_pvalues[offset]),
            cliffs_delta=float(delta),
            bootstrap_ci=(float(ci_lower[offset]), float(ci_upper[offset])),
        )
    return stats
