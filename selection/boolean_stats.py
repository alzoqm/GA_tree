# /selection/boolean_stats.py
"""Boolean event statistics and inference utilities with vectorised operations.

This module computes per-feature contingency table statistics for boolean
features against a target label. It mirrors the public API of the legacy
implementation while improving vectorisation (Cliff's delta, odds/relative
risk, G-test) and keeping statistical outputs identical to the previous
behaviour, including exact Fisher p-values and block bootstrap confidence
intervals driven by the provided RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Prefer the project-provided survival function when available.
    from .math_utils import chi2_sf  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback only when math_utils missing
    def chi2_sf(x: np.ndarray | float, df: float = 1.0) -> np.ndarray:
        """Wilson–Hilferty approximation of the χ² survival function for df=1."""

        values = np.asarray(x, dtype=float)
        df = float(df)
        z = (np.power(values / df, 1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        return 0.5 * np.erfc(z / math.sqrt(2.0))


_CHI2_SF_MODULE = getattr(chi2_sf, "__module__", "")


try:
    from .utils import block_bootstrap_indices  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - minimal fallback for environments lacking utils
    def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=int)
        block_size = max(1, min(int(block_size), n))
        num_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max(1, n - block_size + 1), size=num_blocks)
        offsets = np.arange(block_size)
        blocks = starts[:, None] + offsets[None, :]
        valid = blocks < n
        indices = blocks[valid]
        if indices.size < n:
            padding = rng.integers(0, n, size=n - indices.size)
            indices = np.concatenate([indices, padding])
        return indices[:n].astype(int, copy=False)


@dataclass(slots=True)
class BooleanFeatureStats:
    feature: str
    counts: Dict[str, int]  # keys: a, b, c, d (2x2 contingency table)
    odds_ratio: float
    relative_risk: float
    g_statistic: float
    g_pvalue: float
    fisher_pvalue: float
    cliffs_delta: float
    bootstrap_ci: Tuple[float, float]


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test (hypergeometric tail) for a 2x2 table."""

    total = a + b + c + d
    if total == 0:
        return 1.0

    row1 = a + b
    col1 = a + c
    min_x = max(0, row1 + col1 - total)
    max_x = min(row1, col1)
    if min_x == max_x:
        return 1.0

    def _log_comb(n: int, k: int) -> float:
        return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)

    log_den = _log_comb(total, row1)
    log_base = _log_comb(col1, a) + _log_comb(total - col1, row1 - a) - log_den
    base_prob = math.exp(log_base)
    cutoff = base_prob + 1e-15
    cumulative = base_prob

    prob = base_prob
    x = a
    tail_shift = total - col1 - row1
    while x > min_x:
        ratio = (x * (tail_shift + x)) / ((col1 - x + 1) * (row1 - x + 1))
        prob *= ratio
        x -= 1
        if prob <= cutoff:
            cumulative += prob

    prob = base_prob
    x = a
    while x < max_x:
        ratio = ((col1 - x) * (row1 - x)) / ((x + 1) * (tail_shift + x + 1))
        prob *= ratio
        x += 1
        if prob <= cutoff:
            cumulative += prob

    return min(1.0, cumulative)


def _fisher_exact_vectorized(
    a: Sequence[int],
    b: Sequence[int],
    c: Sequence[int],
    d: Sequence[int],
) -> np.ndarray:
    """Exact Fisher p-values for many 2x2 tables (Python loop over features)."""

    return np.array(
        [_fisher_exact(int(ai), int(bi), int(ci), int(di)) for ai, bi, ci, di in zip(a, b, c, d)],
        dtype=float,
    )


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


def _cliffs_delta_all(binary01: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorised Cliff's delta via Mann–Whitney U for every feature."""

    n, p = binary01.shape
    if n == 0 or p == 0:
        return np.zeros(p, dtype=float)

    ranks = pd.Series(target).rank(method="average").to_numpy(dtype=np.float64, copy=False)
    n1 = binary01.sum(axis=0, dtype=np.int64)
    n0 = n - n1

    # Sum of ranks for the exposed group for each feature.
    R1 = binary01.T @ ranks
    U1 = R1 - (n1.astype(np.float64) * (n1.astype(np.float64) + 1.0)) / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (2.0 * U1) / (n1 * n0) - 1.0

    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    delta[(n1 == 0) | (n0 == 0)] = 0.0
    return delta


def _chi2_sf_df1(values: np.ndarray) -> np.ndarray:
    """Vectorised χ² survival function for df=1 using erfc to avoid math.warnings."""

    arr = np.asarray(values, dtype=np.float64)
    result = np.ones_like(arr, dtype=np.float64)
    mask = arr > 0
    if np.any(mask):
        sqrt_term = np.sqrt(arr[mask] / 2.0)
        result[mask] = np.array([math.erfc(float(value)) for value in sqrt_term], dtype=np.float64)
    return np.clip(result, 0.0, 1.0)


def _normal_diff_ci(binary01: np.ndarray, target: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Normal-approximation CI for mean(target|x=1) - mean(target|x=0) per feature."""

    n, p = binary01.shape
    if n == 0 or p == 0:
        zeros = np.zeros(p, dtype=float)
        return zeros.copy(), zeros.copy()

    ones = binary01.sum(axis=0, dtype=np.int64)
    zeros_count = n - ones

    pos_sum = binary01.T @ target
    pos_sum2 = binary01.T @ (target * target)

    total_sum = float(np.sum(target))
    total_sum2 = float(np.sum(target * target))

    neg_sum = total_sum - pos_sum
    neg_sum2 = total_sum2 - pos_sum2

    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = np.divide(pos_sum, ones, out=np.zeros_like(pos_sum), where=ones > 0)
        mu0 = np.divide(neg_sum, zeros_count, out=np.zeros_like(neg_sum), where=zeros_count > 0)
        diff = mu1 - mu0

    var1 = np.zeros_like(pos_sum, dtype=np.float64)
    mask1 = ones > 1
    var1[mask1] = (pos_sum2[mask1] - (pos_sum[mask1] ** 2) / ones[mask1]) / (ones[mask1] - 1)

    var0 = np.zeros_like(neg_sum, dtype=np.float64)
    mask0 = zeros_count > 1
    var0[mask0] = (neg_sum2[mask0] - (neg_sum[mask0] ** 2) / zeros_count[mask0]) / (zeros_count[mask0] - 1)

    se = np.zeros_like(diff, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        se[mask1] += var1[mask1] / ones[mask1]
        se[mask0] += var0[mask0] / zeros_count[mask0]
        se = np.sqrt(se)

    z = _ndtri(1.0 - alpha / 2.0)
    ci_lo = diff - z * se
    ci_hi = diff + z * se
    return ci_lo, ci_hi


def _ndtri(p: float) -> float:
    """Inverse standard normal CDF (Wichura-like rational approximation)."""

    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        return np.nan

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0
    )


def _bootstrap_mean_diff(
    binary01: np.ndarray,
    target: np.ndarray,
    samples: int,
    block_size: int,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap CI for mean(target|x=1) - mean(target|x=0)."""

    n, p = binary01.shape
    if samples <= 0 or n == 0 or p == 0:
        zeros = np.zeros(p, dtype=float)
        return zeros.copy(), zeros.copy()

    diffs = np.zeros((samples, p), dtype=np.float32)
    for draw in range(samples):
        idx = block_bootstrap_indices(n, block_size, rng)
        if idx.size == 0:
            continue
        sample_binary = binary01[idx]
        sample_target = target[idx]
        pos_counts = sample_binary.sum(axis=0, dtype=np.int64)
        pos_sum = np.einsum("ij,i->j", sample_binary, sample_target, dtype=np.float64)
        sample_total = float(np.sum(sample_target))
        sample_size = sample_binary.shape[0]
        neg_counts = sample_size - pos_counts
        neg_sum = sample_total - pos_sum

        pos_mean = np.divide(
            pos_sum,
            pos_counts,
            out=np.zeros_like(pos_sum),
            where=pos_counts > 0,
        )
        neg_mean = np.divide(
            neg_sum,
            neg_counts,
            out=np.zeros_like(neg_sum),
            where=neg_counts > 0,
        )
        diffs[draw] = (pos_mean - neg_mean).astype(np.float32)

    sorted_diffs = np.sort(diffs, axis=0)
    low_idx = max(0, int((alpha / 2.0) * samples) - 1)
    high_idx = max(0, min(samples - 1, int((1.0 - alpha / 2.0) * samples) - 1))
    ci_lower = sorted_diffs[low_idx].astype(np.float64)
    ci_upper = sorted_diffs[high_idx].astype(np.float64)
    return ci_lower, ci_upper


def compute_boolean_feature_stats(
    df: pd.DataFrame,
    target: pd.Series,
    features: Iterable[str],
    *,
    bootstrap: Mapping[str, int] | None = None,
    event_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    fisher_mode: str = "auto",
    fisher_max_terms: int = 1024,  # kept for API compatibility; currently unused
) -> Dict[str, BooleanFeatureStats]:
    """Compute contingency-table statistics for boolean features.

    Args mirror the legacy implementation. ``fisher_mode`` and ``fisher_max_terms``
    are accepted for backward compatibility; Fisher's exact test is always
    computed exactly (two-sided) unless ``fisher_mode == 'off'``.
    """

    feature_list = [name for name in features if name in df.columns]
    if not feature_list:
        return {}

    index = df.index.intersection(target.index)
    if index.empty:
        return {}

    if rng is None:
        rng = np.random.default_rng()
    params = bootstrap or {}
    samples = int(params.get("samples", 0))
    block_size = int(params.get("block_size", 32))
    alpha = float(params.get("alpha", 0.05))

    frame = df.loc[index, feature_list]
    target_aligned = target.loc[index].to_numpy(dtype=np.float64, copy=False)
    indicator = (target_aligned > event_threshold).astype(np.int8, copy=False)

    values = frame.to_numpy(dtype=np.float32, copy=False)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    values_int = values.astype(np.int16, copy=False)
    binary = np.clip(values_int, 0, 1).astype(np.uint8, copy=False)

    n_obs, _ = binary.shape
    target_pos = indicator.astype(np.int32, copy=False)
    ones = binary.sum(axis=0, dtype=np.int64)
    a = binary.T @ target_pos
    b = ones - a
    target_total = int(target_pos.sum(dtype=np.int64))
    zeros = n_obs - ones
    c = target_total - a
    d = zeros - c

    g_stats = _g_test_vectorized(a, b, c, d)

    g_pvalues: np.ndarray | None = None
    if _CHI2_SF_MODULE != "selection.math_utils":
        try:
            candidate = np.asarray(chi2_sf(g_stats, 1.0), dtype=np.float64)
            if candidate.shape == g_stats.shape:
                g_pvalues = np.clip(candidate, 0.0, 1.0)
        except Exception:  # pragma: no cover - fallback when provided function lacks vector support
            g_pvalues = None

    if g_pvalues is None:
        g_pvalues = _chi2_sf_df1(g_stats)

    odds_arr = _odds_ratio(a, b, c, d)
    rr_arr = _relative_risk(a, b, c, d)
    cliffs = _cliffs_delta_all(binary, target_aligned)

    if fisher_mode == "off":
        fisher_p = g_pvalues.copy()
    else:
        fisher_p = _fisher_exact_vectorized(a, b, c, d)

    ci_lower_norm, ci_upper_norm = _normal_diff_ci(binary, target_aligned, alpha)
    if samples > 0:
        ci_lower_bs, ci_upper_bs = _bootstrap_mean_diff(
            binary.astype(np.uint8, copy=False),
            target_aligned.astype(np.float64, copy=False),
            samples,
            block_size,
            alpha,
            rng,
        )
        ci_lower = ci_lower_bs
        ci_upper = ci_upper_bs
    else:
        ci_lower = ci_lower_norm
        ci_upper = ci_upper_norm

    results: Dict[str, BooleanFeatureStats] = {}
    for j, feature in enumerate(feature_list):
        results[feature] = BooleanFeatureStats(
            feature=feature,
            counts={
                "a": int(a[j]),
                "b": int(b[j]),
                "c": int(c[j]),
                "d": int(d[j]),
            },
            odds_ratio=float(odds_arr[j]),
            relative_risk=float(rr_arr[j]),
            g_statistic=float(g_stats[j]),
            g_pvalue=float(g_pvalues[j]),
            fisher_pvalue=float(fisher_p[j]),
            cliffs_delta=float(cliffs[j]),
            bootstrap_ci=(float(ci_lower[j]), float(ci_upper[j])),
        )
    return results
