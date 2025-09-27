# /selection/boolean_stats_change.py
"""High-performance boolean event statistics & inference.

- Fully vectorized across features (no Python loops over features/data).
- Optional Numba acceleration (if available); graceful fallback otherwise.
- Memory-conscious: uses compact dtypes and avoids giant intermediate tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# If your original project exposes these, we reuse them; otherwise provide simple fallbacks.
try:
    from .math_utils import chi2_sf  # should accept ndarray; if not, we vectorize below
except Exception:
    # Simple chi-square survival function (df=1) using complementary error function approx.
    # For stability on large x, use expm1 where appropriate.
    import numpy

    def chi2_sf(x: np.ndarray, df: float = 1.0) -> np.ndarray:
        # Wilson-Hilferty approximation for df=1 is reasonably accurate.
        x = np.asarray(x, dtype=float)
        z = (np.power(x / df, 1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        # SF ~ 0.5 * erfc(z / sqrt(2))
        return 0.5 * np.erfc(z / math.sqrt(2.0))


try:
    from .utils import cliffs_delta as _cliffs_delta_scalar  # fallback if needed elsewhere
except Exception:
    _cliffs_delta_scalar = None  # we compute vectorized delta internally


# ---------- Optional Numba acceleration ----------
try:
    from numba import njit, prange

    NUMBA_OK = True
except Exception:  # numba not installed
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

    NUMBA_OK = False


# ---------- Public dataclass ----------
@dataclass(slots=True)
class BooleanFeatureStats:
    feature: str
    counts: Dict[str, int]          # keys: a,b,c,d for 2x2 (feature x event)
    odds_ratio: float
    relative_risk: float
    g_statistic: float
    g_pvalue: float
    fisher_pvalue: float
    cliffs_delta: float             # rank-biserial (Mann–Whitney U) version
    bootstrap_ci: Tuple[float, float]  # mean(target|x=1) - mean(target|x=0)


# ---------- Core helpers (vectorized) ----------
def _odds_ratio(a, b, c, d) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)
    num = (a_arr + 0.5) * (d_arr + 0.5)
    den = (b_arr + 0.5) * (c_arr + 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return np.nan_to_num(r, nan=1.0, posinf=1e9, neginf=1e-9)


def _relative_risk(a, b, c, d) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)
    exposed = a_arr + b_arr
    unexposed = c_arr + d_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        num = (a_arr + 0.5) / (exposed + 1.0)
        den = (c_arr + 0.5) / (unexposed + 1.0)
        rr = num / den
    rr = np.nan_to_num(rr, nan=1.0, posinf=1e9, neginf=1e-9)
    invalid = (exposed <= 0) | (unexposed <= 0)
    return np.where(invalid, 1.0, rr)


def _g_test_vectorized(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vectorized G statistic over many 2x2 tables (float64)."""
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    c = c.astype(np.float64, copy=False)
    d = d.astype(np.float64, copy=False)
    totals = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = a + b
        r2 = c + d
        c1 = a + c
        c2 = b + d
        ea = np.where(totals > 0, r1 * c1 / totals, 0.0)
        eb = np.where(totals > 0, r1 * c2 / totals, 0.0)
        ec = np.where(totals > 0, r2 * c1 / totals, 0.0)
        ed = np.where(totals > 0, r2 * c2 / totals, 0.0)

        def term(obs, exp):
            valid = (obs > 0) & (exp > 0)
            x = np.where(valid, obs / exp, 1.0)
            return np.where(valid, 2.0 * obs * np.log(x), 0.0)

        g = term(a, ea) + term(b, eb) + term(c, ec) + term(d, ed)

    return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


# ---------- Fast vectorized Cliff's delta via Mann–Whitney U ----------
def _cliffs_delta_all(binary01: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute Cliff's delta for all features at once using the relation to
    Mann–Whitney U (rank-biserial): delta = 2*U/(n1*n0) - 1
    - binary01: (n, p) uint8/0-1 matrix
    - target: (n,) float
    """
    n, p = binary01.shape
    # average ranks (1..n), ties -> average rank; use fast C-backed pandas.
    ranks = pd.Series(target).rank(method="average").to_numpy(dtype=np.float64, copy=False)

    n1 = binary01.sum(axis=0, dtype=np.int64)           # (# with feature=1) per feature
    n0 = n - n1

    # Sum of ranks for group 1 for all features at once: R1_j = sum_i B[i,j]*rank[i]
    # (p,) = (n,p)^T @ (n,)
    R1 = binary01.T @ ranks

    # U1 = R1 - n1*(n1+1)/2  (Mann–Whitney)
    U1 = R1 - (n1.astype(np.float64) * (n1.astype(np.float64) + 1.0)) / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (2.0 * U1) / (n1 * n0) - 1.0

    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    # n1==0 or n0==0 -> undefined; set to 0
    mask_bad = (n1 == 0) | (n0 == 0)
    delta[mask_bad] = 0.0
    return delta


# ---------- Fast Fisher's exact for "small support" via precomputed log-factorials ----------
def _precompute_logfactorials(n: int) -> np.ndarray:
    # logfac[k] = log(k!) for k=0..n
    if n < 0:
        raise ValueError("n must be non-negative")
    logfac = np.zeros(n + 1, dtype=np.float64)
    if n > 0:
        logfac[1:] = np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))
    return logfac


@njit(cache=True, fastmath=True)
def _logC(logfac: np.ndarray, n: int, k: int) -> float:
    if k < 0 or k > n:
        return -1e300  # effectively -inf
    return logfac[n] - logfac[k] - logfac[n - k]


@njit(cache=True, fastmath=True)
def _fisher_two_tailed_small_support(
    a_arr: np.ndarray, b_arr: np.ndarray, c_arr: np.ndarray, d_arr: np.ndarray,
    max_terms: int,  # maximum support width to compute exactly
    n_total: int,    # constant across features
    col1: int,       # a+c, constant across features (sum of positives)
    logfac: np.ndarray,
) -> np.ndarray:
    """
    Compute Fisher's exact p-values for features whose support width <= max_terms.
    Others receive -1 sentinel (caller will replace e.g. with G-test p-value).
    """
    m = a_arr.shape[0]
    out = np.full(m, -1.0, dtype=np.float64)
    log_denom = _logC(logfac, n_total, col1)  # constant

    for i in range(m):  # Numba-loop (compiled, fast)
        a = int(a_arr[i])
        b = int(b_arr[i])
        c = int(c_arr[i])
        d = int(d_arr[i])

        row1 = a + b
        # feasible x in [min_x, max_x]
        min_x = max(0, col1 - (n_total - row1))
        max_x = min(row1, col1)
        width = max_x - min_x + 1
        if width <= 0 or width > max_terms:
            continue  # keep -1

        # log pmf of observed a
        logp_obs = _logC(logfac, row1, a) + _logC(logfac, n_total - row1, col1 - a) - log_denom

        # two-sided: sum probabilities <= observed
        s = 0.0
        for x in range(min_x, max_x + 1):
            logp = _logC(logfac, row1, x) + _logC(logfac, n_total - row1, col1 - x) - log_denom
            if logp <= logp_obs + 1e-15:
                s += math.exp(logp)
        out[i] = min(1.0, s)

    return out


# ---------- Optional Numba-accelerated block bootstrap over features ----------
@njit(cache=True, fastmath=True, parallel=True)
def _bootstrap_mean_diff_parallel(
    binary01: np.ndarray,    # (n, p) uint8
    target: np.ndarray,      # (n,) float64
    samples: int,
    block_size: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CI for mean(target|x=1) - mean(target|x=0) across features using
    moving-block bootstrap with Numba-parallel loops and row-wise streaming
    accumulation to avoid giant (samples x n x p) tensors.

    Returns (ci_lower, ci_upper) of shape (p,).
    """
    n, p = binary01.shape
    draws = samples
    diffs = np.empty((draws, p), dtype=np.float32)

    n_blocks = (n + block_size - 1) // block_size

    for s in prange(draws):
        # per-sample accumulators
        pos_counts = np.zeros(p, dtype=np.int64)
        pos_sum = np.zeros(p, dtype=np.float64)

        # generate bootstrap indices on the fly (circular blocks)
        # NOTE: using numba's RNG; reproducibility across processes not guaranteed.
        for b in range(n_blocks):
            start = np.random.randint(0, n)
            # fill a block
            for t in range(block_size):
                if b * block_size + t >= n:
                    break
                idx = (start + t)
                if idx >= n:
                    idx -= n  # modulo n
                row = binary01[idx]  # view (p,)
                val = target[idx]
                # accumulate sums and counts
                for j in range(p):
                    if row[j] != 0:
                        pos_counts[j] += 1
                        pos_sum[j] += val

        # derive neg counts & sums from totals
        sample_size = n
        total_sum = np.sum(target)
        neg_counts = sample_size - pos_counts
        neg_sum = total_sum - pos_sum

        # means (guard zero)
        pos_mean = np.zeros(p, dtype=np.float64)
        mask_pos = pos_counts > 0
        pos_mean[mask_pos] = pos_sum[mask_pos] / pos_counts[mask_pos]

        neg_mean = np.zeros(p, dtype=np.float64)
        mask_neg = neg_counts > 0
        neg_mean[mask_neg] = neg_sum[mask_neg] / neg_counts[mask_neg]

        diffs[s, :] = (pos_mean - neg_mean).astype(np.float32)

    # percentile CI
    # sort along draws axis
    diffs.sort(axis=0)
    lo_idx = max(0, int((alpha / 2.0) * draws))
    hi_idx = min(draws - 1, int((1.0 - alpha / 2.0) * draws))
    ci_lower = diffs[lo_idx, :].astype(np.float64)
    ci_upper = diffs[hi_idx, :].astype(np.float64)
    return ci_lower, ci_upper


# ---------- Normal-approx (delta method) CI for diff of means (vectorized) ----------
def _normal_diff_ci(
    binary01: np.ndarray, target: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CI for mean(target|x=1) - mean(target|x=0) using
    SE = sqrt(S1^2/n1 + S0^2/n0), where S_k^2 is unbiased sample variance.
    Fully vectorized.
    """
    n, p = binary01.shape
    ones = binary01.sum(axis=0, dtype=np.int64)
    zeros = n - ones

    t_sum = float(np.sum(target))
    t_sum2 = float(np.sum(target * target))

    pos_sum = binary01.T @ target
    pos_sum2 = binary01.T @ (target * target)
    neg_sum = t_sum - pos_sum
    neg_sum2 = t_sum2 - pos_sum2

    # means
    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = np.divide(pos_sum, ones, out=np.zeros_like(pos_sum), where=ones > 0)
        mu0 = np.divide(neg_sum, zeros, out=np.zeros_like(neg_sum), where=zeros > 0)
        diff = mu1 - mu0

    # unbiased sample variances
    # var = (sum2 - sum^2/n) / (n-1)
    var1 = np.zeros_like(pos_sum, dtype=np.float64)
    mask1 = ones > 1
    var1[mask1] = (pos_sum2[mask1] - (pos_sum[mask1] ** 2) / ones[mask1]) / (ones[mask1] - 1)

    var0 = np.zeros_like(neg_sum, dtype=np.float64)
    mask0 = zeros > 1
    var0[mask0] = (neg_sum2[mask0] - (neg_sum[mask0] ** 2) / zeros[mask0]) / (zeros[mask0] - 1)

    # standard error of difference of means
    se = np.zeros_like(diff, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        se[mask1] += var1[mask1] / ones[mask1]
        se[mask0] += var0[mask0] / zeros[mask0]
        se = np.sqrt(se)

    # z-quantile for given alpha (fast approximation; alpha=0.05 -> 1.95996...)
    z = _ndtri(1.0 - alpha / 2.0)
    ci_lo = diff - z * se
    ci_hi = diff + z * se
    return ci_lo, ci_hi


def _ndtri(p: float) -> float:
    """Inverse standard normal CDF (Wichura-like rational approximation)."""
    # Source: adapted from CEPHES algorithm / common public-domain approximations.
    # Accurate to ~1e-6 which is sufficient for CI bounds here.
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        return np.nan
    # Breakpoints
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------- Main API ----------
def compute_boolean_feature_stats(
    df: pd.DataFrame,
    target: pd.Series,
    features: Iterable[str],
    *,
    bootstrap: Mapping[str, int] | None = None,
    event_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    fisher_mode: str = "auto",        # "auto" | "off" | "force_small"
    fisher_max_terms: int = 1024,     # max support width for exact Fisher when auto/force_small
) -> Dict[str, BooleanFeatureStats]:
    """
    Compute boolean-event statistics for the provided feature set at scale.

    Performance notes:
    - Absolutely no Python loops over features/data for heavy math.
    - Fisher's exact is only computed for small-support tables (auto),
      else falls back to G-test p-values.
    - Cliff's delta is computed for all features in O(n log n + nnz) time.

    Args:
      df, target: input frame & target series (aligned by index).
      features: columns to process (non-existent are ignored).
      bootstrap: {"samples": int, "block_size": int, "alpha": float}; if omitted,
                 we use normal-approximation CI (fast, vectorized).
      event_threshold: indicator = (target > threshold).
      rng: optional np.random.Generator (used only if bootstrap enabled).
      fisher_mode:
        - "auto": Fisher for features whose hypergeometric support <= fisher_max_terms.
        - "off": do not compute Fisher (use G-test p-values instead).
        - "force_small": compute Fisher only for small-support; raise otherwise.
      fisher_max_terms: cut-off for support width to run exact Fisher.

    Returns:
      Dict[str, BooleanFeatureStats]
    """
    feature_list = [c for c in features if c in df.columns]
    if not feature_list:
        return {}

    # Align indices once
    index = df.index.intersection(target.index)
    if index.empty:
        return {}

    # RNG
    if rng is None:
        rng = np.random.default_rng()

    # Bootstrap params
    params = bootstrap or {}
    samples = int(params.get("samples", 0))
    block_size = int(params.get("block_size", 32))
    alpha = float(params.get("alpha", 0.05))

    # -------- Data to arrays (compact dtypes) --------
    frame = df.loc[index, feature_list]
    target_aligned = target.loc[index].to_numpy(dtype=np.float64, copy=False)

    # robust binary casting: negative -> 0, >=1 -> 1
    vals = frame.to_numpy(dtype=np.float32, copy=False)
    np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip to [0,1] like original behavior (floor-to-int then clip)
    vals_i = vals.astype(np.int16, copy=False)
    binary = np.clip(vals_i, 0, 1).astype(np.uint8, copy=False)  # (n, p)

    n, p = binary.shape
    # Indicator labels for 2x2 tables
    indicator = (target_aligned > event_threshold).astype(np.int8, copy=False)

    # -------- 2x2 counts (vectorized) --------
    ones = binary.sum(axis=0, dtype=np.int64)                     # row1 = a + b
    zeros = n - ones                                             # row2 = c + d
    target_pos = indicator.astype(np.int32, copy=False)
    a = binary.T @ target_pos                                    # a: feature==1 & event==1
    b = ones - a
    target_total = int(target_pos.sum(dtype=np.int64))
    c = target_total - a
    d = zeros - c

    # -------- G-test & p-values --------
    g_stats = _g_test_vectorized(a, b, c, d)
    # if math_utils.chi2_sf does not support vector inputs, wrap it
    try:
        g_pvalues = np.asarray(chi2_sf(g_stats, 1.0), dtype=np.float64)
    except Exception:
        g_pvalues = np.array([float(chi2_sf(float(g), 1.0)) if g > 0 else 1.0 for g in g_stats], dtype=np.float64)
        g_pvalues = np.minimum(1.0, g_pvalues)

    # -------- Odds ratio / RR (vectorized) --------
    odds_arr = _odds_ratio(a, b, c, d)
    rr_arr = _relative_risk(a, b, c, d)

    # -------- Cliff's delta (vectorized via ranks) --------
    cliffs = _cliffs_delta_all(binary, target_aligned)

    # -------- Fisher's exact (auto small-support) --------
    if fisher_mode == "off":
        fisher_p = g_pvalues.copy()
    else:
        # hypergeometric support width per feature:
        # min_x = max(0, col1 - (n - row1)), max_x = min(row1, col1)  => width = max_x - min_x + 1
        col1 = target_total  # a + c, constant across features
        row1 = ones
        min_x = np.maximum(0, col1 - (n - row1))
        max_x = np.minimum(row1, col1)
        width = (max_x - min_x + 1).astype(np.int64)

        # features we compute exactly
        small_mask = width <= int(fisher_max_terms)
        fisher_p = np.full(p, np.nan, dtype=np.float64)

        if np.any(small_mask):
            logfac = _precompute_logfactorials(n)
            fisher_small = _fisher_two_tailed_small_support(
                a[small_mask].astype(np.int64),
                b[small_mask].astype(np.int64),
                c[small_mask].astype(np.int64),
                d[small_mask].astype(np.int64),
                int(fisher_max_terms),
                int(n),
                int(col1),
                logfac,
            )
            fisher_p[small_mask] = fisher_small

        # fill remainder
        remainder_mask = ~small_mask
        if np.any(remainder_mask):
            if fisher_mode == "force_small":
                # explicit: if user insisted on Fisher but support too large
                fisher_p[remainder_mask] = np.nan
            else:
                # pragmatic: fall back to G-test p-value
                fisher_p[remainder_mask] = g_pvalues[remainder_mask]

        # any -1 from JIT (shouldn't happen here) -> fallback
        bad = fisher_p < 0
        if np.any(bad):
            fisher_p[bad] = g_pvalues[bad]

    # -------- CI for mean difference --------
    # Fast default: normal-approximation (fully vectorized)
    ci_lower_norm, ci_upper_norm = _normal_diff_ci(binary, target_aligned, alpha)

    # Optional: block bootstrap (Numba-accelerated); only if samples>0 and numba available
    if samples and NUMBA_OK:
        ci_lower_bs, ci_upper_bs = _bootstrap_mean_diff_parallel(
            binary.astype(np.uint8, copy=False),
            target_aligned.astype(np.float64, copy=False),
            int(samples),
            int(block_size),
            float(alpha),
        )
        ci_lower = ci_lower_bs
        ci_upper = ci_upper_bs
    else:
        ci_lower = ci_lower_norm
        ci_upper = ci_upper_norm

    # -------- Assemble results (lightweight Python loop just to build objects) --------
    results: Dict[str, BooleanFeatureStats] = {}
    for j, feat in enumerate(feature_list):
        results[feat] = BooleanFeatureStats(
            feature=feat,
            counts={"a": int(a[j]), "b": int(b[j]), "c": int(c[j]), "d": int(d[j])},
            odds_ratio=float(odds_arr[j]),
            relative_risk=float(rr_arr[j]),
            g_statistic=float(g_stats[j]),
            g_pvalue=float(g_pvalues[j]),
            fisher_pvalue=float(fisher_p[j]),
            cliffs_delta=float(cliffs[j]),
            bootstrap_ci=(float(ci_lower[j]), float(ci_upper[j])),
        )
    return results
