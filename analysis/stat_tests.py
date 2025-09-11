"""
Statistical post-analysis utilities: Probabilistic Sharpe Ratio (PSR) and
Deflated Sharpe Ratio (DSR) following Bailey & López de Prado.

These functions are intended for:
- PSR: Fitness/evaluation objective during evolution (per-strategy time-series)
- DSR: Post-analysis significance check after model selection (across many trials)

Notes
-----
- PSR uses the sample Sharpe ratio (non-annualized) and corrects for non-normality
  using skewness and kurtosis of the returns distribution.
- DSR adjusts PSR by a trials-based deflation term to account for multiple testing.
  This version uses a common closed-form approximation for the deflation term and
  allows an optional correlation parameter between trials.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Optional

import numpy as np
# Avoid optional numpy.special dependency; use math.erf and a lightweight erfinv


def _normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF using math.erf; vectorized for arrays."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    arr = np.asarray(x, dtype=float)
    # Vectorize math.erf for array inputs
    vec_erf = np.vectorize(lambda t: math.erf(float(t)))
    return 0.5 * (1.0 + vec_erf(arr / np.sqrt(2.0)))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    sr_benchmark: float = 0.0,
) -> float:
    """Compute PSR for a single strategy from per-period returns.

    Parameters
    ----------
    returns : array-like
        Per-period arithmetic returns.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Number of return periods per year (for risk-free scaling only).
    sr_benchmark : float
        Sharpe benchmark (SR*). Use 0.0 for zero skill.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return 0.0
    period_rf = risk_free_rate / float(periods_per_year)
    ex = r - period_rf
    n = ex.size
    mu = ex.mean()
    sd = ex.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    sr_hat = mu / sd

    xc = ex - mu
    m2 = (xc ** 2).sum() / (n - 1)
    m3 = (xc ** 3).sum() / (n - 1)
    m4 = (xc ** 4).sum() / (n - 1)
    eps = 1e-12
    S = m3 / (max(m2, eps) ** 1.5)
    K = m4 / (max(m2, eps) ** 2.0)

    denom = math.sqrt(max(1.0 - S * sr_hat + ((K - 1.0) / 4.0) * (sr_hat ** 2), eps))
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1.0) / denom if denom > 0 else 0.0
    return float(_normal_cdf(z))


def deflated_sharpe_ratio(
    sr_hat: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_trials: int = 1,
    rho: float = 0.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR).

    Parameters
    ----------
    sr_hat : float
        Sample Sharpe ratio (non-annualized, per-period).
    n_obs : int
        Number of return observations used to compute sr_hat.
    skew : float
        Sample skewness of returns used for sr_hat.
    kurtosis : float
        Sample kurtosis (Pearson, i.e., normal=3.0) of returns used for sr_hat.
    n_trials : int
        Number of trials/strategies considered (for selection bias deflation).
    rho : float
        Average correlation between trials in [0, 1). 0 if unknown.
    sr_benchmark : float
        Benchmark Sharpe SR* (typically 0.0).

    Returns
    -------
    float
        Probability in [0, 1] that the selected SR exceeds the deflated benchmark.

    Notes
    -----
    - z = ((sr_hat - SR*) * sqrt(n-1)) / sqrt(1 - skew*sr_hat + ((kurtosis-1)/4)*sr_hat^2)
    - z_star ≈ (1 + (skew*sr_hat)/3 - ((kurtosis-1)/36)*sr_hat^2) * Phi^{-1}(1 - 1/n_trials)
    - DSR = Phi( z - sqrt(1 - rho) * z_star )
    """
    if n_obs < 3 or n_trials < 1:
        return 0.0
    eps = 1e-12
    c3 = max(1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * (sr_hat ** 2), eps)
    z = ((sr_hat - sr_benchmark) * math.sqrt(n_obs - 1.0)) / math.sqrt(c3)

    # Deflation term (expected max inflation under multiple testing)
    # Include non-normal correction and inter-trial correlation adjustment.
    q = 1.0 - 1.0 / float(max(n_trials, 1))
    z_alpha = float(np.quantile(np.random.normal(size=10_000_0), q)) if False else math.sqrt(2) * erfinv_clamped(2*q - 1.0)
    non_normal_correction = (1.0 + (skew * sr_hat) / 3.0 - ((kurtosis - 1.0) / 36.0) * (sr_hat ** 2))
    z_star = non_normal_correction * z_alpha
    z_deflated = z - math.sqrt(max(0.0, 1.0 - rho)) * z_star
    return float(_normal_cdf(z_deflated))


def erfinv_clamped(x: float) -> float:
    """Approximate inverse error function with domain clamp to (-1,1)."""
    # Winitzki approximation (no numpy.special dependency)
    x = max(-0.999999, min(0.999999, float(x)))
    a = 0.147
    ln = math.log((1.0 - x) * (1.0 + x))
    s = (2.0 / (math.pi * a)) + (ln / 2.0)
    res = math.copysign(math.sqrt(math.sqrt(s * s - (ln / a)) - s), x)
    return res


def window_level_dsr(
    sr_selected: float,
    n_obs: int,
    sr_all_candidates: Iterable[float],
    skew: float = 0.0,
    kurtosis: float = 3.0,
    rho: float = 0.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Convenience wrapper to compute DSR for a window using all candidates' SRs.

    Parameters
    ----------
    sr_selected : float
        Selected strategy Sharpe in the window.
    n_obs : int
        Number of OOS observations in this window.
    sr_all_candidates : Iterable[float]
        Sharpe ratios of all evaluated candidates in the window.
    skew, kurtosis, rho : float
        Optional non-normality and correlation parameters.
    """
    sr_all = np.asarray(list(sr_all_candidates), dtype=float)
    n_trials = int(np.sum(np.isfinite(sr_all))) if sr_all.size else 1
    return deflated_sharpe_ratio(
        sr_hat=sr_selected,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        n_trials=max(n_trials, 1),
        rho=rho,
        sr_benchmark=sr_benchmark,
    )
