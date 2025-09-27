# math_utils.py
"""
Fast, vectorized regularized gamma & chi-square CDF for very large arrays.

- Uses SciPy if available (fastest).
- Else falls back to Numba-jitted implementations (parallel, fastmath).
- Provides memory-friendly, fully vectorized APIs with broadcasting and `out=`.

References for algorithms:
- Series expansion for P(a,x) when x < a+1
- Modified Lentz continued fraction for Q(a,x) when x >= a+1
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

# Optional accelerators
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

try:
    # SciPy path (fastest, uses Boost algorithms under the hood)
    from scipy.special import gammainc as _sp_gammainc, gammaincc as _sp_gammaincc  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_TINY = 1e-300  # To avoid division underflow in continued fraction
_DEFAULT_TOL = 1e-12
_DEFAULT_MAXITER = 512


def _broadcast(a: ArrayLike, x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a)
    x_arr = np.asarray(x)
    a_b, x_b = np.broadcast_arrays(a_arr, x_arr)
    return a_b, x_b


def _ensure_positive_a(a: np.ndarray) -> None:
    if np.any(a <= 0):
        raise ValueError("parameter 'a' must be positive")


# -----------------------------------------------------------------------------
# Public API (vectorized, broadcasting, memory-friendly with `out=`)
# -----------------------------------------------------------------------------

def regularized_lower_gamma(
    a: ArrayLike,
    x: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute P(a, x) for scalars or ndarrays with full NumPy broadcasting.
    Returns an ndarray (or writes into `out`).

    Shapes:
      - a, x can be scalars or arrays; they broadcast to a common shape.
      - If `out` is given, it must match the broadcasted shape.
    """
    a_b, x_b = _broadcast(a, x)
    _ensure_positive_a(a_b)

    if _HAS_SCIPY:
        res = _sp_gammainc(a_b, x_b)
        if out is not None:
            out[...] = res
            return out
        return res

    # Numba path
    out_arr = out if (out is not None and out.shape == a_b.shape) else np.empty_like(
        a_b, dtype=np.result_type(a_b, x_b, np.float64)
    )
    if _HAS_NUMBA:
        _rlg_batch(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    else:
        _rlg_batch_py(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    return out_arr


def regularized_upper_gamma(
    a: ArrayLike,
    x: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Q(a, x) = 1 - P(a, x) with full broadcasting.
    Uses the continued fraction directly in the large-x region for stable tails.
    """
    a_b, x_b = _broadcast(a, x)
    _ensure_positive_a(a_b)

    if _HAS_SCIPY:
        res = _sp_gammaincc(a_b, x_b)
        if out is not None:
            out[...] = res
            return out
        return res

    out_arr = out if (out is not None and out.shape == a_b.shape) else np.empty_like(
        a_b, dtype=np.result_type(a_b, x_b, np.float64)
    )
    if _HAS_NUMBA:
        _rug_batch(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    else:
        _rug_batch_py(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    return out_arr


def chi2_cdf(
    x: ArrayLike,
    k: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Chi-square CDF F(x; k) = P(k/2, x/2), vectorized + broadcasting.
    """
    a = np.asarray(k, dtype=np.result_type(k, np.float64)) / 2.0
    xx = np.asarray(x, dtype=np.result_type(x, np.float64)) / 2.0
    return regularized_lower_gamma(a, xx, tol=tol, max_iter=max_iter, out=out)


def chi2_sf(
    x: ArrayLike,
    k: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Chi-square survival function S(x; k) = 1 - F(x; k) = Q(k/2, x/2), vectorized + broadcasting.
    Uses the upper regularized gamma directly for numerical stability in extreme right tails.
    """
    a = np.asarray(k, dtype=np.result_type(k, np.float64)) / 2.0
    xx = np.asarray(x, dtype=np.result_type(x, np.float64)) / 2.0
    return regularized_upper_gamma(a, xx, tol=tol, max_iter=max_iter, out=out)


# -----------------------------------------------------------------------------
# Numba implementations (fast path)
#   - Series for x < a + 1   -> P(a,x)
#   - Modified Lentz CF for x >= a + 1 -> Q(a,x), then P = 1 - Q
# -----------------------------------------------------------------------------

if _HAS_NUMBA:  # pragma: no cover

    @njit(fastmath=True)
    def _log_prefactor(a: float, x: float) -> float:
        # log( e^{-x} x^a / Gamma(a) )
        return -x + a * math.log(x) - math.lgamma(a)

    @njit(fastmath=True)
    def _series_P(a: float, x: float, tol: float, max_iter: int) -> float:
        # P(a,x) via series sum_{n>=0} (x^n / (a)_n) / e^x * ...
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(max_iter):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * tol:
                break
        return math.exp(_log_prefactor(a, x)) * total

    @njit(fastmath=True)
    def _cf_Q(a: float, x: float, tol: float, max_iter: int) -> float:
        # Q(a,x) via modified Lentz continued fraction
        b = x + 1.0 - a
        if abs(b) < _TINY:
            b = _TINY
        c = 1.0 / _TINY
        d = 1.0 / b
        h = d
        for i in range(1, max_iter + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < _TINY:
                d = _TINY
            c = b + an / c
            if abs(c) < _TINY:
                c = _TINY
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < tol:
                break
        return math.exp(_log_prefactor(a, x)) * h

    @njit(fastmath=True)
    def _P_scalar(a: float, x: float, tol: float, max_iter: int) -> float:
        if x <= 0.0:
            return 0.0
        if x < a + 1.0:
            res = _series_P(a, x, tol, max_iter)
            if res < 0.0:
                return 0.0
            if res > 1.0:
                return 1.0
            return res
        comp = _cf_Q(a, x, tol, max_iter)
        val = 1.0 - comp
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @njit(fastmath=True)
    def _Q_scalar(a: float, x: float, tol: float, max_iter: int) -> float:
        if x <= 0.0:
            return 1.0
        if x < a + 1.0:
            p = _series_P(a, x, tol, max_iter)
            q = 1.0 - p
        else:
            q = _cf_Q(a, x, tol, max_iter)
        if q < 0.0:
            return 0.0
        if q > 1.0:
            return 1.0
        return q

    @njit(parallel=True, fastmath=True)
    def _rlg_batch(a_flat: np.ndarray, x_flat: np.ndarray, out_flat: np.ndarray, tol: float, max_iter: int) -> None:
        n = a_flat.size
        for i in prange(n):
            out_flat[i] = _P_scalar(float(a_flat[i]), float(x_flat[i]), tol, max_iter)

    @njit(parallel=True, fastmath=True)
    def _rug_batch(a_flat: np.ndarray, x_flat: np.ndarray, out_flat: np.ndarray, tol: float, max_iter: int) -> None:
        n = a_flat.size
        for i in prange(n):
            out_flat[i] = _Q_scalar(float(a_flat[i]), float(x_flat[i]), tol, max_iter)


# -----------------------------------------------------------------------------
# Pure-Python fallback (only used if SciPy & Numba both unavailable)
# -----------------------------------------------------------------------------

def _series_P_py(a: float, x: float, tol: float, max_iter: int) -> float:
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(max_iter):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * tol:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * total


def _cf_Q_py(a: float, x: float, tol: float, max_iter: int) -> float:
    b = x + 1.0 - a
    if abs(b) < _TINY:
        b = _TINY
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _P_scalar_py(a: float, x: float, tol: float, max_iter: int) -> float:
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        res = _series_P_py(a, x, tol, max_iter)
        return min(1.0, max(0.0, res))
    comp = _cf_Q_py(a, x, tol, max_iter)
    return min(1.0, max(0.0, 1.0 - comp))


def _Q_scalar_py(a: float, x: float, tol: float, max_iter: int) -> float:
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        p = _series_P_py(a, x, tol, max_iter)
        q = 1.0 - p
    else:
        q = _cf_Q_py(a, x, tol, max_iter)
    return min(1.0, max(0.0, q))


def _rlg_batch_py(a_flat: np.ndarray, x_flat: np.ndarray, out_flat: np.ndarray, tol: float, max_iter: int) -> None:
    n = a_flat.size
    for i in range(n):
        out_flat[i] = _P_scalar_py(float(a_flat[i]), float(x_flat[i]), tol, max_iter)


def _rug_batch_py(a_flat: np.ndarray, x_flat: np.ndarray, out_flat: np.ndarray, tol: float, max_iter: int) -> None:
    n = a_flat.size
    for i in range(n):
        out_flat[i] = _Q_scalar_py(float(a_flat[i]), float(x_flat[i]), tol, max_iter)
