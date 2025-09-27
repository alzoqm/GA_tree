# math_utils.py
"""
Vectorised regularised gamma and chi-square helpers with optional accelerators.

- Uses SciPy when available (fastest, high accuracy).
- Falls back to Numba-jitted code, otherwise to pure Python.
- Supports broadcasting, an ``out=`` argument, and scalar returns for compatibility.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

# Optional accelerators
try:  # pragma: no cover - import guard
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

try:  # pragma: no cover - SciPy availability differs per environment
    from scipy.special import gammainc as _sp_gammainc, gammaincc as _sp_gammaincc  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


_TINY = 1e-300
_DEFAULT_TOL = 1e-12
_DEFAULT_MAXITER = 512


def _broadcast(a: ArrayLike, x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a)
    x_arr = np.asarray(x)
    return np.broadcast_arrays(a_arr, x_arr)


def _ensure_positive_a(a: np.ndarray) -> None:
    if np.any(a <= 0):
        raise ValueError("parameter 'a' must be positive")


def _maybe_scalar(result: np.ndarray | np.floating | float, out: np.ndarray | None) -> np.ndarray | float:
    if out is None:
        if isinstance(result, np.ndarray) and result.shape == ():
            return float(result)
        if isinstance(result, np.floating):
            return float(result)
    return result


def regularized_lower_gamma(
    a: ArrayLike,
    x: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray | float:
    """Compute P(a, x) with broadcasting and optional ``out=``."""

    a_b, x_b = _broadcast(a, x)
    _ensure_positive_a(a_b)

    if _HAS_SCIPY:
        res = _sp_gammainc(a_b, x_b)
        if out is not None:
            out[...] = res
            return out
        return _maybe_scalar(res, out)

    shape = a_b.shape
    out_arr = out if (out is not None and out.shape == shape) else np.empty(
        shape, dtype=np.result_type(a_b, x_b, np.float64)
    )
    if _HAS_NUMBA:  # pragma: no branch - runtime flag
        _rlg_batch(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    else:
        _rlg_batch_py(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    if out is not None:
        return out
    if out_arr.shape == ():
        return float(out_arr)
    return out_arr


def regularized_upper_gamma(
    a: ArrayLike,
    x: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray | float:
    """Compute Q(a, x) with broadcasting and optional ``out=``."""

    a_b, x_b = _broadcast(a, x)
    _ensure_positive_a(a_b)

    if _HAS_SCIPY:
        res = _sp_gammaincc(a_b, x_b)
        if out is not None:
            out[...] = res
            return out
        return _maybe_scalar(res, out)

    shape = a_b.shape
    out_arr = out if (out is not None and out.shape == shape) else np.empty(
        shape, dtype=np.result_type(a_b, x_b, np.float64)
    )
    if _HAS_NUMBA:  # pragma: no branch
        _rug_batch(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    else:
        _rug_batch_py(a_b.ravel(), x_b.ravel(), out_arr.ravel(), tol, max_iter)
    if out is not None:
        return out
    if out_arr.shape == ():
        return float(out_arr)
    return out_arr


def chi2_cdf(
    x: ArrayLike,
    k: ArrayLike,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAXITER,
    out: np.ndarray | None = None,
) -> np.ndarray | float:
    """Chi-square CDF F(x; k) = P(k/2, x/2)."""

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
) -> np.ndarray | float:
    """Chi-square survival S(x; k) = Q(k/2, x/2)."""

    a = np.asarray(k, dtype=np.result_type(k, np.float64)) / 2.0
    xx = np.asarray(x, dtype=np.result_type(x, np.float64)) / 2.0
    return regularized_upper_gamma(a, xx, tol=tol, max_iter=max_iter, out=out)


if _HAS_NUMBA:  # pragma: no cover - exercised indirectly

    @njit(fastmath=True)
    def _log_prefactor(a: float, x: float) -> float:
        return -x + a * math.log(x) - math.lgamma(a)

    @njit(fastmath=True)
    def _series_P(a: float, x: float, tol: float, max_iter: int) -> float:
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

