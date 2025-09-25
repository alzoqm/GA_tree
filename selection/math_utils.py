# /selection/math_utils.py
"""Mathematical helpers (gamma functions, chi-square CDF) without SciPy."""

from __future__ import annotations

import math


def regularized_lower_gamma(a: float, x: float, *, tol: float = 1e-10, max_iter: int = 512) -> float:
    """Compute the regularized lower incomplete gamma P(a, x)."""

    if x <= 0:
        return 0.0
    if a <= 0:
        raise ValueError("parameter 'a' must be positive")
    if x < a + 1:
        # Series expansion
        ap = a
        delta = term = 1.0 / a
        total = term
        for _ in range(max_iter):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * tol:
                break
        result = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return min(1.0, max(0.0, result))
    # Continued fraction for the complement
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    prefactor = math.exp(-x + a * math.log(x) - math.lgamma(a))
    complement = prefactor * h
    return min(1.0, max(0.0, 1.0 - complement))


def chi2_cdf(x: float, k: float) -> float:
    if x <= 0:
        return 0.0
    return regularized_lower_gamma(k / 2.0, x / 2.0)


def chi2_sf(x: float, k: float) -> float:
    return 1.0 - chi2_cdf(x, k)
