# /selection/utils.py
"""Shared statistical utilities for the selection pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

# Lightweight cache keyed by (buffer id, shape, dtype, normalize flag)
_GRAM_CACHE: Dict[Tuple[int, Tuple[int, ...], str, bool], np.ndarray] = {}


def compute_gram_matrix(matrix: np.ndarray, *, normalize: bool = False, use_cache: bool = True) -> np.ndarray:
    """Compute the Gram matrix X @ Xᵀ with optional mean normalization and caching."""

    array = np.asarray(matrix, dtype=float)
    if normalize:
        array = array - array.mean(axis=0, keepdims=True)
    if not use_cache:
        return array @ array.T
    signature = (array.__array_interface__["data"][0], array.shape, str(array.dtype), normalize)
    cached = _GRAM_CACHE.get(signature)
    if cached is not None:
        return cached
    product = array @ array.T
    _GRAM_CACHE[signature] = product
    # Simple cache eviction to avoid unbounded growth
    if len(_GRAM_CACHE) > 64:
        # remove oldest entry
        first_key = next(iter(_GRAM_CACHE.keys()))
        _GRAM_CACHE.pop(first_key, None)
    return product


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate block bootstrap indices suitable for dependent time-series data."""

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


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size between two samples."""

    x_sorted = np.sort(np.asarray(x, dtype=float))
    y_sorted = np.sort(np.asarray(y, dtype=float))
    nx = x_sorted.size
    ny = y_sorted.size
    if nx == 0 or ny == 0:
        return 0.0
    i = j = 0
    greater = lesser = 0
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            greater += nx - i
            j += 1
        elif x_sorted[i] < y_sorted[j]:
            lesser += ny - j
            i += 1
        else:
            i += 1
            j += 1
    return float((greater - lesser) / (nx * ny))
