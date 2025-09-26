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

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return 0.0

    # Remove non-finite observations to avoid propagating NaNs.
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    nx = x_arr.size
    ny = y_arr.size
    if nx == 0 or ny == 0:
        return 0.0

    x_sorted = np.sort(x_arr)
    y_sorted = np.sort(y_arr)

    # ``left`` counts how many y are strictly less than each x, ``right`` counts how many
    # are less-or-equal; the difference gives the strictly greater count.
    left = np.searchsorted(y_sorted, x_sorted, side="left").astype(np.int64, copy=False)
    right = np.searchsorted(y_sorted, x_sorted, side="right").astype(np.int64, copy=False)

    greater = left.sum(dtype=np.int64)
    lesser = (ny - right).sum(dtype=np.int64)
    if greater == 0 and lesser == 0:
        return 0.0

    delta = (greater - lesser) / float(nx * ny)
    return float(max(-1.0, min(1.0, delta)))
