# /selection/diversity_fast.py
"""Diversity-aware selection utilities (Stage 2) — optimized for large-scale speed & memory.

- Partial-correlation similarity via precision (inverse covariance) matrix
- Facility-Location with Lazy Greedy (accelerated submodular maximization)
- DPP greedy MAP with incremental Cholesky updates (fast log-det gains)
- Vectorized family-cap handling and candidate pre-filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import heapq
import math

import numpy as np
import pandas as pd


# ----------------------------- Result dataclass -----------------------------

@dataclass(slots=True)
class DiversitySelectionResult:
    method: str
    selected: List[str]
    coverage: Dict[str, float]
    discarded: List[str]
    similarity: str = "partial_correlation"
    diagnostics: Dict[str, float] = field(default_factory=dict)


# ----------------------------- Utilities -----------------------------------

def _family_key(feature: str) -> str:
    # Fast path; avoid splitting strings repeatedly in loops
    if "_" in feature:
        return feature.split("_", 1)[0]
    return feature


def _prepare_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, List[str]]:
    """Extract numeric matrix (n_samples x n_features) with rows of finite values only.

    - Keeps only existing columns
    - Drops all-NaN columns
    - Casts to float32 by default (halves memory)
    - Drops rows with any non-finite value across the selected columns (vectorized)
    """
    existing = [c for c in features if c in df.columns]
    if not existing:
        return np.empty((0, 0), dtype=dtype), []

    subset = df[existing]
    usable = subset.columns[subset.notna().any(axis=0)].tolist()
    if not usable:
        return np.empty((0, 0), dtype=dtype), []

    matrix = subset[usable].to_numpy(dtype=dtype, copy=False)
    # mask rows that are fully finite across usable features
    mask = np.isfinite(matrix).all(axis=1)
    if mask.any():
        matrix = matrix[mask]
    else:
        matrix = np.empty((0, len(usable)), dtype=dtype)
    return matrix, usable


def _partial_correlation_similarity(
    X: np.ndarray,
    jitter: float = 1e-6,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute |partial-correlation| similarity via precision matrix.

    Steps (all vectorized):
      1) center X
      2) covariance = (X^T X)/(n-1) + jitter*I
      3) precision = inv(cov) in float64 for stability, cast back to dtype
      4) partial corr = -precision / sqrt(diag diag^T), diag set to 1
      5) return abs(partial)

    Note: Precision (inverse covariance) encodes conditional relationships / partial correlations. 
    """
    n, d = X.shape
    if n == 0 or d == 0:
        return np.empty((0, 0), dtype=dtype)

    # center
    Xc = X - X.mean(axis=0, keepdims=True)

    denom = max(n - 1, 1)
    # covariance in target dtype, but invert in float64 for numerical stability
    cov = (Xc.T @ Xc) / np.float32(denom)
    cov = cov.astype(np.float64, copy=False)
    # regularize for PD
    cov.flat[:: d + 1] += float(jitter)

    # precision (stable)
    precision = np.linalg.inv(cov)  # float64
    diag = np.diag(precision)
    # Avoid divide-by-zero: clip tiny values
    denom_mat = np.sqrt(np.outer(diag, diag))
    np.maximum(denom_mat, 1e-12, out=denom_mat)

    partial = -precision / denom_mat
    np.fill_diagonal(partial, 1.0)
    partial = np.abs(partial).astype(dtype, copy=False)
    # Ensure finite
    return np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)


def _family_arrays(
    names: Sequence[str],
    family_cap: Optional[int],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Vectorized precomputation of family IDs and counters."""
    if family_cap is None:
        return None, None, None
    families = np.fromiter((_family_key(n) for n in names), count=len(names), dtype=object)
    uniq = {}
    ids = np.empty(len(names), dtype=np.int32)
    c = 0
    for i, f in enumerate(families):
        if f not in uniq:
            uniq[f] = c
            c += 1
        ids[i] = uniq[f]
    counts = np.zeros(len(uniq), dtype=np.int32)
    return ids, counts, uniq


def _weights_from_effects(
    feature_names: Sequence[str],
    effect_scores: Mapping[str, float],
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    w = np.array([abs(effect_scores.get(n, 0.0)) for n in feature_names], dtype=dtype)
    if not np.any(w > 0):
        w = np.ones_like(w)
    else:
        # normalize to [0,1] for numeric stability in kernels
        mx = np.max(w)
        if mx > 0:
            w = w / mx
    return w


def _candidate_pool_by_weight(
    weights: np.ndarray,
    top_k: int,
    *,
    max_candidates: Optional[int] = None,
) -> np.ndarray:
    """Return mask of candidates kept for selection, preferring high weights.

    - Default keeps up to max(8*top_k, 4000) or all if smaller.
    - If top_k is None, keep all.
    """
    n = len(weights)
    if top_k is None or n <= 4000:
        return np.ones(n, dtype=bool)
    if max_candidates is None:
        max_candidates = max(8 * int(top_k), 4000)
    max_candidates = min(n, max_candidates)
    # Argpartition is O(n)
    idx = np.argpartition(-weights, kth=max_candidates - 1)[:max_candidates]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


# ----------------------------- Facility Location (Lazy Greedy) -------------

def _facility_location_lazy_greedy(
    S: np.ndarray,  # similarity (n x n), non-negative
    W: np.ndarray,  # weights (n,)
    k: int,
    *,
    family_ids: Optional[np.ndarray] = None,
    family_counts: Optional[np.ndarray] = None,
    family_cap: Optional[int] = None,
    tol: float = 1e-10,
    dtype: np.dtype = np.float32,
) -> Tuple[List[int], np.ndarray]:
    """Accelerated greedy via lazy evaluations.

    Gain(j | coverage) = sum_i W[i] * max(0, S[i,j] - coverage[i])

    Heap stores upper bounds; true gain is recomputed only as needed.
    """
    n = S.shape[0]
    coverage = np.zeros(n, dtype=dtype)

    # Initial upper bounds: coverage=0 -> gains = S^T @ W
    gains_ub = S.T @ W  # (n,)
    # Avoid selecting negative/NaN
    gains_ub = np.nan_to_num(gains_ub, nan=-np.inf, neginf=-np.inf)

    selected_mask = np.zeros(n, dtype=bool)
    heap = [(-float(g), int(j)) for j, g in enumerate(gains_ub)]
    heapq.heapify(heap)

    selected: List[int] = []
    iters = 0
    while len(selected) < k and heap:
        iters += 1
        neg_g, j = heapq.heappop(heap)
        if selected_mask[j]:
            continue
        if family_cap is not None and family_ids is not None and family_counts is not None:
            if family_counts[family_ids[j]] >= family_cap:
                continue

        # Recompute true marginal gain for j under current coverage (single vector op)
        diff = S[:, j] - coverage
        # clamp negative to 0
        np.maximum(diff, 0.0, out=diff)
        g_true = float(W @ diff)

        # Lazy check: if still >= next best upper bound, accept; else push back
        if heap:
            next_best = -heap[0][0]
            if g_true < next_best - 1e-12:
                heapq.heappush(heap, (-g_true, j))
                continue

        # Stopping condition: if marginal gain ~0 and we already selected something
        if g_true <= tol and len(selected) > 0:
            break

        # Accept j
        selected.append(j)
        selected_mask[j] = True
        # Update coverage vectorized
        cov_col = S[:, j]
        coverage = np.maximum(coverage, cov_col, out=coverage)
        # Update family counts
        if family_cap is not None and family_ids is not None and family_counts is not None:
            family_counts[family_ids[j]] += 1

    return selected, coverage


# ----------------------------- DPP Greedy with Cholesky --------------------

def _dpp_greedy_cholesky(
    K: np.ndarray,              # PSD kernel (n x n), float32
    k: int,
    *,
    family_ids: Optional[np.ndarray] = None,
    family_counts: Optional[np.ndarray] = None,
    family_cap: Optional[int] = None,
    jitter: float = 1e-6,
    dtype: np.dtype = np.float32,
) -> Tuple[List[int], np.ndarray]:
    """Fast greedy MAP for k-DPP with incremental Cholesky updates.

    Based on Chen et al. (NeurIPS 2018): maintain V (n x r) s.t.
      q = diag(K) - sum(V^2, axis=1)
      select argmax q (subject to constraints), update:
        new_col = (K[:, j] - V @ V[j, :]) / sqrt(q[j])
        V = [V, new_col]
        q -= new_col**2
    """
    n = K.shape[0]
    diag = np.diag(K).astype(dtype, copy=False)
    q = diag.copy()  # residual variances
    selected_mask = np.zeros(n, dtype=bool)

    # V will store projection columns incrementally (n x r)
    V = np.empty((n, 0), dtype=dtype)

    selected: List[int] = []
    for _ in range(k):
        # Mask out ineligible
        available = ~selected_mask
        if family_cap is not None and family_ids is not None and family_counts is not None:
            fam_ok = family_counts[family_ids] < int(family_cap)
            available &= fam_ok

        if not np.any(available):
            break

        # choose best available by residual
        j = int(np.argmax(np.where(available, q, -np.inf)))
        gain = float(q[j])
        if not np.isfinite(gain) or gain <= 1e-12:
            break

        # select
        selected.append(j)
        selected_mask[j] = True
        if family_cap is not None and family_ids is not None and family_counts is not None:
            family_counts[family_ids[j]] += 1

        # compute new column (vectorized):
        # proj = V @ V[j, :].T  (shape: n,)
        if V.shape[1] == 0:
            proj = np.zeros(n, dtype=dtype)
        else:
            proj = V @ V[j, :]

        denom = math.sqrt(max(gain, 1e-12))
        new_col = (K[:, j].astype(dtype, copy=False) - proj) / denom
        # Append column
        V = np.concatenate([V, new_col[:, None]], axis=1)
        # Update residuals
        q -= new_col * new_col

        # Ensure numerical safety
        np.maximum(q, 0.0, out=q)

    # Return residual coverage proxy as max similarity later
    return selected, q


# ----------------------------- Public APIs ---------------------------------

def facility_location_diversity(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
    effect_scores: Mapping[str, float],
    *,
    top_k: Optional[int],
    family_cap: Optional[int],
    similarity: str = "partial_correlation",
    # advanced knobs:
    jitter: float = 1e-6,
    dtype: np.dtype = np.float32,
    max_candidates: Optional[int] = None,
) -> DiversitySelectionResult:
    """Select a diverse subset of features via facility-location objective (lazy greedy).

    - similarity: "partial_correlation" (default) or "corr"
    """
    X, features = _prepare_matrix(df, candidate_features, dtype=dtype)
    if len(features) == 0 or X.shape[0] == 0:
        return DiversitySelectionResult(
            method="facility_location",
            selected=[],
            coverage={},
            discarded=list(candidate_features),
            similarity=similarity,
        )

    # Similarity matrix (n_features x n_features) in float32
    if similarity in ("partial_correlation", "partial", "pcorr"):
        S = _partial_correlation_similarity(X, jitter=jitter, dtype=dtype)
    elif similarity in ("corr", "pearson"):
        # fast Pearson |corr|
        Xc = X - X.mean(axis=0, keepdims=True)
        std = np.sqrt(np.maximum(np.sum(Xc * Xc, axis=0) / float(max(X.shape[0]-1, 1)), 1e-12))
        Xn = Xc / std
        S = (Xn.T @ Xn) / np.float32(max(X.shape[0]-1, 1))
        S = np.abs(S.astype(dtype, copy=False))
        np.fill_diagonal(S, 1.0)
    else:
        raise ValueError(f"Unknown similarity mode: {similarity}")

    n = len(features)
    W = _weights_from_effects(features, effect_scores, dtype=dtype)

    # Candidate pre-filter to reduce n (O(n)) — keeps up to ~8*k or 4000 (whichever larger)
    if top_k is None:
        limit = n
        keep_mask = np.ones(n, dtype=bool)
    else:
        limit = max(1, min(int(top_k), n))
        keep_mask = _candidate_pool_by_weight(W, limit, max_candidates=max_candidates)

    # Restrict to pool
    if not np.all(keep_mask):
        idx = np.flatnonzero(keep_mask)
        S = S[np.ix_(idx, idx)]
        W = W[idx]
        features_pool = [features[i] for i in idx]
    else:
        features_pool = features

    # Family caps (vectorized)
    fam_ids, fam_counts, fam_map = _family_arrays(features_pool, family_cap)

    # Lazy Greedy selection (no O(n) loops inside; per-iteration vector ops only)
    selected_idx_pool, cov = _facility_location_lazy_greedy(
        S, W, k=limit,
        family_ids=fam_ids, family_counts=fam_counts, family_cap=family_cap, dtype=dtype
    )

    # Map back to original feature names
    selected = [features_pool[i] for i in selected_idx_pool]
    coverage_map = {features_pool[i]: float(cov[i]) for i in range(len(features_pool))}
    discarded = [name for name in features if name not in selected]

    diagnostics = {
        "similarity": similarity,
        "family_cap": float(family_cap) if family_cap is not None else 0.0,
        "top_k": float(limit),
        "n_features_input": float(n),
        "n_features_pool": float(len(features_pool)),
        "dtype": str(dtype),
        "jitter": float(jitter),
        "lazy_greedy": 1.0,
    }

    return DiversitySelectionResult(
        method="facility_location_lazy",
        selected=selected,
        coverage=coverage_map,
        discarded=discarded,
        similarity=similarity,
        diagnostics=diagnostics,
    )


def determinantal_point_process_diversity(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
    effect_scores: Mapping[str, float],
    *,
    top_k: Optional[int],
    family_cap: Optional[int],
    lambda_scale: float = 1.0,
    jitter: float = 1e-6,
    dtype: np.dtype = np.float32,
    max_candidates: Optional[int] = None,
    similarity: str = "partial_correlation",
) -> DiversitySelectionResult:
    """Select a diverse subset using a k-DPP greedy MAP with fast Cholesky updates.

    DPP kernel:
        K = exp(-lambda_scale * (1 - S)) * (w w^T) + jitter*I
    """
    X, features = _prepare_matrix(df, candidate_features, dtype=dtype)
    if len(features) == 0 or X.shape[0] == 0:
        return DiversitySelectionResult(
            method="dpp",
            selected=[],
            coverage={},
            discarded=list(candidate_features),
            similarity=similarity,
        )

    # Similarity S \in [0,1]
    if similarity in ("partial_correlation", "partial", "pcorr"):
        S = _partial_correlation_similarity(X, jitter=jitter, dtype=dtype)
    elif similarity in ("corr", "pearson"):
        Xc = X - X.mean(axis=0, keepdims=True)
        std = np.sqrt(np.maximum(np.sum(Xc * Xc, axis=0) / float(max(X.shape[0]-1, 1)), 1e-12))
        Xn = Xc / std
        S = (Xn.T @ Xn) / np.float32(max(X.shape[0]-1, 1))
        S = np.abs(S.astype(dtype, copy=False))
        np.fill_diagonal(S, 1.0)
    else:
        raise ValueError(f"Unknown similarity mode: {similarity}")

    n = len(features)
    W = _weights_from_effects(features, effect_scores, dtype=dtype)

    # Candidate pre-filter
    if top_k is None:
        limit = n
        keep_mask = np.ones(n, dtype=bool)
    else:
        limit = max(1, min(int(top_k), n))
        keep_mask = _candidate_pool_by_weight(W, limit, max_candidates=max_candidates)

    if not np.all(keep_mask):
        idx = np.flatnonzero(keep_mask)
        S = S[np.ix_(idx, idx)]
        W = W[idx]
        features_pool = [features[i] for i in idx]
    else:
        features_pool = features

    # Build DPP kernel (vectorized)
    # exp(-lambda*(1-S)) is monotone in S and in [exp(-lambda), 1]
    K = np.exp(-np.float32(lambda_scale) * (1.0 - S), dtype=dtype)
    ww = np.outer(W, W).astype(dtype, copy=False)
    K *= ww
    # jitter on diagonal
    K.flat[:: K.shape[0] + 1] += np.float32(jitter)

    # Family caps (vectorized)
    fam_ids, fam_counts, _ = _family_arrays(features_pool, family_cap)

    # Greedy MAP with fast Cholesky-like updates (vectorized per-iter)
    selected_idx_pool, _residual = _dpp_greedy_cholesky(
        K, k=limit,
        family_ids=fam_ids, family_counts=fam_counts, family_cap=family_cap,
        jitter=jitter, dtype=dtype
    )

    if not selected_idx_pool:
        # Fallback to lazy facility-location (still vectorized)
        fallback = facility_location_diversity(
            df, candidate_features, effect_scores,
            top_k=top_k, family_cap=family_cap, dtype=dtype, similarity=similarity
        )
        return DiversitySelectionResult(
            method="dpp_fallback_facility_location",
            selected=fallback.selected,
            coverage=fallback.coverage,
            discarded=fallback.discarded,
            similarity=fallback.similarity,
            diagnostics=fallback.diagnostics,
        )

    # Compute coverage vector once (max similarity to any selected)
    # coverage_i = max_j S[i, sel_j]
    sel = np.array(selected_idx_pool, dtype=int)
    coverage_vec = np.max(S[:, sel], axis=1) if sel.size > 0 else np.zeros(S.shape[0], dtype=dtype)

    selected = [features_pool[i] for i in selected_idx_pool]
    coverage_map = {features_pool[i]: float(coverage_vec[i]) for i in range(len(features_pool))}
    discarded = [name for name in features if name not in selected]

    diagnostics = {
        "similarity": similarity,
        "lambda_scale": float(lambda_scale),
        "jitter": float(jitter),
        "family_cap": float(family_cap) if family_cap is not None else 0.0,
        "top_k": float(limit),
        "n_features_input": float(n),
        "n_features_pool": float(len(features_pool)),
        "dtype": str(dtype),
        "greedy_cholesky": 1.0,
    }

    return DiversitySelectionResult(
        method="dpp_greedy_fast",
        selected=selected,
        coverage=coverage_map,
        discarded=discarded,
        similarity=similarity,
        diagnostics=diagnostics,
    )
