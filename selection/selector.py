# /selection/selector.py
"""대표 피처 선택 + 공선성 진단 (고속/저메모리 버전).

핵심 최적화
- 대규모 후보군에 대해 '부분상관 전체행렬' 같은 O(p^3) 연산을 없앰.
- 대표 피처 선택은 '상관 임계치 기반의 그리디'를 벡터화된 GEMV만으로 처리.
- 행/열 서브샘플링 + 상위점수 사전-프리필터로 후보 수를 대폭 축소.
- VIF와 (자기) partial R^2는 최종 소수 피처 집합에 대해서만
  상관행렬 역행렬의 대각(diag(inv(C)))을 한 번에 계산해 획득.
- 타깃 기반 partial R^2는 [y|X]의 정규화 상관행렬을 한 번만 역행렬화하여
  -Ω_{y,j}/sqrt(Ω_{y,y}Ω_{j,j}) 공식을 통해 전 피처 일괄 산출.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# 데이터클래스 (동일/호환)
# =========================

@dataclass(slots=True)
class SelectionDiagnostics:
    vif: Dict[str, float]
    partial_r2: Dict[str, float]
    discarded_high_vif: List[str]
    discarded_high_corr: List[str]


@dataclass(frozen=True, slots=True)
class StabilitySelectionConfig:
    """Targetless stability selection 설정."""
    subsamples: int
    sample_fraction: float
    cutoff: float
    max_features: Optional[int]
    random_state: Optional[int]


@dataclass(slots=True)
class StabilitySelectionOutcome:
    """Unsupervised stability selection 결과."""
    frequencies: Dict[str, float]
    subsamples: int
    cutoff: float
    selected_features: List[str]
    support_counts: Dict[str, int]


# =========================
# 내부 유틸 (벡터화 + 메모리 최적)
# =========================

def _get_existing(frame: pd.DataFrame, features: Sequence[str]) -> List[str]:
    return [f for f in features if f in frame.columns]


def _row_subsample_indices(n: int, fraction: float, cap_rows: Optional[int], rng: np.random.Generator) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return np.arange(n)
    k = max(1, int(round(n * fraction)))
    if cap_rows is not None:
        k = min(k, int(cap_rows))
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def _zscore_matrix(frame: pd.DataFrame, cols: List[str], row_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """Frame[cols]에서 유한값만 남기고 표준화(float32), 분산 0 컬럼 제거."""
    if not cols:
        return np.empty((0, 0), dtype=np.float32), []
    sub = frame.loc[:, cols]
    if row_idx is not None:
        sub = sub.iloc[row_idx]

    # 유한 값 행만 유지 (NA/inf drop). 대규모에서는 행비율 약간 줄어도 상관 추정은 안정적.
    values = sub.to_numpy(copy=False)
    mask = np.isfinite(values).all(axis=1)
    if not mask.any():
        return np.empty((0, 0), dtype=np.float32), []
    values = values[mask].astype(np.float32, copy=False)

    # 표준화 (ddof=1). 분산 0(상수) 컬럼 제거.
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    values -= mean
    # 분산 0 방지용 작은 epsilon
    var = values.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    ok = var > 1e-12
    if not ok.any():
        return np.empty((0, 0), dtype=np.float32), []
    values = values[:, ok]
    std = np.sqrt(var[ok])
    values /= std

    kept_cols = [c for c, flag in zip(cols, ok.tolist()) if flag]
    return values, kept_cols


def _gemv_corr_all(Z: np.ndarray, col_j: int, inv_den: float) -> np.ndarray:
    """Z는 표준화된 (n x p). col_j와 모든 열의 상관계수 벡터를 반환."""
    # corr = (Z[:, j]^T @ Z) / (n-1)
    # dot은 float32에서 BLAS 사용, inv_den = 1.0/(n-1)
    v = (Z[:, col_j].T @ Z) * inv_den  # shape (p,)
    return np.asarray(v).ravel()


def _auto_max_candidates(p: int, max_features: Optional[int]) -> int:
    # 효과점수 기반 프리필터 크기 (상수 튜닝 가능)
    if max_features is None:
        return min(p, 2000)   # 상한
    return min(p, max(8 * max_features, 1000))


def _pick_top_by_scores(candidates: List[str], scores: Mapping[str, float] | None, k: int) -> List[str]:
    s = scores or {}
    ranked = sorted(candidates, key=lambda name: (s.get(name, 0.0), name), reverse=True)
    return ranked[:k]


def _pick_top_by_variance(frame: pd.DataFrame, candidates: List[str], k: int, rng: np.random.Generator,
                          row_fraction: float = 0.1, row_cap: int = 20000) -> List[str]:
    """효과점수가 없을 때, 분산 큰 피처 우선 프리픽."""
    n = frame.shape[0]
    ridx = _row_subsample_indices(n, fraction=row_fraction, cap_rows=row_cap, rng=rng)
    sub = frame.loc[ridx, candidates]
    # 결측/inf를 0으로 채움 -> 분산 계산에 편향 가능하나 프리필터 목적에는 충분
    v = sub.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    var = v.var(axis=0).astype(np.float32)
    order = np.argsort(-var, kind="mergesort")
    sel = [candidates[i] for i in order[:k]]
    return sel


# =========================
# 대표 피처 선택 (고속)
# =========================

def greedy_representative_selection(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
    effect_scores: Mapping[str, float] | None,
    *,
    corr_cap: float = 0.9,
    max_features: Optional[int] = None,
    partial_alpha: float = 0.01,
    # 새 파라미터 (대규모 최적화)
    row_sample_fraction: float = 0.15,
    row_sample_cap: int = 50000,
    max_candidates: Optional[int] = None,
    random_state: Optional[int] = None,
) -> tuple[List[str], List[str]]:
    """상관 임계치(corr_cap) 하에서 대표 피처를 고속 그리디로 선택.

    대규모 최적화 포인트:
    - '효과점수'가 있으면 상위 K개로 후보를 먼저 축소.
      없으면 분산 상위 K개로 축소.
    - Z 표준화 행렬(서브샘플 행만)만 메모리에 유지 (float32).
    - 선택될 때마다 1회 GEMV로 전체 후보와의 상관을 일괄 갱신(루프 중첩 없음).

    반환: (selected, discarded_by_corr)
    """
    seed = random_state if random_state is not None else 0
    rng = np.random.default_rng(seed)
    _ = partial_alpha  # API 호환성을 위한 자리표시자

    candidates = _get_existing(df, candidate_features)
    if not candidates:
        return [], []

    # 프리필터 규모 결정
    p = len(candidates)
    K = max_candidates if max_candidates is not None else _auto_max_candidates(p, max_features)

    if effect_scores:
        pool = _pick_top_by_scores(candidates, effect_scores, K)
    else:
        pool = _pick_top_by_variance(df, candidates, K, rng=rng)

    if not pool:
        return [], []

    # 행 서브샘플 후 표준화
    ridx = _row_subsample_indices(df.shape[0], fraction=row_sample_fraction, cap_rows=row_sample_cap, rng=rng)
    Z, usable = _zscore_matrix(df, pool, row_idx=ridx)
    if Z.size == 0:
        return [], []

    inv_den = 1.0 / max(1, (Z.shape[0] - 1))
    p_use = len(usable)
    name_to_idx = {n: i for i, n in enumerate(usable)}

    # 선택 순서는 효과점수(있으면) 또는 이름/분산 기반
    if effect_scores:
        order = sorted(usable, key=lambda n: (effect_scores.get(n, 0.0), n), reverse=True)
    else:
        order = usable[:]  # 이미 분산 상위로 정렬됨
    order_idx = np.array([name_to_idx[n] for n in order], dtype=np.int32)

    selected: List[str] = []
    discarded: List[str] = []
    banned = np.zeros(p_use, dtype=bool)  # corr_cap을 넘으면 True
    # 자기 자신과의 상관 1.0 때문에, 선택 즉시 본인 포함 ban 처리 로직 필요.

    limit = max_features if max_features is not None else p_use
    for j_idx in order_idx:
        if len(selected) >= limit:
            break
        if banned[j_idx]:
            continue
        # 채택
        fname = usable[j_idx]
        selected.append(fname)

        # 새로 선택된 열과 전체 후보의 상관을 한 번에 계산(GEMV)
        corr_vec = np.abs(_gemv_corr_all(Z, j_idx, inv_den))
        # 임계 이상은 전부 금지
        newly_banned = corr_vec >= float(corr_cap) - 1e-12
        # 본인 포함 모두 ban
        banned |= newly_banned

    # 버려진 것은(상관으로 인해 선택 못 한) = 풀에서 선택 안 된 + banned인 것
    selected_set = set(selected)
    for name in usable:
        if name not in selected_set:
            discarded.append(name)

    return selected, discarded


# =========================
# VIF / Partial R^2 (소수 피처 대상, 정확 계산)
# =========================

def _corr_from_z(Z: np.ndarray) -> np.ndarray:
    n = Z.shape[0]
    inv_den = 1.0 / max(1, (n - 1))
    # C = Z^T Z / (n-1)
    C = (Z.T @ Z) * inv_den
    # 수치적 안정성
    np.fill_diagonal(C, 1.0)
    return C


def _diag_inv_spd(C: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """SPD 행렬 C(+ridge I)의 역행렬 대각과 Cholesky 하삼각 invL 반환.
    diag(inv(C)) = sum(invL**2, axis=1)
    """
    p = C.shape[0]
    C = C.astype(np.float64, copy=True)
    C.flat[:: p + 1] += ridge  # 정칙성 확보
    L = np.linalg.cholesky(C)  # C = L @ L.T
    I = np.eye(p, dtype=np.float64)
    invL = np.linalg.solve(L, I)  # L @ X = I -> X = L^{-1}
    # inv(C) = invL.T @ invL, 대각은 invL의 행 제곱합
    diag_inv = np.einsum("ij,ij->j", invL, invL)
    return diag_inv.astype(np.float64), invL  # double precision 보존


def _compute_vif_and_self_r2_from_features(df: pd.DataFrame, features: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """선택된 소수 피처에 대해 VIF 및 self R^2를 정확 계산."""
    if not features:
        return {}, {}
    Z, usable = _zscore_matrix(df, features)
    if Z.size == 0:
        return {}, {}
    C = _corr_from_z(Z)
    diag_inv, _ = _diag_inv_spd(C, ridge=1e-6)
    # VIF_j = diag(inv(C))_j
    # self R^2_j = 1 - 1 / VIF_j
    vif = {name: float(diag_inv[i]) for i, name in enumerate(usable)}
    self_r2 = {name: float(max(0.0, 1.0 - 1.0 / max(vif[name], 1e-12))) for name in usable}
    return vif, self_r2


def _partial_r2_with_target(df: pd.DataFrame, target_col: str, features: List[str]) -> Dict[str, float]:
    """타깃에 대한 partial R^2 (일괄) — 정규화 후 [y|X] 상관행렬의 정밀행렬(precision) 활용."""
    usable = _get_existing(df, [target_col] + features)
    if len(usable) <= 1 or target_col not in usable:
        return {}

    # 표준화 (타깃 포함)
    Z, kept = _zscore_matrix(df, usable)
    if Z.size == 0:
        return {}

    # 타깃 인덱스
    t_idx = kept.index(target_col)

    C = _corr_from_z(Z)
    diag_inv, invL = _diag_inv_spd(C, ridge=1e-6)

    # inv(C) = invL.T @ invL 를 직접 구성하지 않고도
    # off-diagonal 원소(타깃-피처)를 얻으려면 전치 곱이 필요하지만,
    # 수가 많지 않으므로 여기서는 한 번 구성.
    invC = invL.T @ invL  # (p x p)

    # partial corr: ρ_{y,j·rest} = -Ω_{y,j} / sqrt(Ω_{y,y} Ω_{j,j})
    p_corr = {}
    d = np.sqrt(diag_inv)
    denom_y = d[t_idx]
    for j, name in enumerate(kept):
        if j == t_idx:
            continue
        val = -invC[t_idx, j] / (denom_y * d[j])
        # partial R^2 = (partial corr)^2
        p_corr[name] = float(max(0.0, min(1.0, val * val)))
    return p_corr


def enforce_vif_and_partial_r2(
    df: pd.DataFrame,
    target_column: Optional[str],
    features: Sequence[str],
    vif_threshold: float,
    *,
    use_target_for_partial: bool = False,
) -> tuple[List[str], SelectionDiagnostics]:
    """VIF 한계 초과 피처 drop + partial R^2 진단.

    변경 사항:
    - 느린 '한 개씩 VIF 재계산' 루프 제거.
    - 선택된(또는 입력된) 소수 피처 집합에 대해 단 1~2회 역행렬 계산으로
      전 피처의 VIF와 partial/self R^2 동시 산출.
    """
    working = _get_existing(df, list(features))
    if not working:
        diag = SelectionDiagnostics(vif={}, partial_r2={}, discarded_high_vif=[], discarded_high_corr=[])
        return [], diag

    vif, self_r2 = _compute_vif_and_self_r2_from_features(df, working)

    # 임계 초과 전체를 한 번에 제거(보수적). 필요 시 상한을 두고 재귀 제거 가능.
    over = [f for f in working if vif.get(f, 1.0) > vif_threshold]
    kept = [f for f in working if f not in over]

    # Partial R^2 계산 (타깃 기반 or self 기반)
    if use_target_for_partial and target_column:
        partial_scores = _partial_r2_with_target(df, target_column, kept)
    else:
        # 자기 설명력(self R^2) 사용
        partial_scores = {f: self_r2.get(f, 0.0) for f in kept}

    diagnostics = SelectionDiagnostics(
        vif={f: vif[f] for f in kept},
        partial_r2=partial_scores,
        discarded_high_vif=over,
        discarded_high_corr=[f for f in working if f not in kept and f not in over],  # 관례상 빈 리스트
    )
    return kept, diagnostics


# =========================
# Unsupervised Stability Selection (가속)
# =========================

def unsupervised_stability_selection(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    effect_scores: Mapping[str, float] | None,
    subsamples: int,
    sample_fraction: float,
    cutoff: float,
    corr_cap: float,
    max_features: Optional[int],
    partial_alpha: float,  # 유지(호환). 현재 경로에서는 ridge 상수로 쓰지 않음.
    random_state: Optional[int],
) -> StabilitySelectionOutcome:
    if not features or frame.empty:
        return StabilitySelectionOutcome(
            frequencies={}, subsamples=0, cutoff=cutoff, selected_features=[], support_counts={}
        )

    subsamples = max(1, int(subsamples))
    rng = np.random.default_rng(random_state)
    counts: Dict[str, int] = {name: 0 for name in features}

    for _ in range(subsamples):
        idx = _row_subsample_indices(frame.shape[0], fraction=sample_fraction, cap_rows=None, rng=rng)
        subset = frame.iloc[idx]

        selected, _ = greedy_representative_selection(
            subset,
            features,
            effect_scores,
            corr_cap=corr_cap,
            max_features=max_features,
            # 안정적/빠른 서브샘플: 내부에서도 추가 서브샘플을 사용
            row_sample_fraction=min(0.5, max(0.1, sample_fraction)),
            row_sample_cap=30000,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        for name in selected:
            counts[name] = counts.get(name, 0) + 1

    freqs = {name: counts.get(name, 0) / subsamples for name in features}
    sorted_features = sorted(features, key=lambda n: (freqs.get(n, 0.0), (effect_scores or {}).get(n, 0.0)), reverse=True)
    selected = [n for n in sorted_features if freqs.get(n, 0.0) >= cutoff]

    limit = max_features or len(sorted_features)
    if not selected:
        selected = sorted_features[:limit]
    elif len(selected) > limit:
        selected = selected[:limit]

    return StabilitySelectionOutcome(
        frequencies=freqs, subsamples=subsamples, cutoff=cutoff, selected_features=selected, support_counts=counts
    )
