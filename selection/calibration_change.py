# /selection/calibration.py
"""Range calibration and boolean handling (vectorized, memory-efficient, with optional chunked exact MAD).

주요 특징
- 칼럼별 루프 제거: 단 한 번(또는 fold 수만큼)의 DataFrame.quantile 호출로 분위수 일괄 계산
- MAD(중위절대편차) 두 모드 제공
  * approx  : IQR 기반 근사 (MAD ≈ 1.4826/1.349 * IQR) → 초고속/저메모리
  * exact_chunked : np.memmap + 행·열 청크 스트리밍으로 '정확한' MAD 계산 (NaN 무시)
- 대용량(수십만 행 × 수천 피처)에서 메모리 피크 최소화

주의
- float 타입이지만 값이 0/1만 있는 컬럼은 불리언으로 간주하지 않는다(고비용 검증 회피).
- exact_chunked 모드는 디스크 I/O를 사용하므로 scratch 디렉터리에 충분한 여유 필요.
"""

from __future__ import annotations

import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CalibrationResult:
    feature_num: Dict[str, tuple[float, float]]
    feature_bool: List[str]
    feature_comparison_map: Dict[str, List[str]]
    diagnostics: Dict[str, float]


def _as_list_of_ndarrays(fold_indices: Optional[Sequence[np.ndarray]]) -> Optional[List[np.ndarray]]:
    if fold_indices is None:
        return None
    out: List[np.ndarray] = []
    for idx in fold_indices:
        a = np.asarray(idx)
        if a.ndim != 1:
            a = a.ravel()
        out.append(a)
    return out


def _detect_boolean_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """불리언 컬럼(진짜 bool dtype + 0/1의 정수 컬럼)만 벡터화로 판정."""
    # 1) bool/BooleanDtype
    is_bool = df[cols].dtypes.apply(pd.api.types.is_bool_dtype)
    bool_cols_dtype = list(pd.Index(cols)[is_bool.values])

    # 2) 정수형 0/1
    int_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.integer)]
    bool_cols_int: List[str] = []
    if int_cols:
        mm = df[int_cols].agg(["min", "max"])
        mask01 = (mm.loc["min"] >= 0) & (mm.loc["max"] <= 1)
        bool_cols_int = list(mm.columns[mask01])

    # 피처 순서 보존
    bool_cols = list(pd.Index(bool_cols_dtype).union(bool_cols_int))
    return [c for c in cols if c in bool_cols]


def _iter_col_blocks(columns: List[str], block_size: int) -> List[List[str]]:
    if block_size <= 0:
        block_size = len(columns)
    return [columns[i : i + block_size] for i in range(0, len(columns), block_size)]


def _mad_exact_chunked(
    df_num: pd.DataFrame,
    medians: pd.Series,
    chunk_rows: int = 20000,
    col_block_size: int = 128,
    scratch_dir: Optional[str] = None,
    delete_scratch: bool = True,
) -> pd.Series:
    """
    '정확한' MAD = median(|x - median(x)|) 을 NaN 무시(nanmedian)로 계산.
    대규모 중간 배열 생성을 피하기 위해 (행 청크 × 열 블록)으로 np.memmap 파일을 사용.

    Parameters
    ----------
    df_num : pd.DataFrame (모든 열: 수치, 불리언 제외)
    medians : 각 열의 중앙값(전체 데이터 기준)
    chunk_rows : 행 청크 크기
    col_block_size : 열 블록 크기 (메모리/디스크 사용량에 맞게 조정)
    scratch_dir : 임시 파일 경로(없으면 시스템 기본 temp)
    delete_scratch : 계산 후 임시 파일 삭제 여부

    Returns
    -------
    pd.Series : 각 열의 정확한 MAD (NaN 무시)
    """
    n_rows, n_cols = df_num.shape
    cols = list(df_num.columns)
    result = pd.Series(index=cols, dtype=float)

    if n_rows == 0:
        result[:] = np.nan
        return result

    # 인덱스 위치 캐시
    col_to_pos = {c: i for i, c in enumerate(df_num.columns)}

    for block_cols in _iter_col_blocks(cols, col_block_size):
        b = len(block_cols)
        if b == 0:
            continue

        # memmap 파일 생성 (n_rows × b)
        scratch_dir = scratch_dir or tempfile.gettempdir()
        fname = os.path.join(scratch_dir, f"mad_dev_{uuid.uuid4().hex}.dat")
        mmap = np.memmap(fname, dtype="float64", mode="w+", shape=(n_rows, b))

        # 행 청크 단위로 절대편차 쓰기 (NaN 유지 → nanmedian에서 무시됨)
        for r0 in range(0, n_rows, max(1, chunk_rows)):
            r1 = min(n_rows, r0 + chunk_rows)
            # values view (복사 없이)
            block_vals = df_num.iloc[r0:r1, [col_to_pos[c] for c in block_cols]].to_numpy(copy=False)
            med = medians[block_cols].to_numpy()
            # broadcasting: (chunk_rows × b) - (b,)
            dev = np.abs(block_vals - med)
            # dev는 NaN 유지
            mmap[r0:r1, :] = dev

        # 열 방향 nanmedian (정확)
        # np.nanmedian는 memmap에 대해 내부적으로 필요한 페이지만 매핑해 처리(디스크 I/O는 늘 수 있음)
        mad_block = np.nanmedian(mmap, axis=0)
        result.loc[block_cols] = mad_block

        # 자원 정리
        try:
            del mmap  # file handle release
        finally:
            if delete_scratch:
                try:
                    os.remove(fname)
                except OSError:
                    pass

    # NaN → 0.0 (모두 NaN이던 열 등)
    return result.fillna(0.0)


def calibrate_ranges(
    df: pd.DataFrame,
    features: Sequence[str],
    quantiles: tuple[float, float] = (0.005, 0.995),
    mad_factor: float | None = 1.4826,
    fold_indices: Optional[Sequence[np.ndarray]] = None,
    *,
    # MAD 모드: "approx" (IQR 기반 근사) | "exact_chunked" (정확/청크)
    mad_mode: str = "approx",
    # exact_chunked 옵션
    chunk_rows: int = 20000,
    col_block_size: int = 128,
    scratch_dir: Optional[str] = None,
    delete_scratch: bool = True,
) -> CalibrationResult:
    """
    대규모 데이터셋에서도 빠르게 동작하도록 벡터화된 범위 보정 함수.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임.
    features : Sequence[str]
        대상 피처 목록(존재하지 않는 컬럼은 자동 무시).
    quantiles : (float, float)
        하한/상한 분위수. 기본 (0.5%, 99.5%).
    mad_factor : float | None
        마진 배수. None 또는 <=0 이면 마진 미적용.
        mad_mode="approx"일 때는 IQR 기반 근사, "exact_chunked"는 정확 계산.
    fold_indices : Optional[Sequence[np.ndarray]]
        각 fold의 행 인덱스 배열 리스트. 지정 시 fold별 분위수를 구해 중앙값(메디안) 앙상블 사용.
    mad_mode : {"approx", "exact_chunked"}
    chunk_rows, col_block_size, scratch_dir, delete_scratch :
        mad_mode="exact_chunked" 전용 파라미터.

    Returns
    -------
    CalibrationResult
    """
    if not features:
        return CalibrationResult({}, [], {}, {})

    # 존재 컬럼만
    exist_cols = [c for c in features if c in df.columns]
    if not exist_cols:
        return CalibrationResult({}, [], {}, {})

    # 불리언 판정 (벡터화)
    feature_bool = _detect_boolean_columns(df, exist_cols)

    # 수치(불리언 제외) 컬럼
    num_cols_all = df[exist_cols].select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols_all if c not in feature_bool]

    # NaN-only 제거
    if num_cols:
        counts = df[num_cols].count()
        num_cols = [c for c in num_cols if counts[c] > 0]

    if not num_cols:
        return CalibrationResult({}, feature_bool, {}, {})

    q_low, q_high = quantiles
    # 한 번(또는 fold 수 만큼) 호출로 필요한 분위수 모두 구함
    q_levels = np.array(sorted({q_low, 0.25, 0.5, 0.75, q_high}), dtype=float)

    fold_list = _as_list_of_ndarrays(fold_indices)

    # ---- 분위수 (fold 고려) ----
    if fold_list is None:
        Q = df[num_cols].quantile(q_levels, numeric_only=True)
        lows = Q.loc[q_low]
        highs = Q.loc[q_high]
        q25 = Q.loc[0.25]
        q75 = Q.loc[0.75]
        med_for_mad = Q.loc[0.5]
    else:
        lows_list: List[pd.Series] = []
        highs_list: List[pd.Series] = []
        q25_list: List[pd.Series] = []
        q75_list: List[pd.Series] = []
        # fold별 분위수
        for idx in fold_list:
            sub = df.iloc[idx][num_cols]
            Qf = sub.quantile(q_levels, numeric_only=True)
            lows_list.append(Qf.loc[q_low])
            highs_list.append(Qf.loc[q_high])
            q25_list.append(Qf.loc[0.25])
            q75_list.append(Qf.loc[0.75])
        # fold 중앙값 앙상블
        lows = pd.concat(lows_list, axis=1).median(axis=1)
        highs = pd.concat(highs_list, axis=1).median(axis=1)
        q25 = pd.concat(q25_list, axis=1).median(axis=1)
        q75 = pd.concat(q75_list, axis=1).median(axis=1)
        # MAD의 중심은 '전체 데이터' 기준 중앙값으로 계산(정확도를 위해 별도 1회 호출)
        med_for_mad = df[num_cols].quantile(0.5, numeric_only=True)

    # ---- MAD/마진 ----
    diagnostics: Dict[str, float] = {}
    if mad_factor is not None and mad_factor > 0:
        if mad_mode == "exact_chunked":
            approx_mad = _mad_exact_chunked(
                df_num=df[num_cols],
                medians=med_for_mad,
                chunk_rows=chunk_rows,
                col_block_size=col_block_size,
                scratch_dir=scratch_dir,
                delete_scratch=delete_scratch,
            )
            mad_series = approx_mad  # 정확 MAD
        else:
            # IQR 기반 근사: MAD ≈ (1.4826/1.349) * IQR
            iqr = (q75 - q25)
            mad_series = (1.4826 / 1.349) * iqr
            mad_series = mad_series.fillna(0.0)

        margin = mad_factor * mad_series
        diagnostics.update({f"mad_{col}": float(mad_series[col]) for col in mad_series.index})
    else:
        margin = pd.Series(0.0, index=pd.Index(num_cols))

    # ---- 최종 범위 ----
    lower = (lows - margin).astype(float)
    upper = (highs + margin).astype(float)

    feature_num: Dict[str, tuple[float, float]] = {col: (float(lower[col]), float(upper[col])) for col in lower.index}

    return CalibrationResult(
        feature_num=feature_num,
        feature_bool=feature_bool,
        feature_comparison_map={},  # 유지
        diagnostics=diagnostics,
    )
