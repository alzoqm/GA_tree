# /selection/integrity.py
"""Integrity gates for feature screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class IntegrityReport:
    """Records what was removed during integrity gating."""

    dropped_missing: List[str]
    dropped_constant: List[str]
    dropped_sparse: List[str]
    dropped_duplicates: List[str]
    dropped_correlated: List[str]

    @property
    def total_dropped(self) -> int:
        return sum(
            len(section)
            for section in (
                self.dropped_missing,
                self.dropped_constant,
                self.dropped_sparse,
                self.dropped_duplicates,
                self.dropped_correlated,
            )
        )


# def apply_integrity_gates(
#     df: pd.DataFrame,
#     feature_columns: Sequence[str],
#     max_missing_ratio: float = 0.2,
#     min_unique_fraction: float = 0.01,
#     max_zero_fraction: float = 0.995,
#     correlation_threshold: float = 0.999,
# ) -> tuple[pd.DataFrame, IntegrityReport]:
#     """Remove pathological columns prior to screening.

#     Args:
#         df: Source dataframe.
#         feature_columns: Target feature names to evaluate.
#         max_missing_ratio: Maximum allowed fraction of missing values.
#         min_unique_fraction: Minimum unique ratio relative to rows to consider non-constant.
#         max_zero_fraction: For sparse (binary) columns, drop if the majority are zero.
#         correlation_threshold: Threshold for dropping perfectly correlated duplicates (>).
#     """

#     feature_list = list(feature_columns)
#     dropped_missing: List[str] = []
#     dropped_constant: List[str] = []
#     dropped_sparse: List[str] = []
#     dropped_duplicates: List[str] = []
#     dropped_correlated: List[str] = []

#     work_df = df.copy()

#     # Missing values filter
#     missing_ratio = work_df[feature_list].isna().mean()
#     for feature, ratio in missing_ratio.items():
#         if ratio > max_missing_ratio:
#             dropped_missing.append(feature)

#     feature_list = [f for f in feature_list if f not in dropped_missing]

#     # Constant / low variability filter
#     n_rows = len(work_df)
#     for feature in list(feature_list):
#         series = work_df[feature]
#         unique_count = series.nunique(dropna=True)
#         if unique_count <= 1:
#             dropped_constant.append(feature)
#             feature_list.remove(feature)
#             continue
#         if unique_count / max(1, n_rows) < min_unique_fraction:
#             dropped_constant.append(feature)
#             feature_list.remove(feature)
#             continue

#     # Sparse boolean / near-zero filters
#     for feature in list(feature_list):
#         series = work_df[feature]
#         if pd.api.types.is_bool_dtype(series) or set(series.dropna().unique()).issubset({0, 1}):
#             zero_ratio = (series == 0).mean()
#             if zero_ratio >= max_zero_fraction or (1 - zero_ratio) >= max_zero_fraction:
#                 dropped_sparse.append(feature)
#                 feature_list.remove(feature)

#     # Duplicate detection
#     if feature_list:
#         column_data = work_df[feature_list].T.drop_duplicates()
#         unique_features = list(column_data.index)
#         dropped_duplicates = [f for f in feature_list if f not in unique_features]
#         feature_list = unique_features

#     # Perfectly correlated removal (keep highest variance representative)
#     if feature_list:
#         corr_matrix = work_df[feature_list].corr().abs()
#         to_drop: set[str] = set()
#         for i, f1 in enumerate(feature_list):
#             if f1 in to_drop:
#                 continue
#             for f2 in feature_list[i + 1 :]:
#                 if f2 in to_drop:
#                     continue
#                 value = corr_matrix.at[f1, f2]
#                 if np.isnan(value):
#                     continue
#                 if value >= correlation_threshold:
#                     var1 = float(work_df[f1].var(ddof=0))
#                     var2 = float(work_df[f2].var(ddof=0))
#                     if var1 >= var2:
#                         to_drop.add(f2)
#                     else:
#                         to_drop.add(f1)
#                         break
#         dropped_correlated = list(sorted(to_drop))
#         feature_list = [f for f in feature_list if f not in to_drop]

#     cleaned_df = work_df[feature_list + [c for c in work_df.columns if c not in feature_columns]]
#     report = IntegrityReport(
#         dropped_missing=dropped_missing,
#         dropped_constant=dropped_constant,
#         dropped_sparse=dropped_sparse,
#         dropped_duplicates=dropped_duplicates,
#         dropped_correlated=dropped_correlated,
#     )
#     return cleaned_df, report

def apply_integrity_gates(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    max_missing_ratio: float = 0.2,
    min_unique_fraction: float = 0.01,
    max_zero_fraction: float = 0.995,
    correlation_threshold: float = 0.999,
) -> tuple[pd.DataFrame, IntegrityReport]:
    """
    Remove pathological columns prior to screening (optimized).

    성능 포인트:
    - 전역 루프 제거, 벡터화 집계 사용 (isna/nunique/min/max, mean 등)
    - 중복 컬럼 검출: 전치(T) 없이 샘플+해시 지문으로 후보 생성 후 전체 데이터에서 확정
    - 상관 컬럼 제거: float32로 표준화 후 행렬곱 기반 상관행렬 계산, 분산 큰 컬럼을 우선 보존하는 그리디 드롭
    - 상관 계산은 숫자형 컬럼에만 수행
    """
    # ---- 준비
    feat = list(feature_columns)
    n_rows = len(df)
    sub = df.loc[:, feat]  # view (불필요한 copy 방지)

    dropped_missing: List[str] = []
    dropped_constant: List[str] = []
    dropped_sparse: List[str] = []
    dropped_duplicates: List[str] = []
    dropped_correlated: List[str] = []

    # ---- 1) 결측 필터 (벡터화)
    missing_ratio = sub.isna().mean(numeric_only=False)
    mask_missing = missing_ratio > max_missing_ratio
    if mask_missing.any():
        dropped_missing = missing_ratio[mask_missing].index.tolist()
        sub = sub.drop(columns=dropped_missing, errors="ignore")

    if sub.shape[1] == 0:
        cleaned_df = df.loc[:, [c for c in df.columns if c not in feature_columns]]
        report = IntegrityReport(
            dropped_missing=dropped_missing,
            dropped_constant=[],
            dropped_sparse=[],
            dropped_duplicates=[],
            dropped_correlated=[],
        )
        return cleaned_df, report

    # ---- 2) 상수/저가변 필터 (벡터화)
    unique_cnt = sub.nunique(dropna=True)
    low_var = (unique_cnt <= 1) | (unique_cnt / max(1, n_rows) < min_unique_fraction)
    if low_var.any():
        dropped_constant = unique_cnt[low_var].index.tolist()
        sub = sub.drop(columns=dropped_constant, errors="ignore")

    # ---- 3) 희소(이진/불리언) 필터 (벡터화)
    if sub.shape[1]:
        # bool 타입
        bool_cols = sub.select_dtypes(include=["bool"]).columns
        if len(bool_cols):
            zero_ratio_bool = (~sub[bool_cols]).mean()  # False 비율
            sparse_bool = (zero_ratio_bool >= max_zero_fraction) | ((1 - zero_ratio_bool) >= max_zero_fraction)
            if sparse_bool.any():
                drop_bool = zero_ratio_bool[sparse_bool].index.tolist()
                dropped_sparse.extend(drop_bool)
                sub = sub.drop(columns=drop_bool, errors="ignore")

        # 수치형에서 (0/1) 이진컬럼 판별
        num_cols_all = sub.select_dtypes(include=[np.number]).columns
        if len(num_cols_all):
            num_part = sub[num_cols_all]
            nunq = num_part.nunique(dropna=True)
            # 이진 후보: 고유값 <=2 이고 min>=0, max<=1
            minv = num_part.min(skipna=True)
            maxv = num_part.max(skipna=True)
            is_binary = (nunq <= 2) & (minv >= 0) & (maxv <= 1)
            bin_cols = num_cols_all[is_binary]
            if len(bin_cols):
                zero_ratio_num = (num_part[bin_cols] == 0).mean()
                sparse_num = (zero_ratio_num >= max_zero_fraction) | ((1 - zero_ratio_num) >= max_zero_fraction)
                if sparse_num.any():
                    drop_num = zero_ratio_num[sparse_num].index.tolist()
                    dropped_sparse.extend(drop_num)
                    sub = sub.drop(columns=drop_num, errors="ignore")

    # ---- 4) 중복 컬럼 제거 (전치 없이 해시 지문 + 확정 검사)
    # 큰 데이터에서 메모리 사용을 줄이기 위해 일부 행만 샘플링하여 후보를 만든 뒤,
    # 후보끼리만 전체 데이터에서 완전 동일성(NA 동치)을 확인합니다.
    if sub.shape[1]:
        # 샘플 크기 (너무 작게 잡지 말고, 8k 행 정도면 충돌 확률↓)
        SAMPLE_N = 8192
        sampled = sub if n_rows <= SAMPLE_N else sub.sample(n=SAMPLE_N, random_state=0)
        # 지문 함수: 64비트 해시들의 합과 XOR을 함께 사용해 충돌 확률을 낮춤
        def fp(col: pd.Series) -> tuple[int, int]:
            arr = pd.util.hash_pandas_object(col, index=False).to_numpy(dtype=np.uint64, copy=False)
            return int(arr.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(arr))

        rep_for: Dict[tuple[int, int], str] = {}
        for c in sampled.columns:
            key = fp(sampled[c])
            rep = rep_for.get(key)
            if rep is None:
                rep_for[key] = c
                continue
            # 전체 데이터에서 NA 동일 취급하며 완전 동일성 확인
            s1, s2 = sub[rep], sub[c]
            equal_all = ((s1 == s2) | (s1.isna() & s2.isna())).all()
            if bool(equal_all):
                dropped_duplicates.append(c)
            else:
                # 희귀 충돌이면 새로운 키로 대표 등록
                rep_for[(key, c)] = c

        if dropped_duplicates:
            sub = sub.drop(columns=dropped_duplicates, errors="ignore")

    # ---- 5) 고상관(=완전/준완전 상관) 컬럼 제거 (수치형만, 행/열 청크 기반)
    # 메모리 사용을 줄이기 위해 두 단계로 통계를 누적해 상관행렬을 구성합니다.
    if sub.shape[1]:
        num_cols = list(sub.select_dtypes(include=[np.number]).columns)
        if num_cols:
            numeric_df = sub[num_cols]
            p = len(num_cols)
            n_total = len(numeric_df)

            # 1차 패스: 각 열의 합, 제곱합, 개수를 추정 (NaN 제외)
            sums = np.zeros(p, dtype=np.float64)
            sumsqs = np.zeros(p, dtype=np.float64)
            counts = np.zeros(p, dtype=np.int64)

            # 64MB 이하로 유지되도록 행 청크 크기 계산
            target_mem = 64 * 1024 * 1024  # 64 MiB
            denom = max(1, p * 4)  # float32 바이트
            chunk_rows = int(target_mem // denom)
            chunk_rows = max(512, min(n_total, chunk_rows)) if n_total else 0
            if chunk_rows == 0:
                chunk_rows = n_total or 1

            for start in range(0, n_total, chunk_rows):
                stop = min(start + chunk_rows, n_total)
                chunk = numeric_df.iloc[start:stop].to_numpy(dtype=np.float64, copy=False)
                if chunk.size == 0:
                    continue
                sums += np.nansum(chunk, axis=0, dtype=np.float64)
                sumsqs += np.nansum(np.square(chunk, dtype=np.float64), axis=0, dtype=np.float64)
                counts += np.sum(~np.isnan(chunk), axis=0, dtype=np.int64)

            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

            sq_term = np.zeros_like(sumsqs)
            valid_counts = counts > 0
            sq_term[valid_counts] = (sums[valid_counts] ** 2) / counts[valid_counts]

            var = np.zeros_like(sumsqs)
            if n_total > 0:
                var[valid_counts] = (sumsqs[valid_counts] - sq_term[valid_counts]) / float(n_total)
            var = np.maximum(var, 0.0)

            std = np.sqrt(var, dtype=np.float64)
            inv_std = np.zeros_like(std)
            nonzero_std = std > 0.0
            inv_std[nonzero_std] = 1.0 / std[nonzero_std]

            means64 = means.astype(np.float64, copy=False)
            inv_std64 = inv_std.astype(np.float64, copy=False)

            # 2차 패스: 표준화된 누적 내적을 통해 상관 행렬 계산
            corr_accum = np.zeros((p, p), dtype=np.float64)

            for start in range(0, n_total, chunk_rows):
                stop = min(start + chunk_rows, n_total)
                chunk = numeric_df.iloc[start:stop].to_numpy(dtype=np.float64, copy=True)
                if chunk.size == 0:
                    continue
                nan_mask = np.isnan(chunk)
                if nan_mask.any():
                    col_index = np.where(nan_mask)[1]
                    chunk[nan_mask] = means64[col_index]
                chunk -= means64
                chunk *= inv_std64
                corr_accum += chunk.T @ chunk

            if n_total:
                corr_matrix = corr_accum / float(n_total)
            else:
                corr_matrix = corr_accum

            np.fill_diagonal(corr_matrix, 0.0)
            abs_corr = np.abs(corr_matrix)

            variances = var.astype(np.float64, copy=False)
            order = np.argsort(-variances)  # 분산 큰 것부터 유지

            keep = np.ones(p, dtype=bool)
            thr = float(correlation_threshold)
            for idx in order:
                if not keep[idx]:
                    continue
                high = abs_corr[idx] >= thr
                if high.any():
                    remove = high & keep
                    remove[idx] = False
                    keep[remove] = False

            dropped_corr_cols = [c for k, c in enumerate(num_cols) if not keep[k]]
            if dropped_corr_cols:
                dropped_correlated.extend(sorted(dropped_corr_cols))
                sub = sub.drop(columns=dropped_corr_cols, errors="ignore")

    # ---- 결과 조립 (특징 컬럼 + 비특징 컬럼 보존)
    kept_features = sub.columns.tolist()
    feature_set = set(feature_columns)
    other_cols = [c for c in df.columns if c not in feature_set]
    cleaned_df = df.loc[:, kept_features + other_cols]

    report = IntegrityReport(
        dropped_missing=dropped_missing,
        dropped_constant=dropped_constant,
        dropped_sparse=dropped_sparse,
        dropped_duplicates=dropped_duplicates,
        dropped_correlated=sorted(dropped_correlated),
    )
    return cleaned_df, report
