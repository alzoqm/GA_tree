"""Integrity gates for feature screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
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


def apply_integrity_gates(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    max_missing_ratio: float = 0.2,
    min_unique_fraction: float = 0.01,
    max_zero_fraction: float = 0.995,
    correlation_threshold: float = 0.999,
) -> tuple[pd.DataFrame, IntegrityReport]:
    """Remove pathological columns prior to screening.

    Args:
        df: Source dataframe.
        feature_columns: Target feature names to evaluate.
        max_missing_ratio: Maximum allowed fraction of missing values.
        min_unique_fraction: Minimum unique ratio relative to rows to consider non-constant.
        max_zero_fraction: For sparse (binary) columns, drop if the majority are zero.
        correlation_threshold: Threshold for dropping perfectly correlated duplicates (>).
    """

    feature_list = list(feature_columns)
    dropped_missing: List[str] = []
    dropped_constant: List[str] = []
    dropped_sparse: List[str] = []
    dropped_duplicates: List[str] = []
    dropped_correlated: List[str] = []

    work_df = df.copy()

    # Missing values filter
    missing_ratio = work_df[feature_list].isna().mean()
    for feature, ratio in missing_ratio.items():
        if ratio > max_missing_ratio:
            dropped_missing.append(feature)

    feature_list = [f for f in feature_list if f not in dropped_missing]

    # Constant / low variability filter
    n_rows = len(work_df)
    for feature in list(feature_list):
        series = work_df[feature]
        unique_count = series.nunique(dropna=True)
        if unique_count <= 1:
            dropped_constant.append(feature)
            feature_list.remove(feature)
            continue
        if unique_count / max(1, n_rows) < min_unique_fraction:
            dropped_constant.append(feature)
            feature_list.remove(feature)
            continue

    # Sparse boolean / near-zero filters
    for feature in list(feature_list):
        series = work_df[feature]
        if pd.api.types.is_bool_dtype(series) or set(series.dropna().unique()).issubset({0, 1}):
            zero_ratio = (series == 0).mean()
            if zero_ratio >= max_zero_fraction or (1 - zero_ratio) >= max_zero_fraction:
                dropped_sparse.append(feature)
                feature_list.remove(feature)

    # Duplicate detection
    if feature_list:
        column_data = work_df[feature_list].T.drop_duplicates()
        unique_features = list(column_data.index)
        dropped_duplicates = [f for f in feature_list if f not in unique_features]
        feature_list = unique_features

    # Perfectly correlated removal (keep highest variance representative)
    if feature_list:
        corr_matrix = work_df[feature_list].corr().abs()
        to_drop: set[str] = set()
        for i, f1 in enumerate(feature_list):
            if f1 in to_drop:
                continue
            for f2 in feature_list[i + 1 :]:
                if f2 in to_drop:
                    continue
                value = corr_matrix.at[f1, f2]
                if np.isnan(value):
                    continue
                if value >= correlation_threshold:
                    var1 = float(work_df[f1].var(ddof=0))
                    var2 = float(work_df[f2].var(ddof=0))
                    if var1 >= var2:
                        to_drop.add(f2)
                    else:
                        to_drop.add(f1)
                        break
        dropped_correlated = list(sorted(to_drop))
        feature_list = [f for f in feature_list if f not in to_drop]

    cleaned_df = work_df[feature_list + [c for c in work_df.columns if c not in feature_columns]]
    report = IntegrityReport(
        dropped_missing=dropped_missing,
        dropped_constant=dropped_constant,
        dropped_sparse=dropped_sparse,
        dropped_duplicates=dropped_duplicates,
        dropped_correlated=dropped_correlated,
    )
    return cleaned_df, report
