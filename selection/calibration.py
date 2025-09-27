# /selection/calibration.py
"""Range calibration and boolean handling (vectorized, memory-efficient, with optional chunked exact MAD).

Key points
- Removes per-column loops by computing quantiles in batch
- Provides two MAD modes: approximate (IQR-based) and exact_chunked (memmap-driven)
- Minimises peak memory footprint for very wide dataframes

Notes
- Float columns with values in {0, 1} are treated as numeric to avoid costly value checks
- The exact_chunked MAD mode creates temporary files in the scratch directory
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CalibrationResult:
    feature_num: Dict[str, tuple[float, float]]
    feature_bool: List[str]
    feature_comparison_map: Dict[str, List[str]]
    diagnostics: Dict[str, float]


def _as_list_of_ndarrays(
    fold_indices: Optional[Sequence[np.ndarray]],
) -> Optional[List[np.ndarray]]:
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
    """Detect true boolean columns and integer 0/1 indicators."""
    if not cols:
        return []

    dtypes = df[cols].dtypes
    bool_cols_dtype = dtypes[dtypes.apply(pd.api.types.is_bool_dtype)].index.tolist()

    int_mask = dtypes.apply(pd.api.types.is_integer_dtype)
    int_cols = dtypes[int_mask].index.tolist()
    bool_cols_int: List[str] = []
    if int_cols:
        extrema = df[int_cols].agg(["min", "max"])
        mask01 = (extrema.loc["min"] >= 0) & (extrema.loc["max"] <= 1)
        bool_cols_int = extrema.columns[mask01].tolist()

    bool_set = set(bool_cols_dtype) | set(bool_cols_int)
    return [c for c in cols if c in bool_set]


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
    """Compute exact MAD via streamed absolute deviations stored in a memmap."""
    n_rows, _ = df_num.shape
    cols = list(df_num.columns)
    result = pd.Series(index=cols, dtype=float)

    if n_rows == 0 or not cols:
        result[:] = np.nan
        return result

    col_to_pos = {column: pos for pos, column in enumerate(df_num.columns)}

    for block_cols in _iter_col_blocks(cols, col_block_size):
        if not block_cols:
            continue

        tmp_dir = scratch_dir or tempfile.gettempdir()
        fname = os.path.join(tmp_dir, f"mad_dev_{uuid.uuid4().hex}.dat")
        mmap = np.memmap(fname, dtype="float64", mode="w+", shape=(n_rows, len(block_cols)))

        try:
            for r0 in range(0, n_rows, max(1, chunk_rows)):
                r1 = min(n_rows, r0 + chunk_rows)
                positions = [col_to_pos[c] for c in block_cols]
                block_vals = df_num.iloc[r0:r1, positions].to_numpy(copy=False)
                med = medians[block_cols].to_numpy()
                mmap[r0:r1, :] = np.abs(block_vals - med)

            mad_block = np.nanmedian(mmap, axis=0)
            result.loc[block_cols] = mad_block
        finally:
            del mmap
            if delete_scratch:
                try:
                    os.remove(fname)
                except OSError:
                    pass

    return result.fillna(0.0)


def calibrate_ranges(
    df: pd.DataFrame,
    features: Sequence[str],
    quantiles: tuple[float, float] = (0.005, 0.995),
    mad_factor: float | None = 1.4826,
    fold_indices: Optional[Sequence[np.ndarray]] = None,
    *,
    mad_mode: str = "approx",
    chunk_rows: int = 20000,
    col_block_size: int = 128,
    scratch_dir: Optional[str] = None,
    delete_scratch: bool = True,
) -> CalibrationResult:
    """Calibrate numeric feature ranges with optional fold-wise quantiles and MAD margins."""
    if not features:
        return CalibrationResult({}, [], {}, {})

    existing = [feature for feature in features if feature in df.columns]
    if not existing:
        return CalibrationResult({}, [], {}, {})

    feature_bool = _detect_boolean_columns(df, existing)

    numeric_all = df[existing].select_dtypes(include=[np.number]).columns.tolist()
    numeric = [col for col in numeric_all if col not in feature_bool]

    if numeric:
        counts = df[numeric].count()
        numeric = [col for col in numeric if counts[col] > 0]

    if not numeric:
        return CalibrationResult({}, feature_bool, {}, {})

    q_low, q_high = quantiles
    q_levels = np.array(sorted({q_low, 0.25, 0.5, 0.75, q_high}), dtype=float)

    fold_list = _as_list_of_ndarrays(fold_indices)
    if not fold_list:
        Q = df[numeric].quantile(q_levels, numeric_only=True)
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
        for idx in fold_list:
            sub = df.iloc[idx][numeric]
            Qf = sub.quantile(q_levels, numeric_only=True)
            lows_list.append(Qf.loc[q_low])
            highs_list.append(Qf.loc[q_high])
            q25_list.append(Qf.loc[0.25])
            q75_list.append(Qf.loc[0.75])
        lows = pd.concat(lows_list, axis=1).median(axis=1)
        highs = pd.concat(highs_list, axis=1).median(axis=1)
        q25 = pd.concat(q25_list, axis=1).median(axis=1)
        q75 = pd.concat(q75_list, axis=1).median(axis=1)
        med_for_mad = df[numeric].quantile(0.5, numeric_only=True)

    diagnostics: Dict[str, float] = {}
    if mad_factor is not None and mad_factor > 0:
        if mad_mode == "exact_chunked":
            mad_series = _mad_exact_chunked(
                df_num=df[numeric],
                medians=med_for_mad,
                chunk_rows=chunk_rows,
                col_block_size=col_block_size,
                scratch_dir=scratch_dir,
                delete_scratch=delete_scratch,
            )
        else:
            iqr = q75 - q25
            mad_series = (1.4826 / 1.349) * iqr
            mad_series = mad_series.fillna(0.0)
        margin = mad_factor * mad_series
        diagnostics.update({f"mad_{col}": float(mad_series[col]) for col in mad_series.index})
    else:
        margin = pd.Series(0.0, index=pd.Index(numeric))

    lower = (lows - margin).astype(float)
    upper = (highs + margin).astype(float)

    feature_num: Dict[str, tuple[float, float]] = {
        col: (float(lower[col]), float(upper[col]))
        for col in lower.index
    }

    return CalibrationResult(
        feature_num=feature_num,
        feature_bool=feature_bool,
        feature_comparison_map={},
        diagnostics=diagnostics,
    )
