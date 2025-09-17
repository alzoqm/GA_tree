"""Support and resistance feature generators."""
from __future__ import annotations

import pandas as pd
import talib

from .utils import EPS, format_col_name, get_ohlcv_arrays


def calculate_support_resistance(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    new_cols: list[str] = []
    _, high_price, low_price, _, _ = get_ohlcv_arrays(df)

    support_ma_col = format_col_name('Support_MA', window)
    resistance_ma_col = format_col_name('Resistance_MA', window)
    df[support_ma_col] = talib.MIN(low_price, timeperiod=window)
    df[resistance_ma_col] = talib.MAX(high_price, timeperiod=window)
    new_cols.extend([support_ma_col, resistance_ma_col])

    high_prev = df['High'].shift(1)
    low_prev = df['Low'].shift(1)
    close_prev = df['Close'].shift(1)

    pivot_point_col = format_col_name('Pivot_Point', window)
    support1_col = format_col_name('Support1_Pivot', window)
    support2_col = format_col_name('Support2_Pivot', window)
    resistance1_col = format_col_name('Resistance1_Pivot', window)
    resistance2_col = format_col_name('Resistance2_Pivot', window)

    df[pivot_point_col] = (high_prev + low_prev + close_prev) / 3
    df[support1_col] = (2 * df[pivot_point_col]) - high_prev
    df[resistance1_col] = (2 * df[pivot_point_col]) - low_prev
    df[support2_col] = df[pivot_point_col] - (high_prev - low_prev)
    df[resistance2_col] = df[pivot_point_col] + (high_prev - low_prev)

    new_cols.extend([
        pivot_point_col,
        support1_col,
        support2_col,
        resistance1_col,
        resistance2_col,
    ])

    return df, new_cols


__all__ = ['calculate_support_resistance']
