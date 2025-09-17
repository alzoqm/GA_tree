"""Cycle-detection helpers and temporal encoders."""
from __future__ import annotations

import pandas as pd
import talib

from .utils import format_col_name, get_ohlcv_arrays


def calculate_ht_dcperiod(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    col_name = 'HT_DCPERIOD'
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.HT_DCPERIOD(close_price)
    return df, [col_name]


def calculate_ht_trendmode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    col_name = 'HT_TRENDMODE'
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.HT_TRENDMODE(close_price)
    return df, [col_name]


def calculate_ht_sine(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sine_col = 'HT_SINE'
    leadsine_col = 'HT_LEADSINE'
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[sine_col], df[leadsine_col] = talib.HT_SINE(close_price)
    return df, [sine_col, leadsine_col]


def cyclic_encode_fn(df: pd.DataFrame, timestamp_col: str, cycle: str) -> tuple[pd.DataFrame, list[str]]:
    """Placeholder for cyclic timestamp encoding (implementation matches legacy stub)."""
    return df, []


__all__ = [
    'calculate_ht_dcperiod',
    'calculate_ht_trendmode',
    'calculate_ht_sine',
    'cyclic_encode_fn',
]
