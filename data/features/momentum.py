"""Momentum-centric feature generators."""
from __future__ import annotations

import pandas as pd
import talib

from .utils import (
    PERCENT_SCALE,
    format_col_name,
    get_ohlcv_arrays,
    scale_by_reference,
)


def calculate_rsi(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('RSI', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.RSI(close_price, timeperiod=window)
    return df, [col_name]


def calculate_stochastic_oscillator(
    df: pd.DataFrame,
    k_window: int,
    d_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    extra = f"{k_window}_{d_window}"
    k_col = format_col_name('%K', extra)
    d_col = format_col_name('%D', extra)

    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)

    slowk, slowd = talib.STOCH(
        high_price,
        low_price,
        close_price,
        fastk_period=k_window,
        slowk_period=d_window,
        slowk_matype=0,
        slowd_period=d_window,
        slowd_matype=0,
    )
    df[k_col] = slowk
    df[d_col] = slowd
    return df, [k_col, d_col]


def calculate_williams_r(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('Williams_R', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.WILLR(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]


def calculate_cci(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('CCI', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.CCI(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]


def calculate_stochrsi(
    df: pd.DataFrame,
    window: int = 14,
    k_window: int = 5,
    d_window: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    extra = f"{window}_{k_window}_{d_window}"
    k_col = format_col_name('STOCHRSI_K', extra)
    d_col = format_col_name('STOCHRSI_D', extra)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    k, d = talib.STOCHRSI(
        close_price,
        timeperiod=window,
        fastk_period=k_window,
        fastd_period=d_window,
        fastd_matype=0,
    )
    df[k_col] = k
    df[d_col] = d
    return df, [k_col, d_col]


def calculate_ppo(
    df: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('PPO', f"{fast_window}_{slow_window}")
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.PPO(
        close_price,
        fastperiod=fast_window,
        slowperiod=slow_window,
        matype=0,
    )
    return df, [col_name]


def calculate_roc(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('ROC', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.ROC(close_price, timeperiod=window)
    return df, [col_name]


def calculate_ultosc(
    df: pd.DataFrame,
    window1: int = 7,
    window2: int = 14,
    window3: int = 28,
) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('ULTOSC', f"{window1}_{window2}_{window3}")
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.ULTOSC(
        high_price,
        low_price,
        close_price,
        timeperiod1=window1,
        timeperiod2=window2,
        timeperiod3=window3,
    )
    return df, [col_name]


def calculate_mom(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('MOM', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    mom_values = talib.MOM(close_price, timeperiod=window)
    mom_series = pd.Series(mom_values, index=df.index)
    prev_close = df['Close'].shift(window)
    df[col_name] = scale_by_reference(mom_series, prev_close, multiplier=PERCENT_SCALE)
    return df, [col_name]


__all__ = [
    'calculate_rsi',
    'calculate_stochastic_oscillator',
    'calculate_williams_r',
    'calculate_cci',
    'calculate_stochrsi',
    'calculate_ppo',
    'calculate_roc',
    'calculate_ultosc',
    'calculate_mom',
]
