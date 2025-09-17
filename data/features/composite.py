"""Composite and cross-feature generators."""
from __future__ import annotations

import pandas as pd

from .utils import EPS, format_col_name


def calculate_rsi_adx_ratio(
    df: pd.DataFrame,
    rsi_window: int,
    adx_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    rsi_col = format_col_name('RSI', rsi_window)
    adx_col = format_col_name('ADX', adx_window)
    new_col = format_col_name('RSI_ADX_Ratio', f"{rsi_window}_{adx_window}")

    if rsi_col in df and adx_col in df:
        df[new_col] = df[rsi_col] / (df[adx_col] + EPS)
        return df, [new_col]
    return df, []


def calculate_macd_atr_ratio(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    signal_window: int,
    atr_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    macd_col = format_col_name('MACD', f"{short_window}_{long_window}")
    atr_col = format_col_name('ATR', atr_window)
    new_col = format_col_name('MACD_ATR_Ratio', f"{short_window}_{long_window}_{atr_window}")

    if macd_col in df and atr_col in df:
        df[new_col] = df[macd_col] / (df[atr_col] + EPS)
        return df, [new_col]
    return df, []


def calculate_short_long_ratio(
    df: pd.DataFrame,
    indicator_name: str,
    short_window: int,
    long_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    if indicator_name == 'BB_Width':
        short_col = format_col_name(indicator_name, f"{short_window}_2")
        long_col = format_col_name(indicator_name, f"{long_window}_2")
    else:
        short_col = format_col_name(indicator_name, short_window)
        long_col = format_col_name(indicator_name, long_window)

    new_col = format_col_name(f"{indicator_name}_Ratio", f"{short_window}_{long_window}")

    if short_col in df and long_col in df:
        df[new_col] = df[short_col] / (df[long_col] + EPS)
        return df, [new_col]
    return df, []


def calculate_indicator_roc(
    df: pd.DataFrame,
    indicator_name: str,
    indicator_window: int,
    roc_period: int,
) -> tuple[pd.DataFrame, list[str]]:
    indicator_col = format_col_name(indicator_name, indicator_window)
    new_col = format_col_name(f"{indicator_name}_ROC", f"{indicator_window}_{roc_period}")

    if indicator_col in df:
        shifted_indicator = df[indicator_col].shift(roc_period)
        df[new_col] = (
            (df[indicator_col] - shifted_indicator) / (shifted_indicator.abs() + EPS) * 100
        )
        return df, [new_col]
    return df, []


def calculate_volume_spike(
    df: pd.DataFrame,
    window: int,
    factor: float,
) -> tuple[pd.DataFrame, list[str]]:
    new_col = format_col_name('Volume_Spike', f"{window}_{factor}")
    avg_volume = df['Volume'].rolling(window=window).mean()
    df[new_col] = df['Volume'] > (avg_volume * factor)
    return df, [new_col]


def calculate_consecutive_candles(
    df: pd.DataFrame,
    period: int,
    direction: str,
) -> tuple[pd.DataFrame, list[str]]:
    if direction == 'bullish':
        is_positive = df['Close'] > df['Open']
        new_col = format_col_name('Consecutive_Bullish', period)
    elif direction == 'bearish':
        is_positive = df['Close'] < df['Open']
        new_col = format_col_name('Consecutive_Bearish', period)
    else:
        return df, []

    consecutive_count = is_positive.rolling(window=period).sum()
    df[new_col] = consecutive_count == period
    return df, [new_col]


def calculate_confirmation_signal(
    df: pd.DataFrame,
    rsi_window: int,
    adx_window: int,
    macd_short: int,
    macd_long: int,
    rsi_threshold: int,
    adx_threshold: int,
    direction: str,
) -> tuple[pd.DataFrame, list[str]]:
    rsi_col = format_col_name('RSI', rsi_window)
    adx_col = format_col_name('ADX', adx_window)
    macd_col = format_col_name('MACD', f"{macd_short}_{macd_long}")

    if not all(col in df for col in [rsi_col, adx_col, macd_col]):
        return df, []

    base_cond = df[adx_col] > adx_threshold

    if direction == 'bullish':
        cond = base_cond & (df[rsi_col] > rsi_threshold) & (df[macd_col] > 0)
        new_col = format_col_name('Bullish_Confirmation', f"{rsi_window}_{adx_window}")
    elif direction == 'bearish':
        cond = base_cond & (df[rsi_col] < rsi_threshold) & (df[macd_col] < 0)
        new_col = format_col_name('Bearish_Confirmation', f"{rsi_window}_{adx_window}")
    else:
        return df, []

    df = df.assign(**{new_col: cond})
    return df, [new_col]


__all__ = [
    'calculate_rsi_adx_ratio',
    'calculate_macd_atr_ratio',
    'calculate_short_long_ratio',
    'calculate_indicator_roc',
    'calculate_volume_spike',
    'calculate_consecutive_candles',
    'calculate_confirmation_signal',
]
