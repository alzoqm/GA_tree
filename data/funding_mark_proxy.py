import os
from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd

# We reuse the existing spot fetcher to avoid duplicating client setup
from data.data_download import fetch_historical_data as fetch_spot_klines

try:
    from binance.client import Client
    from dotenv import load_dotenv
except Exception:  # ImportError or environment without dependencies
    Client = None  # type: ignore
    load_dotenv = None  # type: ignore


def _require_binance_client() -> Client:
    """
    Initialize and return a Binance Client for futures klines.
    Reads API keys from project root .env via python-dotenv if available.
    """
    if Client is None:
        raise RuntimeError("python-binance not available. Please install dependencies.")
    # .env at repo root
    from pathlib import Path
    root_env = Path(__file__).resolve().parents[1] / '.env'
    if load_dotenv is not None:
        load_dotenv(dotenv_path=str(root_env))
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_API_KEY/SECRET not found in .env")
    return Client(api_key, api_secret)


def fetch_perp_klines(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch USDⓈ-M perpetual futures klines for [start_dt, end_dt].
    Returns a DataFrame with columns: 'Open time','Open','High','Low','Close','Volume', 'Close time', ...
    """
    client = _require_binance_client()

    # Map human interval to Binance constants
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH,
    }
    iv = interval_map.get(interval)
    if iv is None:
        raise ValueError(f"Unsupported interval: {interval}")

    start_ms = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    data = client.futures_klines(symbol=symbol, interval=iv, startTime=start_ms, endTime=end_ms)
    cols = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    df = pd.DataFrame(data, columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms', utc=True).dt.tz_convert(None)
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms', utc=True).dt.tz_convert(None)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _infer_bar_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    diffs = pd.Series(index[1:]).reset_index(drop=True) - pd.Series(index[:-1]).reset_index(drop=True)
    # Use median to avoid occasional gaps
    med = pd.to_timedelta(np.median(diffs.values))
    return max(1, int(round(med / pd.Timedelta(minutes=1))))


def _align_ohlcv(perp_df: pd.DataFrame, spot_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    for df in (perp_df, spot_df):
        if 'Open time' not in df.columns:
            raise ValueError("DataFrame must contain 'Open time'")
    p = perp_df.copy()
    s = spot_df.copy()
    p['Open time'] = pd.to_datetime(p['Open time'])
    s['Open time'] = pd.to_datetime(s['Open time'])
    p = p.sort_values('Open time')
    s = s.sort_values('Open time')

    # Inner join by timestamp to ensure perfect alignment
    p = p.set_index('Open time')
    s = s.set_index('Open time')
    idx = p.index.intersection(s.index)
    p2 = p.loc[idx].copy()
    s2 = s.loc[idx].copy()
    bar_minutes = _infer_bar_minutes(idx)
    p2.reset_index(inplace=True)
    s2.reset_index(inplace=True)
    return p2, s2, bar_minutes


def estimate_funding_and_mark(perp_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic estimator for Funding Rate (8h TWAP policy) and Mark Price.

    Returns a DataFrame with columns: ['Open time','fundingRate_est','markPrice_est'].
    No NaNs are produced; pre-boundary funding defaults to 0.0.
    """
    perp, spot, bar_m = _align_ohlcv(perp_df, spot_df)
    # Proxies
    I = spot['Close'].to_numpy(dtype=np.float64)  # Index proxy = spot close
    M = ((perp['High'] + perp['Low'] + 2.0 * perp['Close']) / 4.0).to_numpy(dtype=np.float64)
    last_price = perp['Close'].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(perp['Open time'])

    # Premium per bar
    I_safe = np.maximum(I, 1e-12)
    p = (M - I) / I_safe

    # 8h TWAP window in bars
    N = int(np.ceil(480.0 / float(bar_m)))
    if N <= 0:
        N = 1

    # Rolling mean with fixed window; first N-1 bars will be NaN
    p_series = pd.Series(p, index=ts)
    Pbar = p_series.rolling(window=N, min_periods=N).mean().to_numpy()

    # Constant parameters (fixed by spec)
    I_rate = 0.0001   # 0.01% per 8h
    cap_abs = 0.0005  # +/- 0.05% per 8h

    # Funding raw at each bar (only meaningful at boundaries)
    # F_raw = Pbar + clamp(I_rate - Pbar, -cap, +cap)
    F_raw = np.empty_like(Pbar, dtype=np.float64)
    # Where Pbar is NaN (initial period), set F_raw to NaN; will be handled by schedule fill later
    mask_valid = np.isfinite(Pbar)
    F_raw[mask_valid] = Pbar[mask_valid] + np.clip(I_rate - Pbar[mask_valid], -cap_abs, cap_abs)
    F_raw[~mask_valid] = np.nan

    # Build funding schedule: constant per [boundary, next_boundary)
    dates = pd.DatetimeIndex(ts)
    # Boundaries at 00:00, 08:00, 16:00 UTC
    def _is_boundary(d: pd.Timestamp) -> bool:
        return d.hour in (0, 8, 16) and d.minute == 0

    funding_est = np.zeros_like(F_raw, dtype=np.float64)
    current_F = np.nan
    for i, d in enumerate(dates):
        if _is_boundary(d):
            # Update F only when Pbar is available; otherwise keep previous
            if np.isfinite(F_raw[i]):
                current_F = F_raw[i]
        if not np.isfinite(current_F):
            # Before the first valid boundary, force 0.0 deterministically
            funding_est[i] = 0.0
        else:
            funding_est[i] = float(current_F)

    # Mark Price estimate: median(Price1, Price2, Last)
    # Basis and short SMA window (2.5 minutes -> k bars)
    k = max(1, int(round(2.5 / float(bar_m))))
    basis = (M - I)
    basis_ma = pd.Series(basis, index=ts).rolling(window=k, min_periods=1).mean().to_numpy()
    price2 = I + basis_ma

    # Price1 uses current funding and time to next boundary (hours/8)
    def _next_boundary(d: pd.Timestamp) -> pd.Timestamp:
        h = d.hour
        base = d.replace(minute=0, second=0, microsecond=0)
        if h < 8:
            return base.replace(hour=8)
        elif h < 16:
            return base.replace(hour=16)
        else:
            return (base + pd.Timedelta(days=1)).replace(hour=0)

    time_to_next_h = np.array([
        max(0.0, (_next_boundary(d) - d).total_seconds() / 3600.0) for d in dates
    ], dtype=np.float64)
    price1 = I * (1.0 + funding_est * (time_to_next_h / 8.0))

    # Contract price = perp close
    last = last_price

    # Median of three
    stacked = np.stack([price1, price2, last], axis=1)
    mark_est = np.median(stacked, axis=1)

    out = pd.DataFrame({
        'Open time': ts,
        'fundingRate_est': funding_est.astype(np.float64),
        'markPrice_est': mark_est.astype(np.float64),
    })
    return out

