"""Candlestick pattern feature generators."""
from __future__ import annotations

import talib

from .utils import get_ohlcv_arrays


def calculate_all_candlestick_patterns(df):
    """Evaluate candlestick patterns using TA-Lib and return boolean flags."""
    open_p, high_p, low_p, close_p, _ = get_ohlcv_arrays(df)

    df['InvertedHammers'] = talib.CDLINVERTEDHAMMER(open_p, high_p, low_p, close_p) != 0
    df['Hammers'] = talib.CDLHAMMER(open_p, high_p, low_p, close_p) != 0
    df['HangingMen'] = talib.CDLHANGINGMAN(open_p, high_p, low_p, close_p) != 0
    df['DarkCloudCovers'] = talib.CDLDARKCLOUDCOVER(open_p, high_p, low_p, close_p) != 0
    df['Dojis'] = talib.CDLDOJI(open_p, high_p, low_p, close_p) != 0
    df['DragonflyDojis'] = talib.CDLDRAGONFLYDOJI(open_p, high_p, low_p, close_p) != 0
    df['GravestoneDojis'] = talib.CDLGRAVESTONEDOJI(open_p, high_p, low_p, close_p) != 0
    df['MorningStars'] = talib.CDLMORNINGSTAR(open_p, high_p, low_p, close_p) != 0
    df['MorningStarDojis'] = talib.CDLMORNINGDOJISTAR(open_p, high_p, low_p, close_p) != 0
    df['PiercingPatterns'] = talib.CDLPIERCING(open_p, high_p, low_p, close_p) != 0
    df['ShootingStars'] = talib.CDLSHOOTINGSTAR(open_p, high_p, low_p, close_p) != 0

    df['3BlackCrows'] = talib.CDL3BLACKCROWS(open_p, high_p, low_p, close_p) != 0
    df['3WhiteSoldiers'] = talib.CDL3WHITESOLDIERS(open_p, high_p, low_p, close_p) != 0
    df['3StarsInSouth'] = talib.CDL3STARSINSOUTH(open_p, high_p, low_p, close_p) != 0

    harami = talib.CDLHARAMI(open_p, high_p, low_p, close_p)
    df['BullishHarami'] = harami > 0
    df['BearishHarami'] = harami < 0

    engulfing = talib.CDLENGULFING(open_p, high_p, low_p, close_p)
    df['BullishEngulfing'] = engulfing > 0
    df['BearishEngulfing'] = engulfing < 0

    doji_star = talib.CDLDOJISTAR(open_p, high_p, low_p, close_p)
    df['BullishDojiStar'] = doji_star > 0
    df['BearishDojiStar'] = doji_star < 0

    tasuki_gap = talib.CDLTASUKIGAP(open_p, high_p, low_p, close_p)
    df['BullishTasukiGap'] = tasuki_gap > 0
    df['BearishTasukiGap'] = tasuki_gap < 0

    xside_gap = talib.CDLXSIDEGAP3METHODS(open_p, high_p, low_p, close_p)
    df['BullishXSideGap3Methods'] = xside_gap > 0
    df['BearishXSideGap3Methods'] = xside_gap < 0

    spinning_top = talib.CDLSPINNINGTOP(open_p, high_p, low_p, close_p)
    df['BullishSpinningTop'] = spinning_top > 0
    df['BearishSpinningTop'] = spinning_top < 0

    rise_fall = talib.CDLRISEFALL3METHODS(open_p, high_p, low_p, close_p)
    df['BullishRise3Methods'] = rise_fall > 0
    df['BearishFall3Methods'] = rise_fall < 0

    candle_cols = [
        'InvertedHammers',
        'Hammers',
        'HangingMen',
        'DarkCloudCovers',
        'Dojis',
        'DragonflyDojis',
        'GravestoneDojis',
        'MorningStars',
        'MorningStarDojis',
        'PiercingPatterns',
        'ShootingStars',
        '3BlackCrows',
        '3WhiteSoldiers',
        '3StarsInSouth',
        'BullishHarami',
        'BearishHarami',
        'BullishEngulfing',
        'BearishEngulfing',
        'BullishDojiStar',
        'BearishDojiStar',
        'BullishTasukiGap',
        'BearishTasukiGap',
        'BullishXSideGap3Methods',
        'BearishXSideGap3Methods',
        'BullishSpinningTop',
        'BearishSpinningTop',
        'BullishRise3Methods',
        'BearishFall3Methods',
    ]

    return df, candle_cols


__all__ = ['calculate_all_candlestick_patterns']
