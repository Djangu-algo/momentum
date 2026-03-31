from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import average_true_range, ema, rolling_theil_sen, validate_ohlcv_frame


STATE_CODES = {
    "TREND_BROKEN": 0,
    "MOMENTUM_LOST": 1,
    "DECELERATING": 2,
    "ACCELERATING": 3,
}


def classify_ema_state(d_close: np.ndarray, slope_high: np.ndarray, slope_close: np.ndarray) -> tuple[np.ndarray, list[str]]:
    codes = np.full(d_close.shape, np.nan, dtype=float)
    labels = ["UNKNOWN"] * d_close.size
    for idx in range(d_close.size):
        dc = d_close[idx]
        sh = slope_high[idx]
        sc = slope_close[idx]
        if np.isnan(dc):
            continue
        if dc < 0.0:
            codes[idx] = STATE_CODES["TREND_BROKEN"]
            labels[idx] = "TREND_BROKEN"
        elif np.isnan(sh) or np.isnan(sc):
            continue
        elif sh > 0.0 and sc > 0.0:
            codes[idx] = STATE_CODES["ACCELERATING"]
            labels[idx] = "ACCELERATING"
        elif sh < 0.0 and sc > 0.0:
            codes[idx] = STATE_CODES["DECELERATING"]
            labels[idx] = "DECELERATING"
        elif sh < 0.0 and sc < 0.0:
            codes[idx] = STATE_CODES["MOMENTUM_LOST"]
            labels[idx] = "MOMENTUM_LOST"
        else:
            codes[idx] = STATE_CODES["ACCELERATING"]
            labels[idx] = "ACCELERATING"
    return codes, labels


def add_ema_distance_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    data = validate_ohlcv_frame(frame)
    open_ = data["open"].to_numpy()
    high = data["high"].to_numpy()
    low = data["low"].to_numpy()
    close = data["close"].to_numpy()

    ema_125 = ema(close, config.ema_length)
    atr_14 = average_true_range(high, low, close, config.atr_length)
    atr_safe = np.where((atr_14 == 0.0) | np.isnan(atr_14), np.nan, atr_14)
    d_open = (open_ - ema_125) / atr_safe
    d_high = (high - ema_125) / atr_safe
    d_low = (low - ema_125) / atr_safe
    d_close = (close - ema_125) / atr_safe
    envelope_width = d_high - d_low

    slope_d_high = rolling_theil_sen(d_high, config.slope_window)
    slope_d_close = rolling_theil_sen(d_close, config.slope_window)
    slope_envelope_width = rolling_theil_sen(envelope_width, config.slope_window)
    ema_state_code, ema_state = classify_ema_state(d_close, slope_d_high, slope_d_close)

    return data.with_columns(
        pl.Series("ema_125", ema_125),
        pl.Series("atr_14", atr_14),
        pl.Series("d_open", d_open),
        pl.Series("d_high", d_high),
        pl.Series("d_low", d_low),
        pl.Series("d_close", d_close),
        pl.Series("envelope_width", envelope_width),
        pl.Series("slope_d_high", slope_d_high),
        pl.Series("slope_d_close", slope_d_close),
        pl.Series("slope_envelope_width", slope_envelope_width),
        pl.Series("ema_state_code", ema_state_code),
        pl.Series("ema_state", ema_state),
    )

