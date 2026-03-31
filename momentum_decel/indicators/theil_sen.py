from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import lag_delta, rolling_theil_sen


def rolling_theil_sen_fast(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < window:
        return out

    x = np.arange(window, dtype=float)
    left, right = np.triu_indices(window, k=1)
    denom = x[right] - x[left]

    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        if np.isnan(win).any():
            continue
        slopes = (win[right] - win[left]) / denom
        out[idx] = float(np.median(slopes))
    return out


def add_theil_sen_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    close = frame["close"].to_numpy()
    columns: list[pl.Series] = []
    for window in config.theil_sen_windows:
        slope = rolling_theil_sen(close, window)
        columns.append(pl.Series(f"ts_slope_{window}", slope))
        columns.append(pl.Series(f"ts_slope_fast_{window}", rolling_theil_sen_fast(close, window)))
        for lookback in config.theil_sen_delta_lookbacks:
            columns.append(pl.Series(f"delta_ts_{window}_{lookback}", lag_delta(slope, lookback)))
    return frame.with_columns(columns)

