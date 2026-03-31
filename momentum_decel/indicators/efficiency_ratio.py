from __future__ import annotations

import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import lag_delta, rolling_efficiency_ratio


def add_efficiency_ratio_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    close = frame["close"].to_numpy()
    columns: list[pl.Series] = []
    for window in config.efficiency_windows:
        er = rolling_efficiency_ratio(close, window)
        columns.append(pl.Series(f"er_{window}", er))
        for lookback in config.efficiency_delta_lookbacks:
            columns.append(pl.Series(f"delta_er_{window}_{lookback}", lag_delta(er, lookback)))
    return frame.with_columns(columns)

