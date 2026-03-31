from __future__ import annotations

import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import rolling_r_squared


def add_trend_coherence_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    close = frame["close"].to_numpy()
    columns: list[pl.Series] = []
    for window in config.trend_windows:
        columns.append(pl.Series(f"ols_r2_{window}", rolling_r_squared(close, window)))
    return frame.with_columns(columns)

