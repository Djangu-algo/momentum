from __future__ import annotations

import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import rolling_hurst_rs


def add_hurst_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    hurst = rolling_hurst_rs(frame["close"].to_numpy(), config.hurst_window, config.hurst_lags)
    return frame.with_columns(pl.Series(f"hurst_{config.hurst_window}", hurst))

