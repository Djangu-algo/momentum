from __future__ import annotations

import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import ema, rolling_quadratic_coefficient, rolling_zscore


def add_quadratic_curvature_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    close = frame["close"].to_numpy()
    columns: list[pl.Series] = []
    for window in config.curvature_windows:
        curvature = rolling_quadratic_coefficient(close, window)
        smooth = ema(curvature, config.curvature_smoothing)
        zscore = rolling_zscore(smooth, config.percentile_window)
        columns.extend(
            [
                pl.Series(f"curvature_c_{window}", curvature),
                pl.Series(f"curvature_c_{window}_ema{config.curvature_smoothing}", smooth),
                pl.Series(f"curvature_c_{window}_z", zscore),
            ]
        )
    return frame.with_columns(columns)

