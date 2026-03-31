from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import clip01, rolling_percentile_rank, safe_mean


def add_flattening_score(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    d_close = frame["d_close"].to_numpy()
    trend_alive = 0.4 + 0.6 * clip01((d_close + 1.0) / 2.5)
    compression_score = np.full(frame.height, np.nan, dtype=float)
    if "slope_envelope_width" in frame.columns:
        compression_score = rolling_percentile_rank(-frame["slope_envelope_width"].to_numpy(), config.percentile_window)

    weakening_components = [
        1.0 - _column_or_half(frame, "norm_delta_ts_15_5"),
        1.0 - _column_or_half(frame, "norm_curvature_c_30_z"),
        1.0 - _column_or_half(frame, "norm_slope_d_high"),
        1.0 - _column_or_half(frame, "norm_er_15"),
        compression_score,
    ]
    core = safe_mean(weakening_components)
    score = clip01(core * trend_alive)
    return frame.with_columns(
        pl.Series("flattening_compression_score", compression_score),
        pl.Series("flattening_score", score),
    )


def _column_or_half(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.full(frame.height, 0.5, dtype=float)

