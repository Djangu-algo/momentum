from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import rolling_percentile_rank, safe_mean


def add_inflection_score(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    curvature = _get_normalized_or_rank(frame, "norm_curvature_c_30_z", "curvature_c_30_z", config.percentile_window)
    delta_ts = _get_normalized_or_rank(frame, "norm_delta_ts_15_5", "delta_ts_15_5", config.percentile_window)
    score = safe_mean([curvature, delta_ts])
    return frame.with_columns(pl.Series("inflection_score", score))


def _get_normalized_or_rank(
    frame: pl.DataFrame,
    normalized_column: str,
    raw_column: str,
    percentile_window: int,
) -> np.ndarray:
    if normalized_column in frame.columns:
        return frame[normalized_column].to_numpy()
    if raw_column in frame.columns:
        return rolling_percentile_rank(frame[raw_column].to_numpy(), percentile_window)
    return np.full(frame.height, np.nan, dtype=float)

