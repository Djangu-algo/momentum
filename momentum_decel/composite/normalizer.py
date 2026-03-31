from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.composite.state_machine import STATE_SCORE_MAP
from momentum_decel.utils import normalize_hurst, rolling_percentile_rank


def percentile_rank_series(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_percentile_rank(values, window)


def state_score_series(values: np.ndarray) -> np.ndarray:
    mapper = np.vectorize(lambda value: STATE_SCORE_MAP.get(int(value), np.nan) if not np.isnan(value) else np.nan)
    return mapper(values.astype(float))


def normalize_frame(frame: pl.DataFrame, percentile_window: int) -> pl.DataFrame:
    normalized = frame
    if "slope_d_high" in frame.columns:
        normalized = normalized.with_columns(
            pl.Series("norm_slope_d_high", percentile_rank_series(frame["slope_d_high"].to_numpy(), percentile_window))
        )
    if "ema_state_code" in frame.columns:
        normalized = normalized.with_columns(
            pl.Series("norm_ema_state", state_score_series(frame["ema_state_code"].to_numpy()))
        )
    for column in ("ols_r2_20", "ols_r2_40", "er_15", "er_21"):
        if column in frame.columns:
            normalized = normalized.with_columns(pl.col(column).alias(f"norm_{column}"))
    for column in ("curvature_c_30_z", "curvature_c_40_z", "ts_slope_15", "ts_slope_20", "delta_ts_15_5", "delta_ts_20_5"):
        if column in frame.columns:
            normalized = normalized.with_columns(
                pl.Series(f"norm_{column}", percentile_rank_series(frame[column].to_numpy(), percentile_window))
            )
    for column in frame.columns:
        if column.startswith("hurst_"):
            normalized = normalized.with_columns(pl.Series(f"norm_{column}", normalize_hurst(frame[column].to_numpy())))
    return normalized

