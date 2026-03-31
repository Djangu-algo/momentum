from __future__ import annotations

import polars as pl

from momentum_decel.composite.flattening_score import add_flattening_score
from momentum_decel.composite.inflection_score import add_inflection_score
from momentum_decel.composite.normalizer import normalize_frame
from momentum_decel.composite.recovery_score import add_recovery_score
from momentum_decel.composite.state_machine_v2 import add_advanced_state
from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import safe_mean


COMPOSITE_INPUTS = (
    "norm_slope_d_high",
    "norm_ema_state",
    "norm_ols_r2_20",
    "norm_er_15",
    "norm_curvature_c_30_z",
    "norm_ts_slope_15",
    "norm_delta_ts_15_5",
    "norm_hurst_80",
)


def add_composite_score(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    normalized = normalize_frame(frame, config.percentile_window)
    available = [column for column in COMPOSITE_INPUTS if column in normalized.columns]
    if available:
        arrays = [normalized[column].to_numpy() for column in available]
        score = safe_mean(arrays)
        delta_1d = score.copy()
        delta_5d = score.copy()
        delta_1d[:] = float("nan")
        delta_5d[:] = float("nan")
        delta_1d[1:] = score[1:] - score[:-1]
        delta_5d[5:] = score[5:] - score[:-5]
        normalized = normalized.with_columns(
            pl.Series("momentum_quality", score),
            pl.Series("momentum_quality_delta_1d", delta_1d),
            pl.Series("momentum_quality_delta_5d", delta_5d),
        )
    normalized = add_inflection_score(normalized, config)
    normalized = add_recovery_score(normalized, config)
    normalized = add_flattening_score(normalized, config)
    normalized = add_advanced_state(normalized)
    return normalized
