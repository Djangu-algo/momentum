from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import clip01, rolling_min, safe_mean


def add_recovery_score(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    d_close = frame["d_close"].to_numpy()
    recent_damage = rolling_min(d_close, 60)
    damage_context = np.where(
        np.isnan(recent_damage),
        np.nan,
        np.where(recent_damage < 0.0, 1.0, np.where(recent_damage < 0.75, 0.6, 0.25)),
    )
    close_repair = clip01((d_close + 1.5) / 3.0)
    components = [
        _column_or_nan(frame, "inflection_score"),
        _column_or_nan(frame, "norm_er_15"),
        _column_or_nan(frame, "norm_ema_state"),
        close_repair,
    ]
    core = safe_mean(components)
    score = core * damage_context
    return frame.with_columns(
        pl.Series("recovery_context", damage_context),
        pl.Series("recovery_score", score),
    )


def _column_or_nan(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.full(frame.height, np.nan, dtype=float)

