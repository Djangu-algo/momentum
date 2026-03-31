from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.utils import safe_mean


def add_leadership_score(frame: pl.DataFrame) -> pl.DataFrame:
    arrays = [
        _column(frame, "momentum_quality"),
        _column(frame, "rel_strength_score"),
        _column(frame, "inflection_score"),
        _column(frame, "recovery_score"),
        1.0 - _column(frame, "flattening_score"),
    ]
    score = safe_mean(arrays)
    return frame.with_columns(pl.Series("leadership_score", score))


def _column(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.full(frame.height, np.nan, dtype=float)

