from __future__ import annotations

import numpy as np
import polars as pl


ADVANCED_STATE_CODES = {
    "BROKEN": 0,
    "REPAIRING": 1,
    "RECOVERING": 2,
    "ACCELERATING": 3,
    "FLATTENING": 4,
    "FLATLINING": 5,
}


def add_advanced_state(frame: pl.DataFrame) -> pl.DataFrame:
    rows = frame.height
    d_close = _column(frame, "d_close")
    er_15 = _column(frame, "er_15")
    ts_slope_15 = _column(frame, "ts_slope_15")
    delta_ts_15_5 = _column(frame, "delta_ts_15_5")
    inflection_score = _column(frame, "inflection_score")
    recovery_score = _column(frame, "recovery_score")
    flattening_score = _column(frame, "flattening_score")

    codes = np.full(rows, np.nan, dtype=float)
    labels = ["UNKNOWN"] * rows
    days_in_state = np.full(rows, np.nan, dtype=float)
    prior_label = "UNKNOWN"
    streak = 0

    for idx in range(rows):
        if np.isnan(d_close[idx]) or np.isnan(delta_ts_15_5[idx]):
            continue

        if abs(ts_slope_15[idx]) < 0.02 and er_15[idx] < 0.25 and abs(delta_ts_15_5[idx]) < 0.05:
            label = "FLATLINING"
        elif d_close[idx] < 0.0 and delta_ts_15_5[idx] < 0.0 and inflection_score[idx] < 0.55:
            label = "BROKEN"
        elif d_close[idx] < 0.0 and inflection_score[idx] >= 0.60:
            label = "REPAIRING"
        elif d_close[idx] >= 0.0 and flattening_score[idx] >= 0.65 and delta_ts_15_5[idx] < 0.0:
            label = "FLATTENING"
        elif d_close[idx] >= 0.0 and recovery_score[idx] >= 0.65 and inflection_score[idx] >= 0.60:
            label = "RECOVERING"
        elif d_close[idx] >= 0.0 and inflection_score[idx] >= 0.65 and delta_ts_15_5[idx] > 0.0 and er_15[idx] > 0.35:
            label = "ACCELERATING"
        elif d_close[idx] < 0.0:
            label = "BROKEN"
        else:
            label = "FLATTENING"

        codes[idx] = ADVANCED_STATE_CODES[label]
        labels[idx] = label
        if label == prior_label:
            streak += 1
        else:
            streak = 1
        days_in_state[idx] = streak
        prior_label = label

    return frame.with_columns(
        pl.Series("advanced_state_code", codes),
        pl.Series("advanced_state", labels),
        pl.Series("days_in_advanced_state", days_in_state),
    )


def _column(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.full(frame.height, np.nan, dtype=float)

