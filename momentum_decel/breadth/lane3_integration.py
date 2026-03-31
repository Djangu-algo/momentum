from __future__ import annotations

import numpy as np
import polars as pl


def add_lane3_signal(frame: pl.DataFrame, er_flatness_column: str = "er_flatness_pct") -> pl.DataFrame:
    if er_flatness_column not in frame.columns:
        return frame
    median_slope = frame["median_slope_d_high"].to_numpy()
    slope_component = np.clip(-median_slope, 0.0, None)
    if np.nanmax(slope_component) > 0.0:
        slope_component = slope_component / np.nanmax(slope_component)
    signal = (
        frame[er_flatness_column].to_numpy() * 0.4
        + frame["decel_breadth"].to_numpy() * 0.4
        + slope_component * 0.2
    )
    return frame.with_columns(pl.Series("lane3_signal", signal))

