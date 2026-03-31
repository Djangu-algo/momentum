from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.indicators.ema_distance import classify_ema_state
from momentum_decel.indicators.theil_sen import rolling_theil_sen_fast
from momentum_decel.utils import average_true_range, ema


def compute_universe_deceleration(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for ticker in frame["ticker"].unique().to_list():
        ticker_frame = frame.filter(pl.col("ticker") == ticker).sort("date")
        high = ticker_frame["high"].to_numpy()
        low = ticker_frame["low"].to_numpy()
        close = ticker_frame["close"].to_numpy()
        ema_125 = ema(close, config.ema_length)
        atr_14 = average_true_range(high, low, close, config.atr_length)
        atr_safe = np.where((atr_14 == 0.0) | np.isnan(atr_14), np.nan, atr_14)
        d_high = (high - ema_125) / atr_safe
        d_close = (close - ema_125) / atr_safe
        slope_d_high = rolling_theil_sen_fast(d_high, config.slope_window)
        slope_d_close = rolling_theil_sen_fast(d_close, config.slope_window)
        codes, labels = classify_ema_state(d_close, slope_d_high, slope_d_close)
        rows.append(
            ticker_frame.select("date", "ticker").with_columns(
                pl.Series("d_high", d_high),
                pl.Series("d_close", d_close),
                pl.Series("slope_d_high", slope_d_high),
                pl.Series("ema_state_code", codes),
                pl.Series("ema_state", labels),
            )
        )
    return pl.concat(rows).sort(["date", "ticker"]) if rows else pl.DataFrame()


def aggregate_breadth(frame: pl.DataFrame) -> pl.DataFrame:
    totals = frame.group_by("date").agg(pl.len().alias("n_stocks"), pl.median("slope_d_high").alias("median_slope_d_high"))
    counts = frame.group_by(["date", "ema_state"]).agg(pl.len().alias("count"))
    pivot = counts.pivot(index="date", on="ema_state", values="count").fill_null(0)
    breadth = totals.join(pivot, on="date", how="left").fill_null(0)
    for state in ("ACCELERATING", "DECELERATING", "MOMENTUM_LOST", "TREND_BROKEN"):
        if state not in breadth.columns:
            breadth = breadth.with_columns(pl.lit(0).alias(state))
    return breadth.with_columns(
        (pl.col("ACCELERATING") / pl.col("n_stocks")).alias("pct_accelerating"),
        (pl.col("DECELERATING") / pl.col("n_stocks")).alias("pct_decelerating"),
        (pl.col("MOMENTUM_LOST") / pl.col("n_stocks")).alias("pct_momentum_lost"),
        (pl.col("TREND_BROKEN") / pl.col("n_stocks")).alias("pct_trend_broken"),
        (
            (pl.col("DECELERATING") + pl.col("MOMENTUM_LOST") + pl.col("TREND_BROKEN"))
            / pl.col("n_stocks")
        ).alias("decel_breadth"),
    ).sort("date")

