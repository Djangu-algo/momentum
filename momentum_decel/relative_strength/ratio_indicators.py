from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.utils import ema, lag_delta, rolling_efficiency_ratio, rolling_percentile_rank, rolling_quadratic_coefficient, rolling_r_squared, rolling_theil_sen, rolling_zscore, safe_mean


def add_relative_strength_features(frame: pl.DataFrame, config: IndicatorConfig, benchmark: str = "SPY") -> pl.DataFrame:
    if frame.is_empty() or benchmark not in frame["ticker"].unique().to_list():
        return frame

    benchmark_frame = (
        frame.filter(pl.col("ticker") == benchmark)
        .select("date", pl.col("close").alias("benchmark_close"))
        .sort("date")
    )

    results: list[pl.DataFrame] = []
    for ticker in frame["ticker"].unique().to_list():
        ticker_frame = frame.filter(pl.col("ticker") == ticker).sort("date")
        if ticker == benchmark:
            results.append(_add_benchmark_placeholders(ticker_frame))
            continue

        joined = ticker_frame.join(benchmark_frame, on="date", how="left")
        benchmark_close = joined["benchmark_close"].to_numpy()
        close = joined["close"].to_numpy()
        rel_close = np.where((benchmark_close == 0.0) | np.isnan(benchmark_close), np.nan, close / benchmark_close)

        rel_ema_125 = ema(rel_close, config.ema_length)
        rel_d_close = np.where(rel_ema_125 == 0.0, np.nan, (rel_close - rel_ema_125) / rel_ema_125)
        rel_ols_r2_20 = rolling_r_squared(rel_close, 20)
        rel_er_15 = rolling_efficiency_ratio(rel_close, 15)
        rel_curvature_c_30 = rolling_quadratic_coefficient(rel_close, 30)
        rel_curvature_c_30_ema5 = ema(rel_curvature_c_30, config.curvature_smoothing)
        rel_curvature_c_30_z = rolling_zscore(rel_curvature_c_30_ema5, config.percentile_window)
        rel_ts_slope_15 = rolling_theil_sen(rel_close, 15)
        rel_delta_ts_15_5 = lag_delta(rel_ts_slope_15, 5)

        rel_norm_d_close = rolling_percentile_rank(rel_d_close, config.percentile_window)
        rel_norm_ols_r2_20 = rel_ols_r2_20
        rel_norm_er_15 = rel_er_15
        rel_norm_curvature_c_30_z = rolling_percentile_rank(rel_curvature_c_30_z, config.percentile_window)
        rel_norm_ts_slope_15 = rolling_percentile_rank(rel_ts_slope_15, config.percentile_window)
        rel_norm_delta_ts_15_5 = rolling_percentile_rank(rel_delta_ts_15_5, config.percentile_window)
        rel_strength_score = safe_mean(
            [
                rel_norm_d_close,
                rel_norm_ols_r2_20,
                rel_norm_er_15,
                rel_norm_curvature_c_30_z,
                rel_norm_ts_slope_15,
                rel_norm_delta_ts_15_5,
            ]
        )

        results.append(
            joined.drop("benchmark_close").with_columns(
                pl.Series("rel_close", rel_close),
                pl.Series("rel_ema_125", rel_ema_125),
                pl.Series("rel_d_close", rel_d_close),
                pl.Series("rel_ols_r2_20", rel_ols_r2_20),
                pl.Series("rel_er_15", rel_er_15),
                pl.Series("rel_curvature_c_30", rel_curvature_c_30),
                pl.Series("rel_curvature_c_30_ema5", rel_curvature_c_30_ema5),
                pl.Series("rel_curvature_c_30_z", rel_curvature_c_30_z),
                pl.Series("rel_ts_slope_15", rel_ts_slope_15),
                pl.Series("rel_delta_ts_15_5", rel_delta_ts_15_5),
                pl.Series("rel_norm_d_close", rel_norm_d_close),
                pl.Series("rel_norm_ols_r2_20", rel_norm_ols_r2_20),
                pl.Series("rel_norm_er_15", rel_norm_er_15),
                pl.Series("rel_norm_curvature_c_30_z", rel_norm_curvature_c_30_z),
                pl.Series("rel_norm_ts_slope_15", rel_norm_ts_slope_15),
                pl.Series("rel_norm_delta_ts_15_5", rel_norm_delta_ts_15_5),
                pl.Series("rel_strength_score", rel_strength_score),
            )
        )

    return pl.concat(results).sort(["ticker", "date"])


def _add_benchmark_placeholders(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(1.0).cast(pl.Float64).alias("rel_close"),
        pl.lit(1.0).cast(pl.Float64).alias("rel_ema_125"),
        pl.lit(0.0).cast(pl.Float64).alias("rel_d_close"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_ols_r2_20"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_er_15"),
        pl.lit(0.0).cast(pl.Float64).alias("rel_curvature_c_30"),
        pl.lit(0.0).cast(pl.Float64).alias("rel_curvature_c_30_ema5"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_curvature_c_30_z"),
        pl.lit(0.0).cast(pl.Float64).alias("rel_ts_slope_15"),
        pl.lit(0.0).cast(pl.Float64).alias("rel_delta_ts_15_5"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_norm_d_close"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_norm_ols_r2_20"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_norm_er_15"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_norm_curvature_c_30_z"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_norm_ts_slope_15"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_norm_delta_ts_15_5"),
        pl.lit(0.5).cast(pl.Float64).alias("rel_strength_score"),
    )

