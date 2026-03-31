# Output Reference

This document explains the values produced by the CLI, how they are calculated, and how to interpret them.

## Conventions

- `t` means the current row or day.
- `N` means the rolling window length embedded in the column name.
- `K` means the lag or delta lookback embedded in the column name.
- Distances are generally expressed in ATR units, so `1.0` means "one ATR away from EMA125".
- Many rolling fields are `null`/`NaN` during warmup because not enough history exists yet.

## Snapshot Output

The `snapshot` command prints one row per ticker from the latest available date.

Columns:

- `Ticker`: symbol, for example `SPY` or `XLK`.
- `State`: categorical EMA-distance momentum state.
  - `ACCELERATING`: `slope_d_high > 0` and `slope_d_close > 0`
  - `DECELERATING`: `slope_d_high < 0` and `slope_d_close > 0`
  - `MOMENTUM_LOST`: `slope_d_high < 0` and `slope_d_close < 0`
  - `TREND_BROKEN`: `d_close < 0`
  - `UNKNOWN`: not enough history to classify
- `slope_d_hi`: Theil-Sen slope of `d_high` over the configured slope window.
- `slope_d_cl`: Theil-Sen slope of `d_close` over the configured slope window.
- `R2_20`: rolling `ols_r2_20`.
- `ER_15`: rolling `er_15`.
- `Composite`: `momentum_quality`.
- `Delta_1d`: one-day change in `momentum_quality`.
- `Delta_5d`: five-day change in `momentum_quality`.

Meaning:

- More positive `slope_d_hi` and `slope_d_cl` means price extremes are still expanding away from the trend.
- Falling `R2_20` means the uptrend is becoming less coherent.
- Falling `ER_15` means price is getting choppier.
- Lower `Composite` means lower trend quality and a higher probability of momentum exhaustion.

## `output/indicators/{ticker}.parquet`

This is the main per-ticker output. It contains raw OHLCV, all indicator fields, normalized fields, and the composite score.

### Base Market Data

- `date`: trading date.
- `ticker`: ticker symbol.
- `open`, `high`, `low`, `close`, `volume`: raw daily OHLCV values loaded from the source.

### EMA Distance / State Machine

- `ema_125`: exponential moving average of `close` using the configured EMA length.
  - Calculation: standard EMA with smoothing factor `2 / (length + 1)`.
  - Meaning: the trend anchor.
- `atr_14`: Average True Range using the configured ATR length.
  - Calculation: Wilder-style ATR from true range.
  - Meaning: volatility scale used to normalize distances.
- `d_open`: `(open - ema_125) / atr_14`
- `d_high`: `(high - ema_125) / atr_14`
- `d_low`: `(low - ema_125) / atr_14`
- `d_close`: `(close - ema_125) / atr_14`
  - Meaning: where each price component sits relative to the EMA, scaled by ATR.
  - Positive means above EMA, negative means below EMA.
- `envelope_width`: `d_high - d_low`
  - Meaning: ATR-normalized intraday range around the EMA anchor.
- `slope_d_high`: Theil-Sen slope of `d_high` over the configured slope window.
  - Meaning: whether daily highs are extending farther from the EMA or fading back toward it.
- `slope_d_close`: Theil-Sen slope of `d_close` over the configured slope window.
  - Meaning: whether closes are still pushing away from the EMA or flattening.
- `slope_envelope_width`: Theil-Sen slope of `envelope_width` over the configured slope window.
  - Meaning: whether the price envelope is expanding or compressing.
- `ema_state_code`: numeric state code.
  - `3 = ACCELERATING`
  - `2 = DECELERATING`
  - `1 = MOMENTUM_LOST`
  - `0 = TREND_BROKEN`
- `ema_state`: text version of `ema_state_code`.

Interpretation:

- `DECELERATING` is the early warning state: highs are fading but closes are still above the EMA.
- `MOMENTUM_LOST` is a later warning: both highs and closes are fading.
- `TREND_BROKEN` means the close has already fallen below the EMA anchor.

### Trend Coherence

- `ols_r2_20`
- `ols_r2_40`

Calculation:

- Rolling `R^2` of log-close against time over 20-day and 40-day windows.
- Implemented as rolling correlation squared, which is equivalent to `R^2` in simple linear regression.

Meaning:

- Near `1.0`: price closely follows a straight trend.
- Near `0.0`: price path is noisy or non-linear.
- Falling `R^2` while price is still above EMA can indicate trend deterioration before a breakdown.

### Efficiency Ratio

- `er_15`
- `er_21`

Calculation:

- `ER(N) = abs(close[t] - close[t-N]) / sum(abs(diff(close))) over the last N bars`

Meaning:

- `1.0`: very directional move.
- `0.0`: all path, little net progress.

Delta fields:

- `delta_er_15_5`
- `delta_er_15_10`
- `delta_er_21_5`
- `delta_er_21_10`

Calculation:

- `delta_er_N_K = er_N[t] - er_N[t-K]`

Meaning:

- Positive: efficiency is improving.
- Negative: trend is getting choppier.

### Quadratic Curvature

- `curvature_c_30`
- `curvature_c_40`

Calculation:

- Fit `y = a + bt + ct^2` over the rolling window.
- Store `c`, the quadratic coefficient.

Meaning:

- In an uptrend, negative `c` suggests concavity and deceleration.
- Positive `c` suggests convexity and acceleration.

Smoothed fields:

- `curvature_c_30_ema5`
- `curvature_c_40_ema5`

Calculation:

- 5-day EMA of the raw curvature coefficient.

Meaning:

- Less noisy curvature signal for trend-change inspection.

Z-score fields:

- `curvature_c_30_z`
- `curvature_c_40_z`

Calculation:

- Rolling z-score of the smoothed curvature series over the 252-day normalization window.

Meaning:

- Negative: curvature is weak versus its own trailing history.
- Positive: curvature is strong versus its own trailing history.

### Theil-Sen Slope and Delta

- `ts_slope_15`
- `ts_slope_20`

Calculation:

- Theil-Sen slope of `close` over the given rolling window.
- Robust to outliers because it uses the median of pairwise slopes.

Meaning:

- Positive: uptrend slope.
- Negative: downtrend slope.
- Smaller positive values can still indicate flattening.

Fast variants:

- `ts_slope_fast_15`
- `ts_slope_fast_20`

Calculation:

- Same concept as `ts_slope_*`, but computed with a direct pairwise-slope median implementation.

Meaning:

- Intended as a faster implementation check and a practical reference for scaling.

Delta fields:

- `delta_ts_15_5`
- `delta_ts_15_10`
- `delta_ts_20_5`
- `delta_ts_20_10`

Calculation:

- `delta_ts_N_K = ts_slope_N[t] - ts_slope_N[t-K]`

Meaning:

- Negative: slope is flattening or worsening.
- More negative: faster deceleration.

### Hurst Exponent

- `hurst_80`

Calculation:

- Rolling Hurst exponent over an 80-day window using an R/S-style estimate across lags.

Meaning:

- Above `0.5`: persistent / trending behavior.
- Near `0.5`: random-walk-like behavior.
- Below `0.5`: mean-reverting behavior.

## Normalized Columns

These columns convert raw indicators into comparable 0-to-1 style values.

### Directly Used in the Current Composite

- `norm_slope_d_high`: percentile rank of `slope_d_high` over the 252-day window.
- `norm_ema_state`: mapped from state code:
  - `ACCELERATING -> 1.0`
  - `DECELERATING -> 0.66`
  - `MOMENTUM_LOST -> 0.33`
  - `TREND_BROKEN -> 0.0`
- `norm_ols_r2_20`: direct copy of `ols_r2_20`.
- `norm_er_15`: direct copy of `er_15`.
- `norm_curvature_c_30_z`: percentile rank of `curvature_c_30_z` over the 252-day window.
- `norm_ts_slope_15`: percentile rank of `ts_slope_15` over the 252-day window.
- `norm_delta_ts_15_5`: percentile rank of `delta_ts_15_5` over the 252-day window.
- `norm_hurst_80`: rescaled Hurst value using `(hurst_80 - 0.5) * 2`, clipped to `[0, 1]`.

### Additional Normalized Fields Currently Written But Not Used in the Composite

- `norm_ols_r2_40`
- `norm_er_21`
- `norm_curvature_c_40_z`
- `norm_ts_slope_20`
- `norm_delta_ts_20_5`

These are useful for comparison and future model refinement, but the current composite only uses the eight fields listed above.

## Relative Strength Output Versus SPY

These columns are calculated from the ratio series `ETF_close / SPY_close`.

Base ratio fields:

- `rel_close`: `close / spy_close`
- `rel_ema_125`: EMA125 of `rel_close`
- `rel_d_close`: `(rel_close - rel_ema_125) / rel_ema_125`

Meaning:

- Rising `rel_close` means the ticker is outperforming `SPY`.
- Falling `rel_close` means the ticker is underperforming `SPY`.
- Positive `rel_d_close` means the ratio is above its own relative EMA trend.

Relative trend-quality fields:

- `rel_ols_r2_20`: rolling `R^2` of the ratio series
- `rel_er_15`: efficiency ratio of the ratio series
- `rel_curvature_c_30`: quadratic curvature coefficient on the ratio series
- `rel_curvature_c_30_ema5`: 5-day EMA of relative curvature
- `rel_curvature_c_30_z`: rolling z-score of smoothed relative curvature
- `rel_ts_slope_15`: Theil-Sen slope of the ratio series
- `rel_delta_ts_15_5`: `rel_ts_slope_15[t] - rel_ts_slope_15[t-5]`

Meaning:

- Higher `rel_er_15` means cleaner relative outperformance.
- Negative `rel_curvature_c_30_z` means the relative trend is becoming more concave / decelerating.
- Negative `rel_delta_ts_15_5` means relative slope is flattening or worsening.

Relative normalized fields:

- `rel_norm_d_close`
- `rel_norm_ols_r2_20`
- `rel_norm_er_15`
- `rel_norm_curvature_c_30_z`
- `rel_norm_ts_slope_15`
- `rel_norm_delta_ts_15_5`

Relative composite:

- `rel_strength_score`: equal-weight mean of the six normalized relative fields above

Important special case for `SPY`:

- The relative-strength benchmark is `SPY` itself.
- So for `SPY`, the ratio series is intentionally a placeholder:
  - `rel_close = 1.0`
  - `rel_d_close = 0.0`
  - `rel_er_15 = 0.5`
  - `rel_delta_ts_15_5 = 0.0`
- Therefore the relative-strength row in the `SPY` dashboard is expected to look flat.

## Composite Output

- `momentum_quality`

Calculation:

- Equal-weight mean of the currently selected normalized inputs:
  - `norm_slope_d_high`
  - `norm_ema_state`
  - `norm_ols_r2_20`
  - `norm_er_15`
  - `norm_curvature_c_30_z`
  - `norm_ts_slope_15`
  - `norm_delta_ts_15_5`
  - `norm_hurst_80`

Meaning:

- Near `1.0`: strong, coherent, persistent momentum.
- Near `0.0`: weak, broken, or decaying momentum.
- It is a trend-quality score, not a direct return forecast.

Delta fields:

- `momentum_quality_delta_1d`: `momentum_quality[t] - momentum_quality[t-1]`
- `momentum_quality_delta_5d`: `momentum_quality[t] - momentum_quality[t-5]`

Meaning:

- Positive: composite is improving.
- Negative: composite is deteriorating.

## `output/charts/{ticker}_dashboard.html`

This is the interactive Plotly dashboard for one ticker.

Panels:

- Panel 1: candlestick price with `ema_125`
- Panel 2: ATR-normalized OHLC distance
  - `d_high`, `d_low`, `d_open`, `d_close`
- Panel 3: spacer row for readability
- Panel 4: `slope_d_high`
- Panel 5: `slope_d_close`, `ema_state_code`, `advanced_state_code`
- Panel 6: `ols_r2_20`, `er_15`
- Panel 7: `curvature_c_30_z`, `delta_ts_15_5`
- Panel 8: relative-strength row versus `SPY`
  - `rel_d_close`, `rel_er_15`, `rel_delta_ts_15_5`
- Panel 9: framework scores
  - `momentum_quality`, `inflection_score`, `recovery_score`, `flattening_score`, `leadership_score`
- Panel 10: validation risk / recovery hit-rate row
  - `DD8/20`, `DD8/40`, `DD10/40`, `UP10/40`

Meaning:

- This is the best output for visually examining how the indicators line up through time.
- The HTML graph supports zooming, panning, a range slider, and range selector buttons.
- The relative-strength row is only informative for non-benchmark tickers.
- For `SPY`, the relative-strength row is intentionally flat because it is measured versus itself.
- The bottom validation row uses historical study hit rates rather than raw indicator values.

## `output/charts/sector_heatmap.html`

This is the interactive cross-ticker heatmap of `momentum_quality`.

Axes:

- X-axis: date
- Y-axis: ticker
- Cell value: `momentum_quality`

Meaning:

- Useful for comparing sector deterioration or improvement across the full universe.
- Lower values show up toward the red end of the color scale.

## `output/breadth/decel_states.parquet`

This file contains the per-stock breadth-state detail for the S&P 500 breadth run.

Columns:

- `date`: trading date.
- `ticker`: stock symbol.
- `d_high`: `(high - ema_125) / atr_14` for that stock.
- `d_close`: `(close - ema_125) / atr_14` for that stock.
- `slope_d_high`: fast Theil-Sen slope of `d_high`.
- `ema_state_code`: stock-level state code.
- `ema_state`: stock-level state label.

Meaning:

- This is the raw per-name breadth detail before aggregation.

## `output/breadth/decel_breadth.parquet`

This file aggregates the breadth-state detail by day.

Columns:

- `date`: trading date.
- `n_stocks`: number of stocks included that day.
- `median_slope_d_high`: median `slope_d_high` across the universe.
- `TREND_BROKEN`: count of stocks in `TREND_BROKEN`.
- `UNKNOWN`: count of stocks with insufficient history to classify.
- `ACCELERATING`: count of stocks in `ACCELERATING`.
- `DECELERATING`: count of stocks in `DECELERATING`.
- `MOMENTUM_LOST`: count of stocks in `MOMENTUM_LOST`.
- `pct_accelerating`: `ACCELERATING / n_stocks`
- `pct_decelerating`: `DECELERATING / n_stocks`
- `pct_momentum_lost`: `MOMENTUM_LOST / n_stocks`
- `pct_trend_broken`: `TREND_BROKEN / n_stocks`
- `decel_breadth`: `(DECELERATING + MOMENTUM_LOST + TREND_BROKEN) / n_stocks`

Meaning:

- `decel_breadth` is the main breadth deterioration measure in the current implementation.
- Higher values mean a larger share of the universe is no longer in a healthy accelerating state.
- `UNKNOWN` usually appears during early warmup periods.

## `output/validation/event_study_results.parquet`

This file contains event-level detail for the SPY event study.

Columns:

- `indicator`: indicator being evaluated.
- `peak_date`: detected pre-drawdown peak date.
- `trough_date`: subsequent trough date for the drawdown event.
- `drawdown_pct`: peak-to-trough drawdown size in percent.
- `warning_date`: earliest warning date found in the configured lookback window.
- `lead_days`: `peak_date - warning_date` in days.

Meaning:

- Larger `lead_days` is better if the warnings are still specific enough.
- Missing `warning_date` / `lead_days` means that indicator never warned in the lookback window before that peak.

## `output/validation/event_study_summary.csv`

This is the summary table across all detected drawdown events.

Columns:

- `indicator`: indicator being summarized.
- `events`: number of drawdown events examined.
- `median_lead_days`: median warning lead time across events.
- `false_positive_rate`: fraction of warning days not followed by a drawdown peak within the warning horizon.
- `signal_to_noise`: `median_lead_days / false_positive_rate` when defined.

Meaning:

- Higher `median_lead_days` is better.
- Lower `false_positive_rate` is better.
- Higher `signal_to_noise` is better, but it is only a heuristic ranking metric.

## `output/validation/sector_lead_lag.parquet`

This file compares sector composite warnings against SPY around SPY drawdown peaks.

Columns:

- `peak_date`: SPY peak date for the event.
- `ticker`: sector ETF ticker.
- `composite_lead_days`: days between the sector's first composite warning and the SPY peak.
- `lead_vs_spy`: `composite_lead_days - spy_composite_lead_days`

Meaning:

- Positive `lead_vs_spy`: the sector warned earlier than SPY.
- Negative `lead_vs_spy`: the sector warned later than SPY.
- Useful for lead/lag and sector-rotation analysis.

## Warmup and Missing Values

`NaN` or missing values usually mean one of these:

- not enough history yet for the rolling calculation
- ATR was zero or undefined
- the underlying series itself was undefined at that point

This is expected at the start of every series and should not be treated as a signal by itself.
