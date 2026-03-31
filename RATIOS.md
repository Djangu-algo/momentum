# Ratio Reference

This document lists the ratio-style values used in the system and what they mean.

## 1. ATR-Normalized EMA Distance

These are not raw price ratios, but they are normalized distance ratios relative to the EMA trend anchor.

- `d_open = (open - ema_125) / atr_14`
- `d_high = (high - ema_125) / atr_14`
- `d_low = (low - ema_125) / atr_14`
- `d_close = (close - ema_125) / atr_14`

Meaning:

- `0.0` means price is exactly at `EMA125`
- `1.0` means price is one ATR above `EMA125`
- `-1.0` means price is one ATR below `EMA125`

Used in:

- state machine
- slope / flattening logic
- dashboard ATR-distance row
- recovery features such as `d_close`

## 2. Envelope Width Ratio

- `envelope_width = d_high - d_low`

Meaning:

- ATR-normalized bar width around the EMA anchor
- higher values mean a wider daily range relative to current volatility

## 3. Efficiency Ratio

Used on raw close:

- `er_15`
- `er_21`

Formula:

- `ER(N) = abs(close[t] - close[t-N]) / sum(abs(diff(close))) over the last N bars`

Meaning:

- `1.0` means very directional travel
- `0.0` means very choppy path with little net progress

## 4. Relative-Strength Price Ratio Versus SPY

Defined in [ratio_indicators.py](/home/denys/momentum/momentum_decel/relative_strength/ratio_indicators.py).

Base ratio:

- `rel_close = ETF_close / SPY_close`

Derived from the ratio series:

- `rel_ema_125 = EMA125(rel_close)`
- `rel_d_close = (rel_close - rel_ema_125) / rel_ema_125`
- `rel_ols_r2_20`
- `rel_er_15`
- `rel_curvature_c_30`
- `rel_curvature_c_30_ema5`
- `rel_curvature_c_30_z`
- `rel_ts_slope_15`
- `rel_delta_ts_15_5`

Meaning:

- `rel_close` rising means the ETF is outperforming `SPY`
- `rel_close` falling means it is underperforming `SPY`
- `rel_d_close > 0` means the ETF/SPY ratio is above its own relative EMA

Important constraint:

- this is currently close-based only
- there is no full OHLC relative-ratio candle set yet

## 5. Normalized Relative Ratio Features

These map relative-strength features into comparable `0..1` style ranks:

- `rel_norm_d_close`
- `rel_norm_ols_r2_20`
- `rel_norm_er_15`
- `rel_norm_curvature_c_30_z`
- `rel_norm_ts_slope_15`
- `rel_norm_delta_ts_15_5`

Composite:

- `rel_strength_score = mean(all 6 relative normalized features)`

Meaning:

- higher values mean stronger and cleaner outperformance versus `SPY`

## 6. Validation Hit-Rate Ratios

These are shown on the bottom dashboard row and come from historical validation studies.

Defined in [single_instrument.py](/home/denys/momentum/momentum_decel/dashboard/single_instrument.py), sourced from:

- [exhaustion_study.py](/home/denys/momentum/momentum_decel/validation/exhaustion_study.py)
- [recovery_study.py](/home/denys/momentum/momentum_decel/validation/recovery_study.py)

Ratios:

- `DD8/20`: probability of an `>= 8%` drawdown within 20 trading days
- `DD8/40`: probability of an `>= 8%` drawdown within 40 trading days
- `DD10/40`: probability of an `>= 10%` drawdown within 40 trading days
- `UP10/40`: probability of achieving `>= 10%` upside within 40 trading days

Meaning:

- higher `DD*` means worse forward downside profile
- higher `UP10/40` means stronger historical recovery / follow-through odds

## 7. Group-Relative Drawdown Bands

These are not single-series indicators. They are threshold ratios estimated from historical forward drawdowns by inferred group.

Examples:

- `group_drawdown_threshold_p50_20d`
- `group_drawdown_threshold_p75_20d`
- `group_drawdown_threshold_p90_20d`
- `group_drawdown_threshold_p50_40d`
- `group_drawdown_threshold_p75_40d`
- `group_drawdown_threshold_p90_40d`

Meaning:

- `p50`: typical downside for that group
- `p75`: stressed downside
- `p90`: severe downside

These are written by:

- `recoverystudy`
- `exhaustionstudy`

and shown in the dashboard annotation when available.

## Summary

The main ratio families are:

- ATR-normalized EMA distance ratios
- efficiency ratio
- ETF/SPY relative-strength ratio
- validation hit-rate ratios
- group-relative drawdown threshold ratios
