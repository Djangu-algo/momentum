from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl
from scipy import stats


def validate_ohlcv_frame(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"OHLCV frame missing columns: {missing_str}")
    return frame.sort("date")


def ema(values: Sequence[float], span: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    result = np.full(data.shape, np.nan, dtype=float)
    if span <= 0 or data.size == 0:
        return result

    alpha = 2.0 / (span + 1.0)
    result[0] = data[0]
    for idx in range(1, data.size):
        if math.isnan(data[idx]):
            result[idx] = result[idx - 1]
        elif math.isnan(result[idx - 1]):
            result[idx] = data[idx]
        else:
            result[idx] = alpha * data[idx] + (1.0 - alpha) * result[idx - 1]
    return result


def average_true_range(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    length: int,
) -> np.ndarray:
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    result = np.full(close_arr.shape, np.nan, dtype=float)
    if length <= 0 or close_arr.size == 0:
        return result

    prev_close = np.roll(close_arr, 1)
    prev_close[0] = close_arr[0]
    tr = np.maximum.reduce(
        [
            high_arr - low_arr,
            np.abs(high_arr - prev_close),
            np.abs(low_arr - prev_close),
        ]
    )
    if tr.size < length:
        return result
    result[length - 1] = np.nanmean(tr[:length])
    for idx in range(length, tr.size):
        result[idx] = ((result[idx - 1] * (length - 1)) + tr[idx]) / length
    return result


def rolling_apply(values: Sequence[float], window: int, func) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 0 or arr.size < window:
        return out
    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        if np.isnan(win).any():
            continue
        out[idx] = func(win)
    return out


def rolling_theil_sen(values: Sequence[float], window: int) -> np.ndarray:
    x = np.arange(window, dtype=float)
    return rolling_apply(values, window, lambda win: stats.theilslopes(win, x)[0])


def rolling_r_squared(values: Sequence[float], window: int) -> np.ndarray:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_std = x.std(ddof=0)

    def _calc(win: np.ndarray) -> float:
        log_win = np.log(win)
        y_std = log_win.std(ddof=0)
        if y_std == 0.0:
            return 0.0
        corr = np.mean((x - x_mean) * (log_win - log_win.mean())) / (x_std * y_std)
        return float(np.clip(corr**2, 0.0, 1.0))

    return rolling_apply(values, window, _calc)


def rolling_efficiency_ratio(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 0 or arr.size <= window:
        return out
    for idx in range(window, arr.size):
        win = arr[idx - window : idx + 1]
        if np.isnan(win).any():
            continue
        net = abs(win[-1] - win[0])
        path = np.abs(np.diff(win)).sum()
        out[idx] = 0.0 if path == 0.0 else net / path
    return np.clip(out, 0.0, 1.0)


def lag_delta(values: Sequence[float], lookback: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    out[lookback:] = arr[lookback:] - arr[:-lookback]
    return out


def rolling_quadratic_coefficient(values: Sequence[float], window: int) -> np.ndarray:
    x = np.arange(window, dtype=float)
    return rolling_apply(values, window, lambda win: float(np.polyfit(x, win, 2)[0]))


def rolling_zscore(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 1 or arr.size < window:
        return out
    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        if np.isnan(win).any():
            continue
        std = win.std(ddof=0)
        out[idx] = 0.0 if std == 0.0 else (win[-1] - win.mean()) / std
    return out


def rolling_percentile_rank(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 0 or arr.size < window:
        return out
    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        val = win[-1]
        if np.isnan(win).any():
            continue
        out[idx] = float(np.count_nonzero(win <= val) / window)
    return np.clip(out, 0.0, 1.0)


def rolling_min(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 0 or arr.size < window:
        return out
    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        if np.isnan(win).all():
            continue
        out[idx] = np.nanmin(win)
    return out


def rolling_max(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if window <= 0 or arr.size < window:
        return out
    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        if np.isnan(win).all():
            continue
        out[idx] = np.nanmax(win)
    return out


def clip01(values: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def rolling_hurst_rs(values: Sequence[float], window: int, lags: Iterable[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    lags = tuple(sorted(set(int(lag) for lag in lags if lag > 1)))
    if arr.size < window or not lags:
        return out

    def _rs(chunk: np.ndarray) -> float:
        centered = chunk - chunk.mean()
        cumulative = np.cumsum(centered)
        spread = cumulative.max() - cumulative.min()
        sigma = chunk.std(ddof=0)
        if sigma == 0.0:
            return np.nan
        return float(spread / sigma)

    for idx in range(window - 1, arr.size):
        win = arr[idx - window + 1 : idx + 1]
        if np.isnan(win).any():
            continue
        rs_values: list[float] = []
        lag_values: list[int] = []
        for lag in lags:
            chunk_count = win.size // lag
            if chunk_count < 2:
                continue
            chunk_rs = []
            for chunk_idx in range(chunk_count):
                chunk = win[chunk_idx * lag : (chunk_idx + 1) * lag]
                value = _rs(chunk)
                if not np.isnan(value) and value > 0.0:
                    chunk_rs.append(value)
            if chunk_rs:
                lag_values.append(lag)
                rs_values.append(float(np.mean(chunk_rs)))
        if len(rs_values) < 2:
            continue
        slope, *_ = stats.linregress(np.log(lag_values), np.log(rs_values))
        out[idx] = slope
    return out


def normalize_hurst(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip((arr - 0.5) * 2.0, 0.0, 1.0)


def safe_mean(columns: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(columns)
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(matrix), axis=0)
        summed = np.nansum(matrix, axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=float)
    valid = counts > 0
    result[valid] = summed[valid] / counts[valid]
    return result


def parse_tickers(tickers: Sequence[str] | None, default: Sequence[str]) -> list[str]:
    if not tickers:
        return list(default)
    values = []
    for entry in tickers:
        values.extend(token.strip().upper() for token in entry.split(","))
    return [ticker for ticker in values if ticker]
