from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.validation.event_study import identify_drawdown_events


RECOVERY_SIGNAL_COLUMNS = (
    "recovery_score",
    "inflection_score",
    "momentum_quality",
    "leadership_score",
    "rel_strength_score",
    "d_close",
    "advanced_state_code",
)

FORWARD_HORIZONS = (5, 10, 20, 40)
RECOVERY_HIT_THRESHOLDS = (0.10, 0.20)


def build_recovery_study(
    frame: pl.DataFrame,
    min_drawdown: float = 5.0,
    lookback_days: int = 20,
    forward_horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if frame.is_empty():
        return _empty_detail_frame(), _empty_summary_frame()

    ordered = frame.sort("date")
    if "close" not in ordered.columns or "date" not in ordered.columns:
        raise ValueError("recovery study requires date and close columns")

    events = identify_drawdown_events(ordered, min_drawdown=min_drawdown)
    if not events:
        return _empty_detail_frame(), _empty_summary_frame()

    close = ordered["close"].to_numpy()
    dates = ordered["date"].to_list()

    detail_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    for signal_name, signal_values in _recovery_signal_series(ordered).items():
        anchor_rows = _build_signal_rows(
            signal_name=signal_name,
            values=signal_values,
            ordered=ordered,
            events=events,
            close=close,
            dates=dates,
            lookback_days=lookback_days,
            forward_horizons=forward_horizons,
        )
        detail_rows.extend(anchor_rows)
        signal_rows.extend(
            _summarize_recovery_signal(
                rows=anchor_rows,
                signal_name=signal_name,
                forward_horizons=forward_horizons,
            )
        )

    return _frame_from_rows(detail_rows, _empty_detail_schema()), _frame_from_rows(
        signal_rows, _empty_summary_schema()
    ).sort(["signal_name", "signal_bucket"], nulls_last=True)


def _recovery_signal_series(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for column in RECOVERY_SIGNAL_COLUMNS:
        if column in frame.columns:
            series[column] = frame[column].to_numpy()
    return series


def _build_signal_rows(
    signal_name: str,
    values: np.ndarray,
    ordered: pl.DataFrame,
    events: list[object],
    close: np.ndarray,
    dates: list[object],
    lookback_days: int,
    forward_horizons: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    buckets = _bucket_values(signal_name, values)
    for event in events:
        trough_idx = int(event.trough_idx)
        start_idx = max(0, trough_idx - lookback_days)
        anchor_window = values[start_idx : trough_idx + 1]
        anchor_positions = np.where(~np.isnan(anchor_window))[0]
        if not anchor_positions.size:
            continue
        anchor_idx = start_idx + int(anchor_positions[-1])
        signal_value = values[anchor_idx]
        if np.isnan(signal_value):
            continue

        row: dict[str, object] = {
            "signal_name": signal_name,
            "signal_bucket": buckets[anchor_idx],
            "event_index": trough_idx,
            "signal_index": anchor_idx,
            "signal_date": dates[anchor_idx],
            "trough_date": event.trough_date,
            "trough_close": float(close[trough_idx]),
            "drawdown_pct": float(event.drawdown_pct),
            "signal_value": _as_python_scalar(signal_value),
        }

        for horizon in forward_horizons:
            row[f"forward_return_{horizon}d"] = _forward_return(close, anchor_idx, horizon)
            row[f"max_forward_return_{horizon}d"] = _max_forward_return(close, anchor_idx, horizon)

        for threshold in RECOVERY_HIT_THRESHOLDS:
            horizon = max(forward_horizons)
            row[f"hit_rate_{int(threshold * 100)}pct_{horizon}d"] = _hit_rate(
                row[f"max_forward_return_{horizon}d"], threshold
            )

        rows.append(row)
    return rows


def _summarize_recovery_signal(
    rows: list[dict[str, object]],
    signal_name: str,
    forward_horizons: tuple[int, ...],
) -> list[dict[str, object]]:
    if not rows:
        return []

    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["signal_bucket"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for bucket, bucket_rows in buckets.items():
        summary: dict[str, object] = {
            "signal_name": signal_name,
            "signal_bucket": bucket,
            "events": len(bucket_rows),
            "median_drawdown_pct": _nan_median([row["drawdown_pct"] for row in bucket_rows]),
            "median_signal_value": _nan_median([row["signal_value"] for row in bucket_rows]),
        }

        for horizon in forward_horizons:
            summary[f"median_forward_return_{horizon}d"] = _nan_median(
                [row[f"forward_return_{horizon}d"] for row in bucket_rows]
            )
            summary[f"win_rate_{horizon}d"] = _nan_mean(
                [row[f"forward_return_{horizon}d"] > 0.0 for row in bucket_rows]
            )
            summary[f"median_max_forward_return_{horizon}d"] = _nan_median(
                [row[f"max_forward_return_{horizon}d"] for row in bucket_rows]
            )

        horizon = max(forward_horizons)
        for threshold in RECOVERY_HIT_THRESHOLDS:
            column = f"hit_rate_{int(threshold * 100)}pct_{horizon}d"
            summary[column] = _nan_mean([bool(row[column]) for row in bucket_rows])

        summary_rows.append(summary)

    return summary_rows


def _bucket_values(signal_name: str, values: np.ndarray) -> list[str]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return ["UNKNOWN"] * len(values)

    unique_count = np.unique(valid).size
    if unique_count == 1:
        return ["ALL"] * len(values)

    if unique_count <= 3:
        thresholds = np.nanpercentile(valid, [50])
        labels = ("LOW", "HIGH")
    elif unique_count == 4:
        thresholds = np.nanpercentile(valid, [25, 50, 75])
        labels = ("Q1", "Q2", "Q3", "Q4")
    else:
        thresholds = np.nanpercentile(valid, [25, 50, 75])
        labels = ("Q1", "Q2", "Q3", "Q4")

    buckets: list[str] = []
    for value in values:
        if np.isnan(value):
            buckets.append("UNKNOWN")
            continue
        if len(labels) == 2:
            buckets.append(labels[int(value > thresholds[0])])
            continue
        if len(labels) == 4:
            idx = int(np.digitize(value, thresholds, right=False))
            buckets.append(labels[min(idx, 3)])
            continue
        buckets.append("ALL")
    return buckets


def _forward_return(close: np.ndarray, idx: int, horizon: int) -> float:
    future_idx = idx + horizon
    if future_idx >= len(close) or np.isnan(close[idx]) or np.isnan(close[future_idx]):
        return float("nan")
    return float((close[future_idx] / close[idx]) - 1.0)


def _max_forward_return(close: np.ndarray, idx: int, horizon: int) -> float:
    future = close[idx + 1 : min(len(close), idx + horizon + 1)]
    future = future[~np.isnan(future)]
    if future.size == 0 or np.isnan(close[idx]):
        return float("nan")
    return float((np.max(future) / close[idx]) - 1.0)


def _hit_rate(value: float, threshold: float) -> bool:
    return bool(not np.isnan(value) and value >= threshold)


def _nan_mean(values: list[object]) -> float | None:
    array = np.array(values, dtype=float)
    if array.size == 0:
        return None
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def _nan_median(values: list[object]) -> float | None:
    array = np.array(values, dtype=float)
    if array.size == 0:
        return None
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _as_python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_from_rows(rows: list[dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")


def _empty_detail_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "signal_name": pl.String,
        "signal_bucket": pl.String,
        "event_index": pl.Int64,
        "signal_index": pl.Int64,
        "signal_date": pl.Date,
        "trough_date": pl.Date,
        "trough_close": pl.Float64,
        "drawdown_pct": pl.Float64,
        "signal_value": pl.Float64,
    }
    for horizon in FORWARD_HORIZONS:
        schema[f"forward_return_{horizon}d"] = pl.Float64
        schema[f"max_forward_return_{horizon}d"] = pl.Float64
    for threshold in RECOVERY_HIT_THRESHOLDS:
        schema[f"hit_rate_{int(threshold * 100)}pct_{max(FORWARD_HORIZONS)}d"] = pl.Boolean
    return schema


def _empty_summary_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "signal_name": pl.String,
        "signal_bucket": pl.String,
        "events": pl.Int64,
        "median_drawdown_pct": pl.Float64,
        "median_signal_value": pl.Float64,
    }
    for horizon in FORWARD_HORIZONS:
        schema[f"median_forward_return_{horizon}d"] = pl.Float64
        schema[f"win_rate_{horizon}d"] = pl.Float64
        schema[f"median_max_forward_return_{horizon}d"] = pl.Float64
    for threshold in RECOVERY_HIT_THRESHOLDS:
        schema[f"hit_rate_{int(threshold * 100)}pct_{max(FORWARD_HORIZONS)}d"] = pl.Float64
    return schema


def _empty_detail_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_empty_detail_schema())


def _empty_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_empty_summary_schema())
