from __future__ import annotations

import numpy as np
import polars as pl

from momentum_decel.composite.state_machine_v2 import ADVANCED_STATE_CODES
from momentum_decel.validation.event_study import identify_drawdown_events, warning_masks


EXHAUSTION_SIGNAL_COLUMNS = (
    "ema_state_code",
    "ols_r2_20",
    "er_15",
    "curvature_c_30_z",
    "delta_ts_15_5",
    "hurst_80",
    "momentum_quality",
    "flattening_score",
    "advanced_state_code",
)
FORWARD_HORIZONS = (5, 10, 20, 40)
DRAWDOWN_SEVERITY_THRESHOLDS = (0.05, 0.08, 0.10)


def build_exhaustion_study(
    frame: pl.DataFrame,
    min_drawdown: float = 5.0,
    lookback_days: int = 40,
    warning_horizon: int = 40,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if frame.is_empty():
        return _empty_detail_frame(), _empty_summary_frame()

    ordered = frame.sort("date")
    if "close" not in ordered.columns or "date" not in ordered.columns:
        raise ValueError("exhaustion study requires date and close columns")

    events = identify_drawdown_events(ordered, min_drawdown=min_drawdown)
    if not events:
        return _empty_detail_frame(), _empty_summary_frame()

    peak_indices = [event.peak_idx for event in events]
    detail_rows: list[dict[str, object]] = []
    low = ordered["low"].to_numpy() if "low" in ordered.columns else ordered["close"].to_numpy()
    summary_rows: list[dict[str, object]] = []

    for signal_name, signal_values, mask in _warning_signal_series(ordered).values():
        warning_rows = _build_warning_rows(
            signal_name=signal_name,
            values=signal_values,
            mask=mask,
            events=events,
            peak_indices=peak_indices,
            ordered=ordered,
            low=low,
            lookback_days=lookback_days,
            warning_horizon=warning_horizon,
        )
        detail_rows.extend(warning_rows)
        summary_rows.extend(
            _summarize_warning_signal(
                rows=warning_rows,
                mask=mask,
                signal_name=signal_name,
                peak_indices=peak_indices,
                warning_horizon=warning_horizon,
            )
        )

    return _frame_from_rows(detail_rows, _empty_detail_schema()), _frame_from_rows(
        summary_rows, _empty_summary_schema()
    ).sort(["signal_name", "signal_bucket"], nulls_last=True)


def _warning_signal_series(frame: pl.DataFrame) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    base_masks = warning_masks(frame)
    series: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for name, mask in base_masks.items():
        source_column = "ema_state_code" if name == "ema_state" else name
        if source_column in frame.columns:
            series[name] = (source_column, frame[source_column].to_numpy(), mask)

    if "flattening_score" in frame.columns:
        values = frame["flattening_score"].to_numpy()
        series["flattening_score"] = ("flattening_score", values, values > 0.65)

    if "advanced_state" in frame.columns:
        labels = frame["advanced_state"].to_list()
        codes = np.array([ADVANCED_STATE_CODES.get(str(label), np.nan) for label in labels], dtype=float)
        mask = np.array([label in {"BROKEN", "FLATTENING", "FLATLINING"} for label in labels], dtype=bool)
        series["advanced_state_code"] = ("advanced_state_code", codes, mask)

    return series


def _build_warning_rows(
    signal_name: str,
    values: np.ndarray,
    mask: np.ndarray,
    events: list[object],
    peak_indices: list[int],
    ordered: pl.DataFrame,
    low: np.ndarray,
    lookback_days: int,
    warning_horizon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    buckets = _bucket_values(signal_name, values)
    dates = ordered["date"].to_list()
    close = ordered["close"].to_numpy()

    for event in events:
        peak_idx = int(event.peak_idx)
        start_idx = max(0, peak_idx - lookback_days)
        warning_indices = np.where(mask[start_idx : peak_idx + 1])[0]
        warning_idx = start_idx + int(warning_indices[0]) if warning_indices.size else None
        signal_value = values[warning_idx] if warning_idx is not None else np.nan
        row = {
            "signal_name": signal_name,
            "signal_bucket": buckets[warning_idx] if warning_idx is not None else "UNKNOWN",
            "peak_date": event.peak_date,
            "trough_date": event.trough_date,
            "drawdown_pct": float(event.drawdown_pct),
            "warning_date": dates[warning_idx] if warning_idx is not None else None,
            "warning_value": _as_python_scalar(signal_value) if warning_idx is not None else None,
            "lead_days": (event.peak_date - dates[warning_idx]).days if warning_idx is not None else None,
            "warning_found": warning_idx is not None,
            "warning_horizon_days": warning_horizon,
            "post_peak_close_return_20d": _forward_return(close, peak_idx, 20),
            "post_peak_close_return_40d": _forward_return(close, peak_idx, 40),
        }
        for horizon in FORWARD_HORIZONS:
            row[f"forward_return_{horizon}d"] = _forward_return(close, warning_idx, horizon)
            row[f"forward_min_close_return_{horizon}d"] = _min_forward_return(close, warning_idx, horizon)
            row[f"max_adverse_excursion_{horizon}d"] = _max_adverse_excursion(low, close, warning_idx, horizon)
        for threshold in DRAWDOWN_SEVERITY_THRESHOLDS:
            for horizon in FORWARD_HORIZONS:
                row[f"drawdown_hit_{int(threshold * 100)}pct_{horizon}d"] = _drawdown_hit(
                    row[f"max_adverse_excursion_{horizon}d"],
                    threshold,
                )
        rows.append(row)
    return rows


def _summarize_warning_signal(
    rows: list[dict[str, object]],
    mask: np.ndarray,
    signal_name: str,
    peak_indices: list[int],
    warning_horizon: int,
) -> list[dict[str, object]]:
    if not rows:
        return []

    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["signal_bucket"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    warning_days = int(np.sum(mask))
    false_positive_days = 0
    for idx, is_warning in enumerate(mask):
        if not is_warning:
            continue
        if not any(0 < peak_idx - idx <= warning_horizon for peak_idx in peak_indices):
            false_positive_days += 1

    for bucket, bucket_rows in buckets.items():
        lead_days = [row["lead_days"] for row in bucket_rows if row["lead_days"] is not None]
        warning_values = [row["warning_value"] for row in bucket_rows if row["warning_value"] is not None]
        summary_rows.append(
            {
                "signal_name": signal_name,
                "signal_bucket": bucket,
                "events": len(bucket_rows),
                "warnings": int(sum(bool(row["warning_found"]) for row in bucket_rows)),
                "warning_days": warning_days,
                "coverage_rate": _nan_mean([row["warning_found"] for row in bucket_rows]),
                "false_positive_rate": (false_positive_days / warning_days) if warning_days else None,
                "median_lead_days": _nan_median(lead_days),
                "mean_lead_days": _nan_mean(lead_days),
                "median_drawdown_pct": _nan_median([row["drawdown_pct"] for row in bucket_rows]),
                "median_warning_value": _nan_median(warning_values),
                "median_post_peak_close_return_20d": _nan_median(
                    [row["post_peak_close_return_20d"] for row in bucket_rows]
                ),
                "median_post_peak_close_return_40d": _nan_median(
                    [row["post_peak_close_return_40d"] for row in bucket_rows]
                ),
            }
        )
        for horizon in FORWARD_HORIZONS:
            summary_rows[-1][f"median_forward_return_{horizon}d"] = _nan_median(
                [row[f"forward_return_{horizon}d"] for row in bucket_rows]
            )
            summary_rows[-1][f"median_forward_min_close_return_{horizon}d"] = _nan_median(
                [row[f"forward_min_close_return_{horizon}d"] for row in bucket_rows]
            )
            summary_rows[-1][f"median_max_adverse_excursion_{horizon}d"] = _nan_median(
                [row[f"max_adverse_excursion_{horizon}d"] for row in bucket_rows]
            )
        for threshold in DRAWDOWN_SEVERITY_THRESHOLDS:
            for horizon in FORWARD_HORIZONS:
                column = f"drawdown_hit_{int(threshold * 100)}pct_{horizon}d"
                summary_rows[-1][column] = _nan_mean([_bool_to_float(row[column]) for row in bucket_rows])
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
        idx = int(np.digitize(value, thresholds, right=False))
        buckets.append(labels[min(idx, 3)])
    return buckets


def _forward_return(close: np.ndarray, idx: int, horizon: int) -> float:
    if idx is None:
        return float("nan")
    future_idx = idx + horizon
    if future_idx >= len(close) or np.isnan(close[idx]) or np.isnan(close[future_idx]):
        return float("nan")
    return float((close[future_idx] / close[idx]) - 1.0)


def _min_forward_return(close: np.ndarray, idx: int | None, horizon: int) -> float:
    if idx is None:
        return float("nan")
    future = close[idx + 1 : min(len(close), idx + horizon + 1)]
    future = future[~np.isnan(future)]
    if future.size == 0 or np.isnan(close[idx]):
        return float("nan")
    return float((np.min(future) / close[idx]) - 1.0)


def _max_adverse_excursion(low: np.ndarray, close: np.ndarray, idx: int | None, horizon: int) -> float:
    if idx is None:
        return float("nan")
    future = low[idx + 1 : min(len(low), idx + horizon + 1)]
    future = future[~np.isnan(future)]
    if future.size == 0 or np.isnan(close[idx]):
        return float("nan")
    return float((np.min(future) / close[idx]) - 1.0)


def _drawdown_hit(value: float, threshold: float) -> bool | None:
    if np.isnan(value):
        return None
    return bool(value <= -threshold)


def _nan_mean(values: list[object]) -> float | None:
    array = np.array(values, dtype=float)
    if array.size == 0:
        return None
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def _bool_to_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(bool(value))


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
    schema = {
        "signal_name": pl.String,
        "signal_bucket": pl.String,
        "peak_date": pl.Date,
        "trough_date": pl.Date,
        "drawdown_pct": pl.Float64,
        "warning_date": pl.Date,
        "warning_value": pl.Float64,
        "lead_days": pl.Int64,
        "warning_found": pl.Boolean,
        "warning_horizon_days": pl.Int64,
        "post_peak_close_return_20d": pl.Float64,
        "post_peak_close_return_40d": pl.Float64,
    }
    for horizon in FORWARD_HORIZONS:
        schema[f"forward_return_{horizon}d"] = pl.Float64
        schema[f"forward_min_close_return_{horizon}d"] = pl.Float64
        schema[f"max_adverse_excursion_{horizon}d"] = pl.Float64
    for threshold in DRAWDOWN_SEVERITY_THRESHOLDS:
        for horizon in FORWARD_HORIZONS:
            schema[f"drawdown_hit_{int(threshold * 100)}pct_{horizon}d"] = pl.Boolean
    return schema


def _empty_summary_schema() -> dict[str, pl.DataType]:
    schema = {
        "signal_name": pl.String,
        "signal_bucket": pl.String,
        "events": pl.Int64,
        "warnings": pl.Int64,
        "warning_days": pl.Int64,
        "coverage_rate": pl.Float64,
        "false_positive_rate": pl.Float64,
        "median_lead_days": pl.Float64,
        "mean_lead_days": pl.Float64,
        "median_drawdown_pct": pl.Float64,
        "median_warning_value": pl.Float64,
        "median_post_peak_close_return_20d": pl.Float64,
        "median_post_peak_close_return_40d": pl.Float64,
    }
    for horizon in FORWARD_HORIZONS:
        schema[f"median_forward_return_{horizon}d"] = pl.Float64
        schema[f"median_forward_min_close_return_{horizon}d"] = pl.Float64
        schema[f"median_max_adverse_excursion_{horizon}d"] = pl.Float64
    for threshold in DRAWDOWN_SEVERITY_THRESHOLDS:
        for horizon in FORWARD_HORIZONS:
            schema[f"drawdown_hit_{int(threshold * 100)}pct_{horizon}d"] = pl.Float64
    return schema


def _empty_detail_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_empty_detail_schema())


def _empty_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_empty_summary_schema())
