from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(slots=True)
class DrawdownEvent:
    peak_idx: int
    peak_date: object
    trough_idx: int
    trough_date: object
    drawdown_pct: float


def identify_drawdown_events(frame: pl.DataFrame, min_drawdown: float) -> list[DrawdownEvent]:
    close = frame["close"].to_numpy()
    dates = frame["date"].to_list()
    events: list[DrawdownEvent] = []
    peak_idx = 0
    peak_price = close[0]
    trough_idx = 0
    trough_drawdown = 0.0
    in_drawdown = False

    for idx in range(1, len(close)):
        price = close[idx]
        if price >= peak_price:
            if in_drawdown and trough_drawdown <= -(min_drawdown / 100.0):
                events.append(
                    DrawdownEvent(
                        peak_idx=peak_idx,
                        peak_date=dates[peak_idx],
                        trough_idx=trough_idx,
                        trough_date=dates[trough_idx],
                        drawdown_pct=abs(trough_drawdown) * 100.0,
                    )
                )
            peak_idx = idx
            peak_price = price
            trough_idx = idx
            trough_drawdown = 0.0
            in_drawdown = False
            continue

        drawdown = (price / peak_price) - 1.0
        if not in_drawdown:
            trough_idx = idx
            trough_drawdown = drawdown
            in_drawdown = True
        elif drawdown < trough_drawdown:
            trough_idx = idx
            trough_drawdown = drawdown

    if in_drawdown and trough_drawdown <= -(min_drawdown / 100.0):
        events.append(
            DrawdownEvent(
                peak_idx=peak_idx,
                peak_date=dates[peak_idx],
                trough_idx=trough_idx,
                trough_date=dates[trough_idx],
                drawdown_pct=abs(trough_drawdown) * 100.0,
            )
        )
    return events


def warning_masks(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    if "ema_state_code" in frame.columns:
        masks["ema_state"] = frame["ema_state_code"].to_numpy() < 3
    if "ols_r2_20" in frame.columns:
        masks["ols_r2_20"] = frame["ols_r2_20"].to_numpy() < 0.4
    if "er_15" in frame.columns:
        masks["er_15"] = frame["er_15"].to_numpy() < 0.4
    if "curvature_c_30_z" in frame.columns:
        masks["curvature_c_30_z"] = frame["curvature_c_30_z"].to_numpy() < 0.0
    if "delta_ts_15_5" in frame.columns:
        masks["delta_ts_15_5"] = frame["delta_ts_15_5"].to_numpy() < 0.0
    if "hurst_80" in frame.columns:
        masks["hurst_80"] = frame["hurst_80"].to_numpy() < 0.55
    if "momentum_quality" in frame.columns:
        masks["momentum_quality"] = frame["momentum_quality"].to_numpy() < 0.45
    return masks


def build_event_study(
    frame: pl.DataFrame,
    min_drawdown: float = 5.0,
    lookback_days: int = 40,
    warning_horizon: int = 40,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    events = identify_drawdown_events(frame, min_drawdown=min_drawdown)
    masks = warning_masks(frame)
    dates = frame["date"].to_list()
    peak_indices = [event.peak_idx for event in events]

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for indicator, mask in masks.items():
        leads: list[int] = []
        total_warning_days = 0
        false_positive_days = 0

        for idx, is_warning in enumerate(mask):
            if not is_warning:
                continue
            total_warning_days += 1
            if not any(0 < peak_idx - idx <= warning_horizon for peak_idx in peak_indices):
                false_positive_days += 1

        for event in events:
            start_idx = max(0, event.peak_idx - lookback_days)
            prior_window = np.where(mask[start_idx : event.peak_idx + 1])[0]
            warning_idx = start_idx + int(prior_window[0]) if prior_window.size else None
            lead_days = event.peak_idx - warning_idx if warning_idx is not None else None
            if lead_days is not None:
                leads.append(lead_days)
            detail_rows.append(
                {
                    "indicator": indicator,
                    "peak_date": event.peak_date,
                    "trough_date": event.trough_date,
                    "drawdown_pct": event.drawdown_pct,
                    "warning_date": dates[warning_idx] if warning_idx is not None else None,
                    "lead_days": lead_days,
                }
            )

        false_positive_rate = (false_positive_days / total_warning_days) if total_warning_days else None
        median_lead = float(np.median(leads)) if leads else None
        signal_to_noise = (median_lead / false_positive_rate) if leads and false_positive_rate not in (None, 0.0) else None
        summary_rows.append(
            {
                "indicator": indicator,
                "events": len(events),
                "median_lead_days": median_lead,
                "false_positive_rate": false_positive_rate,
                "signal_to_noise": signal_to_noise,
            }
        )

    return pl.DataFrame(detail_rows), pl.DataFrame(summary_rows).sort("median_lead_days", descending=True, nulls_last=True)


def build_sector_lead_lag(
    combined: pl.DataFrame,
    events: list[DrawdownEvent],
    composite_threshold: float = 0.45,
    lookback_days: int = 40,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    spy = combined.filter(pl.col("ticker") == "SPY").sort("date")
    spy_dates = spy["date"].to_list()
    for event in events:
        spy_window = spy.slice(max(0, event.peak_idx - lookback_days), lookback_days + 1)
        spy_warning = spy_window.filter(pl.col("momentum_quality") < composite_threshold).head(1)
        spy_lead = None
        if spy_warning.height:
            spy_lead = (event.peak_date - spy_warning["date"][0]).days

        for ticker in sorted(combined["ticker"].unique().to_list()):
            ticker_frame = combined.filter(pl.col("ticker") == ticker).sort("date")
            peak_match = ticker_frame.with_row_index("idx").filter(pl.col("date") == event.peak_date)
            if peak_match.is_empty():
                continue
            peak_idx = int(peak_match["idx"][0])
            ticker_window = ticker_frame.slice(max(0, peak_idx - lookback_days), lookback_days + 1)
            warning = ticker_window.filter(pl.col("momentum_quality") < composite_threshold).head(1)
            lead_days = None
            if warning.height:
                lead_days = (event.peak_date - warning["date"][0]).days
            rows.append(
                {
                    "peak_date": event.peak_date,
                    "ticker": ticker,
                    "composite_lead_days": lead_days,
                    "lead_vs_spy": None if lead_days is None or spy_lead is None else lead_days - spy_lead,
                }
            )
    return pl.DataFrame(rows)

