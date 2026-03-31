from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


EXHAUSTION_SIGNAL_MAP: tuple[tuple[str, str], ...] = (
    ("advanced_state_code", "advanced_state_code"),
    ("ema_state_code", "ema_state"),
    ("flattening_score", "flattening_score"),
    ("momentum_quality", "momentum_quality"),
    ("curvature_c_30_z", "curvature_c_30_z"),
    ("delta_ts_15_5", "delta_ts_15_5"),
)
RECOVERY_SIGNAL_MAP: tuple[tuple[str, str], ...] = (
    ("recovery_score", "recovery_score"),
    ("inflection_score", "inflection_score"),
    ("leadership_score", "leadership_score"),
    ("rel_strength_score", "rel_strength_score"),
    ("d_close", "d_close"),
    ("momentum_quality", "momentum_quality"),
)
RATIO_SPECS: tuple[tuple[str, str, str], ...] = (
    ("avoidance_dd8_20d", "DD8/20", "drawdown_hit_8pct_20d"),
    ("avoidance_dd8_40d", "DD8/40", "drawdown_hit_8pct_40d"),
    ("tail_dd10_40d", "DD10/40", "drawdown_hit_10pct_40d"),
    ("recovery_up10_40d", "UP10/40", "hit_rate_10pct_40d"),
)
GROUP_LABEL_COLUMNS: tuple[str, ...] = ("industry", "focus", "sector", "category", "exchange")
THRESHOLD_FILE_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("industry_inferred", "exhaustion_study_group_thresholds_by_industry.csv"),
    ("focus", "exhaustion_study_group_thresholds_by_focus.csv"),
)


@dataclass(slots=True)
class DashboardRiskContext:
    frame: pl.DataFrame
    annotation_text: str | None = None


def build_single_instrument_dashboard(
    frame: pl.DataFrame,
    ticker: str,
    risk_context: DashboardRiskContext | None = None,
) -> go.Figure:
    data = frame.sort("date").to_pandas()
    risk_data = risk_context.frame.sort("date").to_pandas() if risk_context is not None else None
    x_values = data["date"].astype(str)
    risk_x_values = risk_data["date"].astype(str) if risk_data is not None else None
    distance_limit = float(
        max(
            abs(data["d_high"]).max(skipna=True),
            abs(data["d_low"]).max(skipna=True),
            abs(data["d_open"]).max(skipna=True),
            abs(data["d_close"]).max(skipna=True),
            1.0,
        )
    )
    slope_high_limit = float(max(abs(data["slope_d_high"]).max(skipna=True), 0.1))
    slope_close_limit = float(max(abs(data["slope_d_close"]).max(skipna=True), 0.1))
    curvature_limit = _robust_symmetric_limit(data["curvature_c_30_z"] if "curvature_c_30_z" in data.columns else None, 1.0)
    delta_limit = _robust_symmetric_limit(data["delta_ts_15_5"] if "delta_ts_15_5" in data.columns else None, 1.0)

    figure = make_subplots(
        rows=10,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.25, 0.12, 0.03, 0.07, 0.07, 0.09, 0.08, 0.08, 0.11, 0.10],
        subplot_titles=(
            "Price and EMA125",
            "ATR-normalized OHLC distance",
            "",
            "Slope d_high",
            "Slope d_close and state",
            "Trend coherence and efficiency",
            "Curvature and Theil-Sen delta",
            "Relative strength versus SPY",
            "Framework scores",
            "Validation risk / recovery hit rates",
        ),
        specs=[
            [{}],
            [{}],
            [None],
            [{}],
            [{"secondary_y": True}],
            [{}],
            [{"secondary_y": True}],
            [{}],
            [{}],
            [{}],
        ],
    )

    figure.add_trace(
        go.Candlestick(
            x=x_values,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(go.Scatter(x=x_values, y=data["ema_125"], name="EMA125"), row=1, col=1)

    figure.add_trace(go.Scatter(x=x_values, y=data["d_high"], name="d_high", line=dict(color="#1a9850", width=1.0)), row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=data["d_low"],
            name="d_low",
            line=dict(color="#d73027", width=1.0),
            fill="tonexty",
            fillcolor="rgba(120, 120, 120, 0.18)",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(go.Scatter(x=x_values, y=data["d_open"], name="d_open", line=dict(color="#4575b4", width=1.0, dash="dot")), row=2, col=1)
    figure.add_trace(go.Scatter(x=x_values, y=data["d_close"], name="d_close", line=dict(color="#313695", width=1.4)), row=2, col=1)
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.5, row=2, col=1)

    figure.add_trace(go.Scatter(x=x_values, y=data["slope_d_high"], name="slope_d_high"), row=4, col=1)
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.5, row=4, col=1)

    figure.add_trace(go.Scatter(x=x_values, y=data["slope_d_close"], name="slope_d_close"), row=5, col=1)
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=data["ema_state_code"],
            name="ema_state_code",
            mode="markers",
            marker=dict(size=5),
        ),
        row=5,
        col=1,
        secondary_y=True,
    )
    if "advanced_state_code" in data.columns:
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=data["advanced_state_code"],
                name="advanced_state_code",
                mode="markers",
                marker=dict(size=4, symbol="diamond"),
            ),
            row=5,
            col=1,
            secondary_y=True,
        )
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.5, row=5, col=1)

    for column in ("ols_r2_20", "er_15"):
        if column in data.columns:
            figure.add_trace(go.Scatter(x=x_values, y=data[column], name=column), row=6, col=1)

    if "curvature_c_30_z" in data.columns:
        figure.add_trace(
            go.Scatter(x=x_values, y=data["curvature_c_30_z"], name="curvature_c_30_z", line=dict(width=1.5)),
            row=7,
            col=1,
        )
    if "delta_ts_15_5" in data.columns:
        figure.add_trace(
            go.Scatter(x=x_values, y=data["delta_ts_15_5"], name="delta_ts_15_5", line=dict(width=1.5)),
            row=7,
            col=1,
            secondary_y=True,
        )
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.45, row=7, col=1)

    for column in ("rel_d_close", "rel_er_15", "rel_delta_ts_15_5"):
        if column in data.columns:
            figure.add_trace(go.Scatter(x=x_values, y=data[column], name=column), row=8, col=1)
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.4, row=8, col=1)

    for column in ("momentum_quality", "inflection_score", "recovery_score", "flattening_score", "leadership_score"):
        if column in data.columns:
            figure.add_trace(go.Scatter(x=x_values, y=data[column], name=column), row=9, col=1)
    figure.add_hrect(y0=0.0, y1=0.35, fillcolor="#d73027", opacity=0.15, line_width=0, row=9, col=1)
    figure.add_hrect(y0=0.35, y1=0.65, fillcolor="#fee08b", opacity=0.18, line_width=0, row=9, col=1)
    figure.add_hrect(y0=0.65, y1=1.0, fillcolor="#1a9850", opacity=0.12, line_width=0, row=9, col=1)

    if risk_data is not None:
        risk_style = {
            "avoidance_dd8_20d": dict(color="#fdae61", width=1.4),
            "avoidance_dd8_40d": dict(color="#d73027", width=1.8),
            "tail_dd10_40d": dict(color="#7f0000", width=1.4, dash="dot"),
            "recovery_up10_40d": dict(color="#1a9850", width=1.8, dash="dash"),
        }
        display_names = {key: label for key, label, _ in RATIO_SPECS}
        for column, line in risk_style.items():
            if column not in risk_data.columns:
                continue
            figure.add_trace(
                go.Scatter(
                    x=risk_x_values,
                    y=risk_data[column],
                    name=display_names[column],
                    line=line,
                ),
                row=10,
                col=1,
            )
        figure.add_hline(y=0.5, line_width=1, line_color="#666666", opacity=0.35, row=10, col=1)

    figure.update_layout(
        height=2280,
        title=f"{ticker} Momentum Deceleration Dashboard",
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        showlegend=True,
    )
    figure.update_xaxes(
        showspikes=True,
        spikemode="across",
        rangeslider=dict(visible=True, thickness=0.04),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ]
        ),
        row=10,
        col=1,
    )

    figure.update_yaxes(fixedrange=False, row=1, col=1)
    figure.update_yaxes(automargin=True, row=1, col=1)
    figure.update_yaxes(range=[-distance_limit, distance_limit], fixedrange=True, row=2, col=1)
    figure.update_yaxes(range=[-slope_high_limit, slope_high_limit], fixedrange=True, row=4, col=1)
    figure.update_yaxes(range=[-slope_close_limit, slope_close_limit], fixedrange=True, row=5, col=1)
    figure.update_yaxes(
        range=[-0.25, 5.25],
        tickmode="array",
        tickvals=[0, 1, 2, 3, 4, 5],
        fixedrange=True,
        row=5,
        col=1,
        secondary_y=True,
    )
    for row in (6, 8, 9):
        figure.update_yaxes(fixedrange=True, row=row, col=1)
    figure.update_yaxes(range=[-curvature_limit, curvature_limit], fixedrange=True, row=7, col=1)
    figure.update_yaxes(range=[-delta_limit, delta_limit], fixedrange=True, row=7, col=1, secondary_y=True)
    figure.update_yaxes(range=[0.0, 1.0], tickformat=".0%", fixedrange=True, row=10, col=1)

    if risk_context is not None and risk_context.annotation_text:
        figure.add_annotation(
            x=0.995,
            y=0.012,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(80,80,80,0.35)",
            borderwidth=1,
            font=dict(size=11),
            text=risk_context.annotation_text,
        )
    return figure


def save_dashboard(figure: go.Figure, html_path: Path, png_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        html_path,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
        },
    )
    try:
        figure.write_image(png_path)
    except Exception:
        pass


def load_dashboard_validation_context(
    frame: pl.DataFrame,
    ticker: str,
    validation_dir: Path,
    metadata: pl.DataFrame | None = None,
) -> DashboardRiskContext | None:
    if frame.is_empty():
        return None

    exhaustion_summary = _read_csv_if_exists(validation_dir / "exhaustion_study_summary.csv")
    recovery_summary = _read_csv_if_exists(validation_dir / "recovery_study_summary.csv")
    ticker_exhaustion = _filter_ticker_summary(exhaustion_summary, ticker)
    ticker_recovery = _filter_ticker_summary(recovery_summary, ticker)

    risk_series: dict[str, np.ndarray] = {}
    if not ticker_exhaustion.is_empty():
        for output_column, _, ratio_column in RATIO_SPECS[:3]:
            series = _aggregate_ratio_series(frame, ticker_exhaustion, EXHAUSTION_SIGNAL_MAP, ratio_column)
            if not np.isnan(series).all():
                risk_series[output_column] = series
    if not ticker_recovery.is_empty():
        output_column, _, ratio_column = RATIO_SPECS[3]
        series = _aggregate_ratio_series(frame, ticker_recovery, RECOVERY_SIGNAL_MAP, ratio_column)
        if not np.isnan(series).all():
            risk_series[output_column] = series

    annotation_text = _build_risk_annotation(ticker_exhaustion, metadata, validation_dir)
    if not risk_series and not annotation_text:
        return None

    risk_frame = pl.DataFrame({"date": frame["date"]})
    for column, values in risk_series.items():
        risk_frame = risk_frame.with_columns(pl.Series(column, values))
    return DashboardRiskContext(frame=risk_frame, annotation_text=annotation_text)


def _aggregate_ratio_series(
    frame: pl.DataFrame,
    ticker_summary: pl.DataFrame,
    signal_map: tuple[tuple[str, str], ...],
    ratio_column: str,
) -> np.ndarray:
    if ticker_summary.is_empty() or ratio_column not in ticker_summary.columns:
        return np.full(frame.height, np.nan, dtype=float)

    series_values: list[np.ndarray] = []
    for frame_column, summary_signal_name in signal_map:
        if frame_column not in frame.columns:
            continue
        signal_summary = ticker_summary.filter(pl.col("signal_name") == summary_signal_name)
        if signal_summary.is_empty():
            continue
        bucket_map = {
            row["signal_bucket"]: float(row[ratio_column])
            for row in signal_summary.select("signal_bucket", ratio_column).iter_rows(named=True)
            if row["signal_bucket"] is not None and row[ratio_column] is not None
        }
        if not bucket_map:
            continue
        values = frame[frame_column].cast(pl.Float64).to_numpy()
        buckets = _bucket_values(values)
        series_values.append(np.array([bucket_map.get(bucket, np.nan) for bucket in buckets], dtype=float))

    if not series_values:
        return np.full(frame.height, np.nan, dtype=float)
    stacked = np.vstack(series_values)
    counts = np.sum(~np.isnan(stacked), axis=0)
    sums = np.nansum(stacked, axis=0)
    result = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _bucket_values(values: np.ndarray) -> list[str]:
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


def _build_risk_annotation(
    ticker_exhaustion: pl.DataFrame,
    metadata: pl.DataFrame | None,
    validation_dir: Path,
) -> str | None:
    threshold_row = _select_threshold_row(ticker_exhaustion, metadata, validation_dir)
    lines = ["Ratios: DD8/20, DD8/40, DD10/40, UP10/40"]
    if threshold_row is None:
        return "<br>".join(lines)

    group_label = _resolve_group_label(threshold_row)
    if group_label:
        lines.append(f"Group: {group_label}")
    for horizon in (20, 40):
        formatted = _format_threshold_line(threshold_row, horizon)
        if formatted:
            lines.append(formatted)
    return "<br>".join(lines)


def _select_threshold_row(
    ticker_exhaustion: pl.DataFrame,
    metadata: pl.DataFrame | None,
    validation_dir: Path,
) -> dict[str, object] | None:
    if not ticker_exhaustion.is_empty():
        return ticker_exhaustion.row(0, named=True)

    if metadata is None or metadata.is_empty():
        return None

    metadata_row = metadata.row(0, named=True)
    for group_column, file_name in THRESHOLD_FILE_FALLBACKS:
        group_value = metadata_row.get(group_column)
        if not group_value:
            continue
        thresholds = _read_csv_if_exists(validation_dir / file_name)
        if thresholds.is_empty() or group_column.replace("_inferred", "") not in thresholds.columns:
            continue
        threshold_column = group_column.replace("_inferred", "")
        matched = thresholds.filter(pl.col(threshold_column) == group_value)
        if not matched.is_empty():
            row = matched.row(0, named=True)
            row[threshold_column] = group_value
            return row
    return None


def _resolve_group_label(row: dict[str, object]) -> str | None:
    for column in GROUP_LABEL_COLUMNS:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _format_threshold_line(row: dict[str, object], horizon: int) -> str | None:
    p50 = row.get(f"group_drawdown_threshold_p50_{horizon}d")
    p75 = row.get(f"group_drawdown_threshold_p75_{horizon}d")
    p90 = row.get(f"group_drawdown_threshold_p90_{horizon}d")
    if p50 is None and p75 is None and p90 is None:
        return None
    source = row.get(f"threshold_source_{horizon}d")
    suffix = f" ({source})" if source else ""
    return (
        f"{horizon}d p50/p75/p90: "
        f"{_format_percent(p50)} / {_format_percent(p75)} / {_format_percent(p90)}"
        f"{suffix}"
    )


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def _filter_ticker_summary(summary: pl.DataFrame, ticker: str) -> pl.DataFrame:
    if summary.is_empty() or "ticker" not in summary.columns:
        return pl.DataFrame()
    return summary.filter(pl.col("ticker") == ticker)


def _read_csv_if_exists(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


def _robust_symmetric_limit(series, fallback: float) -> float:
    if series is None:
        return float(fallback)
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(max(np.nanpercentile(np.abs(values), 98), fallback))
