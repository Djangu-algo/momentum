from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from momentum_decel.cli import compute_for_tickers
from momentum_decel.config import RuntimeConfig, UNIVERSE
from momentum_decel.data.etf_universe import load_etf_metadata
from momentum_decel.data.loader import DataLoader


SCORE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("momentum_quality", "Momentum", "#1f77b4"),
    ("inflection_score", "Inflection", "#ff7f0e"),
    ("recovery_score", "Recovery", "#2ca02c"),
    ("flattening_score", "Flattening", "#d62728"),
    ("leadership_score", "Leadership", "#9467bd"),
)


@dataclass(slots=True)
class DiscordPanelArtifact:
    ticker: str
    name: str
    latest_date: str
    window_label: str
    figure: go.Figure
    png_bytes: bytes


def build_discord_score_panel(
    frame: pl.DataFrame,
    ticker: str,
    name: str | None = None,
    window_label: str | None = None,
) -> go.Figure:
    ordered = frame.sort("date")
    data = ordered.to_pandas()
    x_values = data["date"].astype(str)
    latest_date = str(x_values.iloc[-1])
    title_name = name or ticker
    state_text = str(data["advanced_state"].iloc[-1]) if "advanced_state" in data.columns else "UNKNOWN"
    label_suffix = f" | {window_label}" if window_label else ""

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
        subplot_titles=("Close and EMA125", "Core Scores"),
    )

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=data["close"],
            name="Close",
            line=dict(color="#111111", width=2.1),
        ),
        row=1,
        col=1,
    )
    if "ema_125" in data.columns:
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=data["ema_125"],
                name="EMA125",
                line=dict(color="#4c78a8", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    for column, label, color in SCORE_SPECS:
        if column not in data.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=data[column],
                name=label,
                line=dict(color=color, width=1.8),
            ),
            row=2,
            col=1,
        )

    figure.add_hrect(y0=0.0, y1=0.35, fillcolor="#d73027", opacity=0.12, line_width=0, row=2, col=1)
    figure.add_hrect(y0=0.35, y1=0.65, fillcolor="#fee08b", opacity=0.15, line_width=0, row=2, col=1)
    figure.add_hrect(y0=0.65, y1=1.0, fillcolor="#1a9850", opacity=0.10, line_width=0, row=2, col=1)

    figure.update_layout(
        height=880,
        width=1600,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=70, r=210, t=110, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        title=dict(
            text=f"{ticker} | {title_name}{label_suffix}<br><sup>As of {latest_date} | State: {state_text}</sup>",
            x=0.01,
            xanchor="left",
        ),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(title_text="Close", row=1, col=1, fixedrange=True)
    figure.update_yaxes(title_text="Score", row=2, col=1, range=[0.0, 1.0], fixedrange=True)

    latest_summary = _latest_score_summary(data)
    if latest_summary:
        figure.add_annotation(
            x=1.01,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(80,80,80,0.35)",
            borderwidth=1,
            font=dict(size=12),
            text=latest_summary,
        )

    return figure


def generate_discord_panel_png(
    ticker: str,
    *,
    config: RuntimeConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    display_days: int = 365,
    window_label: str | None = None,
    width: int = 1600,
    height: int = 880,
    scale: int = 2,
) -> DiscordPanelArtifact:
    runtime = config or RuntimeConfig()
    loader = DataLoader(runtime)
    symbol = ticker.upper()
    combined = compute_for_tickers(loader, runtime, [symbol], start, end)
    frame = combined.filter(pl.col("ticker") == symbol).drop("ticker")
    if frame.is_empty():
        raise ValueError(f"No price history found for {symbol}.")
    visible_frame = _filter_display_window(frame, display_days)

    metadata = load_etf_metadata(loader, [symbol]) if runtime.data_source == "postgres" else pl.DataFrame()
    display_name = _resolve_display_name(symbol, metadata)
    resolved_label = window_label or _window_label(display_days)
    figure = build_discord_score_panel(visible_frame, symbol, name=display_name, window_label=resolved_label)
    png_bytes = figure.to_image(format="png", width=width, height=height, scale=scale)
    latest_date = str(visible_frame["date"].tail(1).item())
    return DiscordPanelArtifact(
        ticker=symbol,
        name=display_name,
        latest_date=latest_date,
        window_label=resolved_label,
        figure=figure,
        png_bytes=png_bytes,
    )


def save_discord_panel_png(
    ticker: str,
    output_path: Path | str,
    *,
    config: RuntimeConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    display_days: int = 365,
    window_label: str | None = None,
    width: int = 1600,
    height: int = 880,
    scale: int = 2,
) -> Path:
    artifact = generate_discord_panel_png(
        ticker,
        config=config,
        start=start,
        end=end,
        display_days=display_days,
        window_label=window_label,
        width=width,
        height=height,
        scale=scale,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(artifact.png_bytes)
    return path


def generate_standard_discord_panels(
    ticker: str,
    *,
    config: RuntimeConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    width: int = 1600,
    height: int = 880,
    scale: int = 2,
) -> dict[str, DiscordPanelArtifact]:
    return {
        "1y": generate_discord_panel_png(
            ticker,
            config=config,
            start=start,
            end=end,
            display_days=365,
            window_label="1Y",
            width=width,
            height=height,
            scale=scale,
        ),
        "60d": generate_discord_panel_png(
            ticker,
            config=config,
            start=start,
            end=end,
            display_days=60,
            window_label="60D",
            width=width,
            height=height,
            scale=scale,
        ),
    }


def save_standard_discord_panels(
    ticker: str,
    output_dir: Path | str,
    *,
    config: RuntimeConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    width: int = 1600,
    height: int = 880,
    scale: int = 2,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = generate_standard_discord_panels(
        ticker,
        config=config,
        start=start,
        end=end,
        width=width,
        height=height,
        scale=scale,
    )
    paths: dict[str, Path] = {}
    for label, artifact in artifacts.items():
        path = output_root / f"{artifact.ticker}_discord_{label}.png"
        path.write_bytes(artifact.png_bytes)
        paths[label] = path
    return paths


def _resolve_display_name(ticker: str, metadata: pl.DataFrame) -> str:
    if not metadata.is_empty():
        row = metadata.row(0, named=True)
        for column in ("description", "industry_inferred", "focus"):
            value = row.get(column)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    return UNIVERSE.get(ticker, ticker)


def _latest_score_summary(data) -> str:
    lines: list[str] = []
    for column, label, _ in SCORE_SPECS:
        if column not in data.columns:
            continue
        value = data[column].iloc[-1]
        if value == value:
            lines.append(f"{label}: {value:.3f}")
    return "<br>".join(lines)


def _filter_display_window(frame: pl.DataFrame, display_days: int | None) -> pl.DataFrame:
    if not display_days or display_days <= 0:
        return frame.sort("date")
    ordered = frame.sort("date")
    latest_date = ordered["date"].tail(1).item()
    cutoff = latest_date - timedelta(days=display_days)
    visible = ordered.filter(pl.col("date") >= cutoff)
    return visible if not visible.is_empty() else ordered


def _window_label(display_days: int | None) -> str:
    if display_days == 365:
        return "1Y"
    if display_days == 60:
        return "60D"
    if display_days is None:
        return "Full"
    return f"{display_days}D"
