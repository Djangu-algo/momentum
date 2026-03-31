from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def build_single_instrument_dashboard(frame: pl.DataFrame, ticker: str) -> go.Figure:
    data = frame.sort("date").to_pandas()
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

    figure = make_subplots(
        rows=8,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.32, 0.15, 0.04, 0.09, 0.09, 0.11, 0.10, 0.14],
        subplot_titles=(
            "Price and EMA125",
            "ATR-normalized OHLC distance",
            "",
            "Slope d_high",
            "Slope d_close and state",
            "Trend coherence and efficiency",
            "Curvature and Theil-Sen delta",
            "Composite momentum quality",
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
        ],
    )

    figure.add_trace(
        go.Candlestick(
            x=data["date"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(go.Scatter(x=data["date"], y=data["ema_125"], name="EMA125"), row=1, col=1)

    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["d_high"],
            name="d_high",
            line=dict(color="#1a9850", width=1.0),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["d_low"],
            name="d_low",
            line=dict(color="#d73027", width=1.0),
            fill="tonexty",
            fillcolor="rgba(120, 120, 120, 0.18)",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["d_open"],
            name="d_open",
            line=dict(color="#4575b4", width=1.0, dash="dot"),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["d_close"],
            name="d_close",
            line=dict(color="#313695", width=1.4),
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.5, row=2, col=1)

    figure.add_trace(go.Scatter(x=data["date"], y=data["slope_d_high"], name="slope_d_high"), row=4, col=1)
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.5, row=4, col=1)

    figure.add_trace(go.Scatter(x=data["date"], y=data["slope_d_close"], name="slope_d_close"), row=5, col=1)
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["ema_state_code"],
            name="ema_state_code",
            mode="markers",
            marker=dict(size=5),
        ),
        row=5,
        col=1,
        secondary_y=True,
    )
    figure.add_hline(y=0.0, line_width=1, line_color="#666666", opacity=0.5, row=5, col=1)

    for column in ("ols_r2_20", "er_15"):
        if column in data.columns:
            figure.add_trace(go.Scatter(x=data["date"], y=data[column], name=column), row=6, col=1)

    if "curvature_c_30_z" in data.columns:
        figure.add_trace(go.Scatter(x=data["date"], y=data["curvature_c_30_z"], name="curvature_c_30_z"), row=7, col=1)
    if "delta_ts_15_5" in data.columns:
        figure.add_trace(
            go.Scatter(x=data["date"], y=data["delta_ts_15_5"], name="delta_ts_15_5"),
            row=7,
            col=1,
            secondary_y=True,
        )

    figure.add_trace(go.Scatter(x=data["date"], y=data["momentum_quality"], name="momentum_quality"), row=8, col=1)
    figure.add_hrect(y0=0.0, y1=0.35, fillcolor="#d73027", opacity=0.15, line_width=0, row=8, col=1)
    figure.add_hrect(y0=0.35, y1=0.65, fillcolor="#fee08b", opacity=0.18, line_width=0, row=8, col=1)
    figure.add_hrect(y0=0.65, y1=1.0, fillcolor="#1a9850", opacity=0.12, line_width=0, row=8, col=1)

    figure.update_layout(
        height=1800,
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
        row=8,
        col=1,
    )

    figure.update_yaxes(fixedrange=False, row=1, col=1)
    figure.update_yaxes(automargin=True, row=1, col=1)
    figure.update_yaxes(range=[-distance_limit, distance_limit], fixedrange=True, row=2, col=1)
    figure.update_yaxes(range=[-slope_high_limit, slope_high_limit], fixedrange=True, row=4, col=1)
    figure.update_yaxes(range=[-slope_close_limit, slope_close_limit], fixedrange=True, row=5, col=1)
    figure.update_yaxes(
        range=[-0.25, 3.25],
        tickmode="array",
        tickvals=[0, 1, 2, 3],
        fixedrange=True,
        row=5,
        col=1,
        secondary_y=True,
    )
    for row in (6, 8):
        figure.update_yaxes(fixedrange=True, row=row, col=1)
    figure.update_yaxes(fixedrange=True, row=7, col=1)
    figure.update_yaxes(fixedrange=True, row=7, col=1, secondary_y=True)
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
