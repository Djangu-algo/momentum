from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl


def build_sector_heatmap(frame: pl.DataFrame) -> go.Figure:
    pivot = (
        frame.select("date", "ticker", "momentum_quality")
        .pivot(index="ticker", on="date", values="momentum_quality")
        .sort("ticker")
    )
    dates = [column for column in pivot.columns if column != "ticker"]
    x_dates = [str(column) for column in dates]
    z = pivot.select(dates).to_numpy()
    y = pivot["ticker"].to_list()

    figure = go.Figure(
        data=
        [
            go.Heatmap(
                z=z,
                x=x_dates,
                y=y,
                colorscale=[
                    [0.0, "#a50026"],
                    [0.5, "#fee08b"],
                    [1.0, "#1a9850"],
                ],
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="Composite"),
            )
        ]
    )
    figure.update_layout(
        title="Cross-Sector Momentum Quality Heatmap",
        xaxis_title="Date",
        yaxis_title="Ticker",
        template="plotly_white",
        height=700,
        dragmode="zoom",
    )
    figure.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ]
        ),
    )
    return figure


def save_heatmap(figure: go.Figure, html_path: Path, png_path: Path) -> None:
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
