from __future__ import annotations

import polars as pl
from rich.console import Console
from rich.table import Table


def build_snapshot_table(frame: pl.DataFrame) -> Table:
    latest = (
        frame.sort(["ticker", "date"])
        .group_by("ticker")
        .tail(1)
        .sort("ticker")
    )

    table = Table(title="Momentum Deceleration Snapshot")
    for column in (
        "Ticker",
        "State",
        "slope_d_hi",
        "slope_d_cl",
        "R2_20",
        "ER_15",
        "Composite",
        "Delta_1d",
        "Delta_5d",
    ):
        table.add_column(column, justify="right" if column not in {"Ticker", "State"} else "left")

    for row in latest.iter_rows(named=True):
        table.add_row(
            row["ticker"],
            row.get("ema_state", "UNKNOWN"),
            _fmt(row.get("slope_d_high")),
            _fmt(row.get("slope_d_close")),
            _fmt(row.get("ols_r2_20")),
            _fmt(row.get("er_15")),
            _fmt(row.get("momentum_quality")),
            _fmt(row.get("momentum_quality_delta_1d"), signed=True),
            _fmt(row.get("momentum_quality_delta_5d"), signed=True),
        )
    return table


def print_snapshot(frame: pl.DataFrame) -> None:
    Console().print(build_snapshot_table(frame))


def _fmt(value: object, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        if value != value:
            return "-"
    except Exception:
        pass
    if signed:
        return f"{float(value):+0.3f}"
    return f"{float(value):0.3f}"

