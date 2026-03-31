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
        "Composite",
        "Inflect",
        "Recover",
        "Flat",
        "Lead",
        "RelStr",
        "EMA",
    ):
        table.add_column(column, justify="right" if column not in {"Ticker", "State"} else "left")

    for row in latest.iter_rows(named=True):
        table.add_row(
            row["ticker"],
            row.get("advanced_state") or row.get("ema_state") or "UNKNOWN",
            _fmt(row.get("momentum_quality")),
            _fmt(row.get("inflection_score")),
            _fmt(row.get("recovery_score")),
            _fmt(row.get("flattening_score")),
            _fmt(row.get("leadership_score")),
            _fmt(row.get("rel_strength_score")),
            row.get("ema_state", "UNKNOWN"),
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
