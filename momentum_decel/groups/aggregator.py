from __future__ import annotations

import polars as pl


STRONG_STATES = ("RECOVERING", "ACCELERATING")
WEAK_STATES = ("BROKEN", "FLATTENING", "FLATLINING")


def latest_group_snapshot(frame: pl.DataFrame, metadata: pl.DataFrame, group_column: str = "focus") -> tuple[pl.DataFrame, pl.DataFrame]:
    latest = (
        frame.sort(["ticker", "date"])
        .group_by("ticker")
        .tail(1)
        .sort("ticker")
    )
    joined = latest.join(metadata, on="ticker", how="left")
    group_summary = joined.group_by(group_column).agg(
        pl.len().alias("n_etfs"),
        pl.median("momentum_quality").alias("median_momentum_quality"),
        pl.median("inflection_score").alias("median_inflection_score"),
        pl.median("recovery_score").alias("median_recovery_score"),
        pl.median("flattening_score").alias("median_flattening_score"),
        pl.median("leadership_score").alias("median_leadership_score"),
        (pl.col("advanced_state").is_in(STRONG_STATES).sum() / pl.len()).alias("pct_strong_states"),
        (pl.col("advanced_state").is_in(WEAK_STATES).sum() / pl.len()).alias("pct_weak_states"),
        (pl.max("leadership_score") - pl.min("leadership_score")).alias("leadership_dispersion"),
    ).sort("median_leadership_score", descending=True, nulls_last=True)
    ranked = joined.sort("leadership_score", descending=True, nulls_last=True)
    return ranked, group_summary
