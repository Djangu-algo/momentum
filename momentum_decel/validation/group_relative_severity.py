from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import polars as pl


GROUP_RELATIVE_QUANTILES: tuple[tuple[str, float], ...] = (
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
)


def attach_group_relative_severity(
    detail: pl.DataFrame,
    metadata: pl.DataFrame,
    group_column: str = "focus",
    horizons: Sequence[int] = (5, 10, 20, 40),
    min_group_samples: int = 12,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if detail.is_empty():
        empty = _empty_threshold_frame(group_column, horizons)
        return detail, empty

    group_meta = _group_metadata(metadata, group_column)
    enriched = detail.join(group_meta, on="ticker", how="left")
    if group_column not in enriched.columns:
        enriched = enriched.with_columns(pl.lit("UNCLASSIFIED").alias(group_column))
    else:
        enriched = enriched.with_columns(
            pl.when(pl.col(group_column).is_null() | (pl.col(group_column).cast(pl.String).str.strip_chars() == ""))
            .then(pl.lit("UNCLASSIFIED"))
            .otherwise(pl.col(group_column).cast(pl.String))
            .alias(group_column)
        )

    thresholds = build_group_thresholds(
        enriched,
        group_column=group_column,
        horizons=horizons,
        min_group_samples=min_group_samples,
    )
    if thresholds.is_empty():
        return enriched, thresholds

    enriched = enriched.join(thresholds, on=group_column, how="left")
    expressions: list[pl.Expr] = []
    for horizon in horizons:
        metric_column = f"max_adverse_excursion_{horizon}d"
        for label, _ in GROUP_RELATIVE_QUANTILES:
            threshold_column = f"group_drawdown_threshold_{label}_{horizon}d"
            expressions.append(
                pl.when(pl.col(metric_column).is_null() | pl.col(threshold_column).is_null())
                .then(None)
                .otherwise((-pl.col(metric_column)) >= pl.col(threshold_column))
                .alias(f"group_relative_hit_{label}_{horizon}d")
            )
    return enriched.with_columns(expressions), thresholds


def build_group_thresholds(
    detail: pl.DataFrame,
    group_column: str = "focus",
    horizons: Sequence[int] = (5, 10, 20, 40),
    min_group_samples: int = 12,
) -> pl.DataFrame:
    if detail.is_empty():
        return _empty_threshold_frame(group_column, horizons)

    groups = sorted({value for value in detail[group_column].to_list() if value})
    if not groups:
        return _empty_threshold_frame(group_column, horizons)

    global_values = {horizon: _severity_values(detail, f"max_adverse_excursion_{horizon}d") for horizon in horizons}
    rows: list[dict[str, object]] = []

    for group_value in groups:
        row: dict[str, object] = {group_column: group_value}
        group_frame = detail.filter(pl.col(group_column) == group_value)
        for horizon in horizons:
            severity_values = _severity_values(group_frame, f"max_adverse_excursion_{horizon}d")
            use_group_values = severity_values.size >= min_group_samples
            threshold_values = severity_values if use_group_values else global_values[horizon]
            threshold_source = "group" if use_group_values else "global_fallback"
            row[f"threshold_source_{horizon}d"] = threshold_source if threshold_values.size else "unavailable"
            row[f"threshold_sample_count_{horizon}d"] = int(severity_values.size)
            for label, quantile in GROUP_RELATIVE_QUANTILES:
                row[f"group_drawdown_threshold_{label}_{horizon}d"] = (
                    float(np.quantile(threshold_values, quantile)) if threshold_values.size else None
                )
        rows.append(row)

    return pl.DataFrame(rows).sort(group_column)


def augment_summary_with_group_relative_hits(
    summary: pl.DataFrame,
    detail: pl.DataFrame,
    group_column: str = "focus",
    horizons: Sequence[int] = (5, 10, 20, 40),
) -> pl.DataFrame:
    if summary.is_empty() or detail.is_empty():
        return summary

    relative_columns = [
        f"group_relative_hit_{label}_{horizon}d"
        for horizon in horizons
        for label, _ in GROUP_RELATIVE_QUANTILES
        if f"group_relative_hit_{label}_{horizon}d" in detail.columns
    ]
    threshold_columns = [
        column
        for column in detail.columns
        if column.startswith("group_drawdown_threshold_") or column.startswith("threshold_source_") or column.startswith("threshold_sample_count_")
    ]

    aggregated = detail.group_by("ticker", "signal_name", "signal_bucket").agg(
        pl.first(group_column).alias(group_column),
        *[pl.col(column).cast(pl.Float64).mean().alias(column) for column in relative_columns],
        *[pl.first(column).alias(column) for column in threshold_columns],
    )
    return summary.join(aggregated, on=["ticker", "signal_name", "signal_bucket"], how="left")


def build_group_relative_summary(
    detail: pl.DataFrame,
    group_column: str = "focus",
    horizons: Sequence[int] = (5, 10, 20, 40),
) -> pl.DataFrame:
    if detail.is_empty():
        return pl.DataFrame(schema={group_column: pl.String, "signal_name": pl.String, "signal_bucket": pl.String, "events": pl.UInt32})

    expressions: list[pl.Expr] = [
        pl.len().alias("events"),
        pl.n_unique("ticker").alias("tickers"),
    ]
    for horizon in horizons:
        forward_column = f"forward_return_{horizon}d"
        if forward_column in detail.columns:
            expressions.append(pl.median(forward_column).alias(f"median_forward_return_{horizon}d"))
        min_close_column = f"forward_min_close_return_{horizon}d"
        if min_close_column in detail.columns:
            expressions.append(pl.median(min_close_column).alias(f"median_forward_min_close_return_{horizon}d"))
        metric_column = f"max_adverse_excursion_{horizon}d"
        if metric_column in detail.columns:
            expressions.append(pl.median(metric_column).alias(f"median_max_adverse_excursion_{horizon}d"))
        for threshold in (5, 8, 10):
            column = f"drawdown_hit_{threshold}pct_{horizon}d"
            if column in detail.columns:
                expressions.append(pl.col(column).cast(pl.Float64).mean().alias(f"absolute_{column}"))
        for label, _ in GROUP_RELATIVE_QUANTILES:
            column = f"group_relative_hit_{label}_{horizon}d"
            if column in detail.columns:
                expressions.append(pl.col(column).cast(pl.Float64).mean().alias(column))

    return detail.group_by(group_column, "signal_name", "signal_bucket").agg(expressions).sort(
        [group_column, "signal_name", "signal_bucket"]
    )


def _group_metadata(metadata: pl.DataFrame, group_column: str) -> pl.DataFrame:
    if metadata.is_empty():
        return pl.DataFrame(schema={"ticker": pl.String, group_column: pl.String})
    fallback_columns = [
        column
        for column in (group_column, "industry_inferred", "focus", "sector", "category")
        if column in metadata.columns
    ]
    if not fallback_columns:
        return metadata.select("ticker").with_columns(pl.lit("UNCLASSIFIED").alias(group_column))

    expression = None
    for column in fallback_columns:
        candidate = pl.when(pl.col(column).is_null() | (pl.col(column).cast(pl.String).str.strip_chars() == "")).then(None).otherwise(
            pl.col(column).cast(pl.String)
        )
        expression = candidate if expression is None else expression.fill_null(candidate)

    return metadata.select(
        "ticker",
        expression.fill_null(pl.lit("UNCLASSIFIED")).alias(group_column),
    )


def _severity_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    values = -frame[column].to_numpy()
    values = values[np.isfinite(values)]
    values = values[values >= 0.0]
    return values.astype(float, copy=False)


def _empty_threshold_frame(group_column: str, horizons: Sequence[int]) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {group_column: pl.String}
    for horizon in horizons:
        schema[f"threshold_source_{horizon}d"] = pl.String
        schema[f"threshold_sample_count_{horizon}d"] = pl.Int64
        for label, _ in GROUP_RELATIVE_QUANTILES:
            schema[f"group_drawdown_threshold_{label}_{horizon}d"] = pl.Float64
    return pl.DataFrame(schema=schema)
