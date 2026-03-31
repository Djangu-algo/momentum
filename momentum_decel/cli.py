from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from momentum_decel.breadth.universe_decel import aggregate_breadth, compute_universe_deceleration
from momentum_decel.composite.leadership_score import add_leadership_score
from momentum_decel.composite.scorer import add_composite_score
from momentum_decel.config import DEFAULT_CHART_START, DEFAULT_TICKERS, IndicatorConfig, RuntimeConfig
from momentum_decel.dashboard.sector_heatmap import build_sector_heatmap, save_heatmap
from momentum_decel.dashboard.single_instrument import build_single_instrument_dashboard, save_dashboard
from momentum_decel.dashboard.snapshot import print_snapshot
from momentum_decel.data.etf_universe import (
    DEFAULT_CATEGORY_PATTERNS,
    DEFAULT_ISSUER_PATTERNS,
    DEFAULT_MIN_AUM,
    DEFAULT_MIN_AVERAGE_VOLUME,
    load_etf_metadata,
    load_sector_industry_etfs,
)
from momentum_decel.data.loader import DataLoader, save_ticker_parquet
from momentum_decel.groups.aggregator import latest_group_snapshot
from momentum_decel.indicators import compute_all_indicators
from momentum_decel.relative_strength.ratio_indicators import add_relative_strength_features
from momentum_decel.utils import parse_tickers
from momentum_decel.validation.event_study import build_event_study, build_sector_lead_lag, identify_drawdown_events
from momentum_decel.validation.exhaustion_study import build_exhaustion_study
from momentum_decel.validation.recovery_study import build_recovery_study


DEFAULT_STUDY_TICKERS = ("SPY",)
GROUP_BY_CHOICES = ("focus", "issuer", "category", "exchange", "sector", "industry")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_runtime_config(args)
    config.ensure_output_dirs()
    args.func(args, config)


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    indicator = IndicatorConfig(
        ema_length=args.ema_length,
        atr_length=args.atr_length,
        slope_window=args.slope_window,
    )
    return RuntimeConfig(
        output_dir=Path(args.output_dir),
        data_source=args.data_source,
        pg_dsn=args.pg_dsn,
        price_table=args.price_table,
        breadth_table=args.breadth_table,
        indicator=indicator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Momentum deceleration CLI")
    parser.set_defaults(func=lambda *_: parser.print_help())
    add_global_options(parser)
    subparsers = parser.add_subparsers(dest="command")

    compute = subparsers.add_parser("compute", help="Compute all indicators.")
    add_common_date_options(compute)
    compute.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    compute.set_defaults(func=run_compute)

    snapshot = subparsers.add_parser("snapshot", help="Print current-day snapshot.")
    add_common_date_options(snapshot)
    snapshot.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    snapshot.set_defaults(func=run_snapshot)

    dashboard = subparsers.add_parser("dashboard", help="Generate single-ticker dashboard.")
    add_common_date_options(dashboard)
    dashboard.add_argument("--ticker", required=True)
    dashboard.set_defaults(func=run_dashboard)

    heatmap = subparsers.add_parser("heatmap", help="Generate sector heatmap.")
    add_common_date_options(heatmap)
    heatmap.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    heatmap.set_defaults(func=run_heatmap)

    eventstudy = subparsers.add_parser("eventstudy", help="Run event study.")
    add_common_date_options(eventstudy)
    eventstudy.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    eventstudy.add_argument("--min-drawdown", type=float, default=5.0)
    eventstudy.set_defaults(func=run_eventstudy)

    breadth = subparsers.add_parser("breadth", help="Compute S&P 500 breadth.")
    add_common_date_options(breadth)
    breadth.set_defaults(func=run_breadth)

    universe = subparsers.add_parser("universe", help="List sector and industry ETF candidates from Postgres.")
    add_universe_filter_options(universe)
    universe.set_defaults(func=run_universe)

    batchscore = subparsers.add_parser("batchscore", help="Score a liquid sector and industry ETF universe.")
    add_common_date_options(batchscore)
    add_universe_filter_options(batchscore)
    batchscore.add_argument("--group-by", default="focus", choices=GROUP_BY_CHOICES)
    batchscore.set_defaults(func=run_batchscore)

    groups = subparsers.add_parser("groups", help="Compute ranked ETF and group summaries.")
    add_common_date_options(groups)
    groups.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    groups.add_argument("--group-by", default="focus", choices=GROUP_BY_CHOICES)
    groups.set_defaults(func=run_groups)

    recoverystudy = subparsers.add_parser("recoverystudy", help="Run recovery study.")
    add_common_date_options(recoverystudy)
    recoverystudy.add_argument("--tickers", nargs="*", default=list(DEFAULT_STUDY_TICKERS))
    recoverystudy.add_argument("--min-drawdown", type=float, default=5.0)
    recoverystudy.set_defaults(func=run_recoverystudy)

    exhaustionstudy = subparsers.add_parser("exhaustionstudy", help="Run exhaustion study.")
    add_common_date_options(exhaustionstudy)
    exhaustionstudy.add_argument("--tickers", nargs="*", default=list(DEFAULT_STUDY_TICKERS))
    exhaustionstudy.add_argument("--min-drawdown", type=float, default=5.0)
    exhaustionstudy.set_defaults(func=run_exhaustionstudy)

    return parser


def add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", default="postgres", choices=("postgres", "yfinance"))
    parser.add_argument("--ema-length", type=int, default=125)
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--slope-window", type=int, default=15)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--pg-dsn", default=RuntimeConfig().pg_dsn)
    parser.add_argument("--price-table", default=RuntimeConfig().price_table)
    parser.add_argument("--breadth-table", default=RuntimeConfig().breadth_table)


def add_common_date_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start")
    parser.add_argument("--end")


def add_universe_filter_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", nargs="*", default=list(DEFAULT_ISSUER_PATTERNS))
    parser.add_argument("--category", nargs="*", default=list(DEFAULT_CATEGORY_PATTERNS))
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--min-average-volume", type=float, default=DEFAULT_MIN_AVERAGE_VOLUME)
    parser.add_argument("--min-aum", type=float, default=DEFAULT_MIN_AUM)


def compute_for_tickers(
    loader: DataLoader,
    config: RuntimeConfig,
    tickers: list[str],
    start: str | None,
    end: str | None,
) -> pl.DataFrame:
    requested_tickers = [ticker.upper() for ticker in tickers]
    load_tickers = list(dict.fromkeys([*requested_tickers, "SPY"]))
    prices = loader.load_prices(tickers=load_tickers, start=start, end=end, data_source=config.data_source)
    results: list[pl.DataFrame] = []
    for ticker in load_tickers:
        ticker_prices = prices.filter(pl.col("ticker") == ticker)
        if ticker_prices.is_empty():
            continue
        computed = compute_all_indicators(ticker_prices.drop("ticker"), config.indicator)
        computed = add_composite_score(computed, config.indicator)
        computed = computed.with_columns(pl.lit(ticker).alias("ticker")).select(["date", "ticker", *[c for c in computed.columns if c != "date"]])
        results.append(computed)
    combined = pl.concat(results).sort(["ticker", "date"]) if results else pl.DataFrame()
    if combined.is_empty():
        return combined
    combined = add_relative_strength_features(combined, config.indicator, benchmark="SPY")
    combined = add_leadership_score(combined)
    return combined.filter(pl.col("ticker").is_in(requested_tickers))


def run_compute(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start, args.end)
    for ticker in tickers:
        ticker_frame = combined.filter(pl.col("ticker") == ticker)
        if ticker_frame.is_empty():
            continue
        save_ticker_parquet(ticker_frame, config.indicators_dir / f"{ticker}.parquet")
    if not combined.is_empty():
        print_snapshot(combined)


def run_snapshot(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start, args.end)
    print_snapshot(combined)


def run_dashboard(args: argparse.Namespace, config: RuntimeConfig) -> None:
    ticker = args.ticker.upper()
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, [ticker], args.start or DEFAULT_CHART_START, args.end)
    ticker_frame = combined.filter(pl.col("ticker") == ticker).drop("ticker")
    figure = build_single_instrument_dashboard(ticker_frame, ticker)
    save_dashboard(
        figure,
        config.charts_dir / f"{ticker}_dashboard.html",
        config.charts_dir / f"{ticker}_dashboard.png",
    )


def run_heatmap(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start or DEFAULT_CHART_START, args.end)
    figure = build_sector_heatmap(combined)
    save_heatmap(figure, config.charts_dir / "sector_heatmap.html", config.charts_dir / "sector_heatmap.png")


def run_eventstudy(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start or "2000-01-01", args.end)
    spy = combined.filter(pl.col("ticker") == "SPY").drop("ticker").sort("date")
    detail, summary = build_event_study(spy, min_drawdown=args.min_drawdown)
    detail.write_parquet(config.validation_dir / "event_study_results.parquet", compression="zstd")
    summary.write_csv(config.validation_dir / "event_study_summary.csv")

    events = identify_drawdown_events(spy, min_drawdown=args.min_drawdown)
    sector = build_sector_lead_lag(combined, events)
    if not sector.is_empty():
        sector.write_parquet(config.validation_dir / "sector_lead_lag.parquet", compression="zstd")
    if summary.height:
        print(summary)


def run_breadth(args: argparse.Namespace, config: RuntimeConfig) -> None:
    loader = DataLoader(config)
    universe = loader.load_breadth_universe(start=args.start, end=args.end)
    detail = compute_universe_deceleration(universe, config.indicator)
    breadth = aggregate_breadth(detail)
    detail.write_parquet(config.breadth_dir / "decel_states.parquet", compression="zstd")
    breadth.write_parquet(config.breadth_dir / "decel_breadth.parquet", compression="zstd")
    print(breadth.tail(10))


def run_universe(args: argparse.Namespace, config: RuntimeConfig) -> None:
    loader = DataLoader(config)
    universe = load_sector_industry_etfs(
        loader,
        issuers=args.issuer,
        categories=args.category,
        limit=args.limit,
        min_average_volume=args.min_average_volume,
        min_aum=args.min_aum,
    )
    print(universe)


def run_batchscore(args: argparse.Namespace, config: RuntimeConfig) -> None:
    loader = DataLoader(config)
    metadata = load_sector_industry_etfs(
        loader,
        issuers=args.issuer,
        categories=args.category,
        limit=args.limit,
        min_average_volume=args.min_average_volume,
        min_aum=args.min_aum,
    )
    if metadata.is_empty():
        print("No ETFs matched the issuer and liquidity filters.")
        return

    tickers = metadata["ticker"].to_list()
    combined = compute_for_tickers(loader, config, tickers, args.start or DEFAULT_CHART_START, args.end)
    if combined.is_empty():
        print("No price history was found for the selected ETF universe.")
        return

    available_tickers = set(combined["ticker"].unique().to_list())
    metadata = metadata.filter(pl.col("ticker").is_in(available_tickers))
    ranked, group_summary = latest_group_snapshot(combined, metadata, group_column=args.group_by)

    metadata.write_parquet(config.universe_dir / "universe_metadata.parquet", compression="zstd")
    combined.write_parquet(config.universe_dir / "universe_history.parquet", compression="zstd")
    ranked.write_parquet(config.universe_dir / "ranked_latest.parquet", compression="zstd")
    group_summary.write_parquet(config.universe_dir / f"group_summary_by_{args.group_by}.parquet", compression="zstd")

    print(group_summary)
    print(
        ranked.select(
            "ticker",
            "advanced_state",
            "momentum_quality",
            "inflection_score",
            "recovery_score",
            "flattening_score",
            "leadership_score",
            "average_volume_10_day",
            "aum",
            args.group_by,
        ).head(30)
    )


def run_groups(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start or "2024-01-01", args.end)
    metadata = load_etf_metadata(loader, tickers)
    ranked, group_summary = latest_group_snapshot(combined, metadata, group_column=args.group_by)
    ranked.write_parquet(config.groups_dir / "ranked_etfs.parquet", compression="zstd")
    group_summary.write_parquet(config.groups_dir / f"group_summary_by_{args.group_by}.parquet", compression="zstd")
    print(group_summary)
    print(
        ranked.select(
            "ticker",
            "advanced_state",
            "momentum_quality",
            "inflection_score",
            "recovery_score",
            "flattening_score",
            "leadership_score",
            args.group_by,
        ).head(20)
    )


def run_recoverystudy(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_STUDY_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start or "2000-01-01", args.end)
    detail_frames: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []

    for ticker in tickers:
        ticker_frame = combined.filter(pl.col("ticker") == ticker).drop("ticker").sort("date")
        if ticker_frame.is_empty():
            continue
        detail, summary = build_recovery_study(ticker_frame, min_drawdown=args.min_drawdown)
        if not detail.is_empty():
            detail_frames.append(detail.with_columns(pl.lit(ticker).alias("ticker")).select("ticker", *detail.columns))
        if not summary.is_empty():
            summary_frames.append(summary.with_columns(pl.lit(ticker).alias("ticker")).select("ticker", *summary.columns))

    detail_frame = pl.concat(detail_frames, how="diagonal") if detail_frames else pl.DataFrame(schema={"ticker": pl.String})
    summary_frame = pl.concat(summary_frames, how="diagonal") if summary_frames else pl.DataFrame(schema={"ticker": pl.String})
    detail_frame.write_parquet(config.validation_dir / "recovery_study_detail.parquet", compression="zstd")
    summary_frame.write_csv(config.validation_dir / "recovery_study_summary.csv")
    if summary_frame.height:
        print(
            summary_frame.sort(
                ["median_forward_return_20d", "drawdown_hit_8pct_20d"],
                descending=[True, False],
                nulls_last=True,
            )
        )


def run_exhaustionstudy(args: argparse.Namespace, config: RuntimeConfig) -> None:
    tickers = parse_tickers(args.tickers, DEFAULT_STUDY_TICKERS)
    loader = DataLoader(config)
    combined = compute_for_tickers(loader, config, tickers, args.start or "2000-01-01", args.end)
    detail_frames: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []

    for ticker in tickers:
        ticker_frame = combined.filter(pl.col("ticker") == ticker).drop("ticker").sort("date")
        if ticker_frame.is_empty():
            continue
        detail, summary = build_exhaustion_study(ticker_frame, min_drawdown=args.min_drawdown)
        if not detail.is_empty():
            detail_frames.append(detail.with_columns(pl.lit(ticker).alias("ticker")).select("ticker", *detail.columns))
        if not summary.is_empty():
            summary_frames.append(summary.with_columns(pl.lit(ticker).alias("ticker")).select("ticker", *summary.columns))

    detail_frame = pl.concat(detail_frames, how="diagonal") if detail_frames else pl.DataFrame(schema={"ticker": pl.String})
    summary_frame = pl.concat(summary_frames, how="diagonal") if summary_frames else pl.DataFrame(schema={"ticker": pl.String})
    detail_frame.write_parquet(config.validation_dir / "exhaustion_study_detail.parquet", compression="zstd")
    summary_frame.write_csv(config.validation_dir / "exhaustion_study_summary.csv")
    if summary_frame.height:
        print(
            summary_frame.sort(
                ["drawdown_hit_8pct_40d", "median_max_adverse_excursion_20d"],
                descending=[True, False],
                nulls_last=True,
            )
        )
