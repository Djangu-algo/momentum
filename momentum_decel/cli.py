from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from momentum_decel.breadth.universe_decel import aggregate_breadth, compute_universe_deceleration
from momentum_decel.composite.scorer import add_composite_score
from momentum_decel.config import DEFAULT_CHART_START, DEFAULT_TICKERS, IndicatorConfig, RuntimeConfig
from momentum_decel.dashboard.sector_heatmap import build_sector_heatmap, save_heatmap
from momentum_decel.dashboard.single_instrument import build_single_instrument_dashboard, save_dashboard
from momentum_decel.dashboard.snapshot import print_snapshot
from momentum_decel.data.loader import DataLoader, save_ticker_parquet
from momentum_decel.indicators import compute_all_indicators
from momentum_decel.utils import parse_tickers
from momentum_decel.validation.event_study import build_event_study, build_sector_lead_lag, identify_drawdown_events


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

    return parser


def add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", default="yfinance", choices=("postgres", "yfinance"))
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


def compute_for_tickers(
    loader: DataLoader,
    config: RuntimeConfig,
    tickers: list[str],
    start: str | None,
    end: str | None,
) -> pl.DataFrame:
    prices = loader.load_prices(tickers=tickers, start=start, end=end, data_source=config.data_source)
    results: list[pl.DataFrame] = []
    for ticker in tickers:
        ticker_prices = prices.filter(pl.col("ticker") == ticker)
        if ticker_prices.is_empty():
            continue
        computed = compute_all_indicators(ticker_prices.drop("ticker"), config.indicator)
        computed = add_composite_score(computed, config.indicator)
        computed = computed.with_columns(pl.lit(ticker).alias("ticker")).select(["date", "ticker", *[c for c in computed.columns if c != "date"]])
        results.append(computed)
    return pl.concat(results).sort(["ticker", "date"]) if results else pl.DataFrame()


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
