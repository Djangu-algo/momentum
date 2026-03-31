from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from momentum_decel.config import RuntimeConfig


def _as_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


class DataLoader:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            if not self.config.pg_dsn:
                raise RuntimeError("Postgres mode requires --pg-dsn or MOMENTUM_DECEL_PG_DSN.")
            self._engine = create_engine(self.config.pg_dsn)
        return self._engine

    def load_prices(
        self,
        tickers: Sequence[str],
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        data_source: str | None = None,
        instruments: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        source = (data_source or self.config.data_source).lower()
        if source == "postgres":
            return self._load_prices_postgres(tickers, start, end, instruments=instruments)
        if source == "yfinance":
            return self._load_prices_yfinance(tickers, start, end)
        raise ValueError(f"Unsupported data source: {source}")

    def _load_prices_postgres(
        self,
        tickers: Sequence[str],
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        instruments: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        start_date = _as_date(start)
        end_date = _as_date(end)
        if end_date is not None:
            end_date = end_date + timedelta(days=1)

        instrument_pairs = _build_instrument_pairs(tickers, instruments)
        values_clause = ", ".join(f"(:ticker_{idx}, :exchange_{idx})" for idx, _ in enumerate(instrument_pairs))
        where_clauses = ["1 = 1"]
        params: dict[str, object] = {}
        for idx, (ticker, exchange) in enumerate(instrument_pairs):
            params[f"ticker_{idx}"] = ticker
            params[f"exchange_{idx}"] = exchange
        if start_date is not None:
            where_clauses.append("p.datetime >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            where_clauses.append("p.datetime < :end_date")
            params["end_date"] = end_date

        query = text(
            f"""
            with instruments(ticker, exchange) as (
                values {values_clause}
            )
            select
                cast(p.datetime as date) as date,
                upper(p.symbol) as ticker,
                upper(p.exchange) as exchange,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume
            from {self.config.price_table} p
            join instruments i
              on upper(p.symbol) = i.ticker
             and (i.exchange is null or upper(coalesce(p.exchange, '')) = i.exchange)
            where {' and '.join(where_clauses)}
            order by ticker, date
            """
        )
        with self.engine.connect() as conn:
            rows = [dict(row._mapping) for row in conn.execute(query, params)]
        return _normalize_price_frame(pl.DataFrame(rows))

    def _load_prices_yfinance(
        self,
        tickers: Sequence[str],
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pl.DataFrame:
        try:
            import pandas as pd
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError("yfinance mode requires pandas and yfinance to be installed.") from exc

        start_date = _as_date(start)
        end_date = _as_date(end)
        download_end = end_date + timedelta(days=1) if end_date else None
        raw = yf.download(
            tickers=list(tickers),
            start=start_date,
            end=download_end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw.empty:
            return _normalize_price_frame(pl.DataFrame(schema={"date": pl.Date, "ticker": pl.String}))

        frames: list[pd.DataFrame] = []
        multi = getattr(raw.columns, "nlevels", 1) > 1
        for ticker in tickers:
            if multi:
                ticker_frame = raw[ticker].copy()
            else:
                ticker_frame = raw.copy()
            ticker_frame = ticker_frame.reset_index()
            ticker_frame["ticker"] = ticker
            frames.append(
                ticker_frame.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )[["date", "ticker", "open", "high", "low", "close", "volume"]]
            )
        combined = pd.concat(frames, ignore_index=True)
        return _normalize_price_frame(pl.from_pandas(combined))

    def load_breadth_universe(
        self,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pl.DataFrame:
        start_date = _as_date(start)
        end_date = _as_date(end)
        if end_date is not None:
            end_date = end_date + timedelta(days=1)
        where_clauses = ["s_p_500_current_past = 1"]
        params: dict[str, object] = {}
        if start_date is not None:
            where_clauses.append("datetime >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            where_clauses.append("datetime < :end_date")
            params["end_date"] = end_date

        query = text(
            f"""
            select
                cast(datetime as date) as date,
                upper(symbol) as ticker,
                open,
                high,
                low,
                close,
                volume,
                s_p_500_current_past
            from {self.config.breadth_table}
            where {' and '.join(where_clauses)}
            order by ticker, date
            """
        )
        with self.engine.connect() as conn:
            rows = [dict(row._mapping) for row in conn.execute(query, params)]
        frame = pl.DataFrame(rows)
        return _normalize_price_frame(frame)


def _normalize_price_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "ticker": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    normalized = frame.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ).select("date", "ticker", "open", "high", "low", "close", "volume")
    return normalized.sort(["ticker", "date"])


def _build_instrument_pairs(tickers: Sequence[str], instruments: pl.DataFrame | None) -> list[tuple[str, str | None]]:
    requested = [ticker.upper() for ticker in tickers]
    exchange_by_ticker: dict[str, str | None] = {}
    if instruments is not None and not instruments.is_empty() and "ticker" in instruments.columns:
        available = instruments
        if "exchange" in available.columns:
            available = (
                available.with_columns(
                    pl.col("ticker").cast(pl.String).str.to_uppercase(),
                    pl.col("exchange").cast(pl.String).str.to_uppercase(),
                )
                .sort(["ticker", "exchange"], nulls_last=True)
                .unique(subset=["ticker"], keep="first", maintain_order=True)
            )
            exchange_by_ticker = {
                row["ticker"]: row.get("exchange")
                for row in available.iter_rows(named=True)
            }
        else:
            exchange_by_ticker = {
                ticker: None
                for ticker in available["ticker"].cast(pl.String).str.to_uppercase().to_list()
            }

    pairs: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for ticker in requested:
        pair = (ticker, exchange_by_ticker.get(ticker))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def save_ticker_parquet(frame: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="zstd")


def load_ticker_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
