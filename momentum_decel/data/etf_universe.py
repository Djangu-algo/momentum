from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from sqlalchemy import text

from momentum_decel.data.loader import DataLoader


DEFAULT_ISSUER_PATTERNS = ("ishares", "blackrock", "spdr", "state street", "first trust", "first state", "invesco")
DEFAULT_CATEGORY_PATTERNS = ("sector", "industry")
DEFAULT_MIN_AVERAGE_VOLUME = 250_000.0
DEFAULT_MIN_AUM = 100_000_000.0
DEFAULT_ALLOWED_EXCHANGES = ("NYSE", "AMEX", "NASDAQ", "CBOE")
INDUSTRY_INFERENCE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("regional bank",), "Regional Banks"),
    (("bank", "banking"), "Banks"),
    (("insurance",), "Insurance"),
    (("broker", "capital markets", "financial services"), "Capital Markets"),
    (("biotech", "biotechnology"), "Biotechnology"),
    (("medical device", "medical devices"), "Medical Devices"),
    (("pharmaceutical", "pharma"), "Pharmaceuticals"),
    (("semiconductor", "sox"), "Semiconductors"),
    (("software",), "Software"),
    (("telecommunications", "telecom"), "Telecommunications"),
    (("aerospace", "defense"), "Aerospace & Defense"),
    (("oil & gas exploration", "exploration & production", "exploration and production"), "Oil & Gas E&P"),
    (("oil services", "oil service", "oil equipment"), "Oil Services"),
    (("metals & mining", "metals and mining"), "Metals & Mining"),
    (("homebuilder", "homebuilders", "home construction"), "Homebuilders"),
    (("retail",), "Retail"),
    (("solar",), "Solar"),
    (("clean energy",), "Clean Energy"),
    (("real estate", "reit"), "Real Estate"),
)


def load_sector_industry_etfs(
    loader: DataLoader,
    issuers: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    limit: int = 100,
    min_average_volume: float = 0.0,
    min_aum: float = 0.0,
    exchanges: Sequence[str] | None = None,
) -> pl.DataFrame:
    issuer_patterns = tuple(_normalize_patterns(issuers or DEFAULT_ISSUER_PATTERNS))
    category_patterns = tuple(_normalize_patterns(categories or DEFAULT_CATEGORY_PATTERNS))
    allowed_exchanges = tuple(_normalize_exchanges(exchanges or DEFAULT_ALLOWED_EXCHANGES))

    issuer_clause = " or ".join(f"lower(issuer) like :issuer_{idx}" for idx, _ in enumerate(issuer_patterns))
    category_clause = " or ".join(
        [
            *(f"lower(category) like :category_{idx}" for idx, _ in enumerate(category_patterns)),
            *(f"lower(focus) like :category_{idx}" for idx, _ in enumerate(category_patterns)),
        ]
    )
    exchange_clause = " or ".join(f"upper(exchange) = :exchange_{idx}" for idx, _ in enumerate(allowed_exchanges))

    params: dict[str, object] = {
        "limit": limit,
        "min_average_volume": float(min_average_volume),
        "min_aum": float(min_aum),
    }
    for idx, pattern in enumerate(issuer_patterns):
        params[f"issuer_{idx}"] = f"%{pattern}%"
    for idx, pattern in enumerate(category_patterns):
        params[f"category_{idx}"] = f"%{pattern}%"
    for idx, exchange in enumerate(allowed_exchanges):
        params[f"exchange_{idx}"] = exchange

    query = text(
        f"""
        with latest as (
            select max(scan_date) as scan_date
            from masterlist
        )
        select
            ticker,
            issuer,
            category,
            focus,
            description,
            exchange,
            asset_class,
            sector,
            industry,
            average_volume_10_day,
            aum,
            scan_date
        from masterlist
        where scan_date = (select scan_date from latest)
          and ({exchange_clause})
          and ({issuer_clause})
          and ({category_clause})
          and lower(coalesce(asset_class, '')) like '%equit%'
          and lower(coalesce(leverage, 'non-leveraged')) = 'non-leveraged'
          and coalesce(average_volume_10_day, 0) >= :min_average_volume
          and coalesce(aum, 0) >= :min_aum
        order by average_volume_10_day desc nulls last, aum desc nulls last, issuer, ticker
        limit :limit
        """
    )
    with loader.engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(query, params)]
    if not rows:
        return _empty_metadata_frame()
    frame = _with_inferred_industry(pl.DataFrame(rows).with_columns(pl.col("scan_date").cast(pl.Date)))
    return _dedupe_latest_symbols(frame)


def load_etf_metadata(loader: DataLoader, tickers: Sequence[str]) -> pl.DataFrame:
    if not tickers:
        return _empty_metadata_frame()
    query = text(
        """
        with latest as (
            select max(scan_date) as scan_date
            from masterlist
        )
        select
            ticker,
            issuer,
            category,
            focus,
            description,
            exchange,
            asset_class,
            sector,
            industry,
            average_volume_10_day,
            aum,
            scan_date
        from masterlist
        where scan_date = (select scan_date from latest)
          and upper(ticker) = any(:tickers)
        order by issuer, ticker
        """
    )
    with loader.engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(query, {"tickers": [ticker.upper() for ticker in tickers]})]
    if not rows:
        return _empty_metadata_frame()
    frame = _dedupe_latest_symbols(_with_inferred_industry(pl.DataFrame(rows).with_columns(pl.col("scan_date").cast(pl.Date))))
    requested = [ticker.upper() for ticker in tickers]
    present = set(frame["ticker"].to_list())
    missing = [ticker for ticker in requested if ticker not in present]
    if missing:
        fallback = pl.DataFrame(
            {
                "ticker": missing,
                "issuer": [None] * len(missing),
                "category": [None] * len(missing),
                "focus": [None] * len(missing),
                "description": [None] * len(missing),
                "exchange": [None] * len(missing),
                "asset_class": [None] * len(missing),
                "sector": [None] * len(missing),
                "industry": [None] * len(missing),
                "industry_inferred": [None] * len(missing),
                "average_volume_10_day": [None] * len(missing),
                "aum": [None] * len(missing),
                "scan_date": [None] * len(missing),
            },
            schema=_empty_metadata_frame().schema,
        )
        frame = pl.concat([frame, fallback], how="diagonal")
    return frame.sort(["issuer", "ticker"], nulls_last=True)


def _normalize_patterns(values: Sequence[str]) -> list[str]:
    return [value.strip().lower() for value in values if value.strip()]


def _normalize_exchanges(values: Sequence[str]) -> list[str]:
    return [value.strip().upper() for value in values if value.strip()]


def _empty_metadata_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "issuer": pl.String,
            "category": pl.String,
            "focus": pl.String,
            "description": pl.String,
            "exchange": pl.String,
            "asset_class": pl.String,
            "sector": pl.String,
            "industry": pl.String,
            "industry_inferred": pl.String,
            "average_volume_10_day": pl.Float64,
            "aum": pl.Float64,
            "scan_date": pl.Date,
        }
    )


def _dedupe_latest_symbols(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.sort(
            ["average_volume_10_day", "aum", "ticker", "exchange"],
            descending=[True, True, False, False],
            nulls_last=True,
        )
        .unique(subset=["ticker"], keep="first", maintain_order=True)
        .sort(["average_volume_10_day", "aum", "ticker"], descending=[True, True, False], nulls_last=True)
    )


def _with_inferred_industry(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(None).cast(pl.String).alias("industry_inferred"))
    return frame.with_columns(
        pl.struct("ticker", "description", "focus", "sector", "industry")
        .map_elements(_infer_industry_label, return_dtype=pl.String)
        .alias("industry_inferred")
    )


def _infer_industry_label(row: dict[str, object]) -> str | None:
    industry = _clean_text(row.get("industry"))
    if industry:
        return industry

    description = _clean_text(row.get("description"))
    ticker = _clean_text(row.get("ticker"))
    focus = _clean_text(row.get("focus"))
    sector = _clean_text(row.get("sector"))
    haystack = " ".join(part for part in (ticker, description, focus, sector) if part).lower()

    for keywords, label in INDUSTRY_INFERENCE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return label

    return focus or sector or None


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
