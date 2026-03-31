from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from sqlalchemy import text

from momentum_decel.data.loader import DataLoader


DEFAULT_ISSUER_PATTERNS = ("ishares", "blackrock", "spdr", "state street", "first trust", "first state", "invesco")
DEFAULT_CATEGORY_PATTERNS = ("sector", "industry")
DEFAULT_MIN_AVERAGE_VOLUME = 250_000.0
DEFAULT_MIN_AUM = 100_000_000.0


def load_sector_industry_etfs(
    loader: DataLoader,
    issuers: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    limit: int = 100,
    min_average_volume: float = 0.0,
    min_aum: float = 0.0,
) -> pl.DataFrame:
    issuer_patterns = tuple(_normalize_patterns(issuers or DEFAULT_ISSUER_PATTERNS))
    category_patterns = tuple(_normalize_patterns(categories or DEFAULT_CATEGORY_PATTERNS))

    issuer_clause = " or ".join(f"lower(issuer) like :issuer_{idx}" for idx, _ in enumerate(issuer_patterns))
    category_clause = " or ".join(
        [
            *(f"lower(category) like :category_{idx}" for idx, _ in enumerate(category_patterns)),
            *(f"lower(focus) like :category_{idx}" for idx, _ in enumerate(category_patterns)),
        ]
    )

    params: dict[str, object] = {
        "limit": limit,
        "min_average_volume": float(min_average_volume),
        "min_aum": float(min_aum),
    }
    for idx, pattern in enumerate(issuer_patterns):
        params[f"issuer_{idx}"] = f"%{pattern}%"
    for idx, pattern in enumerate(category_patterns):
        params[f"category_{idx}"] = f"%{pattern}%"

    query = text(
        f"""
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
        from usa_stocks_sector_etfs
        where scan_date = (select max(scan_date) from usa_stocks_sector_etfs)
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
    return pl.DataFrame(rows).with_columns(pl.col("scan_date").cast(pl.Date)).sort(
        ["average_volume_10_day", "aum", "ticker"],
        descending=[True, True, False],
        nulls_last=True,
    )


def load_etf_metadata(loader: DataLoader, tickers: Sequence[str]) -> pl.DataFrame:
    if not tickers:
        return _empty_metadata_frame()
    query = text(
        """
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
        from usa_stocks_sector_etfs
        where scan_date = (select max(scan_date) from usa_stocks_sector_etfs)
          and upper(ticker) = any(:tickers)
        order by issuer, ticker
        """
    )
    with loader.engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(query, {"tickers": [ticker.upper() for ticker in tickers]})]
    if not rows:
        return _empty_metadata_frame()
    frame = pl.DataFrame(rows).with_columns(pl.col("scan_date").cast(pl.Date))
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
            "average_volume_10_day": pl.Float64,
            "aum": pl.Float64,
            "scan_date": pl.Date,
        }
    )
