from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


UNIVERSE = {
    "SPY": "Broad market",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

DEFAULT_TICKERS = tuple(UNIVERSE.keys())
DEFAULT_CHART_START = "2017-01-01"
DEFAULT_PG_DSN = os.getenv("MOMENTUM_DECEL_PG_DSN", "")
DEFAULT_PRICE_TABLE = os.getenv("MOMENTUM_DECEL_PRICE_TABLE", "pricehistory")
DEFAULT_BREADTH_TABLE = os.getenv("MOMENTUM_DECEL_BREADTH_TABLE", "ndu_complete_price_cl")


@dataclass(slots=True)
class IndicatorConfig:
    ema_length: int = 125
    atr_length: int = 14
    slope_window: int = 15
    trend_windows: tuple[int, ...] = (20, 40)
    efficiency_windows: tuple[int, ...] = (15, 21)
    efficiency_delta_lookbacks: tuple[int, ...] = (5, 10)
    curvature_windows: tuple[int, ...] = (30, 40)
    curvature_smoothing: int = 5
    theil_sen_windows: tuple[int, ...] = (15, 20)
    theil_sen_delta_lookbacks: tuple[int, ...] = (5, 10)
    hurst_window: int = 80
    hurst_lags: tuple[int, ...] = (2, 4, 8, 16, 32)
    percentile_window: int = 252


@dataclass(slots=True)
class RuntimeConfig:
    output_dir: Path = Path("output")
    data_source: str = "yfinance"
    pg_dsn: str = DEFAULT_PG_DSN
    price_table: str = DEFAULT_PRICE_TABLE
    breadth_table: str = DEFAULT_BREADTH_TABLE
    indicator: IndicatorConfig = field(default_factory=IndicatorConfig)

    @property
    def indicators_dir(self) -> Path:
        return self.output_dir / "indicators"

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def breadth_dir(self) -> Path:
        return self.output_dir / "breadth"

    @property
    def validation_dir(self) -> Path:
        return self.output_dir / "validation"

    def ensure_output_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.indicators_dir,
            self.charts_dir,
            self.breadth_dir,
            self.validation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
