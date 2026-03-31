from __future__ import annotations

import polars as pl

from momentum_decel.config import IndicatorConfig
from momentum_decel.indicators.ema_distance import add_ema_distance_indicators
from momentum_decel.indicators.efficiency_ratio import add_efficiency_ratio_indicators
from momentum_decel.indicators.hurst import add_hurst_indicators
from momentum_decel.indicators.quadratic_curvature import add_quadratic_curvature_indicators
from momentum_decel.indicators.theil_sen import add_theil_sen_indicators
from momentum_decel.indicators.trend_coherence import add_trend_coherence_indicators


def compute_all_indicators(frame: pl.DataFrame, config: IndicatorConfig) -> pl.DataFrame:
    enriched = add_ema_distance_indicators(frame, config)
    enriched = add_trend_coherence_indicators(enriched, config)
    enriched = add_efficiency_ratio_indicators(enriched, config)
    enriched = add_quadratic_curvature_indicators(enriched, config)
    enriched = add_theil_sen_indicators(enriched, config)
    enriched = add_hurst_indicators(enriched, config)
    return enriched

