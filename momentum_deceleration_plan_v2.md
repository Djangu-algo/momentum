# Momentum Deceleration Detection System — Implementation Plan v2

## Overview

Six complementary momentum deceleration measures, unified into a composite scoring framework. Designed to detect trend exhaustion *before* price breaks key support levels.

**Target instruments:** SPY + 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLU, XLB, XLRE)
**Data source:** PostgreSQL, Norgate/NDU tables; yfinance fallback for prototyping
**Stack:** Python 3.13, polars, numpy, scipy, sqlalchemy
**Environment:** conda `py313`
**Interface:** CLI via argparse — all outputs to terminal tables + saved PNG/HTML charts

---

## Instrument Universe

| Ticker | Sector | Notes |
|--------|--------|-------|
| SPY | Broad market | Primary benchmark |
| XLK | Technology | |
| XLF | Financials | |
| XLE | Energy | |
| XLV | Health Care | |
| XLI | Industrials | |
| XLC | Communication Services | |
| XLY | Consumer Discretionary | |
| XLP | Consumer Staples | Defensive — expect different decel signature |
| XLU | Utilities | Defensive — expect different decel signature |
| XLB | Materials | |
| XLRE | Real Estate | Rate-sensitive — may lead/lag others |

Hardcoded in `config.py`. No dynamic universe management needed — these are static tickers.

---

## Phase 1: Core Indicators (Per Instrument)

**Goal:** Build six indicator functions, each returning a daily time series per ticker. All operate on a polars DataFrame with columns `date, open, high, low, close, volume`.

### Module: `indicators/`

**Design principle:** Theil-Sen is the default slope estimator everywhere. OLS is used only where R² (goodness-of-fit) is the primary signal, since Theil-Sen has no native R² equivalent.

---

### 1.1 — EMA Distance Envelope

The anchor indicator. Measures how far each OHLC component reaches from EMA(125), normalized by ATR(14).

```
d_high  = (H - EMA125) / ATR14
d_low   = (L - EMA125) / ATR14
d_close = (C - EMA125) / ATR14
d_open  = (O - EMA125) / ATR14
```

**Derived signals:**
- `slope_d_high(N)` — Theil-Sen slope of d_high over trailing N days
- `slope_d_close(N)` — Theil-Sen slope of d_close over trailing N days
- `envelope_width = d_high - d_low` — ATR-normalized daily range relative to trend
- `slope_envelope_width(N)` — Theil-Sen slope; is the range compressing toward EMA?

**State machine:**
| State | Code | Condition | Interpretation |
|-------|------|-----------|----------------|
| ACCELERATING | 3 | slope_d_high > 0 AND slope_d_close > 0 | Trend strengthening |
| DECELERATING | 2 | slope_d_high < 0 AND slope_d_close > 0 | Highs fading, close holds — early warning |
| MOMENTUM_LOST | 1 | slope_d_high < 0 AND slope_d_close < 0 | Both fading — late warning |
| TREND_BROKEN | 0 | d_close crosses below 0 | Below EMA — trend over |

**Parameters:**
- EMA length: 125 (default; also test 100, 150)
- ATR length: 14
- Slope lookback N: 10, 15, 20 days (cross-validate)
- Theil-Sen via `scipy.stats.theilslopes`

---

### 1.2 — Trend Coherence (OLS R²)

OLS is used here *only* because R² (proportion of variance explained by linear trend) has no direct Theil-Sen equivalent. The slope output is discarded in favour of 1.5's Theil-Sen slope.

```python
def rolling_r_squared(close: np.ndarray, window: int) -> np.ndarray:
    """Returns R² series only. Slope comes from Theil-Sen (1.5)."""
    # x = np.arange(window)
    # y = close[i:i+window]
    # r² = 1 - SS_res / SS_tot
```

**Signals:**
- `ols_r2(N)` — how well price conforms to a straight line
- R² declining while price still above EMA = trend losing coherence

**Parameters:**
- Window: 20, 40 days
- Computed on log-prices (constant % interpretation)

**Implementation notes:**
- R² can be computed without full OLS: `r2 = correlation(x, y) ** 2`
- This is computationally trivial — just rolling Pearson correlation squared

---

### 1.3 — Efficiency Ratio (Price-Level)

Non-parametric trend quality. Already computed as ER15 for breadth; same concept on price.

```
ER(N) = |close[t] - close[t-N]| / sum(|close[i] - close[i-1]| for i in t-N+1..t)
```

**Signals:**
- `er(N)` — 1.0 = perfectly straight move, 0.0 = net zero with lots of chop
- `delta_er = er(t) - er(t-K)` — is efficiency improving or degrading?

**Parameters:**
- N: 15, 21
- Delta lookback K: 5, 10

**Implementation notes:**
- Pure polars: net move = `col("close").diff(N).abs()`, path = `col("close").diff(1).abs().rolling_sum(N)`
- Bounded [0, 1] — no normalization needed
- Cheapest indicator computationally

---

### 1.4 — Quadratic Curvature Coefficient

Direct parametric curvature estimate. Fit `y = a + bt + ct²` over rolling window.

**Signals:**
- `c < 0` in uptrend = concave = decelerating
- `c > 0` in uptrend = convex = accelerating
- `c` flipping sign = inflection point

**Parameters:**
- Window: 30, 40 days
- Smooth c with 5-day EMA before use (noisy raw)
- Normalize: z-score of c over trailing 252 days

**Implementation notes:**
- `np.polyfit(x, y, 2)` per window
- Likely candidate for pruning in Phase 4 — keep for now, benchmark against others

---

### 1.5 — Theil-Sen Slope Delta

Robust slope → take its change over time. The primary slope measure (replaces OLS slope).

```
ts_slope(t, N) = theil_sen_slope(close[t-N:t])
delta_ts(t, N, K) = ts_slope(t, N) - ts_slope(t-K, N)
```

**Signals:**
- `ts_slope` — robust trend direction/magnitude
- `delta_ts < 0` = slope flattening or turning negative
- Magnitude of delta_ts = rate of deceleration

**Parameters:**
- Slope window N: 15, 20
- Delta lookback K: 5, 10

**Implementation notes:**
- `scipy.stats.theilslopes` per window
- O(N²) per window — fine for 12 tickers, ~1500 windows each
- For the S&P 500 breadth extension (Phase 3), precompute via vectorized pairwise slopes with numpy broadcasting:
  ```python
  # All pairwise slopes in one shot for a window
  y = prices[i:i+N]
  x = np.arange(N)
  # slopes[j,k] = (y[j] - y[k]) / (x[j] - x[k]) for j > k
  diffs_y = y[:, None] - y[None, :]
  diffs_x = x[:, None] - x[None, :]
  mask = np.triu_indices(N, k=1)
  slopes = diffs_y[mask] / diffs_x[mask]
  median_slope = np.median(slopes)
  ```

---

### 1.6 — Rolling Hurst Exponent

Autocorrelation structure measure. H > 0.5 = trending, H → 0.5 = random walk.

**Signals:**
- `H > 0.6` and declining toward 0.5 = trend character fading
- `H < 0.5` = mean-reverting regime

**Parameters:**
- Window: 80 days (compromise between stability and responsiveness)
- Lags: [2, 4, 8, 16, 32]
- Method: R/S (classic)

**Implementation notes:**
- Most computationally expensive — but fine for 12 tickers
- Likely candidate for pruning alongside 1.4
- Reuse existing implementation if available in your codebase

---

## Phase 2: Composite Scoring & Dashboard

### 2.1 — Normalization

| Indicator | Raw range | Normalization |
|-----------|-----------|---------------|
| EMA dist slope_d_high | unbounded | Percentile rank over trailing 252 days |
| EMA dist state machine | {0,1,2,3} | Direct mapping: 1.0 / 0.66 / 0.33 / 0.0 |
| OLS R² | [0, 1] | Direct use |
| Efficiency Ratio | [0, 1] | Direct use |
| Quadratic c | unbounded | Percentile rank over trailing 252 days |
| Theil-Sen slope | unbounded | Percentile rank over trailing 252 days |
| Theil-Sen delta | unbounded | Percentile rank over trailing 252 days |
| Hurst H | [0, 1] | Rescale: (H - 0.5) * 2, clipped to [0, 1] |

### 2.2 — Composite Score

```
momentum_quality = equal_weight_mean(normalized_indicators)
```

Refined after Phase 4 pruning. Max 4 free parameters in the composite.

### 2.3 — Dashboard Layout

**Per-instrument view** (12 tickers × 6 panels):

```
Panel 1: Price + EMA125 + OHLC distance bands (shaded)
Panel 2: d_high, d_close, d_low (ATR-normalized) as line series
Panel 3: slope_d_high, slope_d_close + state coloring (Theil-Sen)
Panel 4: R² + Efficiency Ratio (both [0,1], same axis)
Panel 5: Quadratic c + Theil-Sen delta (dual axis)
Panel 6: Composite momentum quality score with threshold bands
```

**Cross-sector heatmap view:**
```
Rows: 12 tickers (SPY + 11 sectors)
Columns: date (trailing 60 days)
Cell color: composite momentum_quality score (green → yellow → red)
```

Allows visual scan of which sectors are decelerating first.

**Tech:** Plotly subplots, saved as HTML (interactive) + PNG (static). Dash only if interactive controls needed later.

---

## Phase 3: Universe Breadth Aggregation

**Goal:** Extend EMA distance indicators to all S&P 500 constituents, aggregate into breadth-of-deceleration measures for Lane 3.

### 3.1 — Per-Stock Computation

For each constituent on each day:
- `d_high`, `d_close` (ATR-normalized distance from stock's own EMA125)
- `slope_d_high(15)` — trailing 15-day Theil-Sen slope
- `ema_state` — {ACCELERATING, DECELERATING, MOMENTUM_LOST, TREND_BROKEN}

**Performance note:** 500 stocks × ~6500 trading days × Theil-Sen at N=15 = 500 × 6500 × 105 pairs ≈ 341M median computations. Use the vectorized numpy broadcasting approach from 1.5. Expect ~2-5 minutes on a modern machine. Alternatively, use the `slope_d_high` shortcut: compute Theil-Sen on 15 points (105 pairs) per stock per day — this is the inner loop to optimize.

### 3.2 — Breadth Aggregates

```
pct_accelerating   = count(state == ACCEL) / N_stocks
pct_decelerating   = count(state == DECEL) / N_stocks
pct_momentum_lost  = count(state == LOST)  / N_stocks
pct_trend_broken   = count(state == BROKEN) / N_stocks
```

**Derived:**
- `decel_breadth = pct_decelerating + pct_momentum_lost + pct_trend_broken`
- `median_slope_d_high` across universe — robust aggregate momentum trajectory

### 3.3 — Lane 3 Integration

Existing Lane 3 measures *how many stocks are flat* (ER15 breadth). Adding *how many stocks are decelerating* gives the leading indicator to flatness.

**Proposed Lane 3 enhancement:**
```
Lane 3 signal = f(ER15_flatness_pct, decel_breadth, median_slope_d_high)
```

High ER15 flatness + high decel_breadth + negative median_slope_d_high = high-confidence fragility.

---

## Phase 4: Walk-Forward Validation

### 4.1 — Event Study

- Identify top-20 SPY drawdowns (>5% peak-to-trough) since 2000
- For each drawdown, record peak date
- Per indicator: how many days before peak did it flip to warning?
- Per indicator: false positive rate (warnings not followed by >5% drawdown within 40 days)

### 4.2 — Sector Lead/Lag Analysis

For each historical drawdown, record which sector ETFs' composite scores deteriorated first. Build a table:

```
Drawdown | First sector to warn | Lead time vs SPY composite
2020 Feb | XLY (-12 days)       | XLY warned 5 days before SPY
2022 Jan | XLK (-8 days)        | XLK warned 3 days before SPY
...
```

This reveals whether sector rotation into defensives (XLP, XLU strengthening while XLK, XLY weaken) is itself a reliable deceleration signal — measurable as divergence in composite scores across sectors.

### 4.3 — Indicator Pruning

For each indicator, compute:
- Median lead time (days before peak)
- False positive rate
- Signal-to-noise: lead_time / false_positive_rate
- Marginal contribution: does adding this indicator improve composite lead time beyond the best 3?

**Expected survivors:** EMA distance state machine, R², ER, Theil-Sen delta
**Expected pruning:** Quadratic curvature (noisy), Hurst (lagging)

### 4.4 — Composite Optimization

- Blocked time-series CV (match existing ER15 validation methodology)
- Max 4 free parameters in composite
- Target: composite that warns 5-15 days before peak with < 30% false positive rate

---

## CLI Interface

### Entry Point: `cli.py`

```
python cli.py <command> [options]
```

### Commands

```
python cli.py compute --tickers SPY XLK XLF --start 2020-01-01 --end 2026-03-31
    Compute all indicators for specified tickers (default: all 12).
    Saves results to parquet: output/indicators/{ticker}.parquet

python cli.py dashboard --ticker SPY --start 2025-10-01
    Generate 6-panel single-instrument dashboard.
    Saves: output/charts/{ticker}_dashboard.html + .png

python cli.py heatmap --start 2025-10-01
    Generate cross-sector composite heatmap.
    Saves: output/charts/sector_heatmap.html + .png

python cli.py snapshot
    Print current-day state for all 12 tickers as terminal table:
    Ticker | State | slope_d_high | slope_d_close | R² | ER | Composite | Δ1d | Δ5d

python cli.py eventstudy --min-drawdown 5.0 --start 2000-01-01
    Run event study across historical drawdowns.
    Saves: output/validation/event_study_results.parquet + summary table to stdout

python cli.py breadth --start 2020-01-01
    Compute per-stock decel states across S&P 500 universe.
    Saves: output/breadth/decel_breadth.parquet
    Requires PostgreSQL connection.
```

### Global Options

```
--data-source   postgres | yfinance    (default: yfinance for prototyping)
--ema-length    int                     (default: 125)
--atr-length    int                     (default: 14)
--slope-window  int                     (default: 15)
--output-dir    path                    (default: ./output)
```

### Terminal Output Style

`snapshot` command outputs a rich-formatted table:

```
┌────────┬──────────────┬─────────────┬──────────────┬──────┬──────┬───────────┬───────┬───────┐
│ Ticker │ State        │ slope_d_hi  │ slope_d_cl   │ R²   │ ER   │ Composite │ Δ1d   │ Δ5d   │
├────────┼──────────────┼─────────────┼──────────────┼──────┼──────┼───────────┼───────┼───────┤
│ SPY    │ DECEL        │ -0.042      │  0.011       │ 0.34 │ 0.28 │ 0.41      │ -0.03 │ -0.11 │
│ XLK    │ MOM_LOST     │ -0.087      │ -0.031       │ 0.18 │ 0.15 │ 0.22      │ -0.05 │ -0.18 │
│ XLP    │ ACCEL        │  0.023      │  0.019       │ 0.71 │ 0.62 │ 0.78      │ +0.02 │ +0.06 │
│ ...    │              │             │              │      │      │           │       │       │
└────────┴──────────────┴─────────────┴──────────────┴──────┴──────┴───────────┴───────┴───────┘
```

Use `rich` library for terminal table formatting.

---

## Implementation Sequence

| Step | Deliverable | Est. Effort | Dependencies |
|------|-------------|-------------|--------------|
| 1 | `indicators/` — 6 indicator functions, tested on SPY | 1 session | numpy, scipy, polars |
| 2 | `cli.py compute` + `cli.py snapshot` — 12 tickers | 0.5 session | Step 1 |
| 3 | `cli.py dashboard` — 6-panel chart per ticker | 1 session | Step 1 |
| 4 | `cli.py heatmap` — cross-sector composite view | 0.5 session | Step 2 |
| 5 | `cli.py eventstudy` — indicator comparison + pruning | 1 session | Step 2 |
| 6 | Prune indicators, finalize composite weights | 0.5 session | Step 5 results |
| 7 | `cli.py breadth` — S&P 500 universe decel breadth | 1 session | Step 6, PostgreSQL |
| 8 | Lane 3 integration + composite scoring | 1 session | Step 7, existing breadth framework |
| 9 | Walk-forward validation (expanding window, blocked CV) | 1 session | Step 8 |
| 10 | Production daily update hook | 0.5 session | Step 9 |

**Total: ~8 sessions**

---

## File Structure

```
momentum_decel/
├── cli.py                          # argparse entry point
├── config.py                       # all parameters, ticker list
├── indicators/
│   ├── __init__.py
│   ├── ema_distance.py             # 1.1 — EMA envelope + state machine
│   ├── trend_coherence.py          # 1.2 — Rolling R² (OLS for R² only)
│   ├── efficiency_ratio.py         # 1.3 — Price-level ER
│   ├── quadratic_curvature.py      # 1.4 — Curvature coefficient
│   ├── theil_sen.py                # 1.5 — Theil-Sen slope + delta (primary slope measure)
│   └── hurst.py                    # 1.6 — Rolling Hurst
├── composite/
│   ├── __init__.py
│   ├── normalizer.py               # Percentile rank / scaling
│   ├── scorer.py                   # Composite momentum quality
│   └── state_machine.py            # EMA distance state transitions
├── breadth/
│   ├── __init__.py
│   ├── universe_decel.py           # Per-stock decel computation
│   └── lane3_integration.py        # Breadth aggregation for Lane 3
├── validation/
│   ├── __init__.py
│   ├── event_study.py              # Drawdown lead-time analysis
│   └── walk_forward.py             # Blocked time-series CV
├── dashboard/
│   ├── __init__.py
│   ├── single_instrument.py        # 6-panel Plotly dashboard
│   ├── sector_heatmap.py           # Cross-sector composite heatmap
│   └── snapshot.py                 # Terminal table (rich)
├── data/
│   ├── __init__.py
│   └── loader.py                   # PostgreSQL / yfinance data access
└── output/
    ├── indicators/                 # {ticker}.parquet
    ├── charts/                     # .html + .png
    ├── breadth/                    # decel_breadth.parquet
    └── validation/                 # event_study_results.parquet
```

---

## Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Slope estimator | Theil-Sen everywhere except R² | Robust to gaps/outliers; O(N²) acceptable for 12 tickers |
| OLS retained for | R² computation only | Theil-Sen has no native goodness-of-fit; R² = corr(x,y)² is trivial |
| Price transform | Log-prices for R²; levels for EMA distance | R² on log = constant % interpretation; EMA distance on levels = natural ATR normalization |
| EMA vs SMA | EMA (default), SMA as CLI flag for testing | EMA reacts faster; if SMA proves better in event study, switch |
| Normalization | Percentile rank (252-day trailing) | Non-parametric, handles fat tails, no distributional assumptions |
| Sector ETF set | All 11 SPDR sectors + SPY | Complete sector decomposition; defensives (XLP, XLU) as rotation benchmarks |
| Dashboard tech | Plotly (HTML + PNG export) | Interactive exploration + static archival; no Dash server needed |
| Terminal output | rich library | Clean tables, color coding for states |
| Storage format | zstd-compressed parquet | Consistent with existing pipeline |

---

## Open Questions

1. **Should sector relative momentum (XLK/SPY ratio distance from its own EMA) be a separate indicator?** This captures rotation, not absolute deceleration. Potentially valuable but adds complexity. Park for v2.

2. **Interaction with S2 credit overlay:** Credit overlay catches macro stress; this system catches equity-specific momentum degradation. Track both signals on same timeline in event study to measure lead/lag relationship.

3. **OHLC for Theil-Sen slopes or close only?** Plan uses close for Theil-Sen slope/delta (1.5) and OHLC for EMA distance (1.1). This keeps the two approaches cleanly separated — 1.1 extracts information from the intra-day range, 1.5 measures inter-day trend. Don't mix.

4. **Parquet output granularity:** One file per ticker with all indicators as columns, or one file per indicator across all tickers? Per-ticker is simpler for dashboard; per-indicator is better for cross-sector analysis. **Decision: per-ticker**, with a `concat_all()` utility for cross-sector views.
