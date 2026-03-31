# Momentum Deceleration System Usage

## Setup

Install the package in the current environment:

```bash
python3 -m pip install -e .
```

Optional environment variables:

```bash
export MOMENTUM_DECEL_PG_DSN='postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME'
export MOMENTUM_DECEL_PRICE_TABLE='pricehistory'
export MOMENTUM_DECEL_BREADTH_TABLE='ndu_complete_price_cl'
```

## Data Sources

- `postgres`: primary source for ETF and breadth work
- `yfinance`: fallback for quick prototyping

For the ETF lane, the code uses `pricehistory`.
For S&P 500 breadth, the code uses `ndu_complete_price_cl` because it already carries `s_p_500_current_past`.

## Main Commands

Compute indicators and write parquet files:

```bash
python3 cli.py --data-source postgres compute --tickers SPY XLK XLF --start 2020-01-01
```

Print the latest snapshot table:

```bash
python3 cli.py --data-source postgres snapshot --tickers SPY XLK XLF --start 2024-01-01
```

Generate a long-range interactive Plotly dashboard for one ticker.
If `--start` is omitted, `dashboard` defaults to `2017-01-01`.

```bash
python3 cli.py --data-source postgres dashboard --ticker SPY
```

Generate a long-range interactive cross-sector heatmap.
If `--start` is omitted, `heatmap` defaults to `2017-01-01`.

```bash
python3 cli.py --data-source postgres heatmap
```

Run the event study:

```bash
python3 cli.py --data-source postgres eventstudy --start 2000-01-01 --min-drawdown 5.0
```

Compute S&P 500 breadth:

```bash
python3 cli.py --data-source postgres breadth --start 2025-01-01
```

## Long-Range HTML Graphs

The HTML outputs are Plotly files with:

- drag-to-zoom
- scroll-wheel zoom
- range selector buttons
- range sliders

To build graphs from `2017-01-01` through the latest available data in Postgres:

```bash
python3 cli.py --data-source postgres dashboard --ticker SPY --start 2017-01-01
python3 cli.py --data-source postgres heatmap --start 2017-01-01
```

Output files:

- `output/charts/SPY_dashboard.html`
- `output/charts/sector_heatmap.html`

## Output Layout

- `output/indicators/{ticker}.parquet`
- `output/charts/{ticker}_dashboard.html`
- `output/charts/{ticker}_dashboard.png`
- `output/charts/sector_heatmap.html`
- `output/charts/sector_heatmap.png`
- `output/validation/event_study_results.parquet`
- `output/validation/event_study_summary.csv`
- `output/validation/sector_lead_lag.parquet`
- `output/breadth/decel_states.parquet`
- `output/breadth/decel_breadth.parquet`

Detailed field-by-field explanations live in `OUTPUTS.md`.

## Notes

- Omitting `--end` means “up to the latest data available from the source.”
- HTML charts are the preferred format for multi-year exploration.
- PNG export is still written for static review, but the interactive HTML is the primary output for long windows.
