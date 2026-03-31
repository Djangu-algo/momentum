# Momentum Framework Extension Plan v1

## Goal

Extend the current momentum deceleration system into a broader framework that can answer 3 distinct questions:

1. Is this symbol recovering after damage or drawdown?
2. Is this symbol flattening or exhausting after an advance?
3. Is this sector or industry group leading or lagging relative to the market and to peers?

The implementation should preserve the existing indicator stack, reuse the current CLI/data/charting structure, and add new scores, states, relative-strength logic, and validation studies.

## Design Principles

- Keep the current indicator layer as the base feature set.
- Do not overload one composite score with multiple use cases.
- Separate price-level momentum from relative momentum.
- Use group breadth plus group median, not group median alone.
- Validate recovery, exhaustion, and leadership separately.
- Treat sectors first, then add industries with the same framework.

## Target Outputs

At the end of this extension, the system should be able to produce:

- per-symbol `Recovery Score`
- per-symbol `Flattening Score`
- per-symbol `Leadership Score`
- per-symbol `Inflection Score`
- state labels:
  - `Broken`
  - `Repairing`
  - `Recovering`
  - `Accelerating`
  - `Flattening`
  - `Flatlining`
- group-level summary outputs for sectors and later industries
- relative-strength diagnostics versus SPY
- validation reports for recovery, exhaustion, and cross-sectional ranking

## Phase 1: Feature Layer Extension

### 1.1 Add Impulse and Persistence Features

Current code already has the main building blocks:

- EMA distance features
- `slope_d_high`
- `slope_d_close`
- `slope_envelope_width`
- `R^2`
- `ER`
- curvature
- Theil-Sen slope and delta

Add the following derived features:

- change in `slope_d_high`
- change in `slope_d_close`
- change in `slope_envelope_width`
- days since `d_close` crossed above or below zero
- days spent in current EMA state
- rolling fraction of recent days with positive `delta_ts`
- rolling fraction of recent days with positive curvature

Deliverable:

- new derived feature columns in the per-ticker parquet outputs

### 1.2 Add Relative-Strength Series

Build a ratio-series layer using `ETF / SPY`.

For each non-SPY ETF:

- construct price ratio against SPY
- compute the same trend/impulse/curvature/quality features on the ratio series

Suggested files:

- `momentum_decel/relative_strength/__init__.py`
- `momentum_decel/relative_strength/ratio_loader.py`
- `momentum_decel/relative_strength/ratio_indicators.py`

Deliverable:

- relative feature columns such as:
  - `rel_d_close`
  - `rel_er_15`
  - `rel_curvature_c_30_z`
  - `rel_delta_ts_15_5`

## Phase 2: State System Upgrade

### 2.1 Replace or Extend the Current EMA State Machine

The current 4-state EMA system remains useful, but it is not rich enough for recovery and exhaustion analysis.

Add a higher-level state classifier:

- `Broken`
- `Repairing`
- `Recovering`
- `Accelerating`
- `Flattening`
- `Flatlining`

Suggested logic:

- `Broken`: negative position and weak impulse
- `Repairing`: still weak, but impulse and curvature improving
- `Recovering`: near/above EMA with positive impulse and improving quality
- `Accelerating`: positive position, positive impulse, positive curvature, strong quality
- `Flattening`: still elevated position, but impulse and curvature deteriorating
- `Flatlining`: low slope, low ER, low range expansion

Suggested file:

- `momentum_decel/composite/state_machine_v2.py`

Deliverable:

- new columns:
  - `advanced_state_code`
  - `advanced_state`
  - `days_in_advanced_state`

## Phase 3: New Score Layer

### 3.1 Recovery Score

Purpose:

- detect improving momentum after weakness or drawdown

Initial candidate inputs:

- `delta_ts_15_5`
- `curvature_c_30_z`
- change in `ER_15`
- change in `R^2_20`
- `d_close`
- days since EMA recapture

Suggested file:

- `momentum_decel/composite/recovery_score.py`

### 3.2 Flattening Score

Purpose:

- detect stalling or exhaustion while trend may still appear intact

Initial candidate inputs:

- negative `delta_ts_15_5`
- negative curvature
- falling `slope_d_high`
- falling `slope_envelope_width`
- weakening `ER_15`

Suggested file:

- `momentum_decel/composite/flattening_score.py`

### 3.3 Leadership Score

Purpose:

- rank sectors or industries by absolute plus relative strength

Initial candidate inputs:

- current `momentum_quality`
- relative-strength features versus SPY
- breadth of group members improving

Suggested file:

- `momentum_decel/composite/leadership_score.py`

### 3.4 Inflection Score

Purpose:

- isolate the visually strong curvature plus Theil-Sen-delta family

Initial candidate inputs:

- `curvature_c_30_z`
- `delta_ts_15_5`

Variants:

- close-based
- `d_close`-based
- relative-strength-based

Suggested file:

- `momentum_decel/composite/inflection_score.py`

Deliverable:

- per-symbol columns:
  - `recovery_score`
  - `flattening_score`
  - `leadership_score`
  - `inflection_score`

## Phase 4: Group Aggregation

### 4.1 Sector Group Aggregates

Build daily group summaries using:

- median score
- median impulse
- median curvature
- percent of members in strong states
- percent of members in weak states
- dispersion between top and bottom members

Suggested files:

- `momentum_decel/groups/__init__.py`
- `momentum_decel/groups/aggregator.py`

Outputs:

- `sector_group_summary.parquet`
- `sector_group_breadth.parquet`

### 4.2 Industry ETF Expansion

After sector-level group logic is working, extend the universe to industry ETFs.

Requirements:

- define a curated industry ETF universe
- assign each ETF to a higher-level sector or style bucket
- run the same feature and score stack

Deliverable:

- hierarchical dashboards and group-level reports

## Phase 5: Dashboard and CLI Extensions

### 5.1 CLI Commands

Add commands such as:

```bash
python cli.py recovery --start 2020-01-01
python cli.py flattening --start 2020-01-01
python cli.py leadership --start 2020-01-01
python cli.py relative --tickers XLK XLF XLY --start 2020-01-01
python cli.py groups --level sector --start 2020-01-01
```

Suggested CLI additions:

- `run_recovery`
- `run_flattening`
- `run_leadership`
- `run_relative`
- `run_groups`

### 5.2 Dashboard Changes

Extend the single-instrument dashboard with:

- advanced state row
- relative-strength row
- new score row for recovery, flattening, and leadership

Add a group dashboard showing:

- median group score
- breadth of improving members
- breadth of flattening members
- group dispersion

Deliverable:

- new Plotly HTML dashboards for symbol-level and group-level analysis

## Phase 6: Validation Layer

### 6.1 Recovery Study

Question:

- after 5-10% drawdowns, which features best predict positive forward returns?

Target outputs:

- `recovery_study_results.parquet`
- summary table of 20-day and 40-day forward return lifts by recovery bucket

Suggested file:

- `momentum_decel/validation/recovery_study.py`

### 6.2 Exhaustion Study

Question:

- before peaks, which features warn earliest with acceptable false positives?

Target outputs:

- `exhaustion_study_results.parquet`
- summary ranking of flattening and inflection features

Suggested file:

- `momentum_decel/validation/exhaustion_study.py`

### 6.3 Cross-Sectional Ranking Study

Question:

- do top-ranked sectors or industries by Leadership Score outperform bottom-ranked groups?

Target outputs:

- long-short spread series
- hit rate by rebalance horizon
- turnover and persistence diagnostics

Suggested file:

- `momentum_decel/validation/cross_sectional_rank.py`

## Phase 7: Recommended Build Order

### Step 1

Add derived impulse and persistence features.

### Step 2

Implement `Inflection Score` first.

Reason:

- it is the smallest new score
- it uses the strongest visual pair already identified
- it provides an immediate comparison against the current composite

### Step 3

Implement `Recovery Score` and `Flattening Score`.

### Step 4

Implement advanced states.

### Step 5

Add relative-strength features versus SPY.

### Step 6

Implement `Leadership Score`.

### Step 7

Add group aggregation for sectors.

### Step 8

Add new dashboards and CLI entry points.

### Step 9

Run the 3 validation studies.

### Step 10

Expand from sectors to industry ETFs.

## Suggested File Additions

```text
momentum_decel/
├── composite/
│   ├── state_machine_v2.py
│   ├── recovery_score.py
│   ├── flattening_score.py
│   ├── leadership_score.py
│   └── inflection_score.py
├── relative_strength/
│   ├── __init__.py
│   ├── ratio_loader.py
│   └── ratio_indicators.py
├── groups/
│   ├── __init__.py
│   └── aggregator.py
└── validation/
    ├── recovery_study.py
    ├── exhaustion_study.py
    └── cross_sectional_rank.py
```

## Success Criteria

This extension is successful if it can do all of the following:

- identify improving momentum after damage more clearly than the current composite
- identify flattening before breakdown more clearly than the current composite
- distinguish strong sectors from weak sectors on both absolute and relative bases
- summarize group breadth and leadership in a way that survives validation
- provide a clean path from sectors to industry ETFs without changing the framework

## Immediate Recommendation

The first concrete build should be:

1. `inflection_score.py`
2. `recovery_score.py`
3. `flattening_score.py`
4. advanced state machine

That is the shortest path to a materially better system while keeping the current codebase coherent.
