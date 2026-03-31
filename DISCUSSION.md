# Senior Quant Note: Momentum Recovery, Flattening, and Leadership Framework

## Why the Current Composite Is Not Enough

The current composite is useful as a broad trend-quality measure, but it is trying to answer too many questions at once.

In practice there are at least 4 separate problems:

1. Is the symbol structurally strong or weak versus trend?
2. Is momentum improving or fading right now?
3. Is the move convex or concave, meaning accelerating or decelerating?
4. Is the path orderly and persistent, or noisy and low-quality?

Trying to collapse all of that into one scalar loses signal. A better system is to keep the dimensions separate, then build targeted scores and states on top of them.

## Core Axes

For each sector ETF, and later each industry ETF, compute these 4 latent dimensions.

### 1. Trend Position

Measures where price sits relative to the trend anchor.

Inputs:

- `d_close`
- `d_high`
- `ema_state`
- days above or below EMA125
- percent distance from EMA in ATR units

Interpretation:

- Strong positive values mean price is extended above trend.
- Negative values mean the trend is damaged or broken.

### 2. Trend Impulse

Measures whether momentum is improving or fading.

Inputs:

- `delta_ts`
- change in `slope_d_high`
- change in `ER`
- change in `R^2`

Interpretation:

- Positive impulse means thrust is rebuilding.
- Negative impulse means the existing trend is flattening or stalling.

### 3. Trend Curvature

Measures the shape of the move.

Inputs:

- `curvature_c_*`
- smoothed curvature
- curvature ranks or z-scores

Interpretation:

- Positive curvature means convexity and acceleration.
- Negative curvature means concavity and deceleration.

### 4. Trend Quality

Measures whether the move is orderly or noisy.

Inputs:

- `ER`
- `R^2`
- envelope compression or expansion
- optionally Hurst if it survives validation

Interpretation:

- High quality means directional, orderly trend.
- Low quality means chop, instability, or flatlining.

## State System I Would Actually Use

The state machine should move beyond just accelerating/decelerating/broken. I would use these mutually exclusive states.

### Broken

Conditions:

- below EMA
- negative slope
- weak trend quality

Meaning:

- the trend is damaged and not yet repairing

### Repairing

Conditions:

- still weak or below trend
- `delta_ts` turning up
- curvature turning up

Meaning:

- damage remains, but the move is trying to stabilize

### Recovering

Conditions:

- near or above EMA
- positive `delta_ts`
- positive curvature
- improving `ER` and `R^2`

Meaning:

- momentum is improving after prior weakness or drawdown

### Accelerating

Conditions:

- positive slope
- positive `delta_ts`
- positive curvature
- strong trend quality

Meaning:

- strong directional momentum with confirmation

### Flattening

Conditions:

- still above trend or only mildly damaged
- `delta_ts < 0`
- curvature rolling over
- envelope compressing
- `ER` fading

Meaning:

- likely early warning of exhaustion

### Flatlining

Conditions:

- slope near zero
- low `ER`
- low envelope width
- low dispersion of daily movement

Meaning:

- trend has stopped progressing and is not yet meaningfully reversing

## The Three Scores I Would Build

I would not build one master score for every use case. I would build 3 separate scores.

### 1. Recovery Score

Use case:

- "This symbol is improving after damage or drawdown."

Candidate construction:

`Recovery = rank(delta_ts_15_5) + rank(curvature_c_30_z) + rank(change in ER_15) + rank(change in R2_20) + bonus for d_close crossing above 0`

Interpretation:

- high values mean thrust is rebuilding
- should be most useful after `Broken` or `Repairing`, not as a general ranking signal

### 2. Flattening Score

Use case:

- "This uptrend is stalling or exhausting."

Candidate construction:

High when:

- `delta_ts < 0`
- curvature is negative
- `slope_d_high` is falling
- `slope_envelope_width` is falling
- `ER` is weakening

Interpretation:

- this should be the main early-warning score for trend exhaustion

### 3. Leadership Score

Use case:

- "This sector or industry group is stronger than peers."

Candidate construction:

`Leadership = trend quality + relative momentum vs SPY + breadth of members improving`

Interpretation:

- not just whether a symbol is strong, but whether it is strong relative to the market and its peers

## Curvature Plus Theil-Sen Delta

This pair is especially promising.

Why it matters:

- curvature is a second-derivative style measure of inflection
- `delta_ts` measures the change in robust trend slope

Together they form a clean "impulse inflection" family.

### Proposed Inflection Score

`Inflection = 0.5 * rank(curvature_c_30_z) + 0.5 * rank(delta_ts_15_5)`

Variants worth testing:

- `Inflection_close`: on close
- `Inflection_dclose`: on `d_close`
- `Inflection_relative`: on `log(ETF / SPY)`

My prior is that the relative version will be particularly useful for sector rotation and later for industry ETF ranking.

## Relative Strength Must Be Added

For sector ETFs and industry ETFs, price-level momentum alone is not enough. The system should also run on relative series.

Key idea:

- build the same indicator stack on `ETF / SPY`

That lets you ask:

- is XLK strong?
- and separately, is XLK stronger than SPY?

Those are not the same question.

Relative-series indicators should become first-class inputs to Leadership and Recovery.

## Group-Level System

For sectors now, industries later, I would aggregate groups using both central tendency and breadth.

Metrics:

- median `momentum_quality`
- median `delta_ts`
- median curvature
- percent in `Recovering` or `Accelerating`
- percent in `Flattening` or `Broken`
- dispersion between strongest and weakest members

This allows statements like:

- Energy is improving broadly
- Tech remains strong, but leadership is narrowing
- Defensives are stronger than cyclicals in aggregate

Important rule:

- high median with weak breadth is fragile
- rising breadth with rising median impulse is healthy

## Hierarchical Expansion Path

I would expand the system in layers.

### Level 1

- SPY

### Level 2

- SPDR sector ETFs

### Level 3

- industry ETFs

### Level 4, optional

- stock breadth within each sector or industry group

Then the framework can answer:

- is SPY improving?
- are sectors confirming?
- are industries broadening?
- is leadership rotating?

That is much more useful than a flat ranking table.

## Validation Studies I Would Add Next

The current drawdown event study is only one slice of the problem. I would add 3 more studies.

### 1. Recovery Study

Question:

- after 5-10% drawdowns, which features best predict positive 20-day and 40-day forward returns?

Goal:

- validate Recovery Score

### 2. Exhaustion Study

Question:

- before peaks, which features roll over earliest with an acceptable false-positive rate?

Goal:

- validate Flattening Score

### 3. Cross-Sectional Ranking Study

Question:

- if sectors are ranked weekly by Leadership Score, do the top-ranked sectors outperform the bottom-ranked sectors?

Goal:

- validate relative rotation usefulness

## Strongest Practical Suggestions

- add ratio-series indicators on `ETF / SPY`
- split the current composite into `Recovery`, `Flattening`, and `Leadership`
- keep curvature plus `delta_ts` as a dedicated inflection family
- add persistence features such as days in `Recovering` or days in `Flattening`
- for group analysis, always use breadth plus median, never median alone

## Concrete Next Modules

If this framework is implemented in the current repo, the next logical modules are:

- `momentum_decel/composite/recovery_score.py`
- `momentum_decel/composite/flattening_score.py`
- `momentum_decel/composite/leadership_score.py`
- `momentum_decel/composite/inflection_score.py`
- `momentum_decel/relative_strength/ratio_loader.py`
- `momentum_decel/relative_strength/ratio_indicators.py`
- `momentum_decel/groups/aggregator.py`
- `momentum_decel/validation/recovery_study.py`
- `momentum_decel/validation/exhaustion_study.py`
- `momentum_decel/validation/cross_sectional_rank.py`

This would turn the current system from a single composite monitor into a proper trend-recovery, exhaustion, and rotation framework.

