# Data Sources

## Primary dataset

**`raw/ASA All PGA Raw Data - Tourn Level.csv`**

- 36,864 rows, player × tournament grain
- Coverage: 2015–2022, 333 tournaments
- Source: Kaggle [`robikscube/pga-tour-golf-data-20152022`](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022)
- Key columns: `sg_putt`, `sg_arg`, `sg_app`, `sg_ott`, `sg_total`, `finish`, `made_cut`

## Missing data policy

~20.84% of rows have null `sg_*` values (7,683 of 36,864). These are **not**
withdrawals — they're a strokes-gained data-coverage gap (e.g. no ShotLink
for that event/course); many of these rows belong to players who completed
all rounds and made the cut. Withdrawal/DQ is a separate, explicit signal
carried in the `Finish` column (`WD`, `W/D`, `DQ` — 312 rows), which does
not overlap with the null-`sg_*` rows at all.

- **`finish_position` regression target:** drop rows with null `sg_total`,
  then drop rows where `Finish` is non-numeric (`CUT`, `WD`, `W/D`, `DQ`,
  `MDF`) — a missed cut or withdrawal has no finish position to predict.
- **`made_cut` classifier:** exclude withdrawn/DQ'd rows (`Finish` in `WD`,
  `W/D`, `DQ`) so withdrawing isn't counted as a missed cut, and drop rows
  missing any of `sg_putt`/`sg_arg`/`sg_app`/`sg_ott` (unusable as
  classifier features).

Implemented in `src/data/load_and_clean.py`.

## Processed data

**Classification target (primary):** `processed/tournament_classification.csv`
- 28,869 rows (from 36,864 raw)
- Target: `made_cut` (0/1, clean, no nulls)
- Chosen as the primary model target — larger, unbiased sample, clean target
  column.

**Regression target (secondary/stretch):** `processed/tournament_regression.csv`
- 16,606 rows (from 36,864 raw)
- Target: `finish_position` (parsed from `Finish`, tie-prefix stripped, no
  nulls)
- **Known limitation:** this file only contains players who made the cut. It
  cannot answer "what predicts making the cut" — only "given a player made
  the cut, what predicts their finish position." A model trained on this set
  should not be described as predicting general tournament outcome without
  this caveat.

## Reference-only files (not training data)

**`raw/putt_baseline.csv`**, **`raw/shot_baseline.csv`**

Broadie's published expected-strokes curves. Used in a standalone demo
module only — not fed into the main model.

## Rejected

**`raw/OWGR_Ranking.csv`**

Single-week snapshot only (week 11, 2020). Cannot serve as a time-varying
feature across an 8-year dataset.
