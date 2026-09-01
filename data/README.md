# Data Sources

## Primary dataset

**`raw/ASA All PGA Raw Data - Tourn Level.csv`**

- 36,864 rows, player × tournament grain
- Coverage: 2015–2022, 333 tournaments
- Source: Kaggle [`robikscube/pga-tour-golf-data-20152022`](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022)
- Key columns: `sg_putt`, `sg_arg`, `sg_app`, `sg_ott`, `sg_total`, `finish`, `made_cut`

## Missing data policy

~20.84% of rows have null `sg_*` values, from players who withdrew or were
disqualified mid-tournament.

- **Finish-position regression target:** drop these rows entirely.
- **`made_cut` classifier:** treat withdrawal as excluded (not counted as a
  missed cut). Withdrawing mid-tournament is a different event from missing
  the cut on score, and shouldn't be modeled as the same outcome.

## Reference-only files (not training data)

**`raw/putt_baseline.csv`**, **`raw/shot_baseline.csv`**

Broadie's published expected-strokes curves. Used in a standalone demo
module only — not fed into the main model.

## Rejected

**`raw/OWGR_Ranking.csv`**

Single-week snapshot only (week 11, 2020). Cannot serve as a time-varying
feature across an 8-year dataset.
