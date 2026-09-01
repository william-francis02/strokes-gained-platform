"""Leakage-safe train/test splitting for tournament-grain data.

Rows share a tournament id (same course, weather, field), so a random
row-level split would leak tournament-level signal across train and test.
Splitting must be grouped by tournament id.
"""

import pandas as pd


def time_based_group_split(
    df: pd.DataFrame,
    tournament_col: str = "tournament id",
    date_col: str = "date",
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by tournament, holding out the most recent tournaments as test.

    Each tournament's earliest date determines its position in the
    chronological ordering; the last `test_frac` share of tournaments (by
    count) become the test set. No tournament id appears in both outputs.
    """
    tournament_dates = df.groupby(tournament_col)[date_col].min().sort_values()
    n_tournaments = len(tournament_dates)
    n_test = round(n_tournaments * test_frac)

    train_ids = set(tournament_dates.index[: n_tournaments - n_test])
    test_ids = set(tournament_dates.index[n_tournaments - n_test :])
    assert train_ids.isdisjoint(test_ids)

    train_df = df[df[tournament_col].isin(train_ids)]
    test_df = df[df[tournament_col].isin(test_ids)]
    return train_df, test_df
