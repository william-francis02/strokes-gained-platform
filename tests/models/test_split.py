import pandas as pd

from src.models.split import time_based_group_split


def _make_df():
    return pd.DataFrame(
        {
            "tournament id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "date": pd.to_datetime(
                [
                    "2020-01-01", "2020-01-01",
                    "2020-02-01", "2020-02-01",
                    "2020-03-01", "2020-03-01",
                    "2020-04-01", "2020-04-01",
                    "2020-05-01", "2020-05-01",
                ]
            ),
            "value": range(10),
        }
    )


def test_no_tournament_overlap():
    df = _make_df()
    train_df, test_df = time_based_group_split(df, test_frac=0.2)
    train_ids = set(train_df["tournament id"])
    test_ids = set(test_df["tournament id"])
    assert train_ids.isdisjoint(test_ids)


def test_test_set_is_most_recent():
    df = _make_df()
    train_df, test_df = time_based_group_split(df, test_frac=0.2)
    assert test_df["tournament id"].unique().tolist() == [5]
    assert train_df["date"].max() < test_df["date"].min()


def test_all_rows_preserved():
    df = _make_df()
    train_df, test_df = time_based_group_split(df, test_frac=0.2)
    assert len(train_df) + len(test_df) == len(df)
