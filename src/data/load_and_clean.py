"""Load and clean the ASA PGA tournament-level raw data.

Produces two independent processed outputs:
  - a regression dataset (targets: sg_total, finish_position), rows with
    null sg_total dropped
  - a classification dataset (target: made_cut), rows where the player
    withdrew/was disqualified, or is missing sg_* feature components,
    excluded

These filters are not all the same rows: null sg_total is a strokes-gained
data-coverage gap (many such rows are completed, cut-making tournament
entries), while withdrawal/DQ is a distinct signal carried in the `Finish`
column (`WD`, `W/D`, `DQ`).
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/ASA All PGA Raw Data - Tourn Level.csv")
PROCESSED_DIR = Path("data/processed")

EMPTY_COLUMNS = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]

SELECTED_COLUMNS = [
    "player",
    "tournament id",
    "date",
    "sg_putt",
    "sg_arg",
    "sg_app",
    "sg_ott",
    "sg_total",
    "Finish",
    "made_cut",
]

WITHDRAWAL_CODES = ["WD", "W/D", "DQ"]

COMPLETENESS_COLUMNS = ["sg_putt", "sg_arg", "sg_app", "sg_ott"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw tournament-level CSV, unmodified."""
    return pd.read_csv(path)


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the fully-empty Unnamed columns."""
    return df.drop(columns=EMPTY_COLUMNS)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select the columns needed by downstream modeling tasks."""
    return df[SELECTED_COLUMNS]


def parse_finish_position(finish: pd.Series) -> pd.Series:
    """Parse the Finish text column into a numeric finish position.

    Strips a leading "T" (tie marker), so "T18" -> 18 (tied players share
    the same numeric position as an untied finisher, the standard golf
    convention). Non-numeric codes (CUT, WD, W/D, DQ, MDF) don't represent
    a finish position and become null.
    """
    return pd.to_numeric(finish.str.replace(r"^T", "", regex=True), errors="coerce")


def filter_null_sg_total(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a null sg_total (strokes-gained data unavailable)."""
    return df[df["sg_total"].notna()]


def add_finish_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric finish_position column parsed from Finish."""
    return df.assign(finish_position=parse_finish_position(df["Finish"]))


def build_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the sg_total / finish_position regression task.

    Drops rows with a null sg_total (strokes-gained data unavailable for
    that tournament entry). Adds a numeric finish_position column parsed
    from Finish, then drops rows where finish_position is null (non-numeric
    Finish: CUT, WD, W/D, DQ, MDF) -- neither regression target can be null.
    """
    filtered = filter_null_sg_total(df)
    selected = select_columns(filtered)
    with_position = add_finish_position(selected)
    return with_position[with_position["finish_position"].notna()]


def filter_withdrawn(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows where the player withdrew or was disqualified.

    Finish in WD, W/D, DQ, so withdrawal is not counted as a missed cut.
    """
    return df[~df["Finish"].isin(WITHDRAWAL_CODES)]


def filter_incomplete_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any sg_putt/sg_arg/sg_app/sg_ott component.

    A row with no strokes-gained components can't be used as classifier
    input either, same reasoning as the regression sg_total filter.
    """
    return df.dropna(subset=COMPLETENESS_COLUMNS)


def build_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the made_cut classification task.

    Excludes withdrawn/DQ'd players and rows missing sg_* feature
    components.
    """
    filtered = filter_withdrawn(df)
    filtered = filter_incomplete_features(filtered)
    return select_columns(filtered)


def summarize_step(name: str, before: int, after: int) -> None:
    """Print row counts and % dropped for one cleaning step."""
    dropped = before - after
    pct = (dropped / before * 100) if before else 0.0
    print(f"{name}: {before} -> {after} rows ({dropped} dropped, {pct:.2f}%)")


def run(
    raw_path: Path = RAW_PATH, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, and write both processed outputs. Returns (regression_df, classification_df)."""
    raw = load_raw(raw_path)
    n_raw = len(raw)
    print(f"Loaded raw data: {n_raw} rows")

    cleaned = drop_empty_columns(raw)
    summarize_step("Drop empty columns (Unnamed: 2/3/4)", n_raw, len(cleaned))

    sg_total_filtered = filter_null_sg_total(cleaned)
    summarize_step("Regression: drop null sg_total", n_raw, len(sg_total_filtered))

    regression_df = build_regression_target(cleaned)
    summarize_step(
        "Regression: drop null finish_position (non-numeric Finish, e.g. CUT/WD/W-D/DQ/MDF)",
        n_raw,
        len(regression_df),
    )

    withdrawn_filtered = filter_withdrawn(cleaned)
    summarize_step(
        "Classification: exclude withdrawn/DQ (Finish in WD, W/D, DQ)",
        n_raw,
        len(withdrawn_filtered),
    )

    classification_df = select_columns(filter_incomplete_features(withdrawn_filtered))
    summarize_step(
        "Classification: drop rows with incomplete sg_* features",
        n_raw,
        len(classification_df),
    )

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    regression_path = processed_dir / "tournament_regression.csv"
    classification_path = processed_dir / "tournament_classification.csv"
    regression_df.to_csv(regression_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    print(f"Wrote {regression_path} ({len(regression_df)} rows)")
    print(f"Wrote {classification_path} ({len(classification_df)} rows)")

    return regression_df, classification_df


if __name__ == "__main__":
    run()
