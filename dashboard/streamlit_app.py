"""Read-only dashboard over the MLflow runs logged by src/models/train.py.

Three sections: a model comparison table (one row per model, latest run
wins), a feature importance chart for the boosting model, and a live
prediction tool against the exported pipeline in models/. All three read
already-computed values — nothing here retrains a model or recomputes
permutation importance.

Styling: a "Fairway" theme (deep green / sand-gold on warm off-white,
system fonts only — no CDN font loading, so the page renders identically
offline). Layout/table/chart/card styling is an injected CSS block in
main(); Streamlit's own widget accent color (sliders, buttons, etc.) is set
via ../.streamlit/config.toml's [theme] section, since that color is baked
into widgets as computed inline styles that plain CSS can't override.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import EXPERIMENT_NAME, FEATURE_COLUMNS, IMPORTANCE_RUN_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
MODEL_PATH = REPO_ROOT / "models" / "gradient_boosting.joblib"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# Matches the impurity/permutation colors train.py's own matplotlib plot uses
# (plot_feature_importance), so this chart and the PNG artifact agree.
IMPORTANCE_COLORS = ["#4C72B0", "#DD8452"]
# Wide enough to cover the ~99th-percentile range of each strokes-gained
# column (see data/processed/tournament_classification.csv) without forcing
# an unreadably long slider.
INPUT_MIN, INPUT_MAX, INPUT_STEP = -4.0, 4.0, 0.1
# Plain-English names for the sliders; the technical column name (sg_putt
# etc.) still shows underneath each one as a small caption.
FEATURE_LABELS = {
    "sg_putt": "Putting",
    "sg_arg": "Around the Green",
    "sg_app": "Approach",
    "sg_ott": "Off the Tee",
}

FAIRWAY_CSS = """
<style>
:root {
    --fairway-green: #2D6A4F;
    --fairway-green-dark: #1B4332;
    --sand: #C9A227;
    --clay: #9B2226;
    --paper: #F7F5EF;
    --ink: #1F2A24;
    --border: #E3E0D6;
}
.stApp {
    background-color: var(--paper);
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
}
h1, h2, h3 {
    color: var(--fairway-green-dark) !important;
}

/* Title divider */
.title-divider {
    height: 4px;
    width: 100%;
    background-color: var(--sand);
    border-radius: 2px;
    margin: 0.25rem 0 1.75rem 0;
}

/* Section subtitles */
.section-subtitle {
    color: #6b6b63;
    font-size: 0.85rem;
    margin: -0.5rem 0 1rem 0;
}

/* Card sections (st.container(border=True, key="card-...")) */
div[class*="st-key-card-"] {
    background-color: #FFFFFF;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    padding: 24px !important;
    margin-bottom: 28px;
}

/* Buttons */
div[data-testid="stButton"] button {
    background-color: var(--fairway-green);
    color: white;
    border: none;
    border-radius: 6px;
}
div[data-testid="stButton"] button:hover {
    background-color: var(--sand);
    color: var(--fairway-green-dark);
}

/* Comparison table */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}
.comparison-table th {
    text-align: left;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.5rem 0.75rem;
    border-bottom: 3px solid var(--sand);
    color: var(--fairway-green-dark);
}
.comparison-table th.num, .comparison-table td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.comparison-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
}
.comparison-table tr.winner-row {
    background-color: rgba(45, 106, 79, 0.08); /* #2D6A4F @ 8% */
}

/* Prediction result badge */
.prediction-card {
    padding: 1.5rem 2rem;
    border-radius: 10px;
    border-left: 8px solid;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
.prediction-card.made {
    background-color: #E8F3EC;
    border-left-color: var(--fairway-green);
}
.prediction-card.missed {
    background-color: #F6E9E9;
    border-left-color: var(--clay);
}
.prediction-verdict {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.prediction-card.made .prediction-verdict { color: var(--fairway-green-dark); }
.prediction-card.missed .prediction-verdict { color: var(--clay); }
.prediction-proba {
    font-size: 1rem;
    color: #555;
    margin-top: 0.25rem;
}

/* Slider label row (feature name + bold current value) */
.slider-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    margin-bottom: -0.5rem;
}
.slider-value {
    font-weight: 700;
    color: var(--fairway-green-dark);
}
.slider-tech-caption {
    color: #9a9a90;
    font-size: 0.75rem;
    margin: -0.4rem 0 0.9rem 0;
}

/* Slider track/thumb color comes from .streamlit/config.toml's primaryColor
   (#2D6A4F) — Streamlit bakes it into a computed inline background-image
   gradient per render, which plain CSS selectors can't reach. Only the
   default floating value bubble is overridden here, since it's replaced by
   the bold inline value in .slider-label-row. */
div[data-testid="stSliderThumbValue"] {
    display: none;
}
</style>
"""


@st.cache_data(ttl=30)
def load_runs(tracking_uri: str, experiment_name: str) -> pd.DataFrame:
    """All finished runs for the experiment, as returned by MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(experiment_names=[experiment_name])


def latest_per_run_name(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated executions of train.py to one row per model.

    train.py logs a fresh run every time it's executed, so the same model
    accumulates many rows over time. Dedup on params.model_type rather than
    the run name tag: a handful of runs in this experiment predate always
    passing run_name and carry MLflow's autogenerated names (e.g.
    "adaptable-midge-154") instead of "logistic_regression" etc, which would
    otherwise slip past a dedup keyed on the run name and double up a row.
    Keeps the most recent row per model, sorted best-first by test ROC AUC
    to match the comparison table train.py prints to console.
    """
    latest = runs_df.sort_values("start_time", ascending=False).drop_duplicates(
        subset="params.model_type", keep="first"
    )
    return latest.sort_values("metrics.test_roc_auc", ascending=False)


def render_comparison_table(latest_df: pd.DataFrame) -> None:
    st.header("Model comparison")
    st.markdown(
        '<p class="section-subtitle">Three different prediction approaches were '
        "tested — this shows how each performed.</p>",
        unsafe_allow_html=True,
    )

    display_df = pd.DataFrame(
        {
            "run": latest_df["tags.mlflow.runName"],
            "model_type": latest_df["params.model_type"],
            **{
                metric: latest_df[f"metrics.test_{metric}"].round(4) for metric in METRIC_COLUMNS
            },
            "started": latest_df["start_time"].dt.strftime("%Y-%m-%d %H:%M"),
        }
    )

    numeric_cols = set(METRIC_COLUMNS)
    winner_run = display_df.iloc[0]["run"] if not display_df.empty else None

    header_cells = "".join(
        f'<th class="num">{col}</th>' if col in numeric_cols else f"<th>{col}</th>"
        for col in display_df.columns
    )
    body_rows = []
    for _, row in display_df.iterrows():
        row_class = "winner-row" if row["run"] == winner_run else ""
        cells = "".join(
            f'<td class="num">{row[col]}</td>' if col in numeric_cols else f"<td>{row[col]}</td>"
            for col in display_df.columns
        )
        body_rows.append(f'<tr class="{row_class}">{cells}</tr>')

    table_html = (
        f'<table class="comparison-table"><thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_feature_importance(latest_df: pd.DataFrame) -> None:
    st.header("Feature importance")
    st.markdown(
        '<p class="section-subtitle">Which strokes-gained stats mattered most '
        "in predicting the cut.</p>",
        unsafe_allow_html=True,
    )

    boosting_rows = latest_df[latest_df["tags.mlflow.runName"].isin(IMPORTANCE_RUN_NAMES)]
    if boosting_rows.empty:
        st.info(
            f"No run named {' or '.join(IMPORTANCE_RUN_NAMES)} found — "
            "run `python -m src.models.train` first."
        )
        return

    run = boosting_rows.iloc[0]
    impurity_vals = [run[f"metrics.importance_{f}"] for f in FEATURE_COLUMNS]
    perm_vals = [run[f"metrics.perm_importance_{f}"] for f in FEATURE_COLUMNS]
    feature_labels = [FEATURE_LABELS[f] for f in FEATURE_COLUMNS]

    st.caption(f"From the latest '{run['tags.mlflow.runName']}' run")

    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_impurity = ax.bar(
        x - width / 2,
        impurity_vals,
        width,
        label="How much the model relied on this stat",
        color=IMPORTANCE_COLORS[0],
    )
    bars_perm = ax.bar(
        x + width / 2,
        perm_vals,
        width,
        label="How much performance drops without this stat",
        color=IMPORTANCE_COLORS[1],
    )
    ax.bar_label(bars_impurity, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_perm, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontweight="bold")
    ax.set_title(f"Feature importance — {run['tags.mlflow.runName']}")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    # Below the plot rather than the default upper-right: the longer,
    # plain-English legend labels are wide enough to overlap the "Approach"
    # bar's value label up there.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Measured two different ways and cross-checked against each other: how "
        "much the model leaned on each stat while learning, and how much "
        "prediction accuracy drops when that stat's values are scrambled. "
        "Technical names: impurity importance (train set) and permutation "
        "importance (test set)."
    )


@st.cache_resource
def load_model(model_path: Path):
    """The exported pipeline (scaler + boosting model), or None if not yet exported."""
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def render_prediction_tool(model_path: Path) -> None:
    st.header("Predict made cut")
    st.markdown(
        '<p class="section-subtitle">Enter strokes-gained values to get a live '
        "prediction from the exported gradient boosting pipeline.</p>",
        unsafe_allow_html=True,
    )

    model = load_model(model_path)
    if model is None:
        st.info(
            f"No exported model at `{model_path.relative_to(REPO_ROOT)}` — "
            "run `python -m src.models.train` first to fit and export one."
        )
        return

    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.caption(
            f"Using the exported '{meta['run_name']}' pipeline "
            f"(test ROC AUC {meta['test_metrics']['roc_auc']:.4f}, "
            f"exported {meta['exported_at'][:10]})"
        )

    left, right = st.columns([3, 2])

    inputs = {}
    with left:
        for feature in FEATURE_COLUMNS:
            key = f"slider_{feature}"
            current = st.session_state.get(key, 0.0)
            st.markdown(
                f'<div class="slider-label-row"><span>{FEATURE_LABELS[feature]}</span>'
                f'<span class="slider-value">{current:+.2f}</span></div>',
                unsafe_allow_html=True,
            )
            inputs[feature] = st.slider(
                FEATURE_LABELS[feature],
                INPUT_MIN,
                INPUT_MAX,
                current,
                INPUT_STEP,
                key=key,
                label_visibility="collapsed",
            )
            st.markdown(
                f'<p class="slider-tech-caption">{feature}</p>',
                unsafe_allow_html=True,
            )

    input_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    proba_made_cut = model.predict_proba(input_df)[0, 1]
    made_cut = proba_made_cut >= 0.5

    with right:
        st.markdown(
            f"""
            <div class="prediction-card {'made' if made_cut else 'missed'}">
                <div class="prediction-verdict">
                    {"MADE CUT" if made_cut else "MISSED CUT"}
                </div>
                <div class="prediction-proba">
                    {proba_made_cut:.0%} predicted probability of making the cut
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Making the Cut?", layout="wide")
    st.markdown(FAIRWAY_CSS, unsafe_allow_html=True)
    st.title("Making the Cut?")
    st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

    if st.button("Refresh"):
        load_runs.clear()

    runs_df = load_runs(TRACKING_URI, EXPERIMENT_NAME)
    if runs_df.empty:
        st.info(
            f"No runs found for experiment '{EXPERIMENT_NAME}' — "
            "run `python -m src.models.train` first."
        )
    else:
        latest_df = latest_per_run_name(runs_df)
        with st.container(border=True, key="card-comparison"):
            render_comparison_table(latest_df)
        with st.container(border=True, key="card-importance"):
            render_feature_importance(latest_df)

    with st.container(border=True, key="card-prediction"):
        render_prediction_tool(MODEL_PATH)


if __name__ == "__main__":
    main()
