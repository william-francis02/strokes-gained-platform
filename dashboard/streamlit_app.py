"""Two-page dashboard over the MLflow runs logged by src/models/train.py.

Page 1, "Making the Cut": the made_cut classifier — model comparison table,
feature importance chart, live prediction tool against the exported
pipeline in models/.

Page 2, "Predicting Finish Position": the finish_position regression
baseline — same shape (comparison table, live prediction tool), plus a
prominent banner reminding the reader this dataset only contains players
who made the cut (see data/README.md).

Both pages read already-computed values — nothing here retrains a model or
recomputes permutation importance. They share one Fairway-themed CSS block,
injected once in main() before handing off to whichever page is selected,
and share slider state (moving a slider on one page carries the value to
the other, so both predictions stay comparable for the same inputs).

Styling: a "Fairway" theme (deep green / sand-gold on warm off-white,
system fonts only — no CDN font loading, so the page renders identically
offline). Streamlit's own widget accent color (sliders, buttons, etc.) is
set via ../.streamlit/config.toml's [theme] section, since that color is
baked into widgets as computed inline styles that plain CSS can't override.
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

from src.models.train import (
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    IMPORTANCE_RUN_NAMES,
    REGRESSION_EXPERIMENT_NAME,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
MADE_CUT_MODEL_PATH = REPO_ROOT / "models" / "gradient_boosting.joblib"
FINISH_POSITION_MODEL_PATH = REPO_ROOT / "models" / "finish_position_gradient_boosting.joblib"

MADE_CUT_METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
FINISH_POSITION_METRIC_COLUMNS = ["mae", "rmse", "r2"]
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
# Shown on the finish-position page if the exported model's sidecar (which
# normally carries the precise, dynamically-computed bias note logged by
# run_regression()) isn't available yet.
GENERIC_BIAS_NOTE = (
    "This model only knows about players who made the cut. Predictions show "
    "finishing position assuming the player already made the cut — not a "
    "general tournament outcome."
)

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

/* Bento-grid model comparison cards, and the feature-importance "top
   predictor" callout (same classes, reused so both sections read as one
   visual system). Winner/callout card spans the full row via grid-column:
   1 / -1; the rest auto-fit into a row beneath it. */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin: 0.5rem 0 1rem 0;
}
.bento-card {
    background-color: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
}
.bento-card.winner {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, var(--fairway-green) 0%, var(--fairway-green-dark) 100%);
    border: none;
    padding: 26px 30px;
}
.bento-winner-badge {
    display: inline-block;
    background-color: var(--sand);
    color: var(--fairway-green-dark);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 100px;
    padding: 3px 12px;
    margin-bottom: 10px;
    width: fit-content;
}
.bento-model-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--fairway-green-dark);
}
.bento-card.winner .bento-model-name {
    color: #FFFFFF;
    font-size: 1.3rem;
}
.bento-model-type {
    font-size: 0.75rem;
    color: #6b6b63;
    margin-bottom: 6px;
}
.bento-card.winner .bento-model-type {
    color: rgba(255, 255, 255, 0.75);
}
.bento-headline {
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--fairway-green-dark);
    font-size: 2rem;
    margin: 4px 0 2px 0;
    font-variant-numeric: tabular-nums;
}
.bento-card.winner .bento-headline {
    color: var(--sand);
    font-size: 3.4rem;
}
.bento-headline-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b6b63;
    margin-bottom: 10px;
}
.bento-card.winner .bento-headline-label {
    color: rgba(255, 255, 255, 0.75);
}
.bento-supporting {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
}
.bento-supporting-item {
    font-size: 0.78rem;
    color: #6b6b63;
}
.bento-supporting-item .val {
    font-weight: 700;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}
.bento-card.winner .bento-supporting-item {
    color: rgba(255, 255, 255, 0.75);
}
.bento-card.winner .bento-supporting-item .val {
    color: #FFFFFF;
}
.bento-started {
    font-size: 0.7rem;
    color: #9a9a90;
    margin-top: 10px;
}
.bento-card.winner .bento-started {
    color: rgba(255, 255, 255, 0.6);
}

/* Gradient accent rule above the feature-importance chart, echoing the
   bento hero-card top border without touching the chart itself. */
.importance-accent {
    height: 4px;
    background: linear-gradient(90deg, var(--fairway-green), var(--sand), var(--fairway-green));
    border-radius: 2px;
    margin: 6px 0 18px 0;
}

/* Prediction result cards (made/missed badge, and the numeric variant) */
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
.prediction-card.numeric {
    background-color: #E8F3EC;
    border-left-color: var(--fairway-green);
}
.prediction-verdict {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.prediction-card.made .prediction-verdict { color: var(--fairway-green-dark); }
.prediction-card.missed .prediction-verdict { color: var(--clay); }
.prediction-card.numeric .prediction-verdict { color: var(--fairway-green-dark); }
.prediction-proba {
    font-size: 1rem;
    color: #555;
    margin-top: 0.25rem;
}
.prediction-margin {
    font-size: 0.95rem;
    color: #444;
    margin-top: 0.35rem;
}
.prediction-detail {
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

/* Bias banner (finish-position page) — deliberately loud: full-width,
   tinted clay background, solid left border. Reuses the winner-row's
   8%-opacity-tint recipe with clay instead of green. */
.bias-banner {
    background-color: rgba(155, 34, 38, 0.08);
    border-left: 6px solid var(--clay);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0 0 1.5rem 0;
    font-size: 0.95rem;
    color: var(--ink);
}
</style>
"""


@st.cache_data(ttl=30)
def load_runs(tracking_uri: str, experiment_name: str) -> pd.DataFrame:
    """All finished runs for the experiment, as returned by MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(experiment_names=[experiment_name])


def latest_per_run_name(
    runs_df: pd.DataFrame,
    sort_metric: str = "metrics.test_roc_auc",
    ascending: bool = False,
) -> pd.DataFrame:
    """Collapse repeated executions of train.py to one row per model.

    train.py logs a fresh run every time it's executed, so the same model
    accumulates many rows over time. Dedup on params.model_type rather than
    the run name tag: a handful of runs in this experiment predate always
    passing run_name and carry MLflow's autogenerated names (e.g.
    "adaptable-midge-154") instead of "logistic_regression" etc, which would
    otherwise slip past a dedup keyed on the run name and double up a row.
    Keeps the most recent row per model, sorted best-first by sort_metric
    (roc_auc descending for classification, mae ascending for regression) to
    match the comparison table train.py prints to console.
    """
    latest = runs_df.sort_values("start_time", ascending=False).drop_duplicates(
        subset="params.model_type", keep="first"
    )
    return latest.sort_values(sort_metric, ascending=ascending)


def render_comparison_table(
    latest_df: pd.DataFrame, metric_columns: list[str], primary_metric: str
) -> None:
    """Render each model as a bento card; the winner (best primary_metric,
    since latest_df is already sorted best-first — see latest_per_run_name)
    spans the full row with a large headline number, the rest sit smaller
    below it. primary_metric must be one of metric_columns.
    """
    st.header("Model comparison")
    st.markdown(
        '<p class="section-subtitle">Three different prediction approaches were '
        "tested — this shows how each performed.</p>",
        unsafe_allow_html=True,
    )

    supporting_columns = [metric for metric in metric_columns if metric != primary_metric]

    display_df = pd.DataFrame(
        {
            "run": latest_df["tags.mlflow.runName"],
            "model_type": latest_df["params.model_type"],
            **{
                metric: latest_df[f"metrics.test_{metric}"].round(4) for metric in metric_columns
            },
            "started": latest_df["start_time"].dt.strftime("%Y-%m-%d %H:%M"),
        }
    )

    winner_run = display_df.iloc[0]["run"] if not display_df.empty else None

    # display_df is already best-first (latest_df's sort order), so the
    # winner card — grid-column: 1 / -1 in CSS — lands first in the grid
    # and takes the full-width top row; the rest auto-fit beneath it.
    #
    # Built as single-line HTML (no embedded newlines): a blank-ish line
    # here — e.g. from an empty badge_html on a non-winner card — reads to
    # Streamlit's markdown renderer as a blank line inside a raw HTML
    # block, which ends the block early and dumps everything after it as
    # literal indented-code text instead of rendering it.
    cards_html = []
    for _, row in display_df.iterrows():
        is_winner = row["run"] == winner_run
        badge_html = '<div class="bento-winner-badge">Top performer</div>' if is_winner else ""
        supporting_html = "".join(
            f'<span class="bento-supporting-item">{metric} '
            f'<span class="val">{row[metric]}</span></span>'
            for metric in supporting_columns
        )
        cards_html.append(
            f'<div class="bento-card{" winner" if is_winner else ""}">'
            f"{badge_html}"
            f'<div class="bento-model-name">{row["run"]}</div>'
            f'<div class="bento-model-type">{row["model_type"]}</div>'
            f'<div class="bento-headline">{row[primary_metric]}</div>'
            f'<div class="bento-headline-label">{primary_metric}</div>'
            f'<div class="bento-supporting">{supporting_html}</div>'
            f'<div class="bento-started">Run {row["started"]}</div>'
            f"</div>"
        )

    st.markdown(f'<div class="bento-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


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

    # Bento callout for the top-ranked feature (by permutation importance,
    # the test-set measure) — the bar chart below still carries the full,
    # precise comparison across both importance measures; this just answers
    # "which one matters most" at a glance, reusing Part 1's card classes so
    # the two sections read as one visual system.
    top_idx = int(np.argmax(perm_vals))
    st.markdown(
        f'<div class="bento-grid">'
        f'<div class="bento-card winner">'
        f'<div class="bento-winner-badge">Top predictor</div>'
        f'<div class="bento-model-name">{feature_labels[top_idx]}</div>'
        f'<div class="bento-model-type">{FEATURE_COLUMNS[top_idx]}</div>'
        f'<div class="bento-headline">{perm_vals[top_idx]:.3f}</div>'
        f'<div class="bento-headline-label">permutation importance</div>'
        f'<div class="bento-supporting">'
        f'<span class="bento-supporting-item">impurity importance '
        f'<span class="val">{impurity_vals[top_idx]:.3f}</span></span>'
        f"</div>"
        f"</div>"
        f"</div>"
        f'<div class="importance-accent"></div>',
        unsafe_allow_html=True,
    )

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


SHARED_INPUTS_KEY = "shared_feature_inputs"
SHARED_INPUTS_REVISION_KEY = "shared_feature_inputs_rev"


def get_shared_inputs() -> dict:
    """The durable, page-agnostic store for the current slider values.

    Deliberately not a widget-bound session_state key: Streamlit garbage-
    collects a widget's own session_state entry once that widget stops
    being part of the rendered script (which happens on every page
    navigation, since each page calls st.slider from a different call
    site even when given the same key) — verified live, this silently
    resets sliders to their default the moment you flip pages, discarding
    whatever the other page had set. A plain, non-widget dict entry has no
    such lifecycle and survives navigation, so it's the actual source of
    truth; each page's own widgets sync from it (see render_feature_sliders).
    """
    if SHARED_INPUTS_KEY not in st.session_state:
        st.session_state[SHARED_INPUTS_KEY] = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        st.session_state[SHARED_INPUTS_REVISION_KEY] = 0
    return st.session_state[SHARED_INPUTS_KEY]


def reset_sliders_to_zero() -> None:
    get_shared_inputs().update(dict.fromkeys(FEATURE_COLUMNS, 0.0))
    st.session_state[SHARED_INPUTS_REVISION_KEY] += 1


def render_feature_sliders(widget_key_prefix: str) -> dict:
    """Render the 4 strokes-gained sliders plus a reset button; return the values.

    Values are shared across both pages via get_shared_inputs(), so setting
    up a player's stats on one page carries over when you flip to the
    other. widget_key_prefix keeps each page's own slider *widgets*
    independent (avoids both pages fighting over the same widget key).

    Each page's widgets normally just live their own life once created —
    that's what keeps the bold value label showing the live value instead
    of lagging a render behind. But that alone breaks Reset (and stale
    cross-page navigation) whenever a page's widgets already existed before
    the shared store changed underneath them, since an existing widget's
    key= default is ignored on later renders. shared_feature_inputs_rev is
    a version counter bumped only by reset_sliders_to_zero(); each page
    remembers the revision it last synced to (in
    f"{widget_key_prefix}_synced_rev") and force-syncs its own widget keys
    from the shared store whenever that's behind — including on a page's
    very first render, when it hasn't synced anything yet. This is only
    safe because it happens before the st.slider() calls below.
    """
    shared = get_shared_inputs()
    current_rev = st.session_state[SHARED_INPUTS_REVISION_KEY]
    synced_rev_key = f"{widget_key_prefix}_synced_rev"
    first_widget_key = f"{widget_key_prefix}_slider_{FEATURE_COLUMNS[0]}"
    # Two independent reasons a resync is needed, not just one: the revision
    # check catches "Reset was clicked while already on this page" (the
    # widget keys still exist then, just holding stale values); the missing-
    # key check catches "we're back on this page after being away" — a
    # widget's session_state entry is garbage-collected the moment it stops
    # being part of the rendered script, i.e. on every page navigation, so
    # synced_rev_key alone (itself NOT a widget key, so it survives) can
    # claim "already synced to revision 0" while the actual widget keys it's
    # supposedly tracking are long gone — verified live, that combination
    # raises a KeyError below rather than silently misbehaving.
    if first_widget_key not in st.session_state or st.session_state.get(synced_rev_key) != current_rev:
        for feature in FEATURE_COLUMNS:
            st.session_state[f"{widget_key_prefix}_slider_{feature}"] = shared[feature]
        st.session_state[synced_rev_key] = current_rev

    inputs = {}
    for feature in FEATURE_COLUMNS:
        key = f"{widget_key_prefix}_slider_{feature}"
        current = st.session_state[key]
        st.markdown(
            f'<div class="slider-label-row"><span>{FEATURE_LABELS[feature]}</span>'
            f'<span class="slider-value">{current:+.2f}</span></div>',
            unsafe_allow_html=True,
        )
        value = st.slider(
            FEATURE_LABELS[feature],
            INPUT_MIN,
            INPUT_MAX,
            current,
            INPUT_STEP,
            key=key,
            label_visibility="collapsed",
        )
        shared[feature] = value
        inputs[feature] = value
        st.markdown(
            f'<p class="slider-tech-caption">{feature}</p>',
            unsafe_allow_html=True,
        )

    # on_click runs before the rerun that recreates the sliders above, which
    # is the only supported way to change a widget's value via a button —
    # setting session_state after the widget's already been instantiated
    # this run would raise. reset_sliders_to_zero() only touches the shared
    # store + revision counter; the force-sync above is what actually
    # applies it to this page's own widget keys, on this rerun.
    st.button(
        "Reset to 0", key=f"{widget_key_prefix}_reset_sliders", on_click=reset_sliders_to_zero
    )
    return inputs


def ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', 11-13 -> 'Nth'. No tie prefix."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def render_made_cut_prediction(model_path: Path) -> None:
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
    with left:
        inputs = render_feature_sliders("made_cut")

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


def render_finish_position_prediction(model_path: Path) -> None:
    st.header("Predict finish position")
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
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta:
        st.caption(
            f"Using the exported '{meta['run_name']}' pipeline "
            f"(test MAE {meta['test_metrics']['mae']:.2f}, "
            f"test R² {meta['test_metrics']['r2']:.3f}, "
            f"exported {meta['exported_at'][:10]})"
        )

    left, right = st.columns([3, 2])
    with left:
        inputs = render_feature_sliders("finish_position")

    input_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    predicted_position = model.predict(input_df)[0]
    # Finish position can't be below 1st; the raw regression output has no
    # such floor, so the displayed number is clamped for the same reason the
    # ordinal formatting below drops the "T" (tie) prefix — read as a
    # display concern, not a change to what the model actually predicted.
    rounded_position = max(1, round(predicted_position))

    with right:
        margin_html = ""
        if meta:
            mae = meta["test_metrics"]["mae"]
            margin_html = (
                f'<div class="prediction-margin">Typical error: ±{mae:.1f} '
                "positions, based on test performance</div>"
            )
        st.markdown(
            f"""
            <div class="prediction-card numeric">
                <div class="prediction-verdict">{ordinal(rounded_position)}</div>
                {margin_html}
                <div class="prediction-detail">Assumes the player made the cut —
                    see the note above.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_bias_banner(model_path: Path) -> None:
    meta_path = model_path.with_suffix(".json")
    note = GENERIC_BIAS_NOTE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        note = meta.get("data_bias_note", GENERIC_BIAS_NOTE)
    st.markdown(
        f'<div class="bias-banner"><strong>Read this before trusting a number '
        f"below:</strong> {note}</div>",
        unsafe_allow_html=True,
    )


def render_made_cut_page() -> None:
    st.title("Making the Cut")
    st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

    if st.button("Refresh", key="refresh_made_cut"):
        load_runs.clear()

    runs_df = load_runs(TRACKING_URI, EXPERIMENT_NAME)
    if runs_df.empty:
        st.info(
            f"No runs found for experiment '{EXPERIMENT_NAME}' — "
            "run `python -m src.models.train` first."
        )
    else:
        latest_df = latest_per_run_name(runs_df)
        with st.container(border=True, key="card-comparison-made-cut"):
            render_comparison_table(latest_df, MADE_CUT_METRIC_COLUMNS, primary_metric="roc_auc")
        with st.container(border=True, key="card-importance"):
            render_feature_importance(latest_df)

    with st.container(border=True, key="card-prediction-made-cut"):
        render_made_cut_prediction(MADE_CUT_MODEL_PATH)


def render_finish_position_page() -> None:
    st.title("Predicting Finish Position")
    st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)
    render_bias_banner(FINISH_POSITION_MODEL_PATH)

    if st.button("Refresh", key="refresh_finish_position"):
        load_runs.clear()

    runs_df = load_runs(TRACKING_URI, REGRESSION_EXPERIMENT_NAME)
    if runs_df.empty:
        st.info(
            f"No runs found for experiment '{REGRESSION_EXPERIMENT_NAME}' — "
            "run `python -m src.models.train` first."
        )
    else:
        latest_df = latest_per_run_name(
            runs_df, sort_metric="metrics.test_mae", ascending=True
        )
        with st.container(border=True, key="card-comparison-finish-position"):
            render_comparison_table(
                latest_df, FINISH_POSITION_METRIC_COLUMNS, primary_metric="mae"
            )

    with st.container(border=True, key="card-prediction-finish-position"):
        render_finish_position_prediction(FINISH_POSITION_MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Strokes Gained Dashboard", layout="wide")
    st.markdown(FAIRWAY_CSS, unsafe_allow_html=True)

    made_cut_page = st.Page(render_made_cut_page, title="Making the Cut", icon="⛳", default=True)
    finish_position_page = st.Page(
        render_finish_position_page, title="Predicting Finish Position", icon="📈"
    )
    st.navigation([made_cut_page, finish_position_page]).run()


if __name__ == "__main__":
    main()
