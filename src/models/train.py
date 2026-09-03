"""Model comparison for the made_cut classifier: logistic regression baseline,
random forest, and gradient boosting.

Trains against data/processed/tournament_classification.csv, using a
time-based, tournament-grouped split (see src/models/split.py) so no
tournament appears in both train and test. Every model sees the identical
split and features, so the runs are directly comparable. Logs one MLflow run
per model to the same experiment (local file-based tracking under ./mlruns).
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.split import time_based_group_split

# XGBoost is optional: fall back to sklearn's gradient boosting when it is not
# installed, so the third run is always trained either way.
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier

    HAS_XGBOOST = False

DATA_PATH = Path("data/processed/tournament_classification.csv")
FEATURE_COLUMNS = ["sg_putt", "sg_arg", "sg_app", "sg_ott"]
TARGET_COLUMN = "made_cut"
EXPERIMENT_NAME = "made_cut_baseline"
RANDOM_STATE = 42
TEST_FRAC = 0.2
# Hyperparameters logged per run, when the estimator has them.
TUNED_PARAM_KEYS = ("C", "n_estimators", "learning_rate", "max_depth", "n_jobs")
# Feature importance is analysed for the boosting slot only; build_models() names
# it "xgboost" or "gradient_boosting" depending on what is installed.
IMPORTANCE_RUN_NAMES = ("gradient_boosting", "xgboost")
PERM_IMPORTANCE_SCORING = "roc_auc"
PERM_IMPORTANCE_REPEATS = 10
# Where the boosting pipeline is exported for use outside MLflow, and the test
# metric used to decide whether a rerun is allowed to overwrite it.
MODELS_DIR = Path("models")
EXPORT_METRIC_KEY = "roc_auc"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed classification dataset with date parsed."""
    return pd.read_csv(path, parse_dates=["date"])


def build_pipeline(estimator=None, random_state: int = RANDOM_STATE) -> Pipeline:
    """Wrap an estimator behind a StandardScaler.

    Defaults to the logistic regression baseline. Scaling is a no-op for the
    tree models, but keeping one pipeline shape means one signature and one
    served-model contract across every run.
    """
    if estimator is None:
        estimator = LogisticRegression(random_state=random_state)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def build_models(random_state: int = RANDOM_STATE) -> list[tuple[str, str, object]]:
    """The models to compare, as (run_name, model_type, estimator).

    The class split is a mild 59/41, so none of these carry class weighting.
    """
    if HAS_XGBOOST:
        boosting_name = "xgboost"
        boosting = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            eval_metric="logloss",
        )
    else:
        boosting_name = "gradient_boosting"
        boosting = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        )

    return [
        (
            "logistic_regression",
            "LogisticRegression",
            LogisticRegression(random_state=random_state),
        ),
        (
            "random_forest",
            "RandomForestClassifier",
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        ),
        (boosting_name, type(boosting).__name__, boosting),
    ]


def evaluate(y_true: pd.Series, y_pred, y_proba) -> dict:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def majority_class_baseline_accuracy(y_train: pd.Series, y_eval: pd.Series) -> float:
    """Accuracy of always predicting the train set's majority class."""
    majority_class = y_train.mode()[0]
    return (y_eval == majority_class).mean()


def plot_confusion_matrix(y_true, y_pred, out_path: Path) -> Path:
    """Save a confusion matrix plot for the test set predictions."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["missed cut", "made cut"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def compute_feature_importance(pipeline: Pipeline, X_test, y_test) -> dict:
    """Feature importance by two methods, plus whether they agree on the ranking.

    Built-in `feature_importances_` is impurity reduction measured on the
    *training* data, so it reflects what the model leaned on while fitting,
    overfitting included. Permutation importance is measured on the *test* set
    and reflects what actually generalizes. Disagreement between them is the
    interesting signal.
    """
    model = pipeline.named_steps["model"]
    # StandardScaler preserves column order, so positions map to FEATURE_COLUMNS.
    impurity = dict(zip(FEATURE_COLUMNS, (float(v) for v in model.feature_importances_)))

    # Permute the whole pipeline, not the bare estimator: the feature is shuffled
    # in raw units before scaling, which is the thing we actually want to measure.
    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=PERM_IMPORTANCE_SCORING,
        n_repeats=PERM_IMPORTANCE_REPEATS,
        random_state=RANDOM_STATE,
    )
    perm_mean = dict(zip(FEATURE_COLUMNS, (float(v) for v in perm.importances_mean)))
    perm_std = dict(zip(FEATURE_COLUMNS, (float(v) for v in perm.importances_std)))

    impurity_rank = rank_features(impurity)
    perm_rank = rank_features(perm_mean)
    # With only 4 features Spearman is coarse (it can only take a few values);
    # the exact rank-order comparison is the one to trust.
    rho = float(
        spearmanr(
            [impurity[f] for f in FEATURE_COLUMNS],
            [perm_mean[f] for f in FEATURE_COLUMNS],
        ).statistic
    )

    return {
        "impurity": impurity,
        "permutation_mean": perm_mean,
        "permutation_std": perm_std,
        "impurity_rank": impurity_rank,
        "permutation_rank": perm_rank,
        "agree": impurity_rank == perm_rank,
        "spearman": rho,
    }


def rank_features(importance: dict) -> list[str]:
    """Feature names ordered most to least important."""
    return [f for f, _ in sorted(importance.items(), key=lambda item: item[1], reverse=True)]


def print_feature_importance(importance: dict) -> None:
    """Print both methods side by side and state the agreement verdict."""
    print(f"\nFeature importance ({PERM_IMPORTANCE_SCORING}, {PERM_IMPORTANCE_REPEATS} repeats):")
    print(f"{'feature':<10}{'impurity':>10}{'permutation':>14}{'std':>9}")
    for feature in importance["impurity_rank"]:
        print(
            f"{feature:<10}{importance['impurity'][feature]:>10.4f}"
            f"{importance['permutation_mean'][feature]:>14.4f}"
            f"{importance['permutation_std'][feature]:>9.4f}"
        )

    impurity_order = " > ".join(importance["impurity_rank"])
    if importance["agree"]:
        print(f"Both methods agree on ranking: {impurity_order}")
    else:
        print("Methods DISAGREE on ranking:")
        print(f"  impurity (train):    {impurity_order}")
        print(f"  permutation (test):  {' > '.join(importance['permutation_rank'])}")
    print(f"Spearman rho: {importance['spearman']:.4f}")


def plot_feature_importance(importance: dict, out_path: Path) -> Path:
    """Save a two-panel bar chart of the four features ranked by each method.

    Two panels rather than grouped bars: impurity importances sum to 1 while
    permutation importances are in units of ROC AUC drop, so a shared axis would
    flatten the permutation panel to nothing.
    """
    fig, (ax_imp, ax_perm) = plt.subplots(1, 2, figsize=(11, 4))

    imp_features = importance["impurity_rank"][::-1]  # reversed: barh draws bottom-up
    ax_imp.barh(imp_features, [importance["impurity"][f] for f in imp_features], color="#4C72B0")
    ax_imp.set_title("Impurity importance (train)")
    ax_imp.set_xlabel("mean decrease in impurity")

    perm_features = importance["permutation_rank"][::-1]
    ax_perm.barh(
        perm_features,
        [importance["permutation_mean"][f] for f in perm_features],
        xerr=[importance["permutation_std"][f] for f in perm_features],
        color="#DD8452",
    )
    ax_perm.set_title(f"Permutation importance (test, {PERM_IMPORTANCE_REPEATS} repeats)")
    ax_perm.set_xlabel(f"drop in {PERM_IMPORTANCE_SCORING} when shuffled")

    verdict = "agree" if importance["agree"] else "DISAGREE"
    fig.suptitle(f"Feature importance — methods {verdict}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def get_git_commit() -> str | None:
    """Best-effort current git commit hash, for tracing a run back to code state."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def model_hyperparams(estimator) -> dict:
    """The hyperparameters worth putting in the MLflow params table.

    Unset (None) values are dropped so the UI columns stay readable.
    """
    params = estimator.get_params()
    return {k: params[k] for k in TUNED_PARAM_KEYS if params.get(k) is not None}


def export_if_better(
    pipeline: Pipeline,
    run_name: str,
    model_type: str,
    test_metrics: dict,
    commit: str | None,
    out_dir: Path = MODELS_DIR,
    metric_key: str = EXPORT_METRIC_KEY,
) -> bool:
    """Persist the fitted pipeline to disk if it beats the last exported run.

    Writes <run_name>.joblib plus a <run_name>.json sidecar recording the
    metric this comparison used, so a rerun of this script can never clobber
    a better model with a worse one. Both files are written to a temp path
    and moved into place with os.replace, so a crash mid-write can't leave a
    corrupt or mismatched pair on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{run_name}.joblib"
    meta_path = out_dir / f"{run_name}.json"

    new_score = test_metrics[metric_key]
    if meta_path.exists():
        previous_score = json.loads(meta_path.read_text())["test_metrics"][metric_key]
        if new_score <= previous_score:
            print(
                f"Not exporting {run_name}: test_{metric_key}={new_score:.4f} "
                f"does not beat existing export ({previous_score:.4f})"
            )
            return False

    metadata = {
        "run_name": run_name,
        "model_type": model_type,
        "features": FEATURE_COLUMNS,
        "test_metrics": test_metrics,
        "git_commit": commit,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    fd, tmp_model_path = tempfile.mkstemp(dir=out_dir, suffix=".joblib.tmp")
    os.close(fd)
    joblib.dump(pipeline, tmp_model_path)
    os.replace(tmp_model_path, model_path)

    fd, tmp_meta_path = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
    os.close(fd)
    Path(tmp_meta_path).write_text(json.dumps(metadata, indent=2))
    os.replace(tmp_meta_path, meta_path)

    print(f"Exported {run_name} to {model_path} (test_{metric_key}={new_score:.4f})")
    return True


def train_and_log(
    run_name: str,
    model_type: str,
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    split_params: dict,
    data_rows: int,
) -> dict:
    """Fit one model on the shared split and log it as a single MLflow run."""
    pipeline = build_pipeline(estimator)
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate(
        y_train, pipeline.predict(X_train), pipeline.predict_proba(X_train)[:, 1]
    )
    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_proba)
    baseline_accuracy = majority_class_baseline_accuracy(y_train, y_test)

    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics:  {test_metrics}")
    print(f"Majority-class baseline accuracy (test): {baseline_accuracy:.4f}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_type": model_type,
                "features": FEATURE_COLUMNS,
                **split_params,
                **model_hyperparams(estimator),
            }
        )
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("majority_class_baseline_accuracy", baseline_accuracy)

        commit = get_git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("data_rows", data_rows)

        cm_path = Path("confusion_matrix.png")
        plot_confusion_matrix(y_test, test_pred, cm_path)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink()

        if run_name in IMPORTANCE_RUN_NAMES:
            # Reuses the pipeline fitted above — no refit.
            importance = compute_feature_importance(pipeline, X_test, y_test)
            print_feature_importance(importance)

            mlflow.log_metrics({f"importance_{f}": v for f, v in importance["impurity"].items()})
            mlflow.log_metrics(
                {f"perm_importance_{f}": v for f, v in importance["permutation_mean"].items()}
            )
            mlflow.log_metrics(
                {f"perm_importance_std_{f}": v for f, v in importance["permutation_std"].items()}
            )
            mlflow.log_metric("importance_spearman", importance["spearman"])
            mlflow.set_tag("importance_methods_agree", importance["agree"])

            fi_path = Path("feature_importance.png")
            plot_feature_importance(importance, fi_path)
            mlflow.log_artifact(str(fi_path))
            fi_path.unlink()

            export_if_better(pipeline, run_name, model_type, test_metrics, commit)

        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
        )

    return test_metrics


def run(data_path: Path = DATA_PATH, test_frac: float = TEST_FRAC) -> dict[str, dict]:
    """Split once, then train and log every model against that same split.

    Returns test metrics keyed by run name.
    """
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows, {df['tournament id'].nunique()} tournaments")

    train_df, test_df = time_based_group_split(df, test_frac=test_frac)
    print(
        f"Train: {len(train_df)} rows, {train_df['tournament id'].nunique()} tournaments "
        f"({train_df['date'].min().date()} to {train_df['date'].max().date()})"
    )
    print(
        f"Test:  {len(test_df)} rows, {test_df['tournament id'].nunique()} tournaments "
        f"({test_df['date'].min().date()} to {test_df['date'].max().date()})"
    )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    # Logged identically on every run, which is what proves the comparison is fair.
    split_params = {
        "test_frac": test_frac,
        "split_strategy": "time_based_group",
        "random_state": RANDOM_STATE,
        "train_tournaments": train_df["tournament id"].nunique(),
        "test_tournaments": test_df["tournament id"].nunique(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    models = build_models()
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}
    for i, (run_name, model_type, estimator) in enumerate(models, start=1):
        print(f"\n[{i}/{len(models)}] {run_name} ({model_type})")
        results[run_name] = train_and_log(
            run_name,
            model_type,
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            split_params,
            len(df),
        )

    print(f"\nTest-set comparison ({len(models)} runs in experiment '{EXPERIMENT_NAME}'):")
    print(f"{'run':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'roc_auc':>10}")
    for run_name, metrics in sorted(
        results.items(), key=lambda item: item[1]["roc_auc"], reverse=True
    ):
        print(
            f"{run_name:<22}{metrics['accuracy']:>10.4f}{metrics['precision']:>11.4f}"
            f"{metrics['recall']:>9.4f}{metrics['f1']:>9.4f}{metrics['roc_auc']:>10.4f}"
        )

    return results


if __name__ == "__main__":
    run()
