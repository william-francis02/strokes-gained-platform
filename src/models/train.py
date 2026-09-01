"""First baseline model: logistic regression for the made_cut classifier.

Trains against data/processed/tournament_classification.csv, using a
time-based, tournament-grouped split (see src/models/split.py) so no
tournament appears in both train and test. Logs params/metrics/artifacts to
MLflow (local file-based tracking under ./mlruns).
"""

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
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

DATA_PATH = Path("data/processed/tournament_classification.csv")
FEATURE_COLUMNS = ["sg_putt", "sg_arg", "sg_app", "sg_ott"]
TARGET_COLUMN = "made_cut"
EXPERIMENT_NAME = "made_cut_baseline"
RANDOM_STATE = 42
TEST_FRAC = 0.2


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed classification dataset with date parsed."""
    return pd.read_csv(path, parse_dates=["date"])


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Standardized logistic regression: the baseline model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=random_state)),
        ]
    )


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


def run(data_path: Path = DATA_PATH, test_frac: float = TEST_FRAC) -> dict:
    """Load data, split, train the baseline, log to MLflow, and return test metrics."""
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

    pipeline = build_pipeline()
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

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "features": FEATURE_COLUMNS,
                "test_frac": test_frac,
                "split_strategy": "time_based_group",
                "random_state": RANDOM_STATE,
                "train_tournaments": train_df["tournament id"].nunique(),
                "test_tournaments": test_df["tournament id"].nunique(),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            }
        )
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("majority_class_baseline_accuracy", baseline_accuracy)

        commit = get_git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("data_rows", len(df))

        cm_path = Path("confusion_matrix.png")
        plot_confusion_matrix(y_test, test_pred, cm_path)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink()

        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
        )

    return test_metrics


if __name__ == "__main__":
    run()
