"""Standalone, read-only diagnostic: made_cut probability for hand-picked inputs.

Loads models/gradient_boosting.joblib directly (no retraining, no app code)
and prints predict_proba for a few exact input combinations, plus a sweep
over sg_app to see the full response curve rather than isolated points.
"""

from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "gradient_boosting.joblib"
FEATURE_COLUMNS = ["sg_putt", "sg_arg", "sg_app", "sg_ott"]

FIXED_CASES = [
    {"sg_putt": 0.0, "sg_arg": -1.4, "sg_app": -0.2, "sg_ott": -0.9},
    {"sg_putt": 0.0, "sg_arg": -1.7, "sg_app": -2.5, "sg_ott": -0.9},
    {"sg_putt": 0.0, "sg_arg": -1.7, "sg_app": -3.2, "sg_ott": -0.9},
]

SWEEP_BASE = {"sg_putt": 0.0, "sg_arg": -1.7, "sg_ott": -0.9}
SWEEP_START, SWEEP_STOP, SWEEP_STEP = -0.2, -4.0, -0.2


def made_cut_probability(model, row: dict) -> float:
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return model.predict_proba(input_df)[0, 1]


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No exported model at {MODEL_PATH} — run `python -m src.models.train` first."
        )
    model = joblib.load(MODEL_PATH)

    print("Fixed cases:")
    for i, row in enumerate(FIXED_CASES, start=1):
        proba = made_cut_probability(model, row)
        row_str = ", ".join(f"{k}={v}" for k, v in row.items())
        print(f"  {i}. {row_str}  ->  made_cut probability = {proba:.4f}")

    print(
        f"\nsg_app sweep ({SWEEP_START} to {SWEEP_STOP}, step {SWEEP_STEP}), "
        f"holding {', '.join(f'{k}={v}' for k, v in SWEEP_BASE.items())}:"
    )
    sg_app = SWEEP_START
    steps = round((SWEEP_STOP - SWEEP_START) / SWEEP_STEP) + 1
    for _ in range(steps):
        row = {**SWEEP_BASE, "sg_app": round(sg_app, 2)}
        proba = made_cut_probability(model, row)
        print(f"  sg_app={row['sg_app']:>6.2f}  ->  made_cut probability = {proba:.4f}")
        sg_app += SWEEP_STEP


if __name__ == "__main__":
    main()
