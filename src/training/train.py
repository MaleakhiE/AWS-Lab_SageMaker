"""Train the local customer churn baseline model."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from src.processing.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
except ModuleNotFoundError:
    # SageMaker executes this file directly from the source directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from processing.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset


def train_model(frame: pd.DataFrame, n_estimators: int = 200) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(os.environ.get("SM_CHANNEL_TRAIN", "./data")) / "train.csv",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(os.environ.get("SM_MODEL_DIR", "./model")) / "model.joblib",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    model = train_model(load_dataset(args.input), args.n_estimators)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    # Include the serving module in model.tar.gz for the sklearn container.
    inference_source = Path(__file__).resolve().parents[1] / "inference" / "inference.py"
    code_dir = args.output.parent / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inference_source, code_dir / "inference.py")

    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
