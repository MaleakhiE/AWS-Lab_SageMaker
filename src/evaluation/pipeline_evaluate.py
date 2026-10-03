"""Evaluate the model for the SageMaker Pipeline quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

FEATURE_COLUMNS = ["age", "monthly_spend", "tenure", "support_ticket"]
TARGET_COLUMN = "churn"


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["customer_id", *FEATURE_COLUMNS, TARGET_COLUMN]
    missing = sorted(set(required) - set(frame.columns))
    if missing or frame.empty or frame[required].isnull().any().any():
        raise ValueError("Evaluation dataset is missing required data")
    return frame[required].copy()


def evaluate(model_path: Path, data_path: Path) -> dict[str, float]:
    model = joblib.load(model_path)
    frame = load_dataset(data_path)
    actual = frame[TARGET_COLUMN]
    predicted = model.predict(frame[FEATURE_COLUMNS])
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "model.joblib"
    train = load_dataset(args.train)
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    joblib.dump(model, model_path)
    metrics = evaluate(model_path, args.test)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
