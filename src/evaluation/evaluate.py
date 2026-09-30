"""Evaluate a churn model using business-relevant classification metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.processing.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset


def evaluate(model_path: str | Path, data_path: str | Path) -> dict[str, float]:
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
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(args.model, args.data)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
