"""Evaluate the model for the SageMaker Pipeline quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.evaluation.evaluate import evaluate
from src.training.train import train_model
from src.processing.preprocess import load_dataset


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
    joblib.dump(train_model(load_dataset(args.train)), model_path)
    metrics = evaluate(model_path, args.test)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
