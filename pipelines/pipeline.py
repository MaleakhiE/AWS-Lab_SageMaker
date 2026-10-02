"""Local Phase 6 pipeline with a model-quality gate.

The same ordered stages can later be mapped to SageMaker Processing/Training
steps without allowing an unqualified model to proceed to registration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluate import evaluate
from src.processing.preprocess import load_dataset, split_dataset
from src.training.train import train_model


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    min_roc_auc: float = 0.80,
    min_f1: float = 0.75,
) -> dict[str, object]:
    """Run preprocess, train, evaluate, and fail before registration if needed."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = split_dataset(load_dataset(input_path))
    train_path = output / "train.csv"
    test_path = output / "test.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    model = train_model(train_frame)
    model_path = output / "model.joblib"
    import joblib

    joblib.dump(model, model_path)

    metrics = evaluate(model_path, test_path)
    passed = metrics["roc_auc"] >= min_roc_auc and metrics["f1"] >= min_f1
    result = {
        "status": "PASSED" if passed else "REJECTED",
        "metrics": metrics,
        "thresholds": {"roc_auc": min_roc_auc, "f1": min_f1},
        "model_path": str(model_path),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Model quality gate failed: {json.dumps(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-roc-auc", type=float, default=0.80)
    parser.add_argument("--min-f1", type=float, default=0.75)
    args = parser.parse_args()
    result = run_pipeline(
        args.input,
        args.output_dir,
        min_roc_auc=args.min_roc_auc,
        min_f1=args.min_f1,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
