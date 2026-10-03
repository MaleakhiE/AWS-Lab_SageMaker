"""Validate and split the customer churn dataset for local training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["age", "monthly_spend", "tenure", "support_ticket"]
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")
    if frame[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains null values in required columns")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("churn must contain only 0 or 1")
    return frame[REQUIRED_COLUMNS].copy()


def split_dataset(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test frames with a stratified target split."""
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("churn must contain both classes for a stratified split")
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    input_path = args.input
    if input_path.is_dir():
        candidates = sorted(input_path.glob("*.csv"))
        if len(candidates) != 1:
            raise ValueError("Input directory must contain exactly one CSV file")
        input_path = candidates[0]
    train, test = split_dataset(load_dataset(input_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
