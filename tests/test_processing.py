import pandas as pd
import pytest

from src.processing.preprocess import load_dataset, split_dataset


def test_load_and_split_dataset(tmp_path):
    path = tmp_path / "churn.csv"
    pd.DataFrame(
        {
            "customer_id": range(8),
            "age": [20, 21, 30, 31, 40, 41, 50, 51],
            "monthly_spend": [1] * 8,
            "tenure": [1] * 8,
            "support_ticket": [0] * 8,
            "churn": [0, 1] * 4,
        }
    ).to_csv(path, index=False)

    train, test = split_dataset(load_dataset(path), test_size=0.25)

    assert len(train) == 6
    assert len(test) == 2
    assert set(train["churn"]) == {0, 1}


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"customer_id": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(path)
