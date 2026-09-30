import pandas as pd

from src.training.train import train_model


def test_train_model_predicts_binary_labels():
    frame = pd.DataFrame(
        {
            "customer_id": range(8),
            "age": [20, 21, 30, 31, 40, 41, 50, 51],
            "monthly_spend": [1] * 8,
            "tenure": [1] * 8,
            "support_ticket": [0, 1, 0, 1, 0, 1, 0, 1],
            "churn": [0, 1] * 4,
        }
    )

    predictions = train_model(frame, n_estimators=10).predict(frame[[
        "age", "monthly_spend", "tenure", "support_ticket"
    ]])

    assert set(predictions).issubset({0, 1})
