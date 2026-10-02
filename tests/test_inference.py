import json

import numpy as np
import pytest

from src.inference.inference import input_fn, output_fn


def test_input_fn_preserves_feature_order():
    data = input_fn(
        json.dumps({
            "support_ticket": 10,
            "tenure": 4,
            "age": 42,
            "monthly_spend": 1250000,
        }),
        "application/json",
    )

    np.testing.assert_array_equal(data, [[42, 1250000, 4, 10]])


def test_input_fn_rejects_missing_feature():
    with pytest.raises(ValueError, match="Missing features"):
        input_fn(json.dumps({"age": 42}), "application/json")


def test_output_fn_returns_json():
    body, content_type = output_fn(
        {"prediction": "CHURN", "probability": 0.8},
        "application/json",
    )

    assert json.loads(body)["prediction"] == "CHURN"
    assert content_type == "application/json"
