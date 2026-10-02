"""SageMaker scikit-learn inference handlers for churn predictions."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

FEATURE_ORDER = [
    "age",
    "monthly_spend",
    "tenure",
    "support_ticket",
]


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body: str, content_type: str):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    missing = [feature for feature in FEATURE_ORDER if feature not in payload]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    return np.asarray([[float(payload[feature]) for feature in FEATURE_ORDER]])


def predict_fn(input_data, model):
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    return {
        "prediction": "CHURN" if prediction == 1 else "STAY",
        "probability": probability,
    }


def output_fn(prediction, accept: str):
    if accept not in {"application/json", "*/*", None}:
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
