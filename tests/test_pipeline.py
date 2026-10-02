import json

from pipelines.pipeline import run_pipeline


def test_pipeline_writes_metrics_and_passes_quality_gate(tmp_path):
    result = run_pipeline(
        "data/customer_churn.example.csv",
        tmp_path,
        min_roc_auc=0.5,
        min_f1=0.5,
    )

    assert result["status"] == "PASSED"
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["status"] == "PASSED"
    assert (tmp_path / "model.joblib").exists()
