"""Integration tests for the complete forecasting pipeline."""

import json
from pathlib import Path
from src.pipeline import run_pipeline, METRICS_JSON_PATH, FIGURES_DIR


def test_full_pipeline_run():
    summary_df = run_pipeline()

    assert not summary_df.empty
    assert len(summary_df) == 2
    assert set(summary_df["Model"]).issubset(
        {"linear_regression_model", "random_forest_model"}
    )

    # Check metrics JSON output
    assert METRICS_JSON_PATH.exists()
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert len(metrics) == 2

    # Check that diagnostic figures were created
    assert (FIGURES_DIR / "forecast_vs_actual.png").exists()
    assert (FIGURES_DIR / "feature_importance.png").exists()
