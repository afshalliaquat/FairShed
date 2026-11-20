"""Tests for model evaluation metrics and feature importance utilities."""

import numpy as np
import pandas as pd
from src.evaluate import calculate_metrics, evaluate_all_models, extract_feature_importances
from src.models import train_linear_regression


def test_calculate_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    metrics = calculate_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE_%" in metrics
    assert "R2_Score" in metrics
    assert "MBE" in metrics

    # MAE should be 10.0
    assert np.isclose(metrics["MAE"], 10.0)
    # RMSE should be 10.0
    assert np.isclose(metrics["RMSE"], 10.0)
    # MBE should be 0.0 ((+10 -10 +10 -10) / 4)
    assert np.isclose(metrics["MBE"], 0.0)


def test_evaluate_all_models():
    X = pd.DataFrame({"feat": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 4, 6, 8, 10])

    model = train_linear_regression(X, y)
    models_dict = {"linear_model": model}

    summary_df, preds_dict = evaluate_all_models(models_dict, X, y)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 1
    assert "linear_model" in preds_dict
    assert summary_df.loc[0, "MAE"] < 0.01  # Perfect fit


def test_extract_feature_importances_fallback():
    # If a model doesn't have feature_importances_, should return empty DataFrame gracefully
    dummy_model = object()
    fi_df = extract_feature_importances(dummy_model, ["feat_a", "feat_b"])
    assert isinstance(fi_df, pd.DataFrame)
    assert fi_df.empty
