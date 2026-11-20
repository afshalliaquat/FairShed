"""Tests for modeling and train/test splitting."""

import numpy as np
import pandas as pd
from src.models import (
    chronological_split,
    train_linear_regression,
    train_random_forest,
    save_models,
    load_saved_models,
)


def test_chronological_split_prevents_leakage():
    # 100 rows ordered in time
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "feature_a": np.random.randn(100),
            "energy_consumption_mw": 10000.0 + np.arange(100),
        }
    )

    X_train, X_test, y_train, y_test, ts_train, ts_test = chronological_split(
        df, feature_cols=["feature_a"], test_size=0.2
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    # Crucial time-series check: Train end strictly precedes test start
    assert ts_train.max() < ts_test.min()


def test_linear_regression_and_random_forest_fit_predict(tmp_path):
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "lag_1h": np.linspace(10000, 15000, n),
            "hour": np.random.randint(0, 24, size=n),
            "temperature_c": np.random.uniform(5, 30, size=n),
        }
    )
    y = pd.Series(X["lag_1h"] * 0.9 + X["temperature_c"] * 50 + 500)

    lr_model = train_linear_regression(X, y)
    lr_preds = lr_model.predict(X)
    assert len(lr_preds) == n

    rf_model = train_random_forest(X, y, n_estimators=10, max_depth=5)
    rf_preds = rf_model.predict(X)
    assert len(rf_preds) == n

    # Test save and reload
    models_dict = {"linear_regression_model": lr_model, "random_forest_model": rf_model}
    save_models(models_dict, feature_cols=list(X.columns), output_dir=tmp_path)

    loaded_dict, loaded_cols = load_saved_models(models_dir=tmp_path)
    assert "linear_regression_model" in loaded_dict
    assert "random_forest_model" in loaded_dict
    assert loaded_cols == list(X.columns)
