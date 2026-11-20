"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
from src.feature_engineering import create_features


def test_create_features_structure_and_leakage_prevention():
    # Create 300 hours of synthetic data
    timestamps = pd.date_range("2024-01-01 00:00", periods=300, freq="h")
    consumption = 10000.0 + np.sin(np.linspace(0, 50, 300)) * 2000.0
    temps = 15.0 + np.cos(np.linspace(0, 50, 300)) * 5.0

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_consumption_mw": consumption,
            "temperature_c": temps,
        }
    )

    features_df, feature_cols = create_features(df, drop_na=True)

    # 1. Verify key features are present
    expected_features = [
        "hour",
        "dayofweek",
        "hour_sin",
        "hour_cos",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "cooling_degrees",
        "heating_degrees",
    ]
    for feat in expected_features:
        assert feat in feature_cols
        assert feat in features_df.columns

    # 2. Check no NaNs after drop_na
    assert features_df[feature_cols].isna().sum().sum() == 0

    # 3. Critical: Verify that lag_1h strictly matches the previous hour's actual target
    # In features_df, row index i's lag_1h should equal the previous timestamp's energy_consumption_mw
    for i in range(1, 10):
        prev_actual = df.loc[df["timestamp"] == features_df.loc[i, "timestamp"] - pd.Timedelta(hours=1), "energy_consumption_mw"].values[0]
        assert np.isclose(features_df.loc[i, "lag_1h"], prev_actual)


def test_cyclical_encoding_bounds():
    timestamps = pd.date_range("2024-01-01 00:00", periods=24, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_consumption_mw": [10000.0] * 24,
            "temperature_c": [20.0] * 24,
        }
    )
    features_df, _ = create_features(df, drop_na=False)

    assert features_df["hour_sin"].between(-1.0, 1.0).all()
    assert features_df["hour_cos"].between(-1.0, 1.0).all()
