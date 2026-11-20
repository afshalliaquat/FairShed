"""Feature engineering module for the AI-Powered Energy Consumption Forecasting System.

Extracts time-series features without data leakage:
- Calendar features (hour, day of week, month, weekend)
- Cyclical sine/cosine encodings (hour, month)
- Non-leaking lag features (t-1h, t-2h, t-24h, t-168h)
- Non-leaking rolling window statistics (past 24h mean, min, max, std)
- Temperature-derived heating and cooling degree metrics
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

DEFAULT_PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "energy_features.csv"
)


def create_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_col: str = "energy_consumption_mw",
    temp_col: str = "temperature_c",
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate time series features from cleaned energy consumption dataframe.

    All lag and rolling features are shifted by at least 1 time step to guarantee
    zero future data leakage during training and evaluation.

    Args:
        df: Cleaned dataframe with timestamp and target columns.
        timestamp_col: Name of datetime column.
        target_col: Name of target energy consumption column.
        temp_col: Name of temperature column (if present).
        drop_na: Whether to drop initial rows with NaNs resulting from lag window.

    Returns:
        Tuple of (feature_dataframe, feature_column_names)
    """
    data = df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.sort_values(by=timestamp_col).reset_index(drop=True)

    # 1. Calendar Features
    data["hour"] = data[timestamp_col].dt.hour
    data["dayofweek"] = data[timestamp_col].dt.dayofweek  # 0=Mon, 6=Sun
    data["dayofmonth"] = data[timestamp_col].dt.day
    data["month"] = data[timestamp_col].dt.month
    data["quarter"] = data[timestamp_col].dt.quarter
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # 2. Cyclical Encodings (smooth wrap-around for hours and months)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # 3. Non-leaking Lag Features (past values only)
    # lag_1h: immediate previous hour
    # lag_2h: 2 hours ago
    # lag_24h: same hour yesterday
    # lag_168h: same hour last week
    data["lag_1h"] = data[target_col].shift(1)
    data["lag_2h"] = data[target_col].shift(2)
    data["lag_24h"] = data[target_col].shift(24)
    data["lag_168h"] = data[target_col].shift(168)

    # 4. Non-leaking Rolling Window Statistics
    # Note: .shift(1) ensures the current target is not included in the window!
    shifted_target = data[target_col].shift(1)
    data["rolling_mean_24h"] = shifted_target.rolling(window=24).mean()
    data["rolling_std_24h"] = shifted_target.rolling(window=24).std()
    data["rolling_min_24h"] = shifted_target.rolling(window=24).min()
    data["rolling_max_24h"] = shifted_target.rolling(window=24).max()

    # 5. Temperature and Degree Days Features (if temperature is present)
    if temp_col in data.columns:
        data["cooling_degrees"] = np.maximum(0.0, data[temp_col] - 22.0)
        data["heating_degrees"] = np.maximum(0.0, 14.0 - data[temp_col])
        data["temp_lag_1h"] = data[temp_col].shift(1)
        data["temp_lag_24h"] = data[temp_col].shift(24)

    # Clean rows with NaN introduced by 168-hour lag
    if drop_na:
        data = data.dropna().reset_index(drop=True)

    # Identify all feature column names (excluding target, timestamp, and metadata)
    exclude_cols = {timestamp_col, target_col, "data_source"}
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    return data, feature_cols


def build_and_save_features(
    clean_df: pd.DataFrame,
    output_path: Path = DEFAULT_PROCESSED_DATA_PATH,
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate features and save to processed dataset CSV."""
    features_df, feature_cols = create_features(clean_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    return features_df, feature_cols


if __name__ == "__main__":
    from src.data_loader import load_or_create_dataset
    from src.data_cleaner import clean_energy_data

    raw_df, _ = load_or_create_dataset()
    clean_df, _ = clean_energy_data(raw_df)
    features_df, feature_cols = build_and_save_features(clean_df)

    print(f"Features created successfully! Shape: {features_df.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Feature names:\n{feature_cols}")
    print("\nSample rows:")
    print(features_df[[feature_cols[0], feature_cols[1], "energy_consumption_mw"]].head())
