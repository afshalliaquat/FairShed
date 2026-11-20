"""Data cleaning module for the AI-Powered Energy Consumption Forecasting System.

Provides robust cleaning steps:
- Parsing and sorting datetime chronologically
- Handling duplicate timestamps
- Detecting and filling missing hourly intervals via time interpolation
- Capping or flagging non-physical outliers
- Returning clean DataFrames and a diagnostic data quality report
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def clean_energy_data(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_col: str = "energy_consumption_mw",
    temperature_col: str = "temperature_c",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate hourly energy consumption time-series data.

    Args:
        df: Raw DataFrame.
        timestamp_col: Name of the datetime column.
        target_col: Name of energy consumption target column.
        temperature_col: Name of temperature column (if present).

    Returns:
        Tuple of (clean_df, quality_report)
    """
    clean_df = df.copy()

    # 1. Parse Datetime & Sort Chronologically
    clean_df[timestamp_col] = pd.to_datetime(clean_df[timestamp_col])
    clean_df = clean_df.sort_values(by=timestamp_col).reset_index(drop=True)

    initial_rows = len(clean_df)

    # 2. Check and remove duplicate timestamps
    duplicates_count = clean_df.duplicated(subset=[timestamp_col]).sum()
    if duplicates_count > 0:
        clean_df = clean_df.drop_duplicates(subset=[timestamp_col], keep="first").reset_index(drop=True)

    # 3. Set index and ensure strictly uniform hourly frequency
    clean_df = clean_df.set_index(timestamp_col)
    full_range = pd.date_range(start=clean_df.index.min(), end=clean_df.index.max(), freq="h")
    missing_hours_count = len(full_range) - len(clean_df)

    if missing_hours_count > 0:
        clean_df = clean_df.reindex(full_range)

    # 4. Impute missing values with time interpolation and boundary back/forward fill
    interpolated_target_nans = clean_df[target_col].isna().sum() if target_col in clean_df.columns else 0
    clean_df[target_col] = clean_df[target_col].interpolate(method="time").bfill().ffill()

    if temperature_col in clean_df.columns:
        clean_df[temperature_col] = clean_df[temperature_col].interpolate(method="time").bfill().ffill()

    if "data_source" in clean_df.columns:
        clean_df["data_source"] = clean_df["data_source"].ffill().bfill()

    # 5. Sanity Check / Outlier Capping
    # Energy consumption cannot be negative or absurdly extreme (< 0 capped at 0)
    negative_values_count = (clean_df[target_col] < 0).sum()
    if negative_values_count > 0:
        clean_df[target_col] = clean_df[target_col].clip(lower=0.0)

    # Upper bound check (e.g. 5 standard deviations from median)
    median_val = clean_df[target_col].median()
    std_val = clean_df[target_col].std()
    upper_threshold = median_val + 5 * std_val
    outliers_count = (clean_df[target_col] > upper_threshold).sum()
    if outliers_count > 0:
        clean_df[target_col] = clean_df[target_col].clip(upper=upper_threshold)

    # Reset index back to column for easy downstream tabular operations
    clean_df = clean_df.reset_index().rename(columns={"index": timestamp_col})

    quality_report = {
        "initial_rows": int(initial_rows),
        "final_rows": int(len(clean_df)),
        "duplicates_removed": int(duplicates_count),
        "missing_hours_imputed": int(missing_hours_count + interpolated_target_nans),
        "negative_values_clipped": int(negative_values_count),
        "extreme_outliers_clipped": int(outliers_count),
        "start_time": str(clean_df[timestamp_col].min()),
        "end_time": str(clean_df[timestamp_col].max()),
        "target_min": float(clean_df[target_col].min()),
        "target_max": float(clean_df[target_col].max()),
        "target_mean": float(clean_df[target_col].mean()),
    }

    return clean_df, quality_report


if __name__ == "__main__":
    from src.data_loader import load_or_create_dataset

    raw_df, _ = load_or_create_dataset()
    clean_df, report = clean_energy_data(raw_df)
    print("Data Cleaning Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("\nCleaned Data Sample:")
    print(clean_df.head(3))
