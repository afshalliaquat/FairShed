"""Tests for data cleaner module."""

import numpy as np
import pandas as pd
from src.data_cleaner import clean_energy_data


def test_clean_energy_data_deduplication_and_sorting():
    # Construct messy input: unordered timestamps and duplicates
    dates = pd.to_datetime(["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00"])
    loads = [12000.0, 11000.0, 11500.0, 11500.0]
    temps = [5.0, 4.0, 4.5, 4.5]
    raw_df = pd.DataFrame({"timestamp": dates, "energy_consumption_mw": loads, "temperature_c": temps})

    clean_df, report = clean_energy_data(raw_df)

    assert report["duplicates_removed"] == 1
    assert len(clean_df) == 3
    # Verify strict ascending order
    assert clean_df["timestamp"].is_monotonic_increasing


def test_clean_energy_data_missing_hours_imputation():
    # 3-hour span with 01:00 missing
    dates = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"])
    loads = [10000.0, 12000.0]
    temps = [2.0, 4.0]
    raw_df = pd.DataFrame({"timestamp": dates, "energy_consumption_mw": loads, "temperature_c": temps})

    clean_df, report = clean_energy_data(raw_df)

    assert len(clean_df) == 3  # 00:00, 01:00, 02:00
    assert report["missing_hours_imputed"] >= 1
    # Check linear interpolation for 01:00
    mid_load = clean_df.loc[clean_df["timestamp"] == "2024-01-01 01:00", "energy_consumption_mw"].values[0]
    assert np.isclose(mid_load, 11000.0)


def test_clean_energy_data_negative_capping():
    dates = pd.date_range("2024-01-01", periods=3, freq="h")
    loads = [-50.0, 12000.0, 13000.0]
    raw_df = pd.DataFrame({"timestamp": dates, "energy_consumption_mw": loads, "temperature_c": [10.0, 10.0, 10.0]})

    clean_df, report = clean_energy_data(raw_df)

    assert report["negative_values_clipped"] == 1
    assert clean_df["energy_consumption_mw"].min() >= 0.0
