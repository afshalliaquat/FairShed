"""Tests for data loading and synthetic generation."""

import pandas as pd
from src.data_loader import generate_synthetic_energy_data, load_or_create_dataset


def test_generate_synthetic_energy_data_structure():
    df = generate_synthetic_energy_data(
        start_date="2024-01-01 00:00:00",
        end_date="2024-01-07 23:00:00",
        seed=123,
    )

    assert isinstance(df, pd.DataFrame)
    expected_rows = 7 * 24  # 7 days of hourly data
    assert len(df) == expected_rows

    expected_columns = {"timestamp", "energy_consumption_mw", "temperature_c", "data_source"}
    assert expected_columns.issubset(set(df.columns))

    # Energy consumption should be strictly positive and realistic
    assert (df["energy_consumption_mw"] > 0).all()
    assert df["energy_consumption_mw"].mean() > 5000.0


def test_load_or_create_dataset_fallback(tmp_path):
    test_csv = tmp_path / "test_energy.csv"
    df, status_msg = load_or_create_dataset(output_path=test_csv, prefer_public=False)

    assert test_csv.exists()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    assert "timestamp" in df.columns
    assert "energy_consumption_mw" in df.columns
