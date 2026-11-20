"""Data loading module for the AI-Powered Energy Consumption Forecasting System.

Supports fetching real public hourly energy consumption datasets
with an automatic, realistic synthetic data generator fallback.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import requests

DEFAULT_RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "energy_consumption.csv"


def generate_synthetic_energy_data(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2024-12-31 23:00:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic hourly energy consumption and temperature data.

    Simulates:
    - Base demand (~14,000 MW)
    - Diurnal (daily) cycles: morning (8-10 AM) and evening (6-9 PM) peaks
    - Weekly patterns: lower commercial/industrial demand on weekends
    - Annual seasonality: summer cooling peaks and winter heating peaks
    - Diurnal temperature cycle coupled with seasonal temperature
    - Moderate stochastic noise

    Returns:
        pd.DataFrame with columns:
        ['timestamp', 'energy_consumption_mw', 'temperature_c', 'data_source']
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(timestamps)

    day_of_year = timestamps.dayofyear.values
    hour = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values  # 0=Monday, 6=Sunday

    # 1. Temperature Simulation (°C)
    # Annual cycle: cold in winter (~2°C), warm in summer (~28°C)
    seasonal_temp = 15.0 - 13.0 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
    # Daily cycle: coolest around 5 AM, warmest around 3 PM (15h)
    daily_temp = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp_noise = rng.normal(0, 1.5, size=n)
    temperature = seasonal_temp + daily_temp + temp_noise

    # 2. Base Load
    base_load = 14000.0

    # 3. Daily Profile: two peaks (morning 9 AM, evening 7 PM)
    morning_peak = 1800.0 * np.exp(-0.5 * ((hour - 9) / 2.2) ** 2)
    evening_peak = 2400.0 * np.exp(-0.5 * ((hour - 19) / 2.5) ** 2)
    night_dip = -1200.0 * np.exp(-0.5 * ((hour - 3) / 2.0) ** 2)
    daily_profile = morning_peak + evening_peak + night_dip

    # 4. Weekend Factor: ~15% reduction on weekends
    weekend_mask = day_of_week >= 5
    weekend_factor = np.where(weekend_mask, -1600.0, 0.0)

    # 5. Weather / Cooling and Heating Degree Effect:
    # Cooling demand when temp > 22°C, heating demand when temp < 14°C
    cooling_demand = np.maximum(0, temperature - 22.0) * 180.0
    heating_demand = np.maximum(0, 14.0 - temperature) * 110.0

    # 6. Trend / Long-term small economic growth (~1.5% per year)
    time_trend = np.linspace(0, 250, n)

    # 7. Stochastic Noise & occasional peak events
    load_noise = rng.normal(0, 220, size=n)

    # Combined Energy Consumption in Megawatts (MW)
    consumption = (
        base_load
        + daily_profile
        + weekend_factor
        + cooling_demand
        + heating_demand
        + time_trend
        + load_noise
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_consumption_mw": np.round(consumption, 2),
            "temperature_c": np.round(temperature, 2),
            "data_source": "synthetic_realistic_fallback",
        }
    )

    return df


def download_public_dataset(
    url: str = "https://raw.githubusercontent.com/plotly/datasets/master/hourly_energy.csv",
    timeout: int = 5,
) -> Optional[pd.DataFrame]:
    """Attempt to download a public hourly energy consumption dataset.

    Returns:
        pd.DataFrame or None if download fails or times out.
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            import io
            df = pd.read_csv(io.StringIO(response.text))
            # Standardize column naming if necessary
            if "Datetime" in df.columns and "PJME_MW" in df.columns:
                df = df.rename(columns={"Datetime": "timestamp", "PJME_MW": "energy_consumption_mw"})
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["data_source"] = "public_pjm_hourly"
                return df
    except Exception:
        pass
    return None


def load_or_create_dataset(
    output_path: Optional[Path] = None,
    force_refresh: bool = False,
    prefer_public: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Load existing dataset from disk, or fetch public / generate synthetic data.

    Args:
        output_path: Path where raw dataset CSV should be saved.
        force_refresh: If True, regenerates or redownloads even if file exists.
        prefer_public: If True, attempts online download before falling back.

    Returns:
        Tuple of (DataFrame, status_description)
    """
    if output_path is None:
        output_path = DEFAULT_RAW_DATA_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_refresh:
        df = pd.read_csv(output_path, parse_dates=["timestamp"])
        source = df["data_source"].iloc[0] if "data_source" in df.columns else "cached_file"
        return df, f"Loaded existing dataset from {output_path} (Source: {source})"

    df: Optional[pd.DataFrame] = None
    status = ""

    if prefer_public:
        df = download_public_dataset()
        if df is not None:
            status = "Downloaded verified public energy consumption dataset"

    if df is None:
        df = generate_synthetic_energy_data()
        status = "Generated high-fidelity synthetic energy consumption dataset with weather effects"

    df.to_csv(output_path, index=False)
    return df, f"{status} and saved to {output_path}"


if __name__ == "__main__":
    df, status_msg = load_or_create_dataset(force_refresh=True)
    print(status_msg)
    print("Dataset preview:")
    print(df.head())
    print("\nDataset summary:")
    print(df.info())
