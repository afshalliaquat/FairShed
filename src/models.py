"""Model training and evaluation module for the AI-Powered Energy Consumption Forecasting System.

Implements:
- Chronological train/test splitting (zero lookahead leakage)
- Linear Regression with StandardScaler pipeline
- Random Forest Regressor with feature importance extraction
- Model saving and loading via joblib
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def chronological_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "energy_consumption_mw",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform a strict chronological train/test split.

    Crucial for time-series forecasting: training data strictly precedes test data.
    No future data is ever observed during training.

    Args:
        df: Feature dataframe sorted in chronological order.
        feature_cols: List of predictor column names.
        target_col: Name of the energy consumption column.
        test_size: Proportion of the dataset allocated to the test set (default: 0.2).

    Returns:
        (X_train, X_test, y_train, y_test, timestamps_train, timestamps_test)
    """
    n = len(df)
    split_idx = int(n * (1.0 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    timestamps_train = train_df["timestamp"] if "timestamp" in train_df.columns else train_df.index
    timestamps_test = test_df["timestamp"] if "timestamp" in test_df.columns else test_df.index

    return X_train, X_test, y_train, y_test, timestamps_train, timestamps_test


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train Linear Regression pipeline with feature standardization."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 14,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train Random Forest Regressor."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def save_models(
    models_dict: Dict[str, Any],
    feature_cols: List[str],
    output_dir: Path = MODELS_DIR,
) -> None:
    """Save trained models and feature metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models_dict.items():
        save_path = output_dir / f"{name}.joblib"
        joblib.dump(model, save_path)

    metadata_path = output_dir / "features_meta.joblib"
    joblib.dump({"feature_cols": feature_cols}, metadata_path)


def load_saved_models(
    models_dir: Path = MODELS_DIR,
) -> Tuple[Dict[str, Any], List[str]]:
    """Load persisted models and feature metadata from disk."""
    models_dict = {}
    model_names = ["linear_regression_model", "random_forest_model"]
    for name in model_names:
        file_path = models_dir / f"{name}.joblib"
        if file_path.exists():
            models_dict[name] = joblib.load(file_path)

    metadata_path = models_dir / "features_meta.joblib"
    feature_cols = []
    if metadata_path.exists():
        meta = joblib.load(metadata_path)
        feature_cols = meta.get("feature_cols", [])

    return models_dict, feature_cols
