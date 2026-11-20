"""Evaluation metrics and plotting utilities for the AI-Powered Energy Consumption Forecasting System.

Calculates:
- MAE (Mean Absolute Error) in MW
- RMSE (Root Mean Squared Error) in MW
- MAPE (Mean Absolute Percentage Error) in %
- R² (Coefficient of Determination)
- MBE (Mean Bias Error) in MW
- Feature Importances and Model Explanations
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate key regression evaluation metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values from model.

    Returns:
        Dict with MAE, RMSE, MAPE, R2, MBE, Max_Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    # Avoid zero-division in MAPE
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    r2 = float(r2_score(y_true, y_pred))
    mbe = float(np.mean(y_pred - y_true))
    max_err = float(np.max(np.abs(y_true - y_pred)))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_%": round(mape, 2),
        "R2_Score": round(r2, 4),
        "MBE": round(mbe, 2),
        "Max_Error": round(max_err, 2),
    }


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Evaluate all trained models on the test dataset.

    Returns:
        Tuple of (metrics_summary_df, predictions_dict)
    """
    metrics_records = []
    predictions = {}

    for model_name, model in models_dict.items():
        preds = model.predict(X_test)
        predictions[model_name] = preds

        metrics = calculate_metrics(y_test.values, preds)
        metrics["Model"] = model_name
        metrics_records.append(metrics)

    summary_df = pd.DataFrame(metrics_records)
    # Reorder columns with Model first
    cols = ["Model", "MAE", "RMSE", "MAPE_%", "R2_Score", "MBE", "Max_Error"]
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    summary_df = summary_df.sort_values(by="RMSE").reset_index(drop=True)

    return summary_df, predictions


def extract_feature_importances(
    rf_model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and rank feature importances from trained Random Forest Regressor."""
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "importance_pct": np.round(importances * 100.0, 2),
            }
        ).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return fi_df
    return pd.DataFrame(columns=["feature", "importance", "importance_pct"])


def extract_linear_coefficients(
    linear_model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract standardized Linear Regression coefficients and rank by impact."""
    regressor = getattr(linear_model, "named_steps", {}).get("regressor", linear_model)
    coefficients = getattr(regressor, "coef_", None)
    if coefficients is None:
        return pd.DataFrame(columns=["feature", "coefficient", "absolute_coefficient"])

    coefficients = np.asarray(coefficients, dtype=float)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": np.round(coefficients, 2),
            "absolute_coefficient": np.round(np.abs(coefficients), 2),
        }
    ).sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def save_metrics_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save metrics table to JSON for dashboard and report consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dict = summary_df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
