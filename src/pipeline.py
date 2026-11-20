"""End-to-end pipeline runner for the AI-Powered Energy Consumption Forecasting System.

Executes the complete pipeline:
1. Data loading (public dataset with synthetic fallback)
2. Data cleaning & validation
3. Feature engineering without data leakage
4. Chronological train/test split (80/20)
5. Model training (Linear Regression, Random Forest)
6. Model evaluation & metrics serialization (MAE, RMSE, MAPE, R²)
7. Exporting diagnostic comparison plots
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_or_create_dataset
from src.data_cleaner import clean_energy_data
from src.feature_engineering import build_and_save_features
from src.models import (
    chronological_split,
    train_linear_regression,
    train_random_forest,
    save_models,
)
from src.evaluate import (
    evaluate_all_models,
    extract_feature_importances,
    save_metrics_summary,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
METRICS_JSON_PATH = BASE_DIR / "models" / "metrics_summary.json"


def plot_and_save_figures(
    timestamps_test: pd.Series,
    y_test: pd.Series,
    predictions: dict,
    fi_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Generate and save clean Matplotlib charts for portfolio presentation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forecast vs Actual Comparison Chart (Zoomed to 7-day test window)
    plt.figure(figsize=(14, 6))
    # Pick a 168-hour (1 week) snapshot for clean visual comparison
    slice_n = min(168, len(y_test))
    ts_slice = timestamps_test.iloc[:slice_n]
    y_slice = y_test.iloc[:slice_n]

    plt.plot(ts_slice, y_slice, label="Actual Demand (MW)", color="black", linewidth=2.0)
    for model_name, preds in predictions.items():
        clean_name = model_name.replace("_model", "").replace("_", " ").title()
        plt.plot(ts_slice, preds[:slice_n], label=f"{clean_name} Pred", alpha=0.85)

    plt.title("Energy Consumption Forecasting: Actual vs Predicted (1-Week Test Snapshot)", fontsize=14, fontweight="bold")
    plt.xlabel("Datetime", fontsize=11)
    plt.ylabel("Electricity Demand (MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    comparison_fig_path = output_dir / "forecast_vs_actual.png"
    plt.savefig(comparison_fig_path, dpi=200)
    plt.close()

    # 2. Random Forest Top 10 Feature Importances Bar Chart
    if not fi_df.empty:
        plt.figure(figsize=(10, 6))
        top_10 = fi_df.head(10).iloc[::-1]  # Reverse for horizontal bar chart
        plt.barh(top_10["feature"], top_10["importance_pct"], color="#1f77b4", edgecolor="black", alpha=0.85)
        plt.title("Top 10 Feature Importances (Random Forest Regressor)", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance (%)", fontsize=11)
        plt.ylabel("Engineered Feature", fontsize=11)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        fi_fig_path = output_dir / "feature_importance.png"
        plt.savefig(fi_fig_path, dpi=200)
        plt.close()


def run_pipeline() -> pd.DataFrame:
    """Run full training and evaluation workflow."""
    print("=" * 70)
    print("=== AI-POWERED ENERGY CONSUMPTION FORECASTING PIPELINE ===")
    print("=" * 70)

    # 1. Data Ingestion
    print("\n[Step 1/6] Loading data...")
    raw_df, status_msg = load_or_create_dataset()
    print(f"  -> {status_msg}")
    print(f"  -> Raw records: {len(raw_df)}")

    # 2. Data Cleaning
    print("\n[Step 2/6] Cleaning and validating time series...")
    clean_df, report = clean_energy_data(raw_df)
    print(f"  -> Imputed missing hours: {report['missing_hours_imputed']}")
    print(f"  -> Clean records: {report['final_rows']}")

    # 3. Feature Engineering
    print("\n[Step 3/6] Engineering time-series features (zero future leakage)...")
    features_df, feature_cols = build_and_save_features(clean_df)
    print(f"  -> Features engineered: {len(feature_cols)}")
    print(f"  -> Usable samples after lag initialization: {len(features_df)}")

    # 4. Chronological Train/Test Split
    print("\n[Step 4/6] Performing chronological 80/20 train/test split...")
    X_train, X_test, y_train, y_test, ts_train, ts_test = chronological_split(
        features_df, feature_cols, test_size=0.2
    )
    print(f"  -> Training set: {len(X_train)} samples ({ts_train.min()} to {ts_train.max()})")
    print(f"  -> Testing set:  {len(X_test)} samples ({ts_test.min()} to {ts_test.max()})")

    # 5. Model Training
    print("\n[Step 5/6] Training models...")
    print("  -> Fitting Linear Regression Pipeline (StandardScaler + OLS)...")
    linear_model = train_linear_regression(X_train, y_train)

    print("  -> Fitting Random Forest Regressor (100 trees)...")
    rf_model = train_random_forest(X_train, y_train)

    models_dict = {
        "linear_regression_model": linear_model,
        "random_forest_model": rf_model,
    }

    # Save models to disk
    save_models(models_dict, feature_cols)
    print("  -> Models and metadata serialized to models/ directory.")

    # 6. Evaluation & Reporting
    print("\n[Step 6/6] Evaluating models on test set...")
    summary_df, predictions = evaluate_all_models(models_dict, X_test, y_test)
    save_metrics_summary(summary_df, METRICS_JSON_PATH)

    print("\n" + "=" * 70)
    print("=== MODEL EVALUATION PERFORMANCE SUMMARY ===")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Feature Importance analysis
    fi_df = extract_feature_importances(rf_model, feature_cols)
    if not fi_df.empty:
        print("\nTop 5 Most Predictive Features (Random Forest):")
        for _, row in fi_df.head(5).iterrows():
            print(f"  - {row['feature']:<20}: {row['importance_pct']:.2f}%")

    # Save visualization figures
    plot_and_save_figures(ts_test, y_test, predictions, fi_df)
    print(f"\nVisual diagnostic plots saved to: {FIGURES_DIR}")
    print("=" * 70)
    print("Pipeline completed successfully! Ready for the FairShed web GUI.")
    print("=" * 70)

    return summary_df


if __name__ == "__main__":
    run_pipeline()
