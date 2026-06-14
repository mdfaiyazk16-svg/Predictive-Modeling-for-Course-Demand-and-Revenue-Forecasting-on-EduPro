"""
====================================================
EduPro Predictive Modeling Project
Feature Importance Analysis
====================================================

Purpose:
--------
1. Load Best Revenue Model
2. Extract Processed Feature Names
3. Calculate Feature Importance
4. Generate Feature Importance Report
5. Save CSV Output
6. Save Visualization

Input:
------
models/revenue_model.pkl

Outputs:
--------
outputs/feature_importance.csv
outputs/feature_importance.png

Author: EduPro ML Project
====================================================
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================
# Logging Configuration
# ====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================================================
# Create Output Directory
# ====================================================

def create_output_directory():

    output_dir = "outputs"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    return output_dir

# ====================================================
# Load Revenue Model
# ====================================================

def load_model():

    model_path = "models/revenue_model.pkl"

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Revenue model not found: {model_path}"
        )

    model = joblib.load(model_path)

    logging.info(
        "Revenue model loaded successfully."
    )

    return model

# ====================================================
# Extract Feature Importance
# ====================================================

def extract_feature_importance(model):

    preprocessor = model.named_steps["preprocessor"]

    estimator = model.named_steps["model"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # ----------------------------------------
    # Linear Regression
    # ----------------------------------------

    if hasattr(estimator, "coef_"):

        importance_values = np.abs(
            estimator.coef_
        )

        method = (
            "Absolute Linear Regression Coefficients"
        )

    # ----------------------------------------
    # Tree Models
    # ----------------------------------------

    elif hasattr(
        estimator,
        "feature_importances_"
    ):

        importance_values = (
            estimator.feature_importances_
        )

        method = (
            "Tree Feature Importance"
        )

    else:

        raise ValueError(
            "Unsupported model type."
        )

    importance_df = pd.DataFrame({

        "Feature":
        feature_names,

        "Importance":
        importance_values

    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    logging.info(
        f"Importance Method: {method}"
    )

    return importance_df

# ====================================================
# Save CSV
# ====================================================

def save_csv(df):

    output_path = (
        "outputs/feature_importance.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    logging.info(
        f"CSV saved: {output_path}"
    )

# ====================================================
# Save Visualization
# ====================================================

def save_plot(df):

    top_features = df.head(15)

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Revenue Prediction Features"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plot_path = (
        "outputs/feature_importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(
        f"Plot saved: {plot_path}"
    )

# ====================================================
# Display Summary
# ====================================================

def display_summary(df):

    print("\n")
    print("=" * 60)
    print("TOP 15 IMPORTANT FEATURES")
    print("=" * 60)

    print(
        df.head(15)
    )

    print("\n")

    print(
        "Most Important Feature:"
    )

    print(
        df.iloc[0]["Feature"]
    )

# ====================================================
# Main Pipeline
# ====================================================

def run_feature_importance():

    logging.info(
        "Starting Feature Importance Analysis..."
    )

    create_output_directory()

    model = load_model()

    importance_df = (
        extract_feature_importance(
            model
        )
    )

    save_csv(
        importance_df
    )

    save_plot(
        importance_df
    )

    display_summary(
        importance_df
    )

    logging.info(
        "Feature Importance Analysis Completed."
    )

    return importance_df

# ====================================================
# Main Execution
# ====================================================

if __name__ == "__main__":

    run_feature_importance()