"""
====================================================
EduPro Predictive Modeling Project
Revenue Forecasting Model Training
====================================================

Purpose:
--------
1. Load Final ML Dataset
2. Prepare Revenue Features
3. Encode Categories
4. Train Multiple Models
5. Evaluate Performance
6. Select Best Model
7. Save Best Model

Target:
-------
TotalRevenue

Output:
-------
models/revenue_model.pkl
models/revenue_feature_columns.pkl

Author: EduPro ML Project
====================================================
"""

import os
import joblib
import logging
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ====================================================
# Logging
# ====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================================================
# Load Dataset
# ====================================================

def load_dataset(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    logging.info(
        f"Dataset Loaded: {df.shape}"
    )

    return df

# ====================================================
# Prepare Revenue Features
# ====================================================

def prepare_data(df):

    target = "TotalRevenue"

    leakage_columns = [

        # target
        "TotalRevenue",

        # direct revenue leakage
        "EnrollmentCount",

        # identifiers
        "CourseID",
        "YearMonth"
    ]

    X = df.drop(
        columns=leakage_columns,
        errors="ignore"
    )

    y = df[target]

    logging.info(
        f"Feature Matrix Shape: {X.shape}"
    )

    logging.info(
        f"Target Shape: {y.shape}"
    )

    return X, y

# ====================================================
# Build Preprocessor
# ====================================================

def build_preprocessor(X):

    categorical_cols = (
        X.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numerical_cols = (
        X.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )

    return preprocessor

# ====================================================
# Evaluation
# ====================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    logging.info(
        f"{model_name}"
    )

    logging.info(
        f"MAE : {mae:.2f}"
    )

    logging.info(
        f"RMSE: {rmse:.2f}"
    )

    logging.info(
        f"R2  : {r2:.4f}"
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# ====================================================
# Train Models
# ====================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    models = {

        "Linear Regression":
        LinearRegression(),

        "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )
    }

    results = []

    trained_models = {}

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test
        )

        results.append(metrics)

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    return results_df, trained_models

# ====================================================
# Save Best Model
# ====================================================

def save_best_model(
    results_df,
    trained_models
):

    best_model_name = (
        results_df
        .sort_values(
            by="R2",
            ascending=False
        )
        .iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/revenue_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    joblib.dump(
        list(best_model.feature_names_in_),
        "models/revenue_feature_columns.pkl"
    )

    logging.info(
        f"Best Model: {best_model_name}"
    )

    logging.info(
        f"Saved: {model_path}"
    )

    return best_model_name

# ====================================================
# Main Pipeline
# ====================================================

def run_training():

    file_path = (
        "data/processed/final_ml_dataset.csv"
    )

    df = load_dataset(
        file_path
    )

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    logging.info(
        f"Train Shape: {X_train.shape}"
    )

    logging.info(
        f"Test Shape: {X_test.shape}"
    )

    preprocessor = (
        build_preprocessor(X)
    )

    results_df, trained_models = (
        train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )
    )

    print("\n")
    print("=" * 60)
    print("REVENUE MODEL COMPARISON")
    print("=" * 60)
    print(results_df)

    save_best_model(
        results_df,
        trained_models
    )

    return results_df

# ====================================================
# Main
# ====================================================

if __name__ == "__main__":

    run_training()