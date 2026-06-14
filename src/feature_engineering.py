"""
====================================================
EduPro Predictive Modeling Project
Feature Engineering Module
====================================================

Purpose:
--------
1. Create ML Targets
2. Create Course Features
3. Create Teacher Features
4. Create Time Features
5. Create Category Features
6. Generate Final ML Dataset

Input:
------
data/processed/merged_data.csv

Output:
-------
data/processed/final_ml_dataset.csv

Target Variables:
-----------------
EnrollmentCount
TotalRevenue

Author: EduPro ML Project
====================================================
"""

import os
import logging
import pandas as pd
import numpy as np

# ====================================================
# Logging Configuration
# ====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================================================
# Load Merged Dataset
# ====================================================

def load_data(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["TransactionDate"]
    )

    logging.info(
        f"Merged Dataset Loaded: {df.shape}"
    )

    return df

# ====================================================
# Create Time Features
# ====================================================

def create_time_features(df):

    df["Year"] = df["TransactionDate"].dt.year

    df["Month"] = df["TransactionDate"].dt.month

    df["Quarter"] = df["TransactionDate"].dt.quarter

    df["YearMonth"] = (
        df["TransactionDate"]
        .dt.to_period("M")
        .astype(str)
    )

    return df

# ====================================================
# Price Band
# ====================================================

def create_price_band(df):

    bins = [-1, 0, 200, 500, np.inf]

    labels = [
        "Free",
        "Low",
        "Medium",
        "High"
    ]

    df["PriceBand"] = pd.cut(
        df["CoursePrice"],
        bins=bins,
        labels=labels
    )

    return df

# ====================================================
# Duration Bucket
# ====================================================

def create_duration_bucket(df):

    bins = [0, 10, 20, 40, np.inf]

    labels = [
        "Short",
        "Medium",
        "Long",
        "VeryLong"
    ]

    df["DurationBucket"] = pd.cut(
        df["CourseDuration"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

# ====================================================
# Rating Tier
# ====================================================

def create_rating_tier(df):

    bins = [0, 2, 3, 4, 5]

    labels = [
        "Poor",
        "Average",
        "Good",
        "Excellent"
    ]

    df["RatingTier"] = pd.cut(
        df["CourseRating"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

# ====================================================
# Experience Bucket
# ====================================================

def create_experience_bucket(df):

    bins = [0, 5, 10, 15, 50]

    labels = [
        "Junior",
        "Mid",
        "Senior",
        "Expert"
    ]

    df["ExperienceBucket"] = pd.cut(
        df["YearsOfExperience"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

# ====================================================
# Expertise Match Score
# ====================================================

def create_expertise_match(df):

    def match_score(row):

        category = str(
            row["CourseCategory"]
        ).lower()

        expertise = str(
            row["Expertise"]
        ).lower()

        if category in expertise:
            return 1

        return 0

    df["ExpertiseMatchScore"] = (
        df.apply(match_score, axis=1)
    )

    return df

# ====================================================
# Aggregate Monthly Dataset
# ====================================================

def create_monthly_dataset(df):

    group_cols = [
        "CourseID",
        "YearMonth"
    ]

    monthly_df = (
        df.groupby(group_cols)
        .agg(
            EnrollmentCount=(
                "TransactionID",
                "count"
            ),

            TotalRevenue=(
                "Amount",
                "sum"
            ),

            CourseCategory=(
                "CourseCategory",
                "first"
            ),

            CourseType=(
                "CourseType",
                "first"
            ),

            CourseLevel=(
                "CourseLevel",
                "first"
            ),

            CoursePrice=(
                "CoursePrice",
                "first"
            ),

            CourseDuration=(
                "CourseDuration",
                "first"
            ),

            CourseRating=(
                "CourseRating",
                "first"
            ),

            TeacherRating=(
                "TeacherRating",
                "mean"
            ),

            YearsOfExperience=(
                "YearsOfExperience",
                "mean"
            ),

            Expertise=(
                "Expertise",
                "first"
            ),

            PriceBand=(
                "PriceBand",
                "first"
            ),

            DurationBucket=(
                "DurationBucket",
                "first"
            ),

            RatingTier=(
                "RatingTier",
                "first"
            ),

            ExperienceBucket=(
                "ExperienceBucket",
                "first"
            ),

            ExpertiseMatchScore=(
                "ExpertiseMatchScore",
                "mean"
            )
        )
        .reset_index()
    )

    logging.info(
        f"Monthly Dataset Shape: "
        f"{monthly_df.shape}"
    )

    return monthly_df

# ====================================================
# Final Validation
# ====================================================

def validate_dataset(df):

    logging.info(
        f"Final Dataset Shape: {df.shape}"
    )

    logging.info(
        f"Missing Values: "
        f"{df.isnull().sum().sum()}"
    )

    logging.info(
        f"Duplicate Rows: "
        f"{df.duplicated().sum()}"
    )

# ====================================================
# Save Dataset
# ====================================================

def save_dataset(df, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    logging.info(
        f"Saved Dataset:\n{output_path}"
    )

# ====================================================
# Main Pipeline
# ====================================================

def run_feature_engineering():

    input_file = (
        "data/processed/merged_data.csv"
    )

    output_file = (
        "data/processed/final_ml_dataset.csv"
    )

    df = load_data(
        input_file
    )

    df = create_time_features(df)

    df = create_price_band(df)

    df = create_duration_bucket(df)

    df = create_rating_tier(df)

    df = create_experience_bucket(df)

    df = create_expertise_match(df)

    final_df = create_monthly_dataset(df)

    validate_dataset(
        final_df
    )

    save_dataset(
        final_df,
        output_file
    )

    logging.info(
        "Feature Engineering Completed."
    )

    return final_df

# ====================================================
# Main Execution
# ====================================================

if __name__ == "__main__":

    final_df = (
        run_feature_engineering()
    )

    print("\nFinal Dataset Sample:")
    print(final_df.head())