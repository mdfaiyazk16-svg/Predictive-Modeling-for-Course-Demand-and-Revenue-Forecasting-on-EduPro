"""
====================================================
EduPro Predictive Modeling Project
Data Preprocessing Module
====================================================

Purpose:
--------
1. Load Raw Data
2. Validate Data Types
3. Clean Data
4. Validate Relationships
5. Merge Datasets
6. Save Clean Merged Dataset

Input:
------
Courses Sheet
Teachers Sheet
Transactions Sheet
Users Sheet

Output:
-------
data/processed/merged_data.csv

Author: EduPro ML Project
====================================================
"""

import os
import logging
import pandas as pd

from data_loader import load_dataset

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

def create_output_directory(output_path: str):
    """
    Create processed data directory.
    """

    os.makedirs(output_path, exist_ok=True)

    logging.info(
        f"Output directory ready: {output_path}"
    )

# ====================================================
# Clean Courses Data
# ====================================================

def clean_courses(courses_df: pd.DataFrame):

    df = courses_df.copy()

    # Remove accidental spaces
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    # Numeric conversions
    df["CoursePrice"] = pd.to_numeric(
        df["CoursePrice"],
        errors="coerce"
    )

    df["CourseDuration"] = pd.to_numeric(
        df["CourseDuration"],
        errors="coerce"
    )

    df["CourseRating"] = pd.to_numeric(
        df["CourseRating"],
        errors="coerce"
    )

    logging.info(
        f"Courses cleaned. Shape: {df.shape}"
    )

    return df

# ====================================================
# Clean Teachers Data
# ====================================================

def clean_teachers(teachers_df: pd.DataFrame):

    df = teachers_df.copy()

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    df["YearsOfExperience"] = pd.to_numeric(
        df["YearsOfExperience"],
        errors="coerce"
    )

    df["TeacherRating"] = pd.to_numeric(
        df["TeacherRating"],
        errors="coerce"
    )

    logging.info(
        f"Teachers cleaned. Shape: {df.shape}"
    )

    return df

# ====================================================
# Clean Transactions Data
# ====================================================

def clean_transactions(transactions_df: pd.DataFrame):

    df = transactions_df.copy()

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    # Date conversion
    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"],
        errors="coerce"
    )

    # Amount conversion
    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    logging.info(
        f"Transactions cleaned. Shape: {df.shape}"
    )

    return df

# ====================================================
# Clean Users Data
# ====================================================

def clean_users(users_df: pd.DataFrame):

    df = users_df.copy()

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    logging.info(
        f"Users cleaned. Shape: {df.shape}"
    )

    return df

# ====================================================
# Referential Integrity Checks
# ====================================================

def validate_relationships(
    courses_df,
    teachers_df,
    transactions_df,
    users_df
):
    """
    Check foreign-key consistency.
    """

    invalid_courses = (
        ~transactions_df["CourseID"].isin(
            courses_df["CourseID"]
        )
    ).sum()

    invalid_teachers = (
        ~transactions_df["TeacherID"].isin(
            teachers_df["TeacherID"]
        )
    ).sum()

    invalid_users = (
        ~transactions_df["UserID"].isin(
            users_df["UserID"]
        )
    ).sum()

    logging.info(
        f"Invalid Course References: {invalid_courses}"
    )

    logging.info(
        f"Invalid Teacher References: {invalid_teachers}"
    )

    logging.info(
        f"Invalid User References: {invalid_users}"
    )

    if invalid_courses > 0:
        raise ValueError(
            "Invalid CourseID detected."
        )

    if invalid_teachers > 0:
        raise ValueError(
            "Invalid TeacherID detected."
        )

    if invalid_users > 0:
        raise ValueError(
            "Invalid UserID detected."
        )

    logging.info(
        "Relationship validation successful."
    )

# ====================================================
# Merge Dataset
# ====================================================

def merge_datasets(
    courses_df,
    teachers_df,
    transactions_df,
    users_df
):
    
    """
    Merge all sheets into one clean dataset.
    """

    teachers_df = teachers_df.rename(
    columns={
        "Age": "TeacherAge",
        "Gender": "TeacherGender"
        }
    )

    users_df = users_df.rename(
    columns={
        "Age": "UserAge",
        "Gender": "UserGender"
        }
    )

    merged_df = transactions_df.merge(
        courses_df,
        on="CourseID",
        how="left",
        validate="many_to_one"
    )

    merged_df = merged_df.merge(
        teachers_df,
        on="TeacherID",
        how="left",
        validate="many_to_one"
    )

    merged_df = merged_df.merge(
        users_df,
        on="UserID",
        how="left",
        validate="many_to_one"
    )

    logging.info(
        f"Merged Dataset Shape: {merged_df.shape}"
    )

    return merged_df

# ====================================================
# Missing Value Validation
# ====================================================

def validate_missing_values(df):

    missing_count = (
        df.isnull().sum().sum()
    )

    logging.info(
        f"Missing Values After Merge: "
        f"{missing_count}"
    )

    if missing_count > 0:
        logging.warning(
            "Merged dataset contains missing values."
        )

# ====================================================
# Save Dataset
# ====================================================

def save_dataset(df, output_file):

    df.to_csv(
        output_file,
        index=False
    )

    logging.info(
        f"Dataset saved to:\n{output_file}"
    )

# ====================================================
# Main Pipeline
# ====================================================

def run_preprocessing():

    logging.info(
        "Starting preprocessing pipeline..."
    )

    (
        courses_df,
        teachers_df,
        transactions_df,
        users_df
    ) = load_dataset(
        "data/raw/EduPro Online Platform.xlsx"
    )

    courses_df = clean_courses(
        courses_df
    )

    teachers_df = clean_teachers(
        teachers_df
    )

    transactions_df = clean_transactions(
        transactions_df
    )

    users_df = clean_users(
        users_df
    )

    validate_relationships(
        courses_df,
        teachers_df,
        transactions_df,
        users_df
    )

    merged_df = merge_datasets(
        courses_df,
        teachers_df,
        transactions_df,
        users_df
    )

    validate_missing_values(
        merged_df
    )

    output_dir = "data/processed"

    create_output_directory(
        output_dir
    )

    output_file = os.path.join(
        output_dir,
        "merged_data.csv"
    )

    save_dataset(
        merged_df,
        output_file
    )

    logging.info(
        "Preprocessing completed successfully."
    )

    return merged_df

# ====================================================
# Main Execution
# ====================================================

if __name__ == "__main__":

    final_df = run_preprocessing()

    print("\nMerged Dataset Sample:")
    print(final_df.head())