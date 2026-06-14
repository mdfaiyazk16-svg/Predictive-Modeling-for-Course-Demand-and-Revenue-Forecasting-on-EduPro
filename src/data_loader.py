"""
====================================================
EduPro Predictive Modeling Project
Data Loader Module
====================================================

Purpose:
--------
1. Load Excel Dataset
2. Validate Required Sheets
3. Validate Required Columns
4. Basic Data Inspection
5. Return DataFrames for Processing

Input:
------
data/raw/EduPro.xlsx

Output:
-------
courses_df
teachers_df
transactions_df
users_df

Author: EduPro ML Project
====================================================
"""

import os
import logging
import pandas as pd

# ====================================================
# Logging Configuration
# ====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====================================================
# Required Schema
# ====================================================

REQUIRED_SHEETS = [
    "Courses",
    "Teachers",
    "Transactions",
    "Users"
]

REQUIRED_COLUMNS = {
    "Courses": [
        "CourseID",
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "CoursePrice",
        "CourseDuration",
        "CourseRating"
    ],

    "Teachers": [
        "TeacherID",
        "Expertise",
        "YearsOfExperience",
        "TeacherRating"
    ],

    "Transactions": [
        "TransactionID",
        "CourseID",
        "TransactionDate",
        "Amount"
    ],

    "Users": [
        "UserID"
    ]
}

# ====================================================
# Validate File
# ====================================================

def validate_file(file_path: str) -> None:
    """
    Check whether dataset file exists.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}"
        )

    logging.info("Dataset file found.")

# ====================================================
# Validate Sheets
# ====================================================

def validate_sheets(excel_file: pd.ExcelFile) -> None:
    """
    Validate required sheet names.
    """

    available_sheets = excel_file.sheet_names

    missing_sheets = [
        sheet
        for sheet in REQUIRED_SHEETS
        if sheet not in available_sheets
    ]

    if missing_sheets:
        raise ValueError(
            f"Missing sheets: {missing_sheets}"
        )

    logging.info("All required sheets are present.")

# ====================================================
# Validate Columns
# ====================================================

def validate_columns(df: pd.DataFrame,
                     sheet_name: str) -> None:
    """
    Validate required columns in a sheet.
    """

    required_cols = REQUIRED_COLUMNS.get(sheet_name, [])

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in {sheet_name}: {missing_cols}"
        )

    logging.info(
        f"{sheet_name}: Required columns validated."
    )

# ====================================================
# Basic Inspection
# ====================================================

def inspect_dataframe(df: pd.DataFrame,
                      sheet_name: str) -> None:
    """
    Print basic dataset information.
    """

    logging.info(
        f"{sheet_name}: Shape = {df.shape}"
    )

    logging.info(
        f"{sheet_name}: Missing Values = "
        f"{df.isnull().sum().sum()}"
    )

    logging.info(
        f"{sheet_name}: Duplicate Rows = "
        f"{df.duplicated().sum()}"
    )

# ====================================================
# Load Dataset
# ====================================================

def load_dataset(file_path: str):
    """
    Load EduPro Excel dataset.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    tuple:
        courses_df,
        teachers_df,
        transactions_df,
        users_df
    """

    try:

        logging.info(
            "Starting dataset loading..."
        )

        validate_file(file_path)

        excel_file = pd.ExcelFile(
            file_path,
            engine="openpyxl"
        )

        validate_sheets(excel_file)

        # ----------------------------------------
        # Load Sheets
        # ----------------------------------------

        courses_df = pd.read_excel(
            file_path,
            sheet_name="Courses",
            engine="openpyxl"
        )

        teachers_df = pd.read_excel(
            file_path,
            sheet_name="Teachers",
            engine="openpyxl"
        )

        transactions_df = pd.read_excel(
            file_path,
            sheet_name="Transactions",
            engine="openpyxl"
        )

        users_df = pd.read_excel(
            file_path,
            sheet_name="Users",
            engine="openpyxl"
        )

        # ----------------------------------------
        # Validate Columns
        # ----------------------------------------

        validate_columns(
            courses_df,
            "Courses"
        )

        validate_columns(
            teachers_df,
            "Teachers"
        )

        validate_columns(
            transactions_df,
            "Transactions"
        )

        validate_columns(
            users_df,
            "Users"
        )

        # ----------------------------------------
        # Inspection
        # ----------------------------------------

        inspect_dataframe(
            courses_df,
            "Courses"
        )

        inspect_dataframe(
            teachers_df,
            "Teachers"
        )

        inspect_dataframe(
            transactions_df,
            "Transactions"
        )

        inspect_dataframe(
            users_df,
            "Users"
        )

        logging.info(
            "Dataset loaded successfully."
        )

        return (
            courses_df,
            teachers_df,
            transactions_df,
            users_df
        )

    except Exception as e:

        logging.error(
            f"Error while loading dataset: {e}"
        )

        raise

# ====================================================
# Main Testing
# ====================================================

if __name__ == "__main__":

    DATA_PATH = (
        "data/raw/EduPro Online Platform.xlsx"
    )

    courses_df, teachers_df, transactions_df, users_df = (
        load_dataset(DATA_PATH)
    )

    print("\nCourses Sample:")
    print(courses_df.head())

    print("\nTeachers Sample:")
    print(teachers_df.head())

    print("\nTransactions Sample:")
    print(transactions_df.head())

    print("\nUsers Sample:")
    print(users_df.head())