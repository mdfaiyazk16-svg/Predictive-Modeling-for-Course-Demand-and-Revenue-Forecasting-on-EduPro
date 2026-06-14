"""
====================================================
EduPro Predictive Modeling Dashboard
====================================================
"""

import os
import sys
import streamlit as st
import joblib
import pandas as pd
import sklearn
import plotly.express as px

# ====================================================
# Page Config
# ====================================================

st.set_page_config(
    page_title="EduPro Predictive Modeling",
    page_icon="📚",
    layout="wide"
)

# ====================================================
# Header
# ====================================================

st.title("📚 EduPro Predictive Modeling Dashboard")

st.markdown(
    """
Predictive Modeling for Course Demand
and Revenue Forecasting
"""
)

# ====================================================
# Debug Info
# ====================================================

with st.expander("System Information"):

    st.write("Python Version:", sys.version)

    st.write(
        "Scikit-Learn Version:",
        sklearn.__version__
    )

# ====================================================
# Load Dataset
# ====================================================

@st.cache_data
def load_dataset():

    path = "data/processed/final_ml_dataset.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)

# ====================================================
# Load Models
# ====================================================

@st.cache_resource
def load_models():

    demand_model_path = (
        "models/enrollment_model.pkl"
    )

    revenue_model_path = (
        "models/revenue_model.pkl"
    )

    if not os.path.exists(
        demand_model_path
    ):
        raise FileNotFoundError(
            f"Missing file: {demand_model_path}"
        )

    if not os.path.exists(
        revenue_model_path
    ):
        raise FileNotFoundError(
            f"Missing file: {revenue_model_path}"
        )

    demand_model = joblib.load(
        demand_model_path
    )

    revenue_model = joblib.load(
        revenue_model_path
    )

    return demand_model, revenue_model

# ====================================================
# Safe Loading
# ====================================================

try:

    df = load_dataset()

    st.success(
        "Dataset Loaded Successfully"
    )

except Exception as e:

    st.error(
        f"Dataset Loading Error: {e}"
    )

    st.stop()

try:

    demand_model, revenue_model = (
        load_models()
    )

    st.success(
        "Models Loaded Successfully"
    )

except Exception as e:

    st.error(
        f"Model Loading Error: {e}"
    )

    st.stop()

# ====================================================
# Sidebar
# ====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Project Overview",
        "Demand Prediction",
        "Revenue Prediction",
        "Feature Importance",
        "Category Comparison",
        "Model Performance"
    ]
)

# ====================================================
# Project Overview
# ====================================================

if page == "Project Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Courses",
        df["CourseID"].nunique()
    )

    col2.metric(
        "Records",
        len(df)
    )

    col3.metric(
        "Categories",
        df["CourseCategory"].nunique()
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

# ====================================================
# Demand Prediction
# ====================================================

elif page == "Demand Prediction":

    st.header(
        "Course Demand Prediction"
    )

    course_price = st.number_input(
        "Course Price",
        min_value=0.0,
        value=500.0
    )

    course_duration = st.number_input(
        "Course Duration",
        min_value=1.0,
        value=20.0
    )

    course_rating = st.slider(
        "Course Rating",
        1.0,
        5.0,
        4.0
    )

    teacher_rating = st.slider(
        "Teacher Rating",
        1.0,
        5.0,
        4.0
    )

    years_exp = st.number_input(
        "Years Of Experience",
        min_value=0,
        value=5
    )

    if st.button(
        "Predict Demand"
    ):

        st.warning(
            """
Demand model accuracy is limited
(R² ≈ 0.03).

Prediction is for demonstration only.
"""
        )

        st.info(
            "Demand Model Loaded Successfully."
        )

# ====================================================
# Revenue Prediction
# ====================================================

elif page == "Revenue Prediction":

    st.header(
        "Revenue Forecast"
    )

    sample = df.iloc[0].copy()

    course_category = st.selectbox(
        "Course Category",
        sorted(
            df["CourseCategory"].unique()
        )
    )

    course_type = st.selectbox(
        "Course Type",
        sorted(
            df["CourseType"].unique()
        )
    )

    course_level = st.selectbox(
        "Course Level",
        sorted(
            df["CourseLevel"].unique()
        )
    )

    course_price = st.number_input(
        "Course Price",
        min_value=0.0,
        value=500.0
    )

    course_duration = st.number_input(
        "Course Duration",
        min_value=1.0,
        value=20.0
    )

    course_rating = st.slider(
        "Course Rating",
        1.0,
        5.0,
        4.0
    )

    teacher_rating = st.slider(
        "Teacher Rating",
        1.0,
        5.0,
        4.0
    )

    years_exp = st.number_input(
        "Years Of Experience",
        min_value=0,
        value=5
    )

    if st.button(
        "Predict Revenue"
    ):

        try:

            sample[
                "CourseCategory"
            ] = course_category

            sample[
                "CourseType"
            ] = course_type

            sample[
                "CourseLevel"
            ] = course_level

            sample[
                "CoursePrice"
            ] = course_price

            sample[
                "CourseDuration"
            ] = course_duration

            sample[
                "CourseRating"
            ] = course_rating

            sample[
                "TeacherRating"
            ] = teacher_rating

            sample[
                "YearsOfExperience"
            ] = years_exp

            sample = sample.drop(
                [
                    "EnrollmentCount",
                    "TotalRevenue",
                    "CourseID",
                    "YearMonth"
                ],
                errors="ignore"
            )

            pred = revenue_model.predict(
                pd.DataFrame([sample])
            )[0]

            st.success(
                f"Predicted Revenue: ₹ {pred:,.2f}"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

# ====================================================
# Feature Importance
# ====================================================

elif page == "Feature Importance":

    st.header(
        "Feature Importance Analysis"
    )

    file_path = (
        "outputs/feature_importance.csv"
    )

    if os.path.exists(file_path):

        imp_df = pd.read_csv(
            file_path
        )

        st.dataframe(
            imp_df.head(20)
        )

        fig = px.bar(
            imp_df.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.error(
            "feature_importance.csv not found."
        )

# ====================================================
# Category Comparison
# ====================================================

elif page == "Category Comparison":

    st.header(
        "Category Revenue Comparison"
    )

    category_df = (
        df.groupby(
            "CourseCategory"
        )["TotalRevenue"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        category_df,
        x="CourseCategory",
        y="TotalRevenue",
        title="Revenue by Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        category_df.sort_values(
            "TotalRevenue",
            ascending=False
        )
    )

# ====================================================
# Model Performance
# ====================================================

elif page == "Model Performance":

    st.header(
        "Model Performance Summary"
    )

    performance_df = pd.DataFrame({

        "Model": [
            "Demand Model",
            "Revenue Model"
        ],

        "Best Algorithm": [
            "Random Forest",
            "Linear Regression"
        ],

        "R² Score": [
            0.026,
            0.855
        ]
    })

    st.dataframe(
        performance_df
    )

    fig = px.bar(
        performance_df,
        x="Model",
        y="R² Score",
        title="Model Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )