
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="EC2 Instance EDA Dashboard",
    page_icon="☁️",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

COST_COLUMNS = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]


def extract_number(series):
    values = series.astype("string").str.replace(
        ",", "", regex=False
    )
    numbers = values.str.extract(
        r"(-?\d+(?:\.\d+)?)"
    )[0]
    return pd.to_numeric(numbers, errors="coerce")


@st.cache_data
def load_data():
    possible_files = [
        BASE / "ec2dataset.csv",
        BASE / "ec2dataset (1).csv",
        BASE / "Amazon EC2 Instance Comparison.csv"
    ]

    csv_file = next(
        (file for file in possible_files if file.is_file()),
        None
    )

    if csv_file is None:
        csv_files = [
            file for file in BASE.glob("*.csv")
            if file.is_file()
        ]

        if len(csv_files) == 1:
            csv_file = csv_files[0]
        else:
            raise FileNotFoundError(
                "Dataset not found. Please upload ec2dataset.csv "
                "or ec2dataset (1).csv to your GitHub repository."
            )

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    required_columns = [
        "Name",
        "API Name",
        "Instance Memory",
        "vCPUs"
    ] + COST_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for col in COST_COLUMNS:
        df[col] = extract_number(df[col])
        df.loc[df[col] < 0, col] = np.nan

    df["Memory_GiB"] = extract_number(df["Instance Memory"])
    df["vCPU_Count"] = extract_number(df["vCPUs"])

    df["Family"] = (
        df["API Name"]
        .astype("string")
        .str.split(".")
        .str[0]
        .str.lower()
    )

    df["Monthly_Cost_730h"] = df["On Demand"] * 730

    df["Cost_Per_vCPU"] = np.where(
        df["vCPU_Count"] > 0,
        df["On Demand"] / df["vCPU_Count"],
        np.nan
    )

    df["Cost_Per_GiB"] = np.where(
        df["Memory_GiB"] > 0,
        df["On Demand"] / df["Memory_GiB"],
        np.nan
    )

    df["Savings_Percent"] = np.where(
        df["On Demand"] > 0,
        (
            df["On Demand"] - df["Linux Reserved cost"]
        ) / df["On Demand"] * 100,
        np.nan
    )

    return df, csv_file.name


try:
    data, filename = load_data()
except Exception as e:
    st.error(f"Unable to load EC2 dataset: {e}")
    st.stop()


st.title("☁️ EC2 Instance EDA Dashboard")

st.write(
    "Amazon EC2 Cost Analysis and Machine Learning "
    "Prediction Dashboard"
)

st.caption(f"Dataset loaded: {filename}")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Dataset Overview",
        "Cost Analysis",
        "Outlier Detection",
        "Instance Family Comparison",
        "Cost Optimization",
        "Regression Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.write("INFO49971 - Cloud Economics")


if page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Instances", len(data))
    col2.metric("Total Columns", len(data.columns))
    col3.metric(
        "Instance Families",
        data["Family"].nunique()
    )
    col4.metric(
        "Missing On-Demand Prices",
        data["On Demand"].isna().sum()
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(30),
        use_container_width=True
    )

    st.subheader("Dataset Structure")

    st.write("Column Names")
    st.write(data.columns.tolist())

    st.write("Data Types")
    st.dataframe(
        data.dtypes.astype(str).rename("Data Type"),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = data.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        data[COST_COLUMNS].describe(),
        use_container_width=True
    )


elif page == "Cost Analysis":

    st.header("EC2 Pricing Analysis")

    selected_cost = st.selectbox(
        "Select Pricing Option",
        COST_COLUMNS
    )

    valid_prices = data[selected_cost].dropna()

    if valid_prices.empty:
        st.warning("No valid prices are available.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Hourly Cost",
        f"${valid_prices.mean():.4f}"
    )

    col2.metric(
        "Median Hourly Cost",
        f"${valid_prices.median():.4f}"
    )

    col3.metric(
        "Minimum Hourly Cost",
        f"${valid_prices.min():.4f}"
    )

    col4.metric(
        "Maximum Hourly Cost",
        f"${valid_prices.max():.4f}"
    )

    st.subheader("Cost Distribution")

    fig = px.histogram(
        data,
        x=selected_cost,
        nbins=50,
        title=f"{selected_cost} Cost Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Pricing Options Comparison")

    melted = data[COST_COLUMNS].melt(
        var_name="Pricing Type",
        value_name="Hourly Cost"
    ).dropna()

    fig = px.box(
        melted,
        x="Pricing Type",
        y="Hourly Cost",
        color="Pricing Type",
        title="EC2 Hourly Price Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Hourly Costs")

    average = data[COST_COLUMNS].mean().reset_index()
    average.columns = ["Pricing Type", "Average Cost"]

    fig = px.bar(
        average,
        x="Pricing Type",
        y="Average Cost",
        color="Pricing Type",
        title="Average EC2 Cost by Pricing Option"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost Summary Table")

    st.dataframe(
        data[COST_COLUMNS].describe(),
        use_container_width=True
    )


elif page == "Outlier Detection":

    st.header("On-Demand Cost Outlier Detection")

    values = data["On Demand"].dropna()

    if values.empty:
        st.warning("No valid On-Demand prices available.")
        st.stop()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data["On Demand"] < lower_bound) |
        (data["On Demand"] > upper_bound)
    ].copy()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Q1", f"${q1:.4f}")
    col2.metric("Q3", f"${q3:.4f}")
    col3.metric("IQR", f"${iqr:.4f}")
    col4.metric("Outliers", len(outliers))

    st.write(
        f"Upper outlier threshold: ${upper_bound:.4f}/hour"
    )

    fig = px.box(
        data,
        y="On Demand",
        points="outliers",
        title="EC2 Price Outlier Detection"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Outliers")

    st.dataframe(
        outliers[
            [
                "Name",
                "Memory_GiB",
                "vCPU_Count",
                "On Demand"
            ]
        ].sort_values(
            "On Demand",
            ascending=False
        ),
        use_container_width=True
    )

    st.download_button(
        "Download Outliers CSV",
        outliers.to_csv(index=False),
        "ec2_outliers.csv",
        "text/csv"
    )


elif page == "Instance Family Comparison":

    st.header("Instance Family Comparison")

    families = sorted(
        data["Family"].dropna().unique().tolist()
    )

    default_families = [
        family for family in ["t2", "t3"]
        if family in families
    ]

    selected_families = st.multiselect(
        "Choose Instance Families",
        families,
        default=default_families
    )

    filtered = data[
        data["Family"].isin(selected_families)
    ].copy()

    if filtered.empty:
        st.warning("Select at least one family.")
        st.stop()

    st.subheader("Instance Family Cost Distribution")

    fig = px.box(
        filtered,
        x="Family",
        y="On Demand",
        color="Family",
        points="outliers",
        title="On-Demand Price by Instance Family"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost Statistics by Family")

    summary = (
        filtered.groupby("Family")["On Demand"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.subheader("On-Demand vs Reserved Pricing")

    comparison = filtered[
        [
            "Name",
            "Family",
            "On Demand",
            "Linux Reserved cost",
            "Savings_Percent"
        ]
    ].dropna()

    st.dataframe(
        comparison.sort_values("On Demand").head(30),
        use_container_width=True
    )

    st.subheader("Cheapest Instances")

    st.dataframe(
        filtered[
            [
                "Name",
                "Family",
                "Memory_GiB",
                "vCPU_Count",
                "On Demand",
                "Monthly_Cost_730h"
            ]
        ].dropna(
            subset=["On Demand"]
        ).sort_values("On Demand").head(20),
        use_container_width=True
    )


elif page == "Cost Optimization":

    st.header("EC2 Cost Optimization")

    st.write(
        "Find low-cost EC2 instances based on your "
        "minimum CPU and memory requirements."
    )

    col1, col2 = st.columns(2)

    with col1:
        min_cpu = st.number_input(
            "Minimum vCPUs",
            min_value=1,
            value=2,
            step=1
        )

    with col2:
        min_memory = st.number_input(
            "Minimum Memory (GiB)",
            min_value=0.5,
            value=4.0,
            step=0.5
        )

    suitable = data[
        (data["vCPU_Count"] >= min_cpu) &
        (data["Memory_GiB"] >= min_memory) &
        (data["On Demand"] > 0)
    ].copy()

    suitable = suitable.sort_values("On Demand")

    st.subheader("Recommended Low-Cost Instances")

    if suitable.empty:
        st.warning(
            "No instances match the selected requirements."
        )
    else:
        best = suitable.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Lowest-Cost Instance",
            str(best["Name"])
        )

        col2.metric(
            "Hourly Cost",
            f"${best['On Demand']:.4f}"
        )

        col3.metric(
            "Monthly Cost (730 hours)",
            f"${best['Monthly_Cost_730h']:.2f}"
        )

        st.dataframe(
            suitable[
                [
                    "Name",
                    "Family",
                    "vCPU_Count",
                    "Memory_GiB",
                    "On Demand",
                    "Monthly_Cost_730h",
                    "Cost_Per_vCPU",
                    "Cost_Per_GiB"
                ]
            ].head(20),
            use_container_width=True
        )

        fig = px.bar(
            suitable.head(10),
            x="Name",
            y="On Demand",
            color="Family",
            title="10 Lowest-Cost Matching Instances"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reserved Instance Savings")

    savings = data[
        [
            "Name",
            "On Demand",
            "Linux Reserved cost",
            "Savings_Percent"
        ]
    ].dropna()

    savings = savings[
        (savings["On Demand"] > 0) &
        (savings["Savings_Percent"] >= 0)
    ]

    if not savings.empty:
        average_savings = savings["Savings_Percent"].mean()

        st.metric(
            "Average Reserved Price Savings",
            f"{average_savings:.2f}%"
        )

        st.dataframe(
            savings.sort_values(
                "Savings_Percent",
                ascending=False
            ).head(20),
            use_container_width=True
        )

    st.info(
        "Cost comparisons use prices in the uploaded dataset. "
        "Actual AWS prices vary by region, purchase terms, and usage. "
        "Monthly cost assumes 730 running hours."
    )


elif page == "Regression Prediction":

    st.header("Predict EC2 Costs Using Linear Regression")

    cleaned = data.dropna(
        subset=[
            "Memory_GiB",
            "vCPU_Count",
            "On Demand"
        ]
    ).copy()

    cleaned = cleaned[
        (cleaned["Memory_GiB"] > 0) &
        (cleaned["vCPU_Count"] > 0) &
        (cleaned["On Demand"] >= 0)
    ]

    if len(cleaned) < 10:
        st.error("Insufficient valid data for regression.")
        st.stop()

    X = cleaned[["Memory_GiB", "vCPU_Count"]]
    y = cleaned["On Demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    baseline = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline)

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"${mae:.4f}/hour")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"${rmse:.4f}/hour")
    col4.metric("R² Score", f"{r2:.4f}")

    st.metric(
        "Baseline MAE",
        f"${baseline_mae:.4f}/hour"
    )

    if mae < baseline_mae:
        st.success("The model outperforms the baseline.")
    else:
        st.warning(
            "The model does not outperform the baseline."
        )

    st.subheader("Linear Regression Equation")

    intercept = model.intercept_
    memory_coef = model.coef_[0]
    cpu_coef = model.coef_[1]

    st.latex(
        rf"\hat{{y}} = {intercept:.4f}"
        rf" + ({memory_coef:.4f})M"
        rf" + ({cpu_coef:.4f})C"
    )

    st.caption(
        "M = Memory in GiB, C = Number of vCPUs. "
        "The predicted value is the hourly On-Demand cost."
    )

    st.subheader("Actual vs Predicted Costs")

    results = pd.DataFrame({
        "Actual Cost": y_test.to_numpy(),
        "Predicted Cost": predictions
    })

    fig = px.scatter(
        results,
        x="Actual Cost",
        y="Predicted Cost",
        title="Actual vs Predicted On-Demand Costs"
    )

    lower = min(
        results["Actual Cost"].min(),
        results["Predicted Cost"].min()
    )

    upper = max(
        results["Actual Cost"].max(),
        results["Predicted Cost"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Perfect Prediction",
            line=dict(
                color="red",
                dash="dash"
            )
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residual Analysis")

    residuals = y_test.to_numpy() - predictions

    residual_df = pd.DataFrame({
        "Predicted Cost": predictions,
        "Residual": residuals
    })

    fig = px.scatter(
        residual_df,
        x="Predicted Cost",
        y="Residual",
        title="Residuals vs Predicted Prices"
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Fold Cross-Validation")

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        LinearRegression(),
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )

    cv_mae = -cv_scores

    col1, col2 = st.columns(2)

    col1.metric(
        "Cross-Validation MAE",
        f"${cv_mae.mean():.4f}/hour"
    )

    col2.metric(
        "CV Standard Deviation",
        f"{cv_mae.std():.4f}"
    )

    st.subheader("Predict a New EC2 Instance")

    col1, col2 = st.columns(2)

    with col1:
        memory = st.number_input(
            "Memory (GiB)",
            min_value=0.5,
            value=4.0,
            step=0.5
        )

    with col2:
        vcpus = st.number_input(
            "Number of vCPUs",
            min_value=1,
            value=2,
            step=1
        )

    if st.button("Predict Hourly Cost", type="primary"):

        new_instance = pd.DataFrame({
            "Memory_GiB": [memory],
            "vCPU_Count": [vcpus]
        })

        predicted = model.predict(new_instance)[0]

        st.metric(
            "Raw Predicted Hourly Cost",
            f"${predicted:.4f}"
        )

        if predicted < 0:
            st.warning(
                "The linear regression model produced a negative "
                "price. This is not a realistic EC2 cost and "
                "indicates a limitation of the model."
            )
        else:
            st.metric(
                "Estimated Monthly Cost (730 hours)",
                f"${predicted * 730:.2f}"
            )

    st.subheader("Model Prediction Results")

    st.dataframe(
        results.head(30),
        use_container_width=True
    )

    st.download_button(
        "Download Prediction Results",
        results.to_csv(index=False),
        "ec2_regression_predictions.csv",
        "text/csv"
    )


st.divider()

st.caption(
    "EC2 Instance EDA Dashboard | INFO49971 Cloud Economics"
)
