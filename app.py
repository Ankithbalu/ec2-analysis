
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
    values = series.astype("string")
    values = values.str.replace(",", "", regex=False)
    numbers = values.str.extract(
        r"(-?\d+(?:\.\d+)?)",
        expand=False
    )

    return pd.to_numeric(
        numbers,
        errors="coerce"
    ).astype("float64")


@st.cache_data
def load_data():

    possible_files = [
        BASE / "ec2dataset.csv",
        BASE / "ec2dataset (1).csv",
        BASE / "Amazon EC2 Instance Comparison.csv"
    ]

    csv_file = next(
        (f for f in possible_files if f.is_file()),
        None
    )

    if csv_file is None:
        all_csv_files = list(BASE.glob("*.csv"))

        if len(all_csv_files) == 1:
            csv_file = all_csv_files[0]
        else:
            raise FileNotFoundError(
                "EC2 CSV dataset was not found. "
                "Upload ec2dataset.csv or ec2dataset (1).csv."
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
            f"Missing columns: {missing_columns}"
        )

    for col in COST_COLUMNS:
        df[col] = extract_number(df[col])
        df[col] = df[col].mask(df[col] < 0, np.nan)

    df["Memory_GiB"] = extract_number(
        df["Instance Memory"]
    )

    df["vCPU_Count"] = extract_number(
        df["vCPUs"]
    )

    df["Family"] = (
        df["API Name"]
        .fillna("Unknown")
        .astype(str)
        .str.split(".")
        .str[0]
        .str.lower()
    )

    df["Monthly_Cost_730h"] = (
        df["On Demand"] * 730
    )

    df["Cost_Per_vCPU"] = (
        df["On Demand"] /
        df["vCPU_Count"].replace(0, np.nan)
    )

    df["Cost_Per_GiB"] = (
        df["On Demand"] /
        df["Memory_GiB"].replace(0, np.nan)
    )

    df["Savings_Percent"] = np.nan

    valid_savings = (
        df["On Demand"].notna() &
        df["Linux Reserved cost"].notna() &
        (df["On Demand"] > 0)
    )

    df.loc[valid_savings, "Savings_Percent"] = (
        (
            df.loc[valid_savings, "On Demand"] -
            df.loc[valid_savings, "Linux Reserved cost"]
        ) /
        df.loc[valid_savings, "On Demand"]
    ) * 100

    return df, csv_file.name


try:
    data, filename = load_data()

except Exception as e:
    st.error(f"Unable to load EC2 dataset: {e}")
    st.stop()


st.title("☁️ EC2 Instance EDA Dashboard")

st.markdown(
    "Analyze Amazon EC2 instance costs, compare pricing models, "
    "detect outliers, optimize resources, and predict costs "
    "using machine learning."
)

st.caption(f"Dataset loaded successfully: {filename}")

st.sidebar.title("Dashboard Navigation")

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
st.sidebar.caption("INFO49971 - Cloud Economics")


if page == "Dataset Overview":

    st.header("EC2 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Instances", len(data))

    c2.metric(
        "Original Columns",
        len(data.columns) - 7
    )

    c3.metric(
        "Instance Families",
        data["Family"].nunique()
    )

    c4.metric(
        "Missing On-Demand Prices",
        int(data["On Demand"].isna().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(30),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    st.write("Column Names")

    st.write(data.columns.tolist())

    st.write("Data Types")

    types = data.dtypes.astype(str).reset_index()
    types.columns = ["Column", "Data Type"]

    st.dataframe(
        types,
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

    st.header("EC2 Cost Analysis")

    cost_type = st.selectbox(
        "Select Pricing Option",
        COST_COLUMNS
    )

    prices = data[cost_type].dropna()

    if prices.empty:
        st.warning("No valid pricing data available.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Hourly Cost",
        f"${prices.mean():.4f}"
    )

    c2.metric(
        "Median Hourly Cost",
        f"${prices.median():.4f}"
    )

    c3.metric(
        "Minimum Hourly Cost",
        f"${prices.min():.4f}"
    )

    c4.metric(
        "Maximum Hourly Cost",
        f"${prices.max():.4f}"
    )

    st.subheader("Cost Distribution")

    fig = px.histogram(
        data,
        x=cost_type,
        nbins=50,
        title=f"{cost_type} Price Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Pricing Comparison Boxplot")

    melted = data[COST_COLUMNS].melt(
        var_name="Pricing Type",
        value_name="Hourly Cost"
    ).dropna()

    fig = px.box(
        melted,
        x="Pricing Type",
        y="Hourly Cost",
        color="Pricing Type",
        title="EC2 Pricing Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Average Prices")

    average_prices = (
        data[COST_COLUMNS].mean().reset_index()
    )

    average_prices.columns = [
        "Pricing Type",
        "Average Hourly Cost"
    ]

    fig = px.bar(
        average_prices,
        x="Pricing Type",
        y="Average Hourly Cost",
        color="Pricing Type",
        title="Average EC2 Hourly Cost"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Cost Summary Statistics")

    st.dataframe(
        data[COST_COLUMNS].describe(),
        use_container_width=True
    )


elif page == "Outlier Detection":

    st.header("EC2 Cost Outlier Detection")

    values = data["On Demand"].dropna()

    if values.empty:
        st.warning("No On-Demand prices available.")
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

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Q1", f"${q1:.4f}")
    c2.metric("Q3", f"${q3:.4f}")
    c3.metric("IQR", f"${iqr:.4f}")
    c4.metric("Total Outliers", len(outliers))

    st.write(
        f"Upper outlier threshold: ${upper_bound:.4f}"
    )

    fig = px.box(
        data,
        y="On Demand",
        points="outliers",
        title="On-Demand Price Outliers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Outlier Instances")

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
        "Download Outliers",
        data=outliers.to_csv(index=False),
        file_name="ec2_outliers.csv",
        mime="text/csv"
    )


elif page == "Instance Family Comparison":

    st.header("Compare EC2 Instance Families")

    families = sorted(
        data["Family"].dropna().unique().tolist()
    )

    default = [
        f for f in ["t2", "t3"]
        if f in families
    ]

    selected = st.multiselect(
        "Select Families",
        families,
        default=default
    )

    filtered = data[
        data["Family"].isin(selected)
    ].copy()

    if filtered.empty:
        st.warning("Please select instance families.")
        st.stop()

    st.subheader("Family Cost Comparison")

    fig = px.box(
        filtered,
        x="Family",
        y="On Demand",
        color="Family",
        title="On-Demand Prices by Family"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Family Statistics")

    summary = (
        filtered.groupby("Family")["On Demand"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.subheader("T2 / T3 On-Demand vs Reserved")

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
        comparison.sort_values("On Demand").head(20),
        use_container_width=True
    )


elif page == "Cost Optimization":

    st.header("EC2 Cost Optimization")

    st.write(
        "Identify low-cost instances based on minimum "
        "CPU and memory requirements."
    )

    c1, c2 = st.columns(2)

    with c1:
        min_cpu = st.number_input(
            "Minimum vCPUs",
            min_value=1,
            value=2
        )

    with c2:
        min_memory = st.number_input(
            "Minimum Memory (GiB)",
            min_value=0.5,
            value=4.0
        )

    suitable = data[
        (data["vCPU_Count"] >= min_cpu) &
        (data["Memory_GiB"] >= min_memory) &
        (data["On Demand"] > 0)
    ].copy()

    suitable = suitable.sort_values("On Demand")

    if suitable.empty:

        st.warning(
            "No instances match your requirements."
        )

    else:

        best = suitable.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Lowest-Cost Instance",
            str(best["Name"])
        )

        c2.metric(
            "Hourly Cost",
            f"${best['On Demand']:.4f}"
        )

        c3.metric(
            "Monthly Cost (730 Hours)",
            f"${best['Monthly_Cost_730h']:.2f}"
        )

        st.subheader("Top Matching Instances")

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

        st.plotly_chart(
            fig,
            use_container_width=True
        )

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

        st.metric(
            "Average Reserved Savings",
            f"{savings['Savings_Percent'].mean():.2f}%"
        )

        st.dataframe(
            savings.sort_values(
                "Savings_Percent",
                ascending=False
            ).head(20),
            use_container_width=True
        )

    st.info(
        "Prices are based on the uploaded dataset. "
        "Actual AWS prices depend on configuration, "
        "region and purchase terms."
    )


elif page == "Regression Prediction":

    st.header("EC2 Cost Prediction Using Linear Regression")

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
    ].copy()

    if len(cleaned) < 10:
        st.error("Insufficient data for regression.")
        st.stop()

    X = cleaned[
        ["Memory_GiB", "vCPU_Count"]
    ]

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

    mae = mean_absolute_error(
        y_test, predictions
    )

    mse = mean_squared_error(
        y_test, predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test, predictions
    )

    baseline = np.full(
        len(y_test),
        y_train.mean()
    )

    baseline_mae = mean_absolute_error(
        y_test, baseline
    )

    st.subheader("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"${mae:.4f}")
    c2.metric("MSE", f"{mse:.4f}")
    c3.metric("RMSE", f"${rmse:.4f}")
    c4.metric("R² Score", f"{r2:.4f}")

    st.metric(
        "Baseline MAE",
        f"${baseline_mae:.4f}"
    )

    if mae < baseline_mae:
        st.success(
            "Regression performs better than baseline."
        )
    else:
        st.warning(
            "Regression does not outperform the baseline."
        )

    st.subheader("Regression Coefficients")

    st.write(f"Intercept: {model.intercept_:.6f}")

    st.write(
        f"Memory Coefficient: {model.coef_[0]:.6f}"
    )

    st.write(
        f"vCPU Coefficient: {model.coef_[1]:.6f}"
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
        title="Actual vs Predicted EC2 Costs"
    )

    low = min(
        results["Actual Cost"].min(),
        results["Predicted Cost"].min()
    )

    high = max(
        results["Actual Cost"].max(),
        results["Predicted Cost"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect Prediction",
            line=dict(
                color="red",
                dash="dash"
            )
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Residual Analysis")

    residuals = (
        y_test.to_numpy() - predictions
    )

    residual_df = pd.DataFrame({
        "Predicted Cost": predictions,
        "Residual": residuals
    })

    fig = px.scatter(
        residual_df,
        x="Predicted Cost",
        y="Residual",
        title="Regression Residuals"
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("5-Fold Cross Validation")

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

    c1, c2 = st.columns(2)

    c1.metric(
        "Cross-Validation MAE",
        f"${cv_mae.mean():.4f}"
    )

    c2.metric(
        "CV Standard Deviation",
        f"{cv_mae.std():.4f}"
    )

    st.subheader("Predict New EC2 Instance Cost")

    c1, c2 = st.columns(2)

    with c1:
        memory = st.number_input(
            "Memory (GiB)",
            min_value=0.5,
            value=4.0,
            step=0.5
        )

    with c2:
        vcpus = st.number_input(
            "vCPUs",
            min_value=1,
            value=2,
            step=1
        )

    if st.button(
        "Predict Cost",
        type="primary"
    ):

        new_instance = pd.DataFrame({
            "Memory_GiB": [memory],
            "vCPU_Count": [vcpus]
        })

        predicted_cost = model.predict(
            new_instance
        )[0]

        st.metric(
            "Predicted Hourly Cost",
            f"${predicted_cost:.4f}"
        )

        if predicted_cost < 0:

            st.warning(
                "The model predicted a negative price. "
                "This is a limitation of linear regression, "
                "not a realistic EC2 billing price."
            )

        else:

            st.metric(
                "Estimated Monthly Cost",
                f"${predicted_cost * 730:.2f}"
            )

    st.subheader("Prediction Results")

    st.dataframe(
        results,
        use_container_width=True
    )

    st.download_button(
        "Download Regression Results",
        data=results.to_csv(index=False),
        file_name="ec2_regression_results.csv",
        mime="text/csv"
    )


st.divider()

st.caption(
    "INFO49971 - Cloud Economics | EC2 Instance EDA Dashboard"
)
