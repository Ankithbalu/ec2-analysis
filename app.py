import streamlit as st
import pandas as pd
import plotly.express as px

st.title("EC2 Instance EDA Dashboard")

df = pd.read_csv("ec2dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Instances", len(df))
col2.metric("Number of Columns", len(df.columns))
col3.metric("Missing Values", df.isna().sum().sum())

st.subheader("Dataset Structure")

st.write("Columns:")
st.write(df.columns.tolist())

st.subheader("Data Types")
st.text(df.dtypes.to_string())

df["Memory_GiB"] = (
    df["Instance Memory"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)

df["vCPU_Count"] = (
    df["vCPUs"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .astype(float)
)

def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value)

    if "unavailable" in value.lower():
        return None

    try:
        return float(
            value
            .replace("$", "")
            .replace(" hourly", "")
            .replace(",", "")
            .strip()
        )

    except ValueError:
        return None

price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)

df["Monthly_On_Demand"] = (
    df["On Demand_USD"] * 730
)

df["Cost_Per_GiB"] = (
    df["On Demand_USD"] /
    df["Memory_GiB"]
)

df["Cost_Per_vCPU"] = (
    df["On Demand_USD"] /
    df["vCPU_Count"]
)

st.subheader("Monthly On-Demand Cost")

st.dataframe(
    df[
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
)

st.sidebar.header("Filters")

max_memory = float(
    df["Memory_GiB"]
    .dropna()
    .max()
)

memory_filter = st.sidebar.slider(
    "Maximum Memory (GiB)",
    min_value=0.5,
    max_value=max_memory,
    value=max_memory
)

cpu_values = sorted(
    df["vCPU_Count"]
    .dropna()
    .unique()
    .tolist()
)

selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values
)

network_values = sorted(
    df["Network Performance"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_values,
    default=network_values
)

max_price = float(
    df["On Demand_USD"]
    .dropna()
    .max()
)

price_filter = st.sidebar.slider(
    "Maximum Hourly Price ($)",
    min_value=0.0,
    max_value=max_price,
    value=max_price
)

max_monthly_cost = float(
    df["Monthly_On_Demand"]
    .dropna()
    .max()
)

monthly_cost_filter = st.sidebar.slider(
    "Maximum Monthly Cost ($)",
    min_value=0.0,
    max_value=max_monthly_cost,
    value=max_monthly_cost
)

filtered_df = df[
    (df["Memory_GiB"] <= memory_filter) &
    (df["vCPU_Count"].isin(selected_cpu)) &
    (df["Network Performance"].astype(str).isin(selected_network)) &
    (df["On Demand_USD"] <= price_filter) &
    (df["Monthly_On_Demand"] <= monthly_cost_filter)
].copy()

st.subheader("Filtered EC2 Instances")

st.write(
    f"{len(filtered_df)} instances found"
)

st.dataframe(filtered_df)

st.subheader("EC2 Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Instances",
    len(filtered_df)
)

if not filtered_df.empty:

    col2.metric(
        "Avg Memory",
        f"{filtered_df['Memory_GiB'].mean():.2f} GiB"
    )

    col3.metric(
        "Avg vCPUs",
        f"{filtered_df['vCPU_Count'].mean():.1f}"
    )

    col4.metric(
        "Avg Hourly Cost",
        f"${filtered_df['On Demand_USD'].mean():.4f}"
    )

else:

    col2.metric(
        "Avg Memory",
        "N/A"
    )

    col3.metric(
        "Avg vCPUs",
        "N/A"
    )

    col4.metric(
        "Avg Hourly Cost",
        "N/A"
    )

st.subheader("Memory Distribution")

if not filtered_df.empty:

    fig = px.histogram(
        filtered_df,
        x="Memory_GiB",
        nbins=30,
        title="Distribution of EC2 Memory"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.warning(
        "No instances match the selected filters."
    )

st.subheader("vCPU Distribution")

if not filtered_df.empty:

    fig = px.histogram(
        filtered_df,
        x="vCPU_Count",
        title="Distribution of vCPUs"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.subheader("Memory vs vCPUs")

if not filtered_df.empty:

    fig = px.scatter(
        filtered_df,
        x="vCPU_Count",
        y="Memory_GiB",
        hover_name="API Name",
        hover_data=["On Demand_USD"],
        title="EC2 Memory vs vCPU"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.subheader("Memory vs On-Demand Cost")

chart_df = filtered_df.dropna(
    subset=[
        "Memory_GiB",
        "On Demand_USD",
        "vCPU_Count"
    ]
)

if not chart_df.empty:

    fig = px.scatter(
        chart_df,
        x="Memory_GiB",
        y="On Demand_USD",
        hover_name="API Name",
        size="vCPU_Count",
        title="Memory vs EC2 On-Demand Cost"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.subheader("Lowest-Cost EC2 Instances")

cheapest = (
    filtered_df
    .dropna(subset=["On Demand_USD"])
    .sort_values("On Demand_USD")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(cheapest)

st.subheader("Highest-Cost EC2 Instances")

most_expensive = (
    filtered_df
    .dropna(subset=["On Demand_USD"])
    .sort_values(
        "On Demand_USD",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(most_expensive)

st.subheader("EC2 Pricing Comparison")

pricing = filtered_df[
    [
        "Name",
        "API Name",
        "On Demand_USD",
        "Linux Reserved cost_USD",
        "Linux Spot Minimum cost_USD",
        "Windows On Demand cost_USD",
        "Windows Reserved cost_USD"
    ]
]

st.dataframe(pricing)

st.subheader("Instance Pricing Comparison")

available_instances = (
    filtered_df["API Name"]
    .dropna()
    .unique()
)

if len(available_instances) > 0:

    selected_instance = st.selectbox(
        "Select an EC2 Instance",
        available_instances
    )

    instance = filtered_df[
        filtered_df["API Name"] == selected_instance
    ].iloc[0]

    pricing_data = pd.DataFrame(
        {
            "Pricing Model": [
                "On Demand",
                "Linux Reserved",
                "Linux Spot"
            ],

            "Hourly Cost": [
                instance["On Demand_USD"],
                instance["Linux Reserved cost_USD"],
                instance["Linux Spot Minimum cost_USD"]
            ]
        }
    )

    pricing_data = pricing_data.dropna()

    pricing_data["Monthly Cost"] = (
        pricing_data["Hourly Cost"] * 730
    )

    st.dataframe(pricing_data)

    if not pricing_data.empty:

        fig = px.bar(
            pricing_data,
            x="Pricing Model",
            y="Hourly Cost",
            title=f"Pricing Comparison: {selected_instance}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.warning(
        "No EC2 instances are available with the selected filters."
    )

st.subheader("Cost per GiB of Memory")

efficiency = (
    filtered_df
    .dropna(subset=["Cost_Per_GiB"])
    .sort_values("Cost_Per_GiB")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Cost_Per_GiB"
        ]
    ]
    .head(15)
)

st.dataframe(efficiency)

st.subheader("Cost per vCPU")

cpu_efficiency = (
    filtered_df
    .dropna(subset=["Cost_Per_vCPU"])
    .sort_values("Cost_Per_vCPU")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Cost_Per_vCPU"
        ]
    ]
    .head(15)
)

st.dataframe(cpu_efficiency)

st.subheader("Download Data")

csv = filtered_df.to_csv(
    index=False
)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_ec2_instances.csv",
    mime="text/csv"
)
