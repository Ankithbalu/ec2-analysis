
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

data = pd.read_csv(BASE / "ec2dataset.csv")

print("\nSTEP 2: DATASET INFORMATION")
print("Dataset shape:", data.shape)
data.info()
print(data.head())
print("\nColumns:", data.columns.tolist())

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]

required = ["Name", "Instance Memory", "vCPUs"] + cost_columns
missing = [c for c in required if c not in data.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

def extract_number(series):
    values = series.astype("string").str.replace(",", "", regex=False)
    numbers = values.str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(numbers, errors="coerce")

for col in cost_columns:
    data[col] = extract_number(data[col])
    data.loc[data[col] < 0, col] = np.nan

data["Memory_GiB"] = extract_number(data["Instance Memory"])
data["vCPU_Count"] = extract_number(data["vCPUs"])

print("\nSTEP 3: MISSING VALUES")
print(data.isna().sum())

print("\nSTEP 4: SUMMARY STATISTICS")
summary = data[cost_columns].describe()
print(summary.round(4))
summary.to_csv(OUTPUT / "cost_summary.csv")

plt.figure(figsize=(13, 6))
sns.boxplot(data=data[cost_columns])
plt.title("EC2 Hourly Cost Comparison")
plt.xlabel("Pricing Type")
plt.ylabel("Hourly Cost (USD)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "01_cost_boxplot.png", dpi=200)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=np.log1p(data[cost_columns]))
plt.title("EC2 Hourly Cost Distribution (log1p scale)")
plt.ylabel("log(1 + Hourly Cost)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "02_log_cost_boxplot.png", dpi=200)
plt.show()

print("\nSTEP 6: ON-DEMAND OUTLIERS")
prices = data["On Demand"].dropna()
q1 = prices.quantile(0.25)
q3 = prices.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = data[
    (data["On Demand"] < lower) |
    (data["On Demand"] > upper)
].copy()

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Upper bound:", upper)
print("Number of outliers:", len(outliers))
print(outliers[["Name", "On Demand"]].head(20))
outliers.to_csv(OUTPUT / "on_demand_outliers.csv", index=False)

print("\nSTEP 7: RESERVED VS ON-DEMAND")
comparison = data[
    ["Name", "On Demand", "Linux Reserved cost"]
].dropna().copy()

comparison = comparison[
    (comparison["On Demand"] > 0) &
    (comparison["Linux Reserved cost"] >= 0)
]

comparison["Savings_Per_Hour"] = (
    comparison["On Demand"] - comparison["Linux Reserved cost"]
)

comparison["Savings_Percent"] = (
    comparison["Savings_Per_Hour"] /
    comparison["On Demand"] * 100
)

print(comparison.sort_values("On Demand").head(10))
print("\nMean savings percentage:", comparison["Savings_Percent"].mean())
comparison.to_csv(OUTPUT / "reserved_savings.csv", index=False)

plt.figure(figsize=(9, 5))
average_prices = data[cost_columns].mean().sort_values()
sns.barplot(x=average_prices.index, y=average_prices.values)
plt.title("Average Hourly Price by Pricing Option")
plt.ylabel("Average Price (USD/hour)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "03_average_prices.png", dpi=200)
plt.show()

print("\nSTEP 8: INSTANCE FAMILY COMPARISON")

data["Family"] = (
    data["API Name"].astype("string")
    .str.split(".").str[0]
    .str.lower()
)

t2 = data[data["Family"] == "t2"]
t3 = data[data["Family"] == "t3"]

for family_name, family_data in [("T2", t2), ("T3", t3)]:
    print(f"\n{family_name} INSTANCE SUMMARY")
    if family_data.empty:
        print("No instances found.")
        continue

    print(family_data[cost_columns].describe().round(4))

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=family_data[cost_columns], showmeans=True)
    plt.title(f"{family_name} Instance Cost Distribution")
    plt.ylabel("Hourly Price (USD)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"04_{family_name}_costs.png", dpi=200)
    plt.show()

family_comparison = pd.concat([t2, t3])

if not family_comparison.empty:
    print("\n10 LOWEST-COST T2/T3 INSTANCES")
    print(
        family_comparison[
            ["Name", "On Demand", "Linux Reserved cost"]
        ].dropna().sort_values("On Demand").head(10)
    )

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=family_comparison,
        x="Family",
        y="On Demand",
        showmeans=True
    )
    plt.title("T2 vs T3 On-Demand Cost Comparison")
    plt.ylabel("Hourly Price (USD)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "05_t2_vs_t3.png", dpi=200)
    plt.show()

print("\nBONUS: PRICE PER RESOURCE")

efficiency = data[
    (data["On Demand"] > 0) &
    (data["vCPU_Count"] > 0) &
    (data["Memory_GiB"] > 0)
].copy()

efficiency["Cost_Per_vCPU"] = (
    efficiency["On Demand"] / efficiency["vCPU_Count"]
)
efficiency["Cost_Per_GiB"] = (
    efficiency["On Demand"] / efficiency["Memory_GiB"]
)
efficiency["Monthly_Cost_730h"] = efficiency["On Demand"] * 730

print(
    efficiency[
        ["Name", "On Demand", "Cost_Per_vCPU", "Cost_Per_GiB"]
    ].sort_values("Cost_Per_vCPU").head(10)
)

efficiency.to_csv(OUTPUT / "instance_efficiency.csv", index=False)

print("\nPART 1 COMPLETED. Results saved in outputs folder.")
