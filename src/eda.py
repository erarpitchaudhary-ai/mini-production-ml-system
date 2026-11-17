
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------
# 1. LOAD DATA
# ---------------------------------------

DATA_PATH = Path("data/raw/events.csv")

df = pd.read_csv(DATA_PATH)

print("\n========== EDA STARTED ==========")

# ---------------------------------------
# 2. BASIC DATASET INFORMATION
# ---------------------------------------

print("\nDataset shape:", df.shape)

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

# ---------------------------------------
# 3. USER AND PRODUCT ANALYSIS
# ---------------------------------------

total_users = df["visitorid"].nunique()
total_items = df["itemid"].nunique()
total_events = len(df)

print("\n========== USER ANALYSIS ==========")

print("Total users:", total_users)
print("Total products:", total_items)
print("Total interactions:", total_events)

print("\nInteractions per user:")
print(df.groupby("visitorid").size().describe())

print("\nInteractions per product:")
print(df.groupby("itemid").size().describe())

# ---------------------------------------
# 4. EVENT DISTRIBUTION
# ---------------------------------------

event_counts = df["event"].value_counts()

print("\n========== EVENT DISTRIBUTION ==========")
print(event_counts)

print("\nEvent percentages:")
print((event_counts / total_events * 100).round(2))

event_counts.plot(
    kind="bar",
    title="Distribution of User Events",
    xlabel="Event Type",
    ylabel="Number of Events"
)

plt.tight_layout()
plt.savefig("reports/event_distribution.png")
plt.show()

# ---------------------------------------
# 5. TIMESTAMP ANALYSIS
# ---------------------------------------

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    errors="coerce"
)

print("\n========== TIME ANALYSIS ==========")

print("Earliest interaction:", df["datetime"].min())
print("Latest interaction:", df["datetime"].max())

daily_events = (
    df.dropna(subset=["datetime"])
      .set_index("datetime")
      .resample("D")
      .size()
)

daily_events.plot(
    figsize=(12, 5),
    title="Daily User Interaction Trend",
    xlabel="Date",
    ylabel="Number of Events"
)

plt.tight_layout()
plt.savefig("reports/daily_interactions.png")
plt.show()

# ---------------------------------------
# 6. USER ACTIVITY ANALYSIS
# ---------------------------------------

user_activity = df.groupby("visitorid").size()

print("\n========== USER ACTIVITY ==========")

print("Users with only one interaction:",
      (user_activity == 1).sum())

print("Users with more than 10 interactions:",
      (user_activity > 10).sum())

print("Median interactions per user:",
      user_activity.median())

# ---------------------------------------
# 7. PRODUCT POPULARITY
# ---------------------------------------

popular_items = (
    df.groupby("itemid")
      .size()
      .sort_values(ascending=False)
      .head(10)
)

print("\n========== TOP 10 PRODUCTS ==========")
print(popular_items)

popular_items.plot(
    kind="bar",
    title="Top 10 Most Interacted Products",
    xlabel="Product ID",
    ylabel="Interactions"
)

plt.tight_layout()
plt.savefig("reports/top_products.png")
plt.show()

# ---------------------------------------
# 8. INTERACTION MATRIX SPARSITY
# ---------------------------------------


# Correct interaction density

unique_pairs = df[
    ["visitorid", "itemid"]
].drop_duplicates().shape[0]

possible_interactions = total_users * total_items

density = (
    unique_pairs / possible_interactions
) * 100

print("\nUnique user-item pairs:", unique_pairs)

print("Possible user-item combinations:",
      possible_interactions)

print("Correct interaction density:",
      round(density, 6), "%")


print("\n========== MATRIX ANALYSIS ==========")

print("Possible user-item combinations:",
      possible_interactions)

print("Approximate interaction density:",
      round(density * 100, 6), "%")

print("\n========== EDA COMPLETED ==========")
