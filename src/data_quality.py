
import pandas as pd
from pathlib import Path

# Load original data
df = pd.read_csv("data/raw/events.csv")

print("\n========== DATA QUALITY AUDIT ==========")

# 1. Duplicate analysis
duplicates = df[df.duplicated(keep=False)]

print("\nExact duplicate rows:", df.duplicated().sum())

print("\nSample duplicate records:")
print(duplicates.head(10))

# 2. Transaction ID analysis
print("\nTransaction IDs by event:")

print(
    df.groupby("event")["transactionid"]
      .agg(
          total_records="size",
          missing_ids=lambda x: x.isna().sum(),
          available_ids=lambda x: x.notna().sum()
      )
)

# 3. Check unique transaction IDs
print("\nUnique transaction IDs:",
      df["transactionid"].nunique())

# 4. Check repeated user-product pairs
pair_counts = df.groupby(
    ["visitorid", "itemid"]
).size()

print("\nUser-item pair statistics:")
print(pair_counts.describe())

# 5. Check timestamp conversion
df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    errors="coerce"
)

print("\nInvalid timestamps:",
      df["datetime"].isna().sum())

print("\nMonthly event distribution:")

print(
    df.groupby(df["datetime"].dt.to_period("M"))
      .size()
)

print("\n========== AUDIT COMPLETE ==========")
