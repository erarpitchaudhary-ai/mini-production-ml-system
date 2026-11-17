
import pandas as pd
from pathlib import Path

# ---------------------------------------
# 1. DEFINE FILE PATHS
# ---------------------------------------

RAW_PATH = Path("data/raw/events.csv")

OUTPUT_PATH = Path(
    "data/processed/events_clean.csv.gz"
)

# ---------------------------------------
# 2. LOAD RAW DATA
# ---------------------------------------

print("\n========== DATA PREPARATION ==========")

df = pd.read_csv(RAW_PATH)

original_rows = len(df)

print("Original records:", original_rows)

# ---------------------------------------
# 3. REMOVE EXACT DUPLICATES
# ---------------------------------------

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().copy()

print("Removed exact duplicates:", duplicate_count)
print("Records after cleaning:", len(df))

# ---------------------------------------
# 4. CONVERT TIMESTAMP
# ---------------------------------------

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    errors="coerce"
)

invalid_timestamps = df["datetime"].isna().sum()

print("Invalid timestamps:", invalid_timestamps)

if invalid_timestamps > 0:
    raise ValueError("Invalid timestamps detected.")


# ---------------------------------------
# 5. VALIDATE REQUIRED COLUMNS
# ---------------------------------------

# These columns must never contain missing values.
required_columns = [
    "timestamp",
    "visitorid",
    "event",
    "itemid"
]

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Unexpected missing values in required columns."
    )

# Transaction ID validation:
# Missing IDs are expected for views and add-to-cart events.

transaction_events = df[
    df["event"] == "transaction"
]

non_transaction_events = df[
    df["event"] != "transaction"
]

# Every transaction event must have an ID.
if transaction_events["transactionid"].isnull().any():
    raise ValueError(
        "Transaction events contain missing transaction IDs."
    )

# Non-transaction events should not have transaction IDs.
if non_transaction_events["transactionid"].notnull().any():
    raise ValueError(
        "Unexpected transaction IDs in non-transaction events."
    )

print("Required column validation: PASSED")
print("Transaction ID validation: PASSED")


# ---------------------------------------
# 6. SORT CHRONOLOGICALLY
# ---------------------------------------

df = df.sort_values(
    ["timestamp", "visitorid", "itemid"]
).reset_index(drop=True)

# ---------------------------------------
# 7. SAVE CLEANED DATA
# ---------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    compression="gzip"
)

# ---------------------------------------
# 8. FINAL VALIDATION
# ---------------------------------------

print("\n========== FINAL VALIDATION ==========")

print("Final records:", len(df))

print("Unique users:", df["visitorid"].nunique())

print("Unique products:", df["itemid"].nunique())

print("Remaining exact duplicates:",
      df.duplicated().sum())

print("Saved file:", OUTPUT_PATH)

print("\n========== PREPARATION COMPLETE ==========")
