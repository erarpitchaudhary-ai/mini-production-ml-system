
import pandas as pd
from pathlib import Path

# 1. Locate the dataset
DATA_PATH = Path("data/raw/events.csv")

# 2. Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# 3. Load dataset
df = pd.read_csv(DATA_PATH)

# 4. Display basic information
print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# 5. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 6. Unique users and products
print("\nUnique visitors:", df["visitorid"].nunique())
print("Unique products:", df["itemid"].nunique())

# 7. Interaction distribution
print("\nEvent distribution:")
print(df["event"].value_counts())

# 8. Timestamp range
print("\nTimestamp range:")
print(df["timestamp"].min())
print(df["timestamp"].max())

print("\n========== INSPECTION COMPLETE ==========")
