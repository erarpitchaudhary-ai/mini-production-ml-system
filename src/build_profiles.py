
import pandas as pd
from pathlib import Path

# ==========================================
# 1. DEFINE PATHS
# ==========================================

BASE_PATH = Path("data/processed/splits")

TRAIN_PATH = BASE_PATH / "train.csv.gz"

OUTPUT_PATH = Path("data/processed/features")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EVENTS = ["view", "addtocart", "transaction"]

# ==========================================
# 2. LOAD TRAINING DATA ONLY
# ==========================================

print("\n========== PROFILE ENGINEERING ==========")

df = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "timestamp",
        "visitorid",
        "itemid",
        "event"
    ]
)

print("Training interactions:", len(df))

# ==========================================
# 3. BUILD USER EVENT COUNTS
# ==========================================

user_counts = (
    df.groupby(["visitorid", "event"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=EVENTS, fill_value=0)
)

user_counts = user_counts.rename(columns={
    "view": "view_count",
    "addtocart": "addtocart_count",
    "transaction": "transaction_count"
})

# ==========================================
# 4. BUILD USER ACTIVITY FEATURES
# ==========================================

user_activity = df.groupby("visitorid").agg(
    unique_items=("itemid", "nunique"),
    last_interaction_ts=("timestamp", "max")
)

user_profiles = user_counts.join(user_activity)

# ==========================================
# 5. BUILD PRODUCT EVENT COUNTS
# ==========================================

item_counts = (
    df.groupby(["itemid", "event"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=EVENTS, fill_value=0)
)

item_counts = item_counts.rename(columns={
    "view": "view_count",
    "addtocart": "addtocart_count",
    "transaction": "transaction_count"
})

# ==========================================
# 6. BUILD PRODUCT ACTIVITY FEATURES
# ==========================================

item_activity = df.groupby("itemid").agg(
    unique_visitors=("visitorid", "nunique"),
    last_interaction_ts=("timestamp", "max")
)

item_profiles = item_counts.join(item_activity)

# ==========================================
# 7. SAVE FEATURE DATASETS
# ==========================================

user_path = OUTPUT_PATH / "user_profiles.csv.gz"
item_path = OUTPUT_PATH / "item_profiles.csv.gz"

user_profiles.to_csv(
    user_path,
    compression="gzip"
)

item_profiles.to_csv(
    item_path,
    compression="gzip"
)

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n========== USER PROFILES ==========")

print("Total users:", len(user_profiles))
print(user_profiles.head())

print("\n========== PRODUCT PROFILES ==========")

print("Total products:", len(item_profiles))
print(item_profiles.head())

print("\n========== FEATURE VALIDATION ==========")

print("User feature missing values:",
      user_profiles.isnull().sum().sum())

print("Product feature missing values:",
      item_profiles.isnull().sum().sum())

print("\nUser profile shape:", user_profiles.shape)
print("Product profile shape:", item_profiles.shape)

print("\nSaved user profiles:", user_path)
print("Saved product profiles:", item_path)

print("\n========== PROFILE ENGINEERING COMPLETE ==========")
