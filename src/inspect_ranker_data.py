
import pandas as pd

PATH = "data/processed/features/ranker_training_data.csv.gz"

df = pd.read_csv(PATH)

print("\n========== DATASET AUDIT ==========")

# 1. Dataset dimensions
print("\nShape:", df.shape)

# 2. Duplicate user-product pairs
duplicates = df.duplicated(
    subset=["visitorid", "itemid"]
).sum()

print("Duplicate user-item pairs:", duplicates)

# 3. Label distribution
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nPositive percentage:")
print(round(df["label"].mean() * 100, 4), "%")

# 4. Positive users
positive_rows = df[df["label"] == 1]

print("\nUsers with positive candidates:",
      positive_rows["visitorid"].nunique())

# 5. Positive examples per user
print("\nPositive examples per user:")
print(
    positive_rows.groupby("visitorid").size().describe()
)

# 6. Missing values
print("\nMissing values:", df.isnull().sum().sum())

# 7. Negative feature values
feature_columns = [
    "user_event_count",
    "user_unique_items",
    "user_view_count",
    "user_cart_count",
    "user_transaction_count",
    "item_view_count",
    "item_addtocart_count",
    "item_transaction_count",
    "item_unique_visitors",
    "cooccurrence_score",
    "popularity_score"
]

print("\nNegative feature values:")
print((df[feature_columns] < 0).sum())

# 8. Feature statistics
print("\nFeature statistics:")
print(df[feature_columns].describe().round(2))

# 9. Target validation
print("\nUnique labels:", sorted(df["label"].unique()))

assert duplicates == 0, "Duplicate user-item pairs found!"
assert df.isnull().sum().sum() == 0, "Missing values found!"
assert set(df["label"].unique()).issubset({0, 1})

print("\nAUDIT PASSED")
