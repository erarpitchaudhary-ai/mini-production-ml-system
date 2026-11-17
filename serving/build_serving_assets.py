
import pandas as pd
import joblib

from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_PATH = "data/processed/splits/train.csv.gz"
OUTPUT_PATH = "artifacts/serving_assets.joblib"

MAX_HISTORY = 20

# ==========================================
# 2. LOAD AND SPLIT HISTORICAL DATA
# ==========================================

print("\nLoading training data...")

df = pd.read_csv(TRAIN_PATH)

df = df.sort_values("timestamp").reset_index(drop=True)

cutoff = df["timestamp"].quantile(0.80)

history_data = df[
    df["timestamp"] <= cutoff
].copy()

print("\n========== HISTORICAL DATA ==========")

print("Historical interactions:", len(history_data))
print("Historical cutoff:", cutoff)

# ==========================================
# 3. BUILD USER HISTORIES
# ==========================================

print("\nBuilding user histories...")

user_history_lookup = (
    history_data.groupby("visitorid")["itemid"]
    .apply(lambda x: list(pd.unique(x)))
    .to_dict()
)

user_history_lookup = {
    user: items[-MAX_HISTORY:]
    for user, items in user_history_lookup.items()
}

print("Users:", len(user_history_lookup))

# ==========================================
# 4. BUILD USER FEATURES
# ==========================================

print("\nBuilding user features...")

user_stats = history_data.groupby("visitorid").agg(
    user_event_count=("itemid", "size"),
    user_unique_items=("itemid", "nunique"),
    user_view_count=("event", lambda x: (x == "view").sum()),
    user_cart_count=("event", lambda x: (x == "addtocart").sum()),
    user_transaction_count=("event", lambda x: (x == "transaction").sum())
)

user_feature_lookup = user_stats.to_dict("index")

# ==========================================
# 5. BUILD PRODUCT FEATURES
# ==========================================

print("\nBuilding product features...")

item_stats = history_data.groupby("itemid").agg(
    view_count=("event", lambda x: (x == "view").sum()),
    addtocart_count=("event", lambda x: (x == "addtocart").sum()),
    transaction_count=("event", lambda x: (x == "transaction").sum()),
    unique_visitors=("visitorid", "nunique")
).reset_index()

item_stats["popularity_score"] = (
    item_stats["view_count"]
    + 3 * item_stats["addtocart_count"]
    + 5 * item_stats["transaction_count"]
)

item_feature_lookup = item_stats.set_index(
    "itemid"
)[
    [
        "view_count",
        "addtocart_count",
        "transaction_count",
        "unique_visitors"
    ]
].to_dict("index")

popularity_lookup = item_stats.set_index(
    "itemid"
)["popularity_score"].to_dict()

popular_items = (
    item_stats.sort_values(
        "popularity_score",
        ascending=False
    )["itemid"].tolist()
)

print("Products:", len(item_feature_lookup))

# ==========================================
# 6. BUILD ITEM CO-OCCURRENCE
# ==========================================

print("\nBuilding item co-occurrence...")

pair_counts = Counter()

for items in user_history_lookup.values():

    if len(items) < 2:
        continue

    for item_a, item_b in combinations(
        sorted(items), 2
    ):
        pair_counts[(item_a, item_b)] += 1

related_items = defaultdict(list)

for (item_a, item_b), count in pair_counts.items():

    related_items[item_a].append(
        (item_b, count)
    )

    related_items[item_b].append(
        (item_a, count)
    )

for item in related_items:

    related_items[item].sort(
        key=lambda x: x[1],
        reverse=True
    )

print("Unique item pairs:", len(pair_counts))
print("Products with co-occurrence links:", len(related_items))

# ==========================================
# 7. SAVE SERVING ASSETS
# ==========================================

assets = {
    "cutoff": int(cutoff),
    "user_history_lookup": user_history_lookup,
    "user_feature_lookup": user_feature_lookup,
    "item_feature_lookup": item_feature_lookup,
    "popularity_lookup": popularity_lookup,
    "popular_items": popular_items,
    "related_items": dict(related_items)
}

Path("artifacts").mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    assets,
    OUTPUT_PATH,
    compress=3
)

print("\n========== ASSET SUMMARY ==========")

print("Users:", len(user_history_lookup))
print("User feature profiles:", len(user_feature_lookup))
print("Product feature profiles:", len(item_feature_lookup))
print("Popularity entries:", len(popularity_lookup))
print("Co-occurrence entries:", len(related_items))

print("\nSaved:", OUTPUT_PATH)
print("\nSERVING ASSETS CREATED SUCCESSFULLY")
