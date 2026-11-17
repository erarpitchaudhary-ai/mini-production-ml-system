
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# ==========================================
# 1. FILE PATHS AND SETTINGS
# ==========================================

TRAIN_PATH = "data/processed/splits/train.csv.gz"

OUTPUT_PATH = (
    "data/processed/features/ranker_training_data.csv.gz"
)

TOP_K = 100
MAX_HISTORY = 20

POSITIVE_EVENTS = ["addtocart", "transaction"]

# ==========================================
# 2. LOAD AND CHRONOLOGICALLY SPLIT TRAINING
# ==========================================

print("\nLoading training data...")

df = pd.read_csv(TRAIN_PATH)

df = df.sort_values("timestamp").reset_index(drop=True)

cutoff = df["timestamp"].quantile(0.80)

history_data = df[df["timestamp"] <= cutoff].copy()

label_data = df[df["timestamp"] > cutoff].copy()

print("\n========== TEMPORAL SPLIT ==========")

print("Historical interactions:", len(history_data))
print("Future label interactions:", len(label_data))

print("Historical end:", history_data["timestamp"].max())
print("Label period start:", label_data["timestamp"].min())

# ==========================================
# 3. BUILD USER HISTORIES
# ==========================================

print("\nBuilding user histories...")

user_history_lookup = (
    history_data.groupby("visitorid")["itemid"]
    .apply(lambda x: list(pd.unique(x)))
    .to_dict()
)

# Keep the most recent 20 distinct products.
# The data was sorted by timestamp before grouping.

user_history_lookup = {
    user: items[-MAX_HISTORY:]
    for user, items in user_history_lookup.items()
}

print("Users with historical data:", len(user_history_lookup))

# ==========================================
# 4. BUILD PRODUCT POPULARITY
# ==========================================

print("\nCalculating product popularity...")

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

popular_items = (
    item_stats.sort_values(
        "popularity_score",
        ascending=False
    )["itemid"].tolist()
)

popularity_lookup = item_stats.set_index(
    "itemid"
)["popularity_score"].to_dict()

item_feature_lookup = item_stats.set_index(
    "itemid"
).to_dict("index")

print("Products with historical data:", len(item_stats))

# ==========================================
# 5. BUILD ITEM CO-OCCURRENCE
# ==========================================

print("\nBuilding item co-occurrence...")

pair_counts = Counter()

for items in user_history_lookup.values():

    if len(items) < 2:
        continue

    for item_a, item_b in combinations(sorted(items), 2):
        pair_counts[(item_a, item_b)] += 1

related_items = defaultdict(list)

for (item_a, item_b), count in pair_counts.items():

    related_items[item_a].append((item_b, count))
    related_items[item_b].append((item_a, count))

for item in related_items:

    related_items[item].sort(
        key=lambda x: x[1],
        reverse=True
    )

print("Unique item pairs:", len(pair_counts))
print("Products with co-occurrence links:", len(related_items))

# ==========================================
# 6. BUILD USER FEATURES
# ==========================================

print("\nCalculating user features...")

user_stats = history_data.groupby("visitorid").agg(
    user_event_count=("itemid", "size"),
    user_unique_items=("itemid", "nunique"),
    user_view_count=("event", lambda x: (x == "view").sum()),
    user_cart_count=("event", lambda x: (x == "addtocart").sum()),
    user_transaction_count=("event", lambda x: (x == "transaction").sum())
)

user_feature_lookup = user_stats.to_dict("index")

# ==========================================
# 7. IDENTIFY FUTURE POSITIVE INTERACTIONS
# ==========================================

print("\nPreparing future labels...")

future_positive = label_data[
    label_data["event"].isin(POSITIVE_EVENTS)
]

relevant_by_user = (
    future_positive.groupby("visitorid")["itemid"]
    .apply(set)
    .to_dict()
)

print("Users with future positive interactions:", len(relevant_by_user))

total_relevant_pairs = sum(
    len(items) for items in relevant_by_user.values()
)

# ==========================================
# 8. GENERATE CANDIDATES AND FEATURES
# ==========================================

print("\nGenerating ranking training examples...")

records = []

retrieved_positive_pairs = 0

for index, (visitor_id, relevant_items) in enumerate(
    relevant_by_user.items(), start=1
):

    history = user_history_lookup.get(visitor_id, [])

    seen_items = set(history)

    candidate_scores = {}

    # A. Co-occurrence candidates

    for historical_item in history:

        for related_item, count in related_items.get(
            historical_item, []
        ):

            if related_item in seen_items:
                continue

            candidate_scores[related_item] = (
                candidate_scores.get(related_item, 0) + count
            )

    # B. Popularity supplementation

    for item in popular_items:

        if item in seen_items:
            continue

        if item not in candidate_scores:
            candidate_scores[item] = 0

        if len(candidate_scores) >= TOP_K:
            break

    # C. Keep top candidates

    candidates = sorted(
        candidate_scores.items(),
        key=lambda x: (
            x[1],
            popularity_lookup.get(x[0], 0)
        ),
        reverse=True
    )[:TOP_K]

    candidate_ids = {item for item, score in candidates}

    retrieved_positive_pairs += len(
        candidate_ids.intersection(relevant_items)
    )

    # D. Create one row per candidate

    user_features = user_feature_lookup.get(
        visitor_id,
        {
            "user_event_count": 0,
            "user_unique_items": 0,
            "user_view_count": 0,
            "user_cart_count": 0,
            "user_transaction_count": 0
        }
    )

    for item, cooccurrence_score in candidates:

        product_features = item_feature_lookup.get(
            item,
            {
                "view_count": 0,
                "addtocart_count": 0,
                "transaction_count": 0,
                "unique_visitors": 0
            }
        )

        label = int(item in relevant_items)

        records.append({

            "visitorid": visitor_id,
            "itemid": item,

            # User features
            "user_event_count": user_features["user_event_count"],
            "user_unique_items": user_features["user_unique_items"],
            "user_view_count": user_features["user_view_count"],
            "user_cart_count": user_features["user_cart_count"],
            "user_transaction_count": user_features["user_transaction_count"],

            # Product features
            "item_view_count": product_features["view_count"],
            "item_addtocart_count": product_features["addtocart_count"],
            "item_transaction_count": product_features["transaction_count"],
            "item_unique_visitors": product_features["unique_visitors"],

            # Retrieval features
            "cooccurrence_score": cooccurrence_score,
            "popularity_score": popularity_lookup.get(item, 0),

            # Target
            "label": label
        })

    if index % 1000 == 0:
        print("Processed users:", index)

# ==========================================
# 9. SAVE RANKING DATASET
# ==========================================

print("\nCreating final DataFrame...")

ranking_data = pd.DataFrame(records)

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

ranking_data.to_csv(
    OUTPUT_PATH,
    index=False,
    compression="gzip"
)

# ==========================================
# 10. FINAL REPORT
# ==========================================

print("\n========== RANKER DATASET SUMMARY ==========")

print("Total training examples:", len(ranking_data))

print("Positive examples:", int(ranking_data["label"].sum()))

print(
    "Negative examples:",
    int((ranking_data["label"] == 0).sum())
)

print(
    "Positive label percentage:",
    round(ranking_data["label"].mean() * 100, 3),
    "%"
)

print(
    "Candidate recall in internal label period:",
    round(
        retrieved_positive_pairs / total_relevant_pairs,
        4
    ) if total_relevant_pairs else 0
)

print("\nMissing values:")
print(ranking_data.isnull().sum().sum())

print("\nDataset columns:")
print(ranking_data.columns.tolist())

print("\nSaved:", OUTPUT_PATH)
