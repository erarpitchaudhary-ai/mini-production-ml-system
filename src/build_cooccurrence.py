
import pandas as pd
from collections import Counter
from itertools import combinations
from pathlib import Path

# -----------------------------------
# 1. Load training data
# -----------------------------------

TRAIN_PATH = "data/processed/splits/train.csv.gz"
OUTPUT_PATH = "data/processed/candidates/item_cooccurrence.csv.gz"

df = pd.read_csv(TRAIN_PATH)
df = df.sort_values("timestamp")

print("Training interactions:", len(df))

# -----------------------------------
# 2. Build user-item histories
# -----------------------------------

# Remove repeated interactions with the same item
# for the purpose of co-occurrence.
user_items = (
    df.groupby("visitorid")["itemid"]
    .apply(lambda x: list(pd.unique(x))[-20:])
)

print("Users:", len(user_items))
print("Users with 2+ distinct items:",
      sum(len(items) >= 2 for items in user_items))

# -----------------------------------
# 3. Count item pairs
# -----------------------------------

pair_counts = Counter()

for items in user_items:
    if len(items) < 2:
        continue

    for item_a, item_b in combinations(sorted(items), 2):
        pair_counts[(item_a, item_b)] += 1

print("Unique item pairs:", len(pair_counts))

# -----------------------------------
# 4. Convert pairs into a table
# -----------------------------------

records = []

for (item_a, item_b), count in pair_counts.items():
    records.append({
        "item_a": item_a,
        "item_b": item_b,
        "cooccurrence_count": count
    })

cooc = pd.DataFrame(records)

# -----------------------------------
# 5. Save output
# -----------------------------------

Path("data/processed/candidates").mkdir(
    parents=True, exist_ok=True
)

cooc.to_csv(OUTPUT_PATH, index=False, compression="gzip")

print("\nTop 10 co-occurring pairs:")
print(
    cooc.sort_values(
        "cooccurrence_count", ascending=False
    ).head(10).to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
