
import pandas as pd
from pathlib import Path

# ==========================================
# 1. DEFINE PATHS
# ==========================================

BASE = Path("data/processed")

TRAIN_PATH = BASE / "splits/train.csv.gz"

FEATURE_PATH = BASE / "features"

OUTPUT_PATH = BASE / "candidates"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD TRAINING DATA ONLY
# ==========================================

print("\n========== CANDIDATE GENERATION ==========")

df = pd.read_csv(
    TRAIN_PATH,
    usecols=["visitorid", "itemid", "event"]
)

print("Training interactions:", len(df))

# ==========================================
# 3. CALCULATE PRODUCT INTERACTION COUNTS
# ==========================================

counts = (
    df.groupby(["itemid", "event"])
      .size()
      .unstack(fill_value=0)
      .reindex(
          columns=["view", "addtocart", "transaction"],
          fill_value=0
      )
)

# ==========================================
# 4. CALCULATE WEIGHTED POPULARITY
# ==========================================

counts["popularity_score"] = (
    counts["view"]
    + 3 * counts["addtocart"]
    + 5 * counts["transaction"]
)

# ==========================================
# 5. CREATE RANKED PRODUCT LIST
# ==========================================

popularity = (
    counts.reset_index()
          .sort_values(
              ["popularity_score", "itemid"],
              ascending=[False, True]
          )
)

popularity["popularity_rank"] = range(
    1, len(popularity) + 1
)

# ==========================================
# 6. SAVE POPULARITY CANDIDATES
# ==========================================

output_file = OUTPUT_PATH / "popular_items.csv.gz"

popularity.to_csv(
    output_file,
    index=False,
    compression="gzip"
)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== TOP 10 POPULAR PRODUCTS ==========")

print(
    popularity[
        [
            "itemid",
            "view",
            "addtocart",
            "transaction",
            "popularity_score",
            "popularity_rank"
        ]
    ].head(10).to_string(index=False)
)

print("\nTotal candidate products:", len(popularity))

print("\nSaved:", output_file)

print("\n========== CANDIDATE GENERATION COMPLETE ==========")
