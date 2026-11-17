
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD PARTITIONS
# ==========================================

BASE = Path("data/processed/splits")

train = pd.read_csv(
    BASE / "train.csv.gz",
    usecols=["visitorid", "itemid", "event"]
)

validation = pd.read_csv(
    BASE / "validation.csv.gz",
    usecols=["visitorid", "itemid", "event"]
)

test = pd.read_csv(
    BASE / "test.csv.gz",
    usecols=["visitorid", "itemid", "event"]
)

# ==========================================
# 2. EXTRACT UNIQUE USERS AND PRODUCTS
# ==========================================

train_users = train["visitorid"].unique()
train_items = train["itemid"].unique()

# ==========================================
# 3. USER OVERLAP
# ==========================================

def analyse_overlap(name, data):

    users = data["visitorid"].unique()
    items = data["itemid"].unique()

    known_users = np.isin(users, train_users)
    known_items = np.isin(items, train_items)

    print(f"\n========== {name.upper()} ==========")

    print("Unique users:", len(users))
    print("Known training users:", known_users.sum())
    print("New users:", (~known_users).sum())

    print("Unique products:", len(items))
    print("Known training products:", known_items.sum())
    print("New products:", (~known_items).sum())

    # Interaction-level coverage
    known_user_rows = data["visitorid"].isin(train_users)
    known_item_rows = data["itemid"].isin(train_items)

    print("\nInteraction-level coverage:")

    print("Known user + known product:",
          (known_user_rows & known_item_rows).sum())

    print("New user + known product:",
          (~known_user_rows & known_item_rows).sum())

    print("Known user + new product:",
          (known_user_rows & ~known_item_rows).sum())

    print("New user + new product:",
          (~known_user_rows & ~known_item_rows).sum())

    # Future high-intent events
    positive = data[
        data["event"].isin(["addtocart", "transaction"])
    ]

    print("\nFuture positive interactions:",
          len(positive))

    print("Positive interactions with known users:",
          positive["visitorid"].isin(train_users).sum())

    print("Positive interactions with known products:",
          positive["itemid"].isin(train_items).sum())


# ==========================================
# 4. RUN ANALYSIS
# ==========================================

print("\n========== TRAINING COVERAGE ==========")

print("Training users:", len(train_users))
print("Training products:", len(train_items))

analyse_overlap("Validation", validation)
analyse_overlap("Testing", test)

print("\n========== SPLIT ANALYSIS COMPLETE ==========")
