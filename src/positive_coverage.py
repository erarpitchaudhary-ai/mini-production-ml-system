
import pandas as pd
from pathlib import Path

BASE = Path("data/processed/splits")

# Load training reference data
train = pd.read_csv(
    BASE / "train.csv.gz",
    usecols=["visitorid", "itemid"]
)

train_users = set(train["visitorid"].unique())
train_items = set(train["itemid"].unique())

del train

# Analyse future positive interactions
def analyse_positive_coverage(filename):

    df = pd.read_csv(
        BASE / filename,
        usecols=["visitorid", "itemid", "event"]
    )

    # Future high-intent interactions
    positive = df[
        df["event"].isin(["addtocart", "transaction"])
    ].copy()

    known_user = positive["visitorid"].isin(train_users)
    known_item = positive["itemid"].isin(train_items)

    print(f"\n========== {filename.upper()} ==========")

    print("Total positive interactions:", len(positive))

    print("Known user + known product:",
          (known_user & known_item).sum())

    print("Known user + new product:",
          (known_user & ~known_item).sum())

    print("New user + known product:",
          (~known_user & known_item).sum())

    print("New user + new product:",
          (~known_user & ~known_item).sum())

    print("\nUnique positive users:",
          positive["visitorid"].nunique())

    print("Unique positive products:",
          positive["itemid"].nunique())


analyse_positive_coverage("validation.csv.gz")
analyse_positive_coverage("test.csv.gz")

print("\n========== POSITIVE COVERAGE COMPLETE ==========")
