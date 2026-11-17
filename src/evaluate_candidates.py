
import pandas as pd

from recommend import (
    recommend_candidates,
    user_history_lookup
)

# -----------------------------------
# 1. Load validation data
# -----------------------------------

VALIDATION_PATH = "data/processed/splits/validation.csv.gz"

validation = pd.read_csv(VALIDATION_PATH)

print("Validation interactions:", len(validation))

# -----------------------------------
# 2. Define positive interactions
# -----------------------------------

positive_events = ["addtocart", "transaction"]

positive_data = validation[
    validation["event"].isin(positive_events)
].copy()

# Create unique positive user-item pairs
relevant_by_user_raw = (
    positive_data.groupby("visitorid")["itemid"]
    .apply(set)
    .to_dict()
)

# -----------------------------------
# 3. Remove previously seen products
# -----------------------------------

relevant_by_user = {}

for visitor_id, relevant_items in relevant_by_user_raw.items():

    seen_items = set(
        user_history_lookup.get(visitor_id, [])
    )

    unseen_relevant_items = relevant_items - seen_items

    if unseen_relevant_items:
        relevant_by_user[visitor_id] = unseen_relevant_items

raw_pairs = sum(
    len(items) for items in relevant_by_user_raw.values()
)

eligible_pairs = sum(
    len(items) for items in relevant_by_user.values()
)

print("\n========== TARGET FILTERING ==========")

print("Original positive user-item pairs:", raw_pairs)
print("Eligible unseen positive pairs:", eligible_pairs)
print("Excluded previously seen pairs:", raw_pairs - eligible_pairs)
print("Users with eligible targets:", len(relevant_by_user))

# -----------------------------------
# 4. Evaluate candidate retrieval
# -----------------------------------

results = {
    "all": {"users": 0, "hits": 0, "relevant": 0, "recalls": []},
    "warm": {"users": 0, "hits": 0, "relevant": 0, "recalls": []},
    "cold": {"users": 0, "hits": 0, "relevant": 0, "recalls": []}
}

candidate_covered_users = 0

for index, (visitor_id, relevant_items) in enumerate(
    relevant_by_user.items(), start=1
):

    candidates = recommend_candidates(
        visitor_id,
        top_k=100
    )

    candidate_items = set(candidates["itemid"])

    hits = len(candidate_items.intersection(relevant_items))

    recall = hits / len(relevant_items)

    cohort = (
        "warm"
        if visitor_id in user_history_lookup
        else "cold"
    )

    for group in ["all", cohort]:

        results[group]["users"] += 1
        results[group]["hits"] += hits
        results[group]["relevant"] += len(relevant_items)
        results[group]["recalls"].append(recall)

    if len(candidate_items) > 0:
        candidate_covered_users += 1

    if index % 1000 == 0:
        print("Evaluated users:", index)

# -----------------------------------
# 5. Display evaluation metrics
# -----------------------------------

print("\n========== CANDIDATE EVALUATION ==========")

for group in ["all", "warm", "cold"]:

    data = results[group]

    if data["users"] == 0:
        print(f"\n{group.upper()}: No eligible users")
        continue

    micro_recall = data["hits"] / data["relevant"]

    macro_recall = (
        sum(data["recalls"]) / len(data["recalls"])
    )

    print(f"\n{group.upper()} USERS")

    print("Users evaluated:", data["users"])
    print("Relevant products:", data["relevant"])
    print("Retrieved relevant products:", data["hits"])

    print("Micro Recall@100:", round(micro_recall, 4))
    print("Macro Recall@100:", round(macro_recall, 4))

# -----------------------------------
# 6. Candidate coverage
# -----------------------------------

total_users = len(relevant_by_user)

coverage = (
    candidate_covered_users / total_users
    if total_users else 0
)

print("\n========== COVERAGE ==========")

print("Users evaluated:", total_users)
print("Users receiving candidates:", candidate_covered_users)
print("Candidate coverage:", round(coverage, 4))
