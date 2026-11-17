
import pandas as pd
from collections import defaultdict

# -----------------------------------
# 1. File paths
# -----------------------------------

POPULAR_PATH = "data/processed/candidates/popular_items.csv.gz"
COOC_PATH = "data/processed/candidates/item_cooccurrence.csv.gz"
TRAIN_PATH = "data/processed/splits/train.csv.gz"

# -----------------------------------
# 2. Load datasets
# -----------------------------------

popular = pd.read_csv(POPULAR_PATH)
cooc = pd.read_csv(COOC_PATH)
train = pd.read_csv(TRAIN_PATH)

print("Datasets loaded successfully.")

# -----------------------------------
# 3. Precompute user histories
# -----------------------------------

user_history_lookup = (
    train.groupby("visitorid")["itemid"]
    .apply(lambda x: list(pd.unique(x)))
    .to_dict()
)

print("User history lookup created:", len(user_history_lookup))

# -----------------------------------
# 4. Precompute popularity scores
# -----------------------------------

popularity_lookup = popular.set_index(
    "itemid"
)["popularity_score"].to_dict()

# Global popularity ranking
popular_items = popular.sort_values(
    "popularity_score",
    ascending=False
)["itemid"].tolist()

# -----------------------------------
# 5. Build item co-occurrence lookup
# -----------------------------------

related_items = defaultdict(list)

for row in cooc.itertuples(index=False):

    related_items[row.item_a].append(
        (row.item_b, row.cooccurrence_count)
    )

    related_items[row.item_b].append(
        (row.item_a, row.cooccurrence_count)
    )

# Sort related products by association strength
for item in related_items:

    related_items[item].sort(
        key=lambda x: x[1],
        reverse=True
    )

print("Co-occurrence lookup created:", len(related_items))

# -----------------------------------
# 6. Candidate recommendation function
# -----------------------------------

def recommend_candidates(visitor_id, top_k=100):

    """
    Generate product candidates using:

    1. Item co-occurrence
    2. Global popularity

    Returns:
    itemid, cooccurrence_score, popularity_score
    """

    # Retrieve precomputed user history
    user_history = user_history_lookup.get(
        visitor_id, []
    )

    seen_items = set(user_history)

    candidate_scores = {}

    # -----------------------------------
    # A. Co-occurrence candidates
    # -----------------------------------

    for item in user_history[-20:]:

        for related_item, count in related_items.get(item, []):

            if related_item in seen_items:
                continue

            if related_item not in candidate_scores:
                candidate_scores[related_item] = 0

            candidate_scores[related_item] += count

    # -----------------------------------
    # B. Popularity supplementation
    # -----------------------------------

    for item in popular_items:

        if item in seen_items:
            continue

        if item not in candidate_scores:
            candidate_scores[item] = 0

        if len(candidate_scores) >= top_k:
            break

    # -----------------------------------
    # C. Create recommendation table
    # -----------------------------------

    results = pd.DataFrame([

        {
            "itemid": item,
            "cooccurrence_score": score,
            "popularity_score": popularity_lookup.get(item, 0)
        }

        for item, score in candidate_scores.items()

    ])

    if results.empty:
        return results

    # -----------------------------------
    # D. Rank candidates
    # -----------------------------------

    results = results.sort_values(
        ["cooccurrence_score", "popularity_score"],
        ascending=False
    )

    results = results.head(top_k).reset_index(drop=True)

    return results


# -----------------------------------
# 7. Test recommendation function
# -----------------------------------

if __name__ == "__main__":

    # Test a known user
    user_counts = train.groupby("visitorid").size()

    test_user = user_counts[
        user_counts >= 3
    ].index[0]

    recommendations = recommend_candidates(
        test_user,
        top_k=100
    )

    print("\n========== KNOWN USER TEST ==========")

    print("Test visitor:", test_user)

    print("Generated candidates:", len(recommendations))

    print("\nTop 10 candidates:")

    print(
        recommendations.head(10).to_string(index=False)
    )

    # -----------------------------------
    # Test a cold-start user
    # -----------------------------------

    cold_user = -1

    cold_recommendations = recommend_candidates(
        cold_user,
        top_k=100
    )

    print("\n========== COLD-START USER TEST ==========")

    print(
        "Cold-start candidates:",
        len(cold_recommendations)
    )

    print("\nTop 5 cold-start recommendations:")

    print(
        cold_recommendations.head(5).to_string(index=False)
    )
