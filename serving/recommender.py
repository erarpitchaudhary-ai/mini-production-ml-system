
import joblib
import pandas as pd
import numpy as np

from pathlib import Path


# ==========================================
# 1. LOAD MODEL AND SERVING ASSETS
# ==========================================

MODEL_PATH = "artifacts/ranker_logistic_regression.joblib"
ASSETS_PATH = "artifacts/serving_assets.joblib"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not Path(ASSETS_PATH).exists():
    raise FileNotFoundError(f"Serving assets not found: {ASSETS_PATH}")

model_artifact = joblib.load(MODEL_PATH)
assets = joblib.load(ASSETS_PATH)

model = model_artifact["model"]
FEATURES = model_artifact["features"]

user_history_lookup = assets["user_history_lookup"]
user_feature_lookup = assets["user_feature_lookup"]
item_feature_lookup = assets["item_feature_lookup"]
popularity_lookup = assets["popularity_lookup"]
popular_items = assets["popular_items"]
related_items = assets["related_items"]

print("Recommendation engine loaded successfully.")


# ==========================================
# 2. RECOMMENDATION FUNCTION
# ==========================================

def get_recommendations(visitor_id, top_k=10):

    """
    Generate and rank product recommendations.

    Steps:
    1. Retrieve user history.
    2. Generate co-occurrence candidates.
    3. Supplement with popularity candidates.
    4. Construct the 11 model features.
    5. Predict recommendation scores.
    6. Return top-ranked products.
    """

    if not isinstance(visitor_id, (int, np.integer)):
        raise ValueError("visitor_id must be an integer")

    if not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")

    visitor_id = int(visitor_id)

    # -----------------------------------
    # A. Retrieve user history
    # -----------------------------------

    user_history = user_history_lookup.get(
        visitor_id, []
    )

    seen_items = set(user_history)

    candidate_scores = {}

    # -----------------------------------
    # B. Generate co-occurrence candidates
    # -----------------------------------

    for historical_item in user_history:

        for related_item, count in related_items.get(
            historical_item, []
        ):

            if related_item in seen_items:
                continue

            candidate_scores[related_item] = (
                candidate_scores.get(related_item, 0) + count
            )

    # -----------------------------------
    # C. Supplement with popularity
    # -----------------------------------

    for item in popular_items:

        if item in seen_items:
            continue

        if item not in candidate_scores:
            candidate_scores[item] = 0

        if len(candidate_scores) >= 100:
            break

    # -----------------------------------
    # D. Select top 100 candidates
    # -----------------------------------

    candidates = sorted(
        candidate_scores.items(),
        key=lambda x: (
            x[1],
            popularity_lookup.get(x[0], 0)
        ),
        reverse=True
    )[:100]

    if not candidates:
        return pd.DataFrame(
            columns=[
                "itemid",
                "recommendation_score",
                "cooccurrence_score",
                "popularity_score"
            ]
        )

    # -----------------------------------
    # E. Retrieve user features
    # -----------------------------------

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

    # -----------------------------------
    # F. Construct candidate features
    # -----------------------------------

    records = []

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

        records.append({

            "itemid": int(item),

            "user_event_count": user_features["user_event_count"],
            "user_unique_items": user_features["user_unique_items"],
            "user_view_count": user_features["user_view_count"],
            "user_cart_count": user_features["user_cart_count"],
            "user_transaction_count": user_features["user_transaction_count"],

            "item_view_count": product_features["view_count"],
            "item_addtocart_count": product_features["addtocart_count"],
            "item_transaction_count": product_features["transaction_count"],
            "item_unique_visitors": product_features["unique_visitors"],

            "cooccurrence_score": cooccurrence_score,
            "popularity_score": popularity_lookup.get(item, 0)
        })

    candidate_df = pd.DataFrame(records)

    # -----------------------------------
    # G. Apply trained ML ranker
    # -----------------------------------

    X = candidate_df[FEATURES]

    X = np.log1p(X)

    candidate_df["recommendation_score"] = (
        model.predict_proba(X)[:, 1]
    )

    # -----------------------------------
    # H. Return ranked recommendations
    # -----------------------------------

    results = candidate_df.sort_values(
        "recommendation_score",
        ascending=False
    ).head(top_k)

    return results[
        [
            "itemid",
            "recommendation_score",
            "cooccurrence_score",
            "popularity_score"
        ]
    ].reset_index(drop=True)
