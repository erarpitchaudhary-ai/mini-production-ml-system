
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/features/ranker_training_data.csv.gz"
MODEL_PATH = "artifacts/ranker_logistic_regression.joblib"

RANDOM_STATE = 42

FEATURES = [
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

TARGET = "label"

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading ranking dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# ==========================================
# 3. USER-LEVEL TRAIN / VALIDATION SPLIT
# ==========================================

print("\nPreparing user-level split...")

# One row per user, indicating whether that user
# has at least one positive candidate.
user_labels = (
    df.groupby("visitorid")[TARGET]
    .max()
    .reset_index()
)

train_users, validation_users = train_test_split(
    user_labels["visitorid"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=user_labels[TARGET]
)

train_df = df[df["visitorid"].isin(train_users)].copy()
validation_df = df[
    df["visitorid"].isin(validation_users)
].copy()

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))

print("Training positive examples:", int(train_df[TARGET].sum()))
print("Validation positive examples:", int(validation_df[TARGET].sum()))

print("Training users:", train_df["visitorid"].nunique())
print("Validation users:", validation_df["visitorid"].nunique())

# ==========================================
# 4. PREPARE FEATURES
# ==========================================

X_train = np.log1p(train_df[FEATURES])
X_validation = np.log1p(validation_df[FEATURES])

y_train = train_df[TARGET]
y_validation = validation_df[TARGET]

# ==========================================
# 5. BUILD LOGISTIC REGRESSION MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

model.fit(X_train, y_train)

print("Model training completed.")

# ==========================================
# 6. PREDICT VALIDATION SCORES
# ==========================================

validation_df["model_score"] = model.predict_proba(
    X_validation
)[:, 1]

# ==========================================
# 7. CLASSIFICATION METRICS
# ==========================================

print("\n========== CLASSIFICATION METRICS ==========")

if y_validation.nunique() == 2:

    auc = roc_auc_score(
        y_validation,
        validation_df["model_score"]
    )

    average_precision = average_precision_score(
        y_validation,
        validation_df["model_score"]
    )

    print("ROC-AUC:", round(auc, 4))
    print("Average Precision:", round(average_precision, 4))

else:
    print("Only one target class found in validation.")

# ==========================================
# 8. RANKING METRICS
# ==========================================

def calculate_dcg(labels, k):

    labels = np.asarray(labels)[:k]

    discounts = np.log2(
        np.arange(2, len(labels) + 2)
    )

    return np.sum(
        (2 ** labels - 1) / discounts
    )


def evaluate_ranking(data, score_column, k):

    recalls = []
    precisions = []
    ndcgs = []

    # Evaluate only users with at least one relevant item.
    for visitor_id, group in data.groupby("visitorid"):

        relevant_count = int(group[TARGET].sum())

        if relevant_count == 0:
            continue

        ranked = group.sort_values(
            score_column,
            ascending=False
        )

        top_items = ranked.head(k)

        hits = int(top_items[TARGET].sum())

        recall = hits / relevant_count
        precision = hits / k

        dcg = calculate_dcg(
            top_items[TARGET].tolist(),
            k
        )

        ideal_labels = sorted(
            group[TARGET].tolist(),
            reverse=True
        )

        idcg = calculate_dcg(ideal_labels, k)

        ndcg = dcg / idcg if idcg > 0 else 0

        recalls.append(recall)
        precisions.append(precision)
        ndcgs.append(ndcg)

    return {
        "users": len(recalls),
        "precision": np.mean(precisions) if recalls else 0,
        "recall": np.mean(recalls) if recalls else 0,
        "ndcg": np.mean(ndcgs) if recalls else 0
    }


print("\n========== RANKING EVALUATION ==========")

for k in [5, 10, 20]:

    model_metrics = evaluate_ranking(
        validation_df,
        "model_score",
        k
    )

    popularity_metrics = evaluate_ranking(
        validation_df,
        "popularity_score",
        k
    )

    print(f"\nTOP {k}")

    print("Users evaluated:", model_metrics["users"])

    print("Logistic Regression:")
    print("Precision:", round(model_metrics["precision"], 4))
    print("Recall:", round(model_metrics["recall"], 4))
    print("NDCG:", round(model_metrics["ndcg"], 4))

    print("Popularity baseline:")
    print("Precision:", round(popularity_metrics["precision"], 4))
    print("Recall:", round(popularity_metrics["recall"], 4))
    print("NDCG:", round(popularity_metrics["ndcg"], 4))

# ==========================================
# 9. SAVE MODEL
# ==========================================

Path("artifacts").mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_PATH
)

print("\nModel saved:", MODEL_PATH)
print("\nRANKER TRAINING COMPLETED")
