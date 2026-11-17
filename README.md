
# E-Commerce Recommendation & Candidate Ranking System

## 1. Project Overview

A mini production-oriented Machine Learning system for
e-commerce product recommendation and candidate ranking.

The system uses the Retailrocket E-Commerce Dataset.

It combines:
- Popularity-based candidate retrieval.
- Item-to-item co-occurrence recommendations.
- Logistic Regression ranking.
- FastAPI-based inference.
- Cold-start fallback.
- Latency benchmarking and drift monitoring.

## 2. Problem Statement

E-commerce platforms need to recommend relevant products
to users based on their historical interactions.

This project builds a recommendation pipeline that:
- Generates candidate products.
- Ranks candidates using a trained ML model.
- Handles users without interaction history.
- Exposes recommendations through an API.

Positive recommendation outcomes are defined as
add-to-cart and transaction events.

## 3. System Architecture

Raw Events
    |
    v
Data Cleaning & Validation
    |
    v
Chronological Data Splitting
    |
    v
Feature Engineering
    |
    v
Candidate Generation
    |
    v
Logistic Regression Ranker
    |
    v
Model Serialization
    |
    v
FastAPI Inference
    |
    v
Top-K Product Recommendations

## 4. Dataset

Dataset: Retailrocket E-Commerce Dataset

Main file:
- events.csv

Other available files:
- item_properties_part1.csv
- item_properties_part2.csv
- category_tree.csv

The raw dataset is not included in this repository.

Place the downloaded files inside:

data/raw/

## 5. Machine Learning Approach

### Candidate Retrieval

Two retrieval strategies are combined:

1. Item-to-item co-occurrence.
2. Weighted product popularity.

Popularity score:

Views + (3 x Add-to-cart) + (5 x Transactions)

The weights are heuristic choices.

### Ranking Model

Algorithm: Logistic Regression

Preprocessing:
- Log transformation using log1p.
- StandardScaler.
- Class-weight balancing.

The model predicts the probability of a candidate
being a positive recommendation.

Positive labels:
- Add-to-cart.
- Transaction.

### Cold-Start Handling

For users without historical interactions,
the system retrieves candidates using product popularity.

The ranking model then orders those candidates.

## 6. Model Evaluation

Candidate retrieval:
- Recall@100: 6.37%
- Warm-user Recall@100: 9.87%
- Cold-user Recall@100: 5.60%

Ranking model:
- ROC-AUC: 0.7416
- Average Precision: 0.0045

Ranking comparison:

| Metric | Logistic Regression | Popularity |
|---|---:|---:|
| Precision@5 | 0.0459 | 0.0422 |
| Recall@5 | 0.2225 | 0.1972 |
| NDCG@5 | 0.1458 | 0.1265 |
| Precision@10 | 0.0312 | 0.0239 |
| Recall@10 | 0.2928 | 0.2179 |
| NDCG@10 | 0.1701 | 0.1338 |
| Precision@20 | 0.0243 | 0.0179 |
| Recall@20 | 0.4534 | 0.3265 |
| NDCG@20 | 0.2120 | 0.1622 |

Ranking metrics were evaluated on users with
positive validation candidates.

## 7. Production Inference

Framework: FastAPI

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Model health check |
| POST | /predict | Generate recommendations |

Start the API:

python -m uvicorn serving.app:app --reload

API documentation:

http://127.0.0.1:8000/docs

Example request:

{
  "visitor_id": 7,
  "top_k": 5
}

## 8. Performance Benchmark

100 measured sequential requests per scenario,
after 10 warm-up requests.

| Metric | Known User | Cold-Start |
|---|---:|---:|
| Average latency | 19.26 ms | 19.79 ms |
| Median latency | 20.41 ms | 20.78 ms |
| P95 latency | 27.69 ms | 28.09 ms |
| Maximum latency | 44.12 ms | 36.57 ms |

Measurements were obtained on a local development
environment, not under concurrent production load.

## 9. Monitoring

Monitoring compares training and test periods.

Signals:
- Event distribution changes.
- New-user interaction rate.
- New-product interaction rate.

Observed:
- New-user interaction rate: 90.05%.
- New-product interaction rate: 10.83%.

An alert is generated when the new-user interaction
rate exceeds the configured 80% threshold.

## 10. Project Limitations

- Highly imbalanced ranking labels.
- Low candidate retrieval recall.
- Limited item-specific handling for unseen products.
- Popularity can favour frequently interacted products.
- Local latency does not establish production scalability.
- Monitoring uses basic distribution checks rather than
  a comprehensive statistical drift framework.

## 11. Future Improvements

- Improve candidate retrieval recall.
- Add content-based recommendations.
- Explore LightGBM or XGBoost ranking.
- Introduce formal drift metrics such as PSI.
- Add automated retraining and model versioning.
- Conduct concurrent-load and stress testing.

## 12. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Matplotlib
- Seaborn
