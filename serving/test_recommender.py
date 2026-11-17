
from serving.recommender import get_recommendations


# ==========================================
# TEST 1: KNOWN USER
# ==========================================

print("\n========== KNOWN USER TEST ==========")

known_user = 7

recommendations = get_recommendations(
    visitor_id=known_user,
    top_k=10
)

print("Visitor ID:", known_user)
print("Recommendations:", len(recommendations))

print(
    recommendations.to_string(index=False)
)


# ==========================================
# TEST 2: COLD-START USER
# ==========================================

print("\n========== COLD-START USER TEST ==========")

cold_user = -1

cold_recommendations = get_recommendations(
    visitor_id=cold_user,
    top_k=5
)

print("Visitor ID:", cold_user)
print("Recommendations:", len(cold_recommendations))

print(
    cold_recommendations.to_string(index=False)
)


# ==========================================
# TEST 3: OUTPUT VALIDATION
# ==========================================

assert len(recommendations) == 10
assert len(cold_recommendations) == 5

assert recommendations["recommendation_score"].between(
    0, 1
).all()

assert recommendations["itemid"].is_unique

print("\nALL RECOMMENDER TESTS PASSED")
