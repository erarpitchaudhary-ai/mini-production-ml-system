
from pydantic import BaseModel, Field
from typing import List


# ==========================================
# 1. PREDICTION REQUEST
# ==========================================

class RecommendationRequest(BaseModel):

    visitor_id: int = Field(
        ...,
        description="Unique ID of the user",
        examples=[7]
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations requested"
    )


# ==========================================
# 2. INDIVIDUAL RECOMMENDATION
# ==========================================

class RecommendationItem(BaseModel):

    itemid: int

    recommendation_score: float

    cooccurrence_score: int

    popularity_score: int


# ==========================================
# 3. PREDICTION RESPONSE
# ==========================================

class RecommendationResponse(BaseModel):

    visitor_id: int

    recommendations: List[RecommendationItem]

    total_recommendations: int


# ==========================================
# 4. HEALTH RESPONSE
# ==========================================

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool