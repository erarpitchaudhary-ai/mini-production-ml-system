
from fastapi import FastAPI, HTTPException
from serving.recommender import get_recommendations

from serving.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    HealthResponse
)

app = FastAPI(
    title="E-Commerce Recommendation API",
    description="Production-style ML recommendation service",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True
    )


@app.post("/predict", response_model=RecommendationResponse)
def predict(request: RecommendationRequest):

    try:
        recommendations = get_recommendations(
            visitor_id=request.visitor_id,
            top_k=request.top_k
        )

        recommendation_list = recommendations.to_dict(
            orient="records"
        )

        return RecommendationResponse(
            visitor_id=request.visitor_id,
            recommendations=recommendation_list,
            total_recommendations=len(recommendation_list)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(error)}"
        )
