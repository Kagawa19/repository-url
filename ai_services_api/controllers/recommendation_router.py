
from fastapi import APIRouter
from ai_services_api.services.recommendation.app.endpoints import recommendation

api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    recommendation.router,
    prefix="/recommendation",  # Prefix for conversation endpoints
    tags=["recommend"]  # Tag for documentation
)
