from ai_services_api.services.message.app.endpoints import analytics

from fastapi import APIRouter


api_router = APIRouter()

api_router.include_router(
    analytics.router,  # Use analytics_router instead of router
    prefix="/analytics",
    tags=["analytics"]
)
