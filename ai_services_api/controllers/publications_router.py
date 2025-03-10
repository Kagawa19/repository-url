
from fastapi import APIRouter
from ai_services_api.services.centralized_repository.app.endpoints import publications

api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    publications.router,
    prefix="/publications",  # Prefix for conversation endpoints
    tags=["publications"]  # Tag for documentation
)
