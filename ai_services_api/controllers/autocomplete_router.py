
from fastapi import APIRouter
from ai_services_api.services.search.app.endpoints import autocomplete

api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    autocomplete.router,
    prefix="/autocomplete",  # Prefix for conversation endpoints
    tags=["autocomplete"]  # Tag for documentation
)
