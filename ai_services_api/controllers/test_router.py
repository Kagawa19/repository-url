
from fastapi import APIRouter
from ai_services_api.services.tests.endpoints import tests


api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    message.router,
    prefix="/tests",  # Prefix for conversation endpoints
    tags=["tests"]  # Tag for documentation
)
