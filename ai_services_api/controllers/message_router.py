
from fastapi import APIRouter
from ai_services_api.services.message.app.endpoints import message

api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    message.router,
    prefix="/message",  # Prefix for conversation endpoints
    tags=["message"]  # Tag for documentation
)
