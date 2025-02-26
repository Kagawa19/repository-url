from fastapi import APIRouter
from ai_services_api.services.chatbot.app.endpoints import conversation

api_router = APIRouter()

api_router.include_router(
    conversation.router,
    prefix="/conversation",  # Prefix for conversation endpoints
    tags=["conversation"]  # Tag for documentation
)
