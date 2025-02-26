# In app/controllers/chatbot_router.py

from fastapi import APIRouter
from ai_services_api.services.search.app.endpoints import search

api_router = APIRouter()

# Include the conversation router
api_router.include_router(
    search.router,
    prefix="/search",  # Prefix for conversation endpoints
    tags=["search"]  # Tag for documentation
)
