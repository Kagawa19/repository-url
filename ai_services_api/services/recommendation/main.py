# app/services/chatbot/main.py
from fastapi import APIRouter

# Create a router for chatbot routes
recommendation_router = APIRouter()

@recommendation_router.get("/recommendation")
async def recommendation_endpoint():
    return {"message": "This is the chatbot endpoint"}

# Add more chatbot-related routes as needed