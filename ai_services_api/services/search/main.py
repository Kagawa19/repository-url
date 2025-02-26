# app/services/chatbot/main.py
from fastapi import APIRouter

# Create a router for chatbot routes
search_router = APIRouter()

@search_router.get("/search")
async def search_endpoint():
    return {"message": "This is the chatbot endpoint"}

# Add more chatbot-related routes as needed