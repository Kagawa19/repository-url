# app/services/chatbot/main.py
from fastapi import APIRouter

# Create a router for chatbot routes
message_router = APIRouter()

@message_router.get("/message")
async def message_endpoint():
    return {"message": "This is the chatbot endpoint"}

# Add more chatbot-related routes as needed
