from pydantic import BaseModel
from typing import Optional, Text


class ChatRequest(BaseModel):
    
    query: str  # Only the query field remains, with message_id optional


class ChatResponse(BaseModel):
    response: Text  # No changes here