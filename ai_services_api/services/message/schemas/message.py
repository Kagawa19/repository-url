# app/schemas/message.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class MessageBase(BaseModel):
    sender_id: int
    receiver_id: int
    content: str
    subject: Optional[str] = None

class MessageCreate(MessageBase):
    pass

class MessageUpdate(BaseModel):
    content: Optional[str] = None
    subject: Optional[str] = None
    draft: Optional[bool] = None

class MessageInDB(MessageBase):
    id: int
    draft: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MessageResponse(MessageInDB):
    sender_name: Optional[str] = None
    receiver_name: Optional[str] = None

    class Config:
        from_attributes = True

class MessageThread(BaseModel):
    messages: list[MessageResponse]
    total_count: int
    unread_count: int

    class Config:
        from_attributes = True

class MessageWithExpertInfo(MessageResponse):
    sender_designation: Optional[str] = None
    receiver_designation: Optional[str] = None
    sender_theme: Optional[str] = None
    receiver_theme: Optional[str] = None
