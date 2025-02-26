from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from ai_services_api.services.message.models.expert import Base

class Message(Base):
    __tablename__ = "expert_messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("experts_expert.id"))
    receiver_id = Column(Integer, ForeignKey("experts_expert.id"))
    subject = Column(String(255))
    content = Column(Text)
    draft = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
