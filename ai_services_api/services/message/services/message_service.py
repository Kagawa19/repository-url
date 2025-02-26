from sqlalchemy.orm import Session
from ai_services_api.services.message.models.message import Message
from ai_services_api.services.message.models.expert import Expert
from ai_services_api.services.message.schemas.message import MessageCreate, MessageUpdate
from ai_services_api.services.message.services.message_generator import MessageGenerator
from typing import List, Optional

class MessageService:
    def __init__(self, db: Session):
        self.db = db
        self.message_generator = MessageGenerator()

    async def create_draft(
        self,
        sender: Expert,
        receiver: Expert,
        context: str,
        style: Optional[str] = "professional"
    ) -> Message:
        """Create a new draft message"""
        # Generate AI draft
        draft_content = await self.message_generator.generate_draft(
            sender=sender,
            receiver=receiver,
            context=context,
            style=style
        )
        
        # Create new message
        new_message = Message(
            sender_id=sender.id,
            receiver_id=receiver.id,
            content=draft_content,
            draft=True
        )
        
        self.db.add(new_message)
        self.db.commit()
        self.db.refresh(new_message)
        
        return new_message

    async def list_messages(
        self,
        user_id: int,
        is_sender: Optional[bool] = None,
        is_draft: Optional[bool] = None,
        skip: int = 0,
        limit: int = 10
    ) -> List[Message]:
        """List messages with filters"""
        query = self.db.query(Message)
        
        if is_sender is not None:
            if is_sender:
                query = query.filter(Message.sender_id == user_id)
            else:
                query = query.filter(Message.receiver_id == user_id)
        else:
            query = query.filter(
                (Message.sender_id == user_id) | 
                (Message.receiver_id == user_id)
            )

        if is_draft is not None:
            query = query.filter(Message.draft == is_draft)
        
        return query.offset(skip).limit(limit).all()

    async def update_message(
        self,
        message_id: int,
        message_update: MessageUpdate
    ) -> Optional[Message]:
        """Update an existing message"""
        message = self.db.query(Message).filter(Message.id == message_id).first()
        
        if not message:
            return None
            
        update_data = message_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(message, field, value)
            
        self.db.commit()
        self.db.refresh(message)
        
        return message

    async def send_message(self, message_id: int) -> Optional[Message]:
        """Convert a draft to a sent message"""
        message = self.db.query(Message).filter(
            Message.id == message_id,
            Message.draft == True
        ).first()
        
        if not message:
            return None
            
        message.draft = False
        self.db.commit()
        self.db.refresh(message)
        
        return message
