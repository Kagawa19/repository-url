from ai_services_api.services.message.clients.gemini_client import GeminiClient
from ai_services_api.services.message.schemas.expert import ExpertResponse
from typing import Optional

class MessageGenerator:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def generate_draft(
        self,
        sender: ExpertResponse,
        receiver: ExpertResponse,
        context: str,
        style: Optional[str] = "professional"
    ) -> str:
        """
        Generate a draft message based on expert profiles and context
        """
        style_guide = {
            "professional": "formal and professional tone",
            "casual": "friendly yet professional tone",
            "technical": "technical and detailed tone"
        }
        
        prompt = f"""
        Draft a {style_guide.get(style, "professional")} message from {sender.first_name} {sender.last_name} ({sender.designation}) 
        to {receiver.first_name} {receiver.last_name} ({receiver.designation}).
        
        Context about sender:
        - Theme: {sender.theme}
        - Domains: {', '.join(sender.domains or [])}
        - Fields: {', '.join(sender.fields or [])}
        
        Context about receiver:
        - Theme: {receiver.theme}
        - Domains: {', '.join(receiver.domains or [])}
        - Fields: {', '.join(receiver.fields or [])}
        
        Additional context: {context}
        
        Please draft a message that:
        1. Introduces the sender professionally
        2. References shared research interests or potential collaboration areas
        3. Clearly states the purpose of connection
        4. Suggests specific next steps
        5. Maintains appropriate tone throughout
        6. Includes a professional closing
        
        The message should be concise yet comprehensive.
        """
        
        return await self.gemini_client.generate_content(prompt)

    async def generate_follow_up(
        self,
        original_message: str,
        response: str,
        context: str
    ) -> str:
        """
        Generate a follow-up message based on previous conversation
        """
        prompt = f"""
        Generate a follow-up message based on this conversation:
        
        Original Message:
        {original_message}
        
        Their Response:
        {response}
        
        Additional Context for Follow-up:
        {context}
        
        Please create a response that:
        1. Acknowledges their previous message
        2. Addresses any questions or points raised
        3. Moves the conversation forward
        4. Maintains professional tone
        5. Includes clear next steps
        """
        
        return await self.gemini_client.generate_content(prompt)
