import google.generativeai as genai
from ai_services_api.services.message.core.config import get_settings
import asyncio
from typing import Optional, Dict, Any

class GeminiClient:
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

    async def generate_content(
        self, 
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 40,
        max_output_tokens: int = 1024,
    ) -> str:
        """
        Generate content using Gemini AI with specified parameters
        """
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }

            # Run in thread pool since Gemini's API is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")

    async def analyze_expertise(self, expert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze expert's expertise and suggest relevant connections
        """
        prompt = f"""
        Analyze the following expert's profile and provide insights:
        Name: {expert_data.get('first_name')} {expert_data.get('last_name')}
        Theme: {expert_data.get('theme')}
        Domains: {', '.join(expert_data.get('domains', []))}
        Fields: {', '.join(expert_data.get('fields', []))}
        
        Please provide:
        1. Key areas of expertise
        2. Potential collaboration opportunities
        3. Suggested research areas
        """
        
        analysis = await self.generate_content(prompt)
        return {"analysis": analysis}
