import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GeminiPredictor:
    """Prediction service for search suggestions using Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini API client."""
        # Get API key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Set up the model configuration
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.0,  # Use deterministic responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 100,
            }
            # Remove the tools parameter that was causing errors
        )
        
        logger.info("GeminiPredictor initialized successfully")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def predict(self, partial_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get search suggestions for partial query using Gemini API.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query or len(partial_query) < 2:
            # For very short queries, return empty list
            return []
            
        try:
            # Create prompt for Gemini
            prompt = f"""
            You are a search suggestion system for an expert search platform.
            Provide {limit} search suggestions for the partial query: "{partial_query}"
            
            Return ONLY a list of search suggestions that complete or expand the partial query.
            Format your response as a numbered list, one suggestion per line.
            Do not include any explanations or additional text.
            
            The suggestions should be relevant to searching for experts, research, or academic topics.
            """
            
            # Make the API call to Gemini without Google Search grounding
            response = await self.model.generate_content_async(prompt)
            
            # Extract suggestions from the response
            suggestions = []
            
            if hasattr(response, "text"):
                # Try to extract suggestions from text
                text = response.text.strip()
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines[:limit]:
                    # Remove numbering if present
                    if '. ' in line and line[0].isdigit():
                        line = line.split('. ', 1)[1]
                        
                    suggestions.append({
                        "text": line,
                        "source": "gemini",
                        "score": 0.9
                    })
            
            # Ensure we have the requested number of suggestions if possible
            # If Gemini couldn't provide enough, create some basic ones based on the query
            if len(suggestions) < limit and len(partial_query) >= 2:
                common_extensions = [
                    " meaning", " definition", " examples", 
                    " research", " experts", " studies"
                ]
                
                for ext in common_extensions:
                    if len(suggestions) >= limit:
                        break
                        
                    suggestion_text = f"{partial_query}{ext}"
                    # Check if this suggestion is already in the list
                    if not any(s["text"].lower() == suggestion_text.lower() for s in suggestions):
                        suggestions.append({
                            "text": suggestion_text,
                            "source": "fallback",
                            "score": 0.5
                        })
            
            logger.info(f"Generated {len(suggestions)} suggestions for '{partial_query}'")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions from Gemini: {e}")
            # Return empty list on error
            return []
            
    def generate_confidence_scores(self, suggestions: List[Dict[str, Any]]) -> List[float]:
        """
        Generate confidence scores for suggestions.
        
        Args:
            suggestions: List of suggestion dictionaries
            
        Returns:
            List of confidence scores
        """
        # Use provided scores if available, otherwise assign scores based on position
        return [s.get("score", max(0.1, 1.0 - (i * 0.1))) for i, s in enumerate(suggestions)]