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
            },
            tools=[{
                "google_search_retrieval": {}  # Enable Google Search Retrieval
            }]
        )
        
        logger.info("GeminiPredictor initialized successfully")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def predict(self, partial_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get search suggestions for partial query using Gemini API with Google Search.
        
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
            Provide search suggestions for the partial query: "{partial_query}"
            
            Return only relevant search suggestions that start with or closely relate to this partial query.
            Do not include any explanations or additional text.
            
            The response should include exactly {limit} suggestions if possible.
            """
            
            # Make the API call to Gemini with Google Search grounding
            response = await self.model.generate_content_async(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Extract suggestions from the response
            suggestions = []
            
            if hasattr(response, "candidates") and response.candidates:
                # Extract web search queries if available
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        content = candidate.content
                        
                        # Try to extract from grounding metadata first (preferred)
                        if hasattr(content, "parts") and content.parts:
                            for part in content.parts:
                                if hasattr(part, "text") and part.text:
                                    try:
                                        # Parse JSON if possible
                                        suggestions_data = json.loads(part.text)
                                        if isinstance(suggestions_data, list):
                                            suggestions = suggestions_data[:limit]
                                            break
                                    except:
                                        # Not JSON, try to extract manually
                                        pass
                        
                        # Try to extract from web search queries
                        if hasattr(content, "grounding_metadata") and content.grounding_metadata:
                            web_queries = content.grounding_metadata.get("web_search_queries", [])
                            if web_queries:
                                for query in web_queries[:limit]:
                                    suggestions.append({
                                        "text": query,
                                        "source": "gemini_web_search",
                                        "score": 1.0
                                    })
            
            # If no suggestions found but we have text, try to parse it
            if not suggestions and hasattr(response, "text"):
                try:
                    # Try to extract suggestions from text
                    text = response.text.strip()
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    
                    for line in lines[:limit]:
                        # Remove numbering if present
                        if '. ' in line and line[0].isdigit():
                            line = line.split('. ', 1)[1]
                            
                        suggestions.append({
                            "text": line,
                            "source": "gemini_text",
                            "score": 0.9
                        })
                except Exception as e:
                    logger.error(f"Error parsing Gemini text response: {e}")
            
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