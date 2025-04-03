import os
import google.generativeai as genai
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleAutocompletePredictor:
    """
    Prediction service for search suggestions using Google's Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini Autocomplete predictor.
        
        Args:
            api_key: Optional API key. If not provided, will attempt to read from environment.
        """
        # Prioritize passed api_key, then environment variable
        key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not key:
            raise ValueError(
                "No Gemini API key provided. "
                "Please pass the key directly or set GEMINI_API_KEY in your .env file."
            )
        
        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger = logging.getLogger(__name__)
            logger.info("GeminiAutocompletePredictor initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini API: {e}")
            raise
    
    async def predict(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate search suggestions for a partial query using Gemini.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            return []
        
        try:
            # Prompt to generate search suggestions
            # Prompt to generate search suggestions related to APHRC
            prompt = f"""Generate {limit} search query suggestions that are specifically related to the African Population and Health Research Center (APHRC) and expand on or relate to the following partial query:
            "{partial_query}"

            Focus on suggestions that pertain to APHRC's research areas, projects, publications, events, or other relevant aspects of the organization. Each suggestion should be distinct, relevant, and formatted as a numbered list."""

            
            # Generate suggestions
            response = self.model.generate_content(prompt)
            
            # Parse the suggestions
            suggestions = []
            if response.text:
                # Split the response into lines and clean up
                raw_suggestions = [
                    line.strip() 
                    for line in response.text.split('\n') 
                    if line.strip() and not line.strip().startswith('Suggestion')
                ]
                
                # Format suggestions
                for idx, suggestion in enumerate(raw_suggestions[:limit]):
                    suggestions.append({
                        "text": suggestion,
                        "source": "gemini_api",
                        "score": 1.0 - (idx * 0.05)  # Scoring similar to previous implementation
                    })
            
            # Fallback if no suggestions generated
            if not suggestions:
                suggestions = self._generate_fallback_suggestions(partial_query, limit)
            
            return suggestions
        
        except Exception as e:
            logging.error(f"Error generating suggestions with Gemini API: {e}")
            return self._generate_fallback_suggestions(partial_query, limit)
    
    def _generate_fallback_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate fallback suggestions when the API call fails."""
        suggestions = []
        
        if not partial_query:
            return suggestions
        
        # Create some basic fallback suggestions based on the query
        common_extensions = [
            "",              # Just the query itself
            " overview",
            " explained",
            " guide",
            " information",
            " key points",
            " summary",
            " details"
        ]
        
        for i, ext in enumerate(common_extensions):
            if len(suggestions) >= limit:
                break
                
            suggestion_text = f"{partial_query}{ext}".strip()
            if suggestion_text:  # Ensure non-empty suggestion
                suggestions.append({
                    "text": suggestion_text,
                    "source": "fallback",
                    "score": 0.8 - (i * 0.1)  # Decreasing score for later suggestions
                })
        
        return suggestions
    
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

# Example usage
# from gemini_predictor import GeminiAutocompletePredictor
# predictor = GeminiAutocompletePredictor()  # Will read from .env
# suggestions = await predictor.predict('your query')