import requests
import json
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GoogleAutocompletePredictor:
    """Prediction service for search suggestions using Google's Autocomplete API."""
    
    def __init__(self):
        """Initialize the Google Autocomplete predictor."""
        self.url = "https://suggestqueries.google.com/complete/search"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
            )
        }
        logger.info("GoogleAutocompletePredictor initialized successfully")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def predict(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get search suggestions for partial query using Google's Autocomplete API.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            # For empty queries, return empty list
            return []
            
        try:
            # Define the parameters for the GET request
            params = {
                "client": "firefox",  # Client type; can be 'firefox' or 'chrome'
                "q": partial_query,   # The search query keyword
                "hl": "en",           # Language code for the suggestions
                "gl": "us",           # Country code to influence regional suggestions
                "ie": "UTF-8",        # Input encoding
                "oe": "UTF-8",        # Output encoding
                "num": limit          # Number of suggestions to retrieve
            }
            
            # Send the GET request to Google's Autocomplete API
            response = requests.get(self.url, params=params, headers=self.headers)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response and extract the suggestions
                raw_suggestions = json.loads(response.text)[1]
                
                # Format the suggestions into dictionaries
                suggestions = []
                for idx, suggestion in enumerate(raw_suggestions):
                    suggestions.append({
                        "text": suggestion,
                        "source": "google_autocomplete",
                        "score": 1.0 - (idx * 0.05)  # Higher score for earlier suggestions
                    })
                    
                logger.info(f"Generated {len(suggestions)} suggestions for '{partial_query}'")
                return suggestions
            else:
                # Handle error response
                logger.error(f"Google Autocomplete API error: {response.status_code}")
                return self._generate_fallback_suggestions(partial_query, limit)
                
        except Exception as e:
            logger.error(f"Error getting suggestions from Google Autocomplete: {e}")
            # Return fallback suggestions on error
            return self._generate_fallback_suggestions(partial_query, limit)
    
    def _generate_fallback_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate fallback suggestions when the API call fails."""
        suggestions = []
        
        if not partial_query:
            return suggestions
            
        # Create some basic fallback suggestions based on the query
        common_extensions = [
            "",              # Just the query itself
            " expert",
            " research",
            " meaning",
            " studies",
            " specialist",
            " definition",
            " examples"
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