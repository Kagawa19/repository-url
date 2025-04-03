import os
import google.generativeai as genai
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleAutocompletePredictor:
    """
    Prediction service for APHRC-specific search suggestions using Google's Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the APHRC Autocomplete predictor.
        
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
            self.logger = logging.getLogger(__name__)
            self.logger.info("GoogleAutocompletePredictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {e}")
            raise
    
    async def predict(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate APHRC-specific search suggestions for a partial query using Gemini.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            return []
        
        try:
            # Comprehensive APHRC-specific prompt
            prompt = f"""You are an expert at generating search suggestions specifically for the African Population and Health Research Center (APHRC). 

Create {limit} highly targeted search query suggestions that are exclusively related to APHRC. These suggestions must:
- Strictly focus on APHRC's work, research, publications, or initiatives
- Incorporate the partial query: "{partial_query}"
- Ensure each suggestion is unique and directly relevant to APHRC
- Include specific research areas, projects, or publications

Examples of APHRC domains include:
- Population health research
- Urban health
- Maternal and child health
- Education research
- Demographic and health surveys
- Policy research in African contexts

Suggestions should follow this format:
"APHRC [specific research/project/publication domain]"

Provide suggestions that a researcher or professional interested in African population and health research would actually search for."""
            
            # Generate suggestions
            response = self.model.generate_content(prompt)
            
            # Parse the suggestions
            suggestions = []
            if response.text:
                # Split the response into lines and clean up
                raw_suggestions = [
                    line.strip() 
                    for line in response.text.split('\n') 
                    if line.strip() and 
                    not line.strip().startswith(('Suggestion', '•', '-')) and 
                    'APHRC' in line.strip()
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
            
            self.logger.info(f"Generated {len(suggestions)} APHRC-specific suggestions for '{partial_query}'")
            return suggestions
        
        except Exception as e:
            self.logger.error(f"Error generating APHRC suggestions with Gemini API: {e}")
            return self._generate_fallback_suggestions(partial_query, limit)
    
    def _generate_fallback_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate fallback APHRC-specific suggestions when the API call fails."""
        suggestions = []
        
        if not partial_query:
            return suggestions
        
        # APHRC-specific fallback suggestions
        aphrc_extensions = [
            " research projects",
            " population health studies",
            " urban health research",
            " maternal health initiatives",
            " demographic surveys",
            " policy research",
            " publications"
        ]
        
        for i, ext in enumerate(aphrc_extensions):
            if len(suggestions) >= limit:
                break
                
            suggestion_text = f"APHRC {partial_query}{ext}".strip()
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