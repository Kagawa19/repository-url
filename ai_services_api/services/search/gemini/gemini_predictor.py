import os
import google.generativeai as genai
import logging
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleAutocompletePredictor:
    """
    Intelligent prediction service for APHRC search suggestions using Gemini API and FAISS index.
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
            # Configure Gemini API
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.logger = logging.getLogger(__name__)
            
            # Load FAISS index and mapping
            self._load_faiss_index()
            
            self.logger.info("GoogleAutocompletePredictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    def _load_faiss_index(self):
        """Load the FAISS index and expert mapping."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, 'models')
            
            # Paths to index and mapping files
            self.index_path = os.path.join(models_dir, 'expert_faiss_index.idx')
            self.mapping_path = os.path.join(models_dir, 'expert_mapping.pkl')
            
            if not os.path.exists(self.index_path) or not os.path.exists(self.mapping_path):
                self.logger.warning("FAISS index files not found, falling back to Gemini-only mode")
                self.index = None
                self.id_mapping = None
                return
                
            # Load index and mapping
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, 'rb') as f:
                self.id_mapping = pickle.load(f)
                
            self.logger.info("Successfully loaded FAISS index and expert mapping")
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            self.index = None
            self.id_mapping = None
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text (str): Input text to normalize
        
        Returns:
            str: Normalized text
        """
        return str(text).lower().strip()
    
    def _generate_intelligent_suggestions(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate intelligent search suggestions using FAISS index and Gemini API.
        
        Args:
            partial_query (str): Partial search query
            limit (int): Maximum number of suggestions to return
        
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        seen_suggestions = set()
        
        # First try to get suggestions from FAISS index if available
        if self.index and self.id_mapping:
            try:
                # Generate embedding for the partial query using Gemini
                query_embedding = self._get_gemini_embedding(partial_query)
                
                if query_embedding is not None:
                    # Search the FAISS index
                    distances, indices = self.index.search(
                        np.array([query_embedding]).astype(np.float32), 
                        min(limit * 2, self.index.ntotal)
                    )
                    
                    # Process results
                    for i, idx in enumerate(indices[0]):
                        if idx < 0:  # Skip invalid indices
                            continue
                            
                        expert_id = self.id_mapping.get(idx)
                        if expert_id:
                            # Create suggestion based on expert ID
                            suggestion = {
                                "text": f"Expert ID: {expert_id}",
                                "source": "faiss_index",
                                "score": float(1.0 / (1.0 + distances[0][i]))
                            }
                            if suggestion['text'] not in seen_suggestions:
                                suggestions.append(suggestion)
                                seen_suggestions.add(suggestion['text'])
            except Exception as e:
                self.logger.error(f"Error generating suggestions from FAISS index: {e}")
        
        # If we don't have enough suggestions, use Gemini API
        if len(suggestions) < limit:
            try:
                gemini_suggestions = self._generate_gemini_suggestions(partial_query, limit - len(suggestions))
                for suggestion in gemini_suggestions:
                    if suggestion['text'] not in seen_suggestions:
                        suggestions.append(suggestion)
                        seen_suggestions.add(suggestion['text'])
            except Exception as e:
                self.logger.error(f"Error generating Gemini suggestions: {e}")
        
        # Sort suggestions by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return suggestions[:limit]
    
    def _get_gemini_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using Gemini API.
        
        Args:
            text (str): Text to embed
            
        Returns:
            Optional embedding vector
        """
        try:
            # Use Gemini to generate embedding
            response = self.model.embed_content(text)
            if response and hasattr(response, 'embedding'):
                return response.embedding
            return None
        except Exception as e:
            self.logger.error(f"Error getting Gemini embedding: {e}")
            return None
    
    async def predict(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate search suggestions.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            return []
        
        try:
            # Generate intelligent suggestions
            suggestions = self._generate_intelligent_suggestions(partial_query, limit)
            
            # Fallback if no suggestions
            if not suggestions:
                suggestions = self._generate_fallback_suggestions(partial_query, limit)
            
            return suggestions
        
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return self._generate_fallback_suggestions(partial_query, limit)
    
    async def _generate_gemini_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate suggestions using Gemini API.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
        
        Returns:
            List of suggestion dictionaries
        """
        try:
            # Comprehensive suggestion prompt
            prompt = f"""Generate {limit} intelligent search suggestions for a research organization. 
            The query is: "{partial_query}"
            Provide suggestions that help narrow down the search, focusing on:
            - Research domains
            - Relevant expertise areas
            - Common search patterns
            Ensure suggestions are meaningful and helpful."""
            
            # Generate suggestions
            response = await self.model.generate_content_async(prompt)
            
            # Parse the suggestions
            suggestions = []
            if response.text:
                # Split and clean suggestions
                raw_suggestions = [
                    line.strip() 
                    for line in response.text.split('\n') 
                    if line.strip()
                ]
                
                # Convert to suggestion objects
                for idx, suggestion_text in enumerate(raw_suggestions[:limit]):
                    suggestions.append({
                        "text": suggestion_text,
                        "source": "gemini_api",
                        "score": 1.0 - (idx * 0.05)  # Slightly decrease score for later suggestions
                    })
            
            return suggestions
        
        except Exception as e:
            self.logger.error(f"Gemini suggestion generation failed: {e}")
            return []
    
    def _generate_fallback_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate basic fallback suggestions.
        
        Args:
            partial_query: The partial query
            limit: Maximum number of suggestions
        
        Returns:
            List of suggestion dictionaries
        """
        suggestions = [
            {
                "text": f"Search for {partial_query}",
                "source": "fallback",
                "score": 0.5
            }
        ]
        return suggestions[:limit]
    
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