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
            # Try multiple possible paths where models might be located
            possible_paths = [
                # Path in Docker container (mounted volume)
                '/app/ai_services_api/services/search/models',
                # Path in local development
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
                # Alternative path from environment variable
                os.getenv('MODEL_PATH', '/app/models/search'),
                # Absolute path as fallback
                '/app/models'
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'expert_faiss_index.idx')):
                    found_path = path
                    break
            
            if not found_path:
                self.logger.warning("FAISS index files not found in any searched locations")
                self.logger.warning(f"Searched paths: {possible_paths}")
                self.index = None
                self.id_mapping = None
                return
                
            self.logger.info(f"Found models at: {found_path}")
            
            # Paths to index and mapping files
            self.index_path = os.path.join(found_path, 'expert_faiss_index.idx')
            self.mapping_path = os.path.join(found_path, 'expert_mapping.pkl')
            
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
    
    async def _generate_intelligent_suggestions(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
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
                query_embedding = await self._get_gemini_embedding(partial_query)
                
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
                gemini_suggestions = await self._generate_gemini_suggestions(partial_query, limit - len(suggestions))
                for suggestion in gemini_suggestions:
                    if suggestion['text'] not in seen_suggestions:
                        suggestions.append(suggestion)
                        seen_suggestions.add(suggestion['text'])
            except Exception as e:
                self.logger.error(f"Error generating Gemini suggestions: {e}")
        
        # Sort suggestions by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return suggestions[:limit]
    
    async def _get_gemini_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using Gemini API.
        
        Args:
            text (str): Text to embed
            
        Returns:
            Optional embedding vector
        """
        try:
            # Use Gemini to generate embedding
            response = await self.model.embed_content_async(text)
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
            suggestions = await self._generate_intelligent_suggestions(partial_query, limit)
            
            # Fallback if no suggestions
            if not suggestions:
                suggestions = self._generate_fallback_suggestions(partial_query, limit)
            
            return suggestions
        
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return self._generate_fallback_suggestions(partial_query, limit)
    
    async def _generate_gemini_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate structured search suggestions similar to Google's autocomplete.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
        
        Returns:
            List of suggestion dictionaries with rich formatting
        """
        try:
            # Enhanced prompt for structured suggestions
            prompt = f"""Generate {limit} intelligent search suggestions for a research organization's search system. 
            The partial query is: "{partial_query}"
            
            Provide suggestions in this exact format (no numbering, no bullet points):
            
            [Main suggestion topic]
            → [More specific subtopic or detail]
            → [Another relevant subtopic]
            
            [Different main topic]
            → [Specific aspect]
            
            Suggestions should:
            1. Start with the most common/relevant completions of the partial query
            2. Include specialized research areas with proper terminology
            3. Show hierarchical relationships with arrows (→)
            4. Cover diverse aspects (methods, populations, applications)
            5. Use concise but informative phrasing
            6. Never number the suggestions
            
            Example for "cancer":
            Cancer research
            → Genomics and personalized medicine
            → Immunotherapy clinical trials
            Cancer stem cells
            → Drug discovery approaches
            Epigenetic modifications
            → Cancer prevention mechanisms
            Metastatic cancer
            → Advanced imaging techniques
            
            Now generate suggestions for "{partial_query}":"""
            
            # Generate suggestions
            response = await self.model.generate_content_async(prompt)
            
            # Parse the suggestions
            suggestions = []
            if response.text:
                # Split into lines and clean
                raw_lines = [line.strip() for line in response.text.split('\n') if line.strip()]
                
                current_suggestion = ""
                for line in raw_lines:
                    if not line.startswith('→'):  # Main topic
                        if current_suggestion:  # Save previous suggestion
                            suggestions.append({
                                "text": current_suggestion,
                                "source": "gemini_api",
                                "score": 1.0 - (len(suggestions) * 0.03)  # Gradual score decrease
                            })
                        current_suggestion = line
                    else:  # Subtopic
                        current_suggestion += f"\n{line}"
                
                # Add the last suggestion
                if current_suggestion and len(suggestions) < limit:
                    suggestions.append({
                        "text": current_suggestion,
                        "source": "gemini_api",
                        "score": 1.0 - (len(suggestions) * 0.03
                    })
            
            return suggestions[:limit]
        
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