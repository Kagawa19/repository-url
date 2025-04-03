import os
import google.generativeai as genai
import logging
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleAutocompletePredictor:
    """
    Prediction service for APHRC-specific search suggestions using Google's Gemini API and FAISS index.
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
            
            # Load FAISS index and mapping
            current_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(current_dir, 'models', 'expert_faiss_index.idx')
            mapping_path = os.path.join(current_dir, 'models', 'expert_mapping.pkl')
            
            try:
                self.index = faiss.read_index(index_path)
                with open(mapping_path, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                self.logger.info(f"Loaded FAISS index with {self.index.ntotal} entries")
            except Exception as index_err:
                self.logger.warning(f"Could not load FAISS index: {index_err}")
                self.index = None
                self.id_mapping = None
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {e}")
            raise
    
    def _generate_embedding(self, text: str, dimension: int = 384) -> np.ndarray:
        """
        Generate a deterministic embedding for the given text.
        
        Args:
            text (str): Text to embed
            dimension (int): Embedding dimension
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Simple deterministic embedding generation
        import hashlib
        
        # Create a deterministic seed from the text
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.sha256(text_bytes)
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        # Use numpy's random with the seed for deterministic output
        np.random.seed(seed)
        
        # Generate embedding
        embedding = np.zeros(dimension, dtype=np.float32)
        
        # Split text into words and process each word
        words = text.lower().split()
        for word in words:
            word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            word_seed = word_hash % (2**32)
            np.random.seed(word_seed)
            
            # Make it sparse - only activate ~5% of dimensions
            active_dims = np.random.choice(
                dimension, 
                size=max(1, int(dimension * 0.05)), 
                replace=False
            )
            
            # Set values for active dimensions
            for dim in active_dims:
                embedding[dim] = (np.random.random() * 2) - 1
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else embedding
    
    async def predict(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate APHRC-specific search suggestions.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            return []
        
        try:
            # Try FAISS index search if available
            if self.index is not None:
                try:
                    # Generate embedding for the partial query
                    query_embedding = self._generate_embedding(partial_query)
                    
                    # Search FAISS index - get more candidates to filter
                    max_candidates = min(limit * 3, self.index.ntotal)
                    distances, indices = self.index.search(
                        query_embedding.reshape(1, -1).astype(np.float32), 
                        max_candidates
                    )
                    
                    # Collect suggestions from FAISS search
                    suggestions = []
                    seen_suggestions = set()
                    
                    for i, idx in enumerate(indices[0]):
                        if idx < 0:  # Skip invalid indices
                            continue
                        
                        try:
                            # Distance-based scoring
                            distance = float(distances[0][i])
                            base_score = float(1.0 / (1.0 + distance))
                            
                            # Generate FAISS-based suggestions
                            suggestion_texts = [
                                f"APHRC Expert in {partial_query}",
                                f"Research by {partial_query} Expert",
                                f"Specialist in {partial_query}"
                            ]
                            
                            for j, text in enumerate(suggestion_texts):
                                if text not in seen_suggestions:
                                    suggestions.append({
                                        "text": text,
                                        "source": "faiss_index",
                                        "score": base_score - (j * 0.05)
                                    })
                                    seen_suggestions.add(text)
                                
                                # Break if we've reached the limit
                                if len(suggestions) >= limit:
                                    break
                            
                            if len(suggestions) >= limit:
                                break
                        
                        except Exception as e:
                            self.logger.error(f"Error processing FAISS suggestion: {e}")
                    
                    # If we have suggestions from FAISS, return them
                    if suggestions:
                        return suggestions
                
                except Exception as faiss_err:
                    self.logger.warning(f"FAISS index search failed: {faiss_err}")
            
            # Fallback to original Gemini-based suggestion generation
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
            
            return suggestions
        
        except Exception as e:
            self.logger.error(f"Error generating APHRC suggestions: {e}")
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