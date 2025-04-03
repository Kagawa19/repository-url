import os
import google.generativeai as genai
import logging
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

class GoogleAutocompletePredictor:
    """
    Optimized prediction service for search suggestions using Gemini API and FAISS index.
    Provides a Google-like autocomplete experience with minimal latency.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Autocomplete predictor with optimizations for speed.
        
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
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.logger = logging.getLogger(__name__)
            
            # Load FAISS index and mapping
            self._load_faiss_index()
            
            # Initialize cache for suggestions to improve speed
            self.suggestion_cache = {}
            self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
            self.cache_timestamp = {}
            
            # Create a thread pool for parallel processing
            self.executor = ThreadPoolExecutor(max_workers=4)
            
            self.logger.info("GoogleAutocompletePredictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    def _load_faiss_index(self):
        """Load the FAISS index and expert mapping with better error handling."""
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
                
            self.logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            self.index = None
            self.id_mapping = None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison and caching."""
        return str(text).lower().strip()
    
    def _check_cache(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Check if suggestions for this query are in cache and not expired."""
        normalized_query = self._normalize_text(query)
        
        # Check for exact match in cache
        if normalized_query in self.suggestion_cache:
            # Check if cache has expired
            timestamp = self.cache_timestamp.get(normalized_query, 0)
            if time.time() - timestamp < self.cache_expiry:
                self.logger.debug(f"Cache hit for query: {normalized_query}")
                return self.suggestion_cache[normalized_query]
            else:
                # Cache expired, remove it
                del self.suggestion_cache[normalized_query]
                del self.cache_timestamp[normalized_query]
        
        # Check for prefix matches for partial queries
        for cached_query in list(self.suggestion_cache.keys()):
            if normalized_query.startswith(cached_query) and len(normalized_query) - len(cached_query) <= 3:
                # Close enough prefix match
                timestamp = self.cache_timestamp.get(cached_query, 0)
                if time.time() - timestamp < self.cache_expiry:
                    self.logger.debug(f"Prefix cache hit for query: {normalized_query} matched {cached_query}")
                    return self.suggestion_cache[cached_query]
        
        return None
    
    def _update_cache(self, query: str, suggestions: List[Dict[str, Any]]):
        """Update cache with new suggestions."""
        normalized_query = self._normalize_text(query)
        self.suggestion_cache[normalized_query] = suggestions
        self.cache_timestamp[normalized_query] = time.time()
        
        # Cleanup old cache entries
        current_time = time.time()
        expired_keys = [k for k, t in self.cache_timestamp.items() if current_time - t > self.cache_expiry]
        for key in expired_keys:
            if key in self.suggestion_cache:
                del self.suggestion_cache[key]
            if key in self.cache_timestamp:
                del self.cache_timestamp[key]
    
    async def _get_gemini_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Gemini API with timeout and retry logic."""
        try:
            # Use Gemini to generate embedding
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = await asyncio.wait_for(
                        self.model.embed_content_async(text),
                        timeout=2.0  # 2 second timeout
                    )
                    if response and hasattr(response, 'embedding'):
                        return response.embedding
                    return None
                except asyncio.TimeoutError:
                    self.logger.warning(f"Embedding request timed out, attempt {attempt+1}/3")
                    if attempt == 2:  # Last attempt
                        return None
                    await asyncio.sleep(0.5)  # Small delay before retry
        except Exception as e:
            self.logger.error(f"Error getting Gemini embedding: {e}")
            return None
    
    async def _generate_faiss_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate suggestions from FAISS index only - optimized for speed."""
        if not self.index or not self.id_mapping:
            return []
            
        try:
            # Generate embedding for the partial query
            query_embedding = await self._get_gemini_embedding(partial_query)
            
            if query_embedding is None:
                return []
                
            # Search the FAISS index efficiently
            distances, indices = self.index.search(
                np.array([query_embedding]).astype(np.float32), 
                min(limit * 2, self.index.ntotal)
            )
            
            # Process results in parallel for speed
            suggestions = []
            seen_texts = set()
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # Skip invalid indices
                    continue
                    
                expert_id = self.id_mapping.get(idx)
                if not expert_id:
                    continue
                    
                # Create simplified Google-like suggestion
                suggestion_text = f"{partial_query} {expert_id}".strip()
                
                if suggestion_text not in seen_texts:
                    suggestions.append({
                        "text": suggestion_text,
                        "source": "faiss_index",
                        "score": float(1.0 / (1.0 + distances[0][i]))
                    })
                    seen_texts.add(suggestion_text)
                    
                    if len(suggestions) >= limit:
                        break
            
            return suggestions
        except Exception as e:
            self.logger.error(f"Error generating FAISS suggestions: {e}")
            return []
    
    async def _generate_simplified_gemini_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate simplified Gemini suggestions with timeout.
        
        This is a replacement for the previous lengthy hierarchical suggestions with arrows.
        Now it generates simple, one-line, Google-like suggestions.
        """
        try:
            # Simple and direct prompt focused on speed
            prompt = f"""Generate {limit} single-line search suggestions for "{partial_query}" in a research context.
            
            IMPORTANT:
            - Each suggestion must be on its own line
            - No numbering, no arrows, no bullet points
            - Keep each suggestion under 10 words
            - Include relevant research terms
            - No explanations, just the suggestions
            - Prioritize common completions
            
            Suggestions:"""
            
            # Generate with timeout
            try:
                response = await asyncio.wait_for(
                    self.model.generate_content_async(prompt),
                    timeout=3.0  # 3 second timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Gemini suggestion request timed out")
                return []
            
            # Parse simple suggestions
            suggestions = []
            if response and hasattr(response, 'text'):
                # Split into lines and clean
                lines = [line.strip() for line in response.text.split('\n') if line.strip()]
                
                for i, line in enumerate(lines):
                    if i >= limit:
                        break
                    
                    # Remove any numbers, bullets or arrows that might have been added
                    clean_line = line
                    for prefix in ['•', '-', '→', '*']:
                        if clean_line.startswith(prefix):
                            clean_line = clean_line[1:].strip()
                    
                    # Remove numbering like "1. ", "2. "
                    if len(clean_line) > 3 and clean_line[0].isdigit() and clean_line[1:3] in ['. ', ') ']:
                        clean_line = clean_line[3:].strip()
                    
                    suggestions.append({
                        "text": clean_line,
                        "source": "gemini_simplified",
                        "score": 0.85 - (i * 0.03)
                    })
                
            return suggestions
        except Exception as e:
            self.logger.error(f"Simplified Gemini suggestion generation failed: {e}")
            return []
    
    async def predict(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate search suggestions with Google-like experience.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            return []
        
        start_time = time.time()
        normalized_query = self._normalize_text(partial_query)
        
        try:
            # First check cache
            cached_suggestions = self._check_cache(normalized_query)
            if cached_suggestions:
                self.logger.info(f"Returning cached suggestions for '{normalized_query}' ({time.time() - start_time:.3f}s)")
                return cached_suggestions[:limit]
            
            # Generate suggestions from FAISS
            suggestions = await self._generate_faiss_suggestions(normalized_query, limit)
            
            # If we don't have enough from FAISS, try simplified Gemini suggestions
            if len(suggestions) < limit * 0.7:  # If we have less than 70% of requested suggestions
                gemini_suggestion_count = min(limit - len(suggestions), 5)  # Get at most 5 from Gemini
                if gemini_suggestion_count > 0:
                    gemini_suggestions = await self._generate_simplified_gemini_suggestions(normalized_query, gemini_suggestion_count)
                    
                    # Add non-duplicate suggestions
                    existing_texts = {s["text"] for s in suggestions}
                    for suggestion in gemini_suggestions:
                        if suggestion["text"] not in existing_texts:
                            suggestions.append(suggestion)
                            existing_texts.add(suggestion["text"])
            
            # Fallback to simple pattern completion if we still don't have enough
            if not suggestions:
                suggestions = self._generate_pattern_suggestions(normalized_query, limit)
            
            # Update cache with new results
            self._update_cache(normalized_query, suggestions)
            
            # Sort by score for best results first
            suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            self.logger.info(f"Generated {len(suggestions)} suggestions for '{normalized_query}' ({time.time() - start_time:.3f}s)")
            return suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error predicting suggestions: {e}")
            # Ensure we always return something
            return self._generate_pattern_suggestions(normalized_query, limit)
    
    def _generate_pattern_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate simple pattern-based suggestions when other methods fail.
        This is a fast fallback that doesn't require API calls.
        """
        # Common patterns to complete search queries - customize based on your domain
        common_completions = [
            "", 
            " research",
            " methods",
            " framework",
            " meaning",
            " definition",
            " examples",
            " analysis",
            " tools",
            " techniques",
            " case studies",
            " best practices",
            " guidelines",
            " theory",
            " applications"
        ]
        
        suggestions = []
        for i, completion in enumerate(common_completions):
            if len(suggestions) >= limit:
                break
                
            suggestion_text = f"{partial_query}{completion}".strip()
            suggestions.append({
                "text": suggestion_text,
                "source": "pattern",
                "score": 0.9 - (i * 0.05)
            })
            
        return suggestions
    
    def generate_confidence_scores(self, suggestions: List[Dict[str, Any]]) -> List[float]:
        """Generate confidence scores for suggestions."""
        return [s.get("score", max(0.1, 1.0 - (i * 0.05))) for i, s in enumerate(suggestions)]