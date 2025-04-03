import os
import google.generativeai as genai
import logging
import numpy as np
import faiss
import pickle
import redis
import json
import time
import asyncio
from functools import lru_cache
from cachetools import TTLCache
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Implements circuit breaker pattern for external API calls.
    
    This prevents unnecessary calls to failing services and allows
    the system to recover gracefully.
    """
    
    def __init__(self, name, failure_threshold=5, recovery_timeout=30, 
                 retry_timeout=60):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the service protected by this circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            retry_timeout: Time in seconds before resetting failure count
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.retry_timeout = retry_timeout
        
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0
        self.last_attempt_time = 0
    
    async def execute(self, func, *args, **kwargs):
        """
        Execute the function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function or None if circuit is open
        """
        current_time = time.time()
        
        # Check if we should reset failure count
        if current_time - self.last_failure_time > self.retry_timeout:
            if self.failures > 0:
                logger.info(f"Circuit breaker '{self.name}': Resetting failure count")
                self.failures = 0
        
        # Check circuit state
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if current_time - self.last_failure_time > self.recovery_timeout:
                logger.info(f"Circuit breaker '{self.name}': Transitioning to HALF-OPEN")
                self.state = "HALF-OPEN"
            else:
                logger.warning(f"Circuit breaker '{self.name}': Circuit OPEN, request rejected")
                return None
        
        # Try to execute the function
        try:
            self.last_attempt_time = current_time
            result = await func(*args, **kwargs)
            
            # If we got here in HALF-OPEN, close the circuit
            if self.state == "HALF-OPEN":
                logger.info(f"Circuit breaker '{self.name}': Closing circuit after successful execution")
                self.state = "CLOSED"
                self.failures = 0
            
            return result
            
        except Exception as e:
            # Record the failure
            self.failures += 1
            self.last_failure_time = current_time
            
            # Check if we should open the circuit
            if self.failures >= self.failure_threshold:
                if self.state != "OPEN":
                    logger.warning(
                        f"Circuit breaker '{self.name}': Opening circuit after {self.failures} failures"
                    )
                    self.state = "OPEN"
            
            # Re-raise the exception
            raise e

class GoogleAutocompletePredictor:
    """
    Optimized prediction service for search suggestions using Gemini API and FAISS index.
    Provides a Google-like autocomplete experience with minimal latency.
    """
    
    def __init__(self, api_key: str = None, redis_config: Dict = None):
        """
        Initialize the Autocomplete predictor with optimizations for speed.
        
        Args:
            api_key: Optional API key. If not provided, will attempt to read from environment.
            redis_config: Optional Redis configuration for distributed caching.
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
            
            # Set up distributed Redis cache if configured
            self.redis_client = None
            if redis_config:
                self._setup_redis(redis_config)
            
            # Local memory cache with TTL (Time To Live)
            # Using TTLCache instead of manual expiry tracking
            self.cache_ttl = 3600  # 1 hour cache lifetime
            self.cache_maxsize = 1000  # Maximum cache entries
            self.suggestion_cache = TTLCache(maxsize=self.cache_maxsize, ttl=self.cache_ttl)
            
            # Create thread pool for CPU-bound tasks
            self.cpu_executor = ThreadPoolExecutor(max_workers=4)
            
            # Create circuit breakers for external API calls
            self.embedding_circuit = CircuitBreaker("gemini-embedding")
            self.generation_circuit = CircuitBreaker("gemini-generation")
            
            # Load FAISS index and mapping
            self._load_faiss_index()
            
            # Performance metrics
            self.metrics = {
                "api_calls": 0,
                "cache_hits": 0,
                "fallbacks_used": 0,
                "avg_latency": 0,
                "total_requests": 0
            }
            
            self.logger.info("GoogleAutocompletePredictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    def _setup_redis(self, config: Dict):
        """Set up Redis connection for distributed caching."""
        try:
            self.redis_client = redis.Redis(
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                password=config.get('password', None),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Successfully connected to Redis for distributed caching")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
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
            
            # Configure FAISS for parallel search if index is large
            if self.index.ntotal > 10000:
                self.logger.info("Large index detected, enabling parallel search")
                # Create a clone of the index that can be searched in parallel
                self.index = faiss.IndexIDMap(self.index)
                faiss.omp_set_num_threads(4)  # Use 4 threads for search
            
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
    
    async def _check_cache(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Check if suggestions for this query are in cache.
        Checks both Redis (if available) and local cache.
        """
        normalized_query = self._normalize_text(query)
        
        # First check Redis for distributed cache if available
        if self.redis_client:
            try:
                redis_key = f"suggestions:{normalized_query}"
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    self.metrics["cache_hits"] += 1
                    self.logger.debug(f"Redis cache hit for query: {normalized_query}")
                    return json.loads(cached_data)
            except Exception as e:
                self.logger.warning(f"Error checking Redis cache: {e}")
        
        # Then check local cache
        try:
            if normalized_query in self.suggestion_cache:
                self.metrics["cache_hits"] += 1
                self.logger.debug(f"Local cache hit for query: {normalized_query}")
                return self.suggestion_cache[normalized_query]
        except Exception as e:
            self.logger.warning(f"Error checking local cache: {e}")
        
        # Check for prefix matches for partial queries in local cache
        try:
            for cached_query in list(self.suggestion_cache.keys()):
                if normalized_query.startswith(cached_query) and len(normalized_query) - len(cached_query) <= 3:
                    # Close enough prefix match
                    self.metrics["cache_hits"] += 1
                    self.logger.debug(f"Prefix cache hit for query: {normalized_query} matched {cached_query}")
                    return self.suggestion_cache[cached_query]
        except Exception as e:
            self.logger.warning(f"Error checking prefix matches in cache: {e}")
        
        return None
    
    async def _update_cache(self, query: str, suggestions: List[Dict[str, Any]]):
        """Update both Redis and local cache with new suggestions."""
        normalized_query = self._normalize_text(query)
        
        # Update local cache
        self.suggestion_cache[normalized_query] = suggestions
        
        # Update Redis if available
        if self.redis_client:
            try:
                redis_key = f"suggestions:{normalized_query}"
                self.redis_client.set(
                    redis_key, 
                    json.dumps(suggestions),
                    ex=self.cache_ttl
                )
            except Exception as e:
                self.logger.warning(f"Error updating Redis cache: {e}")
    
    @retry(
        retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2)
    )
    async def _get_gemini_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Gemini API with circuit breaker protection."""
        self.metrics["api_calls"] += 1
        
        try:
            # Use circuit breaker to protect against API failures
            embedding_result = await self.embedding_circuit.execute(
                self._execute_embedding_request, text
            )
            
            if embedding_result is None:
                self.logger.warning("Circuit breaker open, using fallback embedding generation")
                # Use fallback embedding generation
                return self._generate_fallback_embedding(text)
            
            return embedding_result
            
        except Exception as e:
            self.logger.error(f"Error getting Gemini embedding after retries: {e}")
            # Use fallback embedding generation
            return self._generate_fallback_embedding(text)
    
    async def _execute_embedding_request(self, text: str) -> Optional[List[float]]:
        """Execute the actual embedding API request with timeout."""
        response = await asyncio.wait_for(
            self.model.embed_content_async(text),
            timeout=2.0  # 2 second timeout
        )
        
        if response and hasattr(response, 'embedding'):
            return response.embedding
        return None
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding when API is unavailable.
        Uses feature hashing to create a consistent vector representation.
        """
        # Use CPU executor for this CPU-bound task
        embedding_dimension = 384  # Match your FAISS index dimension
        
        # Use a method similar to feature hashing to generate embeddings
        import hashlib
        
        # Create a seed from the text for deterministic output
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.md5(text_bytes)
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        # Use numpy's random with the seed for deterministic output
        np.random.seed(seed)
        
        # Generate random values based on text hash
        embedding = np.random.normal(0, 0.1, embedding_dimension).astype(np.float32)
        
        # Normalize to unit length (important for FAISS cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    async def _generate_faiss_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate suggestions from FAISS index with parallel processing."""
        if not self.index or not self.id_mapping:
            return []
            
        try:
            # Generate embedding for the partial query
            query_embedding = await self._get_gemini_embedding(partial_query)
            
            if query_embedding is None:
                return []
            
            # Convert to numpy array if needed
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Ensure we have a 2D array for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Use CPU executor for FAISS search which is CPU-bound
            # This enables asynchronous execution of the search
            search_future = asyncio.get_event_loop().run_in_executor(
                self.cpu_executor,
                self._execute_faiss_search,
                query_embedding,
                limit * 2,  # Get more candidates than needed for filtering
                partial_query
            )
            
            # Wait for the search to complete
            suggestions = await search_future
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating FAISS suggestions: {e}")
            return []
    
    def _execute_faiss_search(self, query_embedding: np.ndarray, k: int, partial_query: str) -> List[Dict[str, Any]]:
        """
        Execute FAISS search for Google-like predictive search suggestions.
        
        This method deliberately does NOT append the partial query to suggestions,
        instead treating the retrieved results as complete prediction options.
        """
        try:
            # Search the FAISS index efficiently
            distances, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(k, self.index.ntotal)
            )
            
            # Process results
            suggestions = []
            seen_texts = set()
            
            # Set of generic research-related terms to use when no specific metadata is available
            prediction_terms = [
                "methods", "analysis", "frameworks", "data collection", 
                "qualitative research", "quantitative analysis", "case studies",
                "clinical trials", "public health", "policy analysis",
                "children's health", "maternal care", "disease prevention"
            ]
            
            # Counter for generic terms
            generic_count = 0
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # Skip invalid indices
                    continue
                    
                expert_id = self.id_mapping.get(idx)
                if not expert_id:
                    continue
                
                # Try to fetch expert metadata from Redis if available
                expert_metadata = None
                if hasattr(self, 'redis_client') and self.redis_client:
                    try:
                        redis_data = self.redis_client.get(f"expert:{expert_id}")
                        if redis_data:
                            expert_metadata = json.loads(redis_data)
                    except Exception as e:
                        self.logger.debug(f"Could not fetch expert metadata from Redis: {e}")
                
                # Create suggestions based on expert knowledge areas
                if expert_metadata and 'knowledge_expertise' in expert_metadata and expert_metadata['knowledge_expertise']:
                    # If we have expertise information, use it directly as predictions
                    expertise_areas = expert_metadata['knowledge_expertise']
                    if isinstance(expertise_areas, str):
                        expertise_areas = [expertise_areas]
                    elif isinstance(expertise_areas, dict):
                        expertise_areas = list(expertise_areas.keys())
                    
                    # Use expertise areas directly as predictions
                    for expertise in expertise_areas[:2]:
                        # Important: Do NOT append partial_query here!
                        # Just use the expertise text directly as a prediction
                        suggestion_text = expertise.lower().strip()
                        
                        # Only add if it contains or is relevant to the partial query
                        # This ensures predictions are related to what the user is typing
                        if partial_query.lower() in suggestion_text or self._text_similarity(partial_query, suggestion_text) > 0.3:
                            if suggestion_text not in seen_texts:
                                suggestions.append({
                                    "text": suggestion_text,
                                    "source": "faiss_expertise",
                                    "score": float(1.0 / (1.0 + distances[0][i]))
                                })
                                seen_texts.add(suggestion_text)
                else:
                    # Without metadata, use generic research-related terms
                    # but only add a limited number of these to avoid dominating results
                    if generic_count < 5:
                        term = prediction_terms[generic_count]
                        generic_count += 1
                        
                        # Don't prepend or append partial_query, just use the term directly
                        suggestion_text = term.lower().strip()
                        
                        if suggestion_text not in seen_texts:
                            suggestions.append({
                                "text": suggestion_text,
                                "source": "generic_prediction",
                                "score": float(0.8 / (1.0 + distances[0][i]))
                            })
                            seen_texts.add(suggestion_text)
                
                # Stop if we have enough suggestions
                if len(suggestions) >= k:
                    break
            
            # If we don't have enough suggestions, add a few generic ones
            if len(suggestions) < min(3, k) and generic_count < len(prediction_terms):
                for term in prediction_terms[generic_count:generic_count+3]:
                    suggestion_text = term.lower().strip()
                    if suggestion_text not in seen_texts and len(suggestions) < k:
                        suggestions.append({
                            "text": suggestion_text,
                            "source": "generic_fallback",
                            "score": 0.7
                        })
                        seen_texts.add(suggestion_text)
            
            return suggestions
        except Exception as e:
            self.logger.error(f"Error in FAISS search execution: {e}")
            return []
        
    def _text_similarity(self, text1, text2):
        """Simple text similarity measure to check relevance."""
        # Convert texts to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    @retry(
        retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=1.5)
    )
    async def _generate_simplified_gemini_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate simplified Gemini suggestions with circuit breaker protection."""
        self.metrics["api_calls"] += 1
        
        try:
            # Use circuit breaker
            suggestions = await self.generation_circuit.execute(
                self._execute_gemini_suggestion_request,
                partial_query,
                limit
            )
            
            if suggestions is None:
                self.logger.warning("Circuit breaker open, using pattern suggestions instead")
                return self._generate_pattern_suggestions(partial_query, limit)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini suggestions after retries: {e}")
            self.metrics["fallbacks_used"] += 1
            return self._generate_pattern_suggestions(partial_query, limit)
    
    async def _execute_gemini_suggestion_request(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Execute the actual Gemini suggestion request with timeout."""
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
        response = await asyncio.wait_for(
            self.model.generate_content_async(prompt),
            timeout=3.0  # 3 second timeout
        )
        
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
        self.metrics["total_requests"] += 1
        
        try:
            # First check cache
            cached_suggestions = await self._check_cache(normalized_query)
            if cached_suggestions:
                processing_time = time.time() - start_time
                self._update_latency_metric(processing_time)
                self.logger.info(f"Returning cached suggestions for '{normalized_query}' ({processing_time:.3f}s)")
                return cached_suggestions[:limit]
            
            # Use asyncio.gather to run multiple async operations concurrently
            # This improves performance by allowing both operations to run in parallel
            faiss_future = self._generate_faiss_suggestions(normalized_query, max(limit - 2, 5))
            gemini_future = self._generate_simplified_gemini_suggestions(normalized_query, 3)
            
            # Wait for both operations to complete
            faiss_suggestions, gemini_suggestions = await asyncio.gather(
                faiss_future, 
                gemini_future
            )
            
            # Combine suggestions, prioritizing FAISS results
            combined_suggestions = []
            seen_texts = set()
            
            # Add FAISS suggestions first (higher priority)
            for suggestion in faiss_suggestions:
                if suggestion["text"] not in seen_texts:
                    combined_suggestions.append(suggestion)
                    seen_texts.add(suggestion["text"])
            
            # Add Gemini suggestions to fill any remaining spots
            remaining_slots = limit - len(combined_suggestions)
            for suggestion in gemini_suggestions[:remaining_slots]:
                if suggestion["text"] not in seen_texts:
                    combined_suggestions.append(suggestion)
                    seen_texts.add(suggestion["text"])
            
            # Fallback to pattern suggestions if we still don't have enough
            if not combined_suggestions:
                self.metrics["fallbacks_used"] += 1
                combined_suggestions = self._generate_pattern_suggestions(normalized_query, limit)
            
            # Update cache with new results
            await self._update_cache(normalized_query, combined_suggestions)
            
            # Sort by score for best results first
            combined_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            processing_time = time.time() - start_time
            self._update_latency_metric(processing_time)
            self.logger.info(f"Generated {len(combined_suggestions)} suggestions for '{normalized_query}' ({processing_time:.3f}s)")
            
            return combined_suggestions[:limit]
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_latency_metric(processing_time)
            self.logger.error(f"Error predicting suggestions: {e}")
            
            # Ensure we always return something
            self.metrics["fallbacks_used"] += 1
            return self._generate_pattern_suggestions(normalized_query, limit)
    
    def _update_latency_metric(self, processing_time: float):
        """Update the average latency metric."""
        # Calculate rolling average
        old_avg = self.metrics["avg_latency"]
        total_requests = self.metrics["total_requests"]
        
        if total_requests <= 1:
            self.metrics["avg_latency"] = processing_time
        else:
            # Update rolling average
            self.metrics["avg_latency"] = old_avg + (processing_time - old_avg) / total_requests
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current performance metrics."""
        return {
            "api_calls": self.metrics["api_calls"],
            "cache_hits": self.metrics["cache_hits"],
            "fallbacks_used": self.metrics["fallbacks_used"],
            "avg_latency_ms": round(self.metrics["avg_latency"] * 1000, 2),
            "total_requests": self.metrics["total_requests"],
            "cache_size": len(self.suggestion_cache),
            "embedding_circuit_state": self.embedding_circuit.state,
            "generation_circuit_state": self.generation_circuit.state
        }
    
    def generate_confidence_scores(self, suggestions: List[Dict[str, Any]]) -> List[float]:
        """Generate confidence scores for suggestions."""
        return [s.get("score", max(0.1, 1.0 - (i * 0.05))) for i, s in enumerate(suggestions)]