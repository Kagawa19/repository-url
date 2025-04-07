import os
import google.generativeai as genai
import logging
import numpy as np
import faiss
import pickle
import redis
from asyncio import to_thread

import json
import time
import aiohttp
import asyncio
import logging
from typing import Optional
import asyncio
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
            self.model = genai.GenerativeModel('gemini-2.0-flash')
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

    def _classify_suggestion_types(self, suggestions: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Classify suggestions into different content types for multi-modal display.
        
        Args:
            suggestions: List of suggestion dictionaries
            query: Original query for context
            
        Returns:
            List of suggestions with content_type field added
        """
        classified = []
        
        # Content type detection patterns
        person_patterns = ["professor", "dr.", "researcher", "expert in"]
        publication_patterns = ["paper", "article", "journal", "conference", "proceedings"]
        dataset_patterns = ["dataset", "data on", "database", "statistics", "survey data"]
        
        for suggestion in suggestions:
            text = suggestion.get("text", "").lower()
            source = suggestion.get("source", "")
            intent = suggestion.get("intent", "")
            
            # Default content type
            content_type = "general"
            
            # Check for existing intent classification
            if intent in ["person", "publication", "theme", "designation"]:
                content_type = intent
            # Check source-based classification
            elif source == "trending":
                content_type = "trending"
            # Check pattern-based classification
            elif any(pattern in text for pattern in person_patterns):
                content_type = "person"
            elif any(pattern in text for pattern in publication_patterns):
                content_type = "publication"
            elif any(pattern in text for pattern in dataset_patterns):
                content_type = "dataset"
            elif self._is_publication_title(text):
                content_type = "publication"
            # Check if likely to be a person name (simplified)
            elif len(text.split()) <= 3 and all(word.istitle() for word in text.split()):
                content_type = "person"
            
            # Add content type to suggestion
            suggestion_copy = suggestion.copy()
            suggestion_copy["content_type"] = content_type
            classified.append(suggestion_copy)
        
        return classified

    def _apply_hybrid_ranking(
        self, suggestions: List[Dict[str, Any]], 
        popularity_weights: Dict[str, float], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid ranking to classified suggestions.
        
        Args:
            suggestions: List of classified suggestion dictionaries
            popularity_weights: Dictionary of popularity weights
            query: Original query for relevance calculation
            
        Returns:
            List of ranked suggestions
        """
        if not suggestions:
            return []
        
        # Deduplicate suggestions
        seen_texts = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            text = suggestion.get("text", "").lower()
            if text in seen_texts:
                continue
            
            seen_texts.add(text)
            unique_suggestions.append(suggestion)
        
        # Apply hybrid ranking
        for suggestion in unique_suggestions:
            text = suggestion.get("text", "").lower()
            base_score = suggestion.get("score", 0.5)
            content_type = suggestion.get("content_type", "general")
            
            # Calculate relevance components
            exact_match_score = 1.0 if query == text else 0.0
            prefix_match_score = 0.8 if text.startswith(query) else 0.0
            substring_match_score = 0.4 if query in text else 0.0
            popularity_score = popularity_weights.get(text, 0.0)
            
            # Length normalization (slightly prefer shorter suggestions)
            length_score = max(0.0, 1.0 - (len(text.split()) / 20))
            
            # Content type boosting (for diversity)
            type_boost = {
                "person": 0.05,          # Slight boost for people
                "publication": 0.05,     # Slight boost for publications
                "trending": 0.10,        # Higher boost for trending
                "dataset": 0.03,         # Small boost for datasets
                "general": 0.00          # No boost for general
            }.get(content_type, 0.0)
            
            # Combine scores with appropriate weights
            hybrid_score = (
                base_score * 0.5 +              # Base relevance score (50%)
                exact_match_score * 0.15 +      # Exact match bonus (15%)
                prefix_match_score * 0.15 +     # Prefix match (15%)
                substring_match_score * 0.05 +  # Contains query (5%)
                popularity_score * 0.1 +        # Historical popularity (10%)
                length_score * 0.05 +           # Length normalization (5%)
                type_boost                      # Content type boost
            )
            
            # Update suggestion score
            suggestion["score"] = min(1.0, hybrid_score)
        
        # Sort by score
        unique_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique_suggestions

    def _ensure_content_diversity(
        self, suggestions: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """
        Ensure diversity of content types in top suggestions.
        
        Args:
            suggestions: List of scored suggestion dictionaries
            limit: Maximum number of suggestions to return
            
        Returns:
            List of diverse suggestions
        """
        if not suggestions:
            return []
        
        # If we have very few suggestions, return all
        if len(suggestions) <= limit / 2:
            return suggestions
        
        # Group by content type
        grouped = {}
        for suggestion in suggestions:
            content_type = suggestion.get("content_type", "general")
            if content_type not in grouped:
                grouped[content_type] = []
            grouped[content_type].append(suggestion)
        
        # Ensure diversity by interleaving content types
        diverse_results = []
        remaining_slots = limit
        
        # First, pick top item from each group (if available)
        for content_type in grouped:
            if grouped[content_type] and remaining_slots > 0:
                diverse_results.append(grouped[content_type][0])
                grouped[content_type] = grouped[content_type][1:]
                remaining_slots -= 1
        
        # Then fill remaining slots based on score
        all_remaining = []
        for content_type in grouped:
            all_remaining.extend(grouped[content_type])
        
        # Sort remaining by score
        all_remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Add until limit reached
        diverse_results.extend(all_remaining[:remaining_slots])
        
        # Final sort by score
        diverse_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return diverse_results

    def _is_publication_title(self, text: str) -> bool:
        """
        Check if text is likely a publication title.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is likely a publication title
        """
        # Publication indicators
        indicators = ["study", "analysis", "review", "survey", "approach", 
                    "framework", "method", "evaluation", "assessment"]
        
        # Check for indicators
        if any(indicator in text.lower() for indicator in indicators):
            return True
        
        # Check if has title pattern (capitalized words, length)
        words = text.split()
        if len(words) >= 4 and any(word[0].isupper() for word in words[1:] if len(word) > 2):
            return True
        
        return False

    async def _detect_query_intent(self, partial_query: str) -> str:
        """
        Detect the intent behind a user query.
        
        Args:
            partial_query: The partial query to analyze
            
        Returns:
            String representing the detected intent
        """
        # Default intent
        default_intent = "general"
        
        try:
            # Check cache for this query intent
            cache_key = f"intent:{self._normalize_text(partial_query)}"
            
            if self.redis_client:
                cached_intent = self.redis_client.get(cache_key)
                if cached_intent:
                    return cached_intent.decode() if isinstance(cached_intent, bytes) else cached_intent
            
            # Simple rule-based intent detection for common patterns
            query_lower = partial_query.lower()
            
            # Intent detection patterns
            intent_patterns = {
                "person": ["who is", "researcher", "expert named", "dr. ", "professor"],
                "publication": ["paper on", "research paper", "publication", "journal", "article about"],
                "theme": ["research on", "research in", "studies on", "field of", "research area"],
                "designation": ["position", "role", "job title", "professor of", "head of"]
            }
            
            # Check each intent pattern
            for intent, patterns in intent_patterns.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        # Cache this intent if Redis available
                        if self.redis_client:
                            self.redis_client.setex(cache_key, 3600, intent)
                        return intent
            
            # Fallback to default
            return default_intent
            
        except Exception as e:
            self.logger.error(f"Error detecting query intent: {e}")
            return default_intent



    async def _get_query_popularity_weights(self, query: str) -> Dict[str, float]:
        """Get popularity weights for queries similar to the input query"""
        try:
            # Initialize with empty weights
            weights = {}
            
            # If Redis is available, try to get popularity data
            if self.redis_client:
                # Try to get exact matches first
                popularity_key = f"query_popularity:{query}"
                popularity_data = self.redis_client.get(popularity_key)
                
                if popularity_data:
                    weights = json.loads(popularity_data)
                else:
                    # Try prefix matching for partial queries
                    query_prefix = query[:min(len(query), 3)]
                    keys = self.redis_client.keys(f"query_popularity:{query_prefix}*")
                    
                    # Combine data from up to 5 similar keys
                    for key in keys[:5]:
                        try:
                            key_data = self.redis_client.get(key)
                            if key_data:
                                key_weights = json.loads(key_data)
                                weights.update(key_weights)
                        except:
                            continue
            
            return weights
        except Exception as e:
            self.logger.error(f"Error getting query popularity weights: {e}")
            return {}

    def _fuzzy_match(self, str1: str, str2: str, max_distance: int = 1) -> bool:
        """
        Determine if two strings match within a maximum edit distance.
        Uses a simplified Levenshtein distance calculation.
        
        Args:
            str1: First string to compare
            str2: Second string to compare
            max_distance: Maximum edit distance allowed
            
        Returns:
            Boolean indicating if strings match within the maximum distance
        """
        # Handle empty strings
        if not str1 and not str2:
            return True
        if not str1 or not str2:
            return False
        
        # Simple length check first
        if abs(len(str1) - len(str2)) > max_distance:
            return False
        
        # For very short strings, check if one is prefix of the other
        if min(len(str1), len(str2)) <= 2:
            return str1.startswith(str2) or str2.startswith(str1)
            
        # For longer strings with 1 max distance, we can use a simplified approach
        if max_distance == 1:
            # Check character by character with at most one difference
            differences = 0
            if len(str1) == len(str2):
                # Same length - check for character substitutions
                for c1, c2 in zip(str1, str2):
                    if c1 != c2:
                        differences += 1
                        if differences > max_distance:
                            return False
                return True
            elif len(str1) == len(str2) + 1:
                # str1 is longer by 1 - check for insertion
                i, j = 0, 0
                while i < len(str1) and j < len(str2):
                    if str1[i] != str2[j]:
                        i += 1
                        differences += 1
                        if differences > max_distance:
                            return False
                    else:
                        i += 1
                        j += 1
                return True
            elif len(str2) == len(str1) + 1:
                # str2 is longer by 1 - check for deletion
                i, j = 0, 0
                while i < len(str1) and j < len(str2):
                    if str1[i] != str2[j]:
                        j += 1
                        differences += 1
                        if differences > max_distance:
                            return False
                    else:
                        i += 1
                        j += 1
                return True
        
        # Fall back to prefix matching for other cases
        return str1.startswith(str2) or str2.startswith(str1)
        
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
    
    
    async def _get_gemini_embedding(self, text: str, retries: int = 3, delay: float = 1.0) -> Optional[list]:
        """
        Get Gemini embedding for the input text with retries.
        """
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    payload = {
                        "input": text
                    }
                    async with session.post(self.api_url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            embedding = data.get("embedding")
                            if embedding:
                                return embedding
                            else:
                                self.logger.warning(f"Embedding not found in response: {data}")
                        else:
                            self.logger.warning(f"Received non-200 status code {response.status}: {await response.text()}")
            except Exception as e:
                self.logger.warning(f"Embedding error (attempt {attempt + 1}): {e}")
            await asyncio.sleep(delay)
        raise RuntimeError("Failed to generate embedding after retries.")

    
    async def _execute_embedding_request(self, text: str) -> Optional[List[float]]:
        """Execute the actual embedding API request with timeout."""
        response = await asyncio.wait_for(
            self.model.generate_content_async(text),
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
                raise RuntimeError("Failed to generate embedding. Gemini API did not return a valid result.")

            # Convert to numpy array if needed
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)

            # Ensure we have a 2D array for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Use to_thread for FAISS search in a background thread
            search_future = to_thread(self._execute_faiss_search, query_embedding, limit * 2, partial_query)

            # Await search completion
            suggestions = await search_future
            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating FAISS suggestions: {e}")
            return []


    
    def _extract_partial_word(self, query: str) -> Tuple[str, str]:
        """
        Extract the last partial word from a query and the query prefix.
        
        Args:
            query: The search query
            
        Returns:
            Tuple containing (query_prefix, partial_word)
        """
        words = query.split()
        if not words:
            return "", ""
        
        query_prefix = " ".join(words[:-1])
        partial_word = words[-1] if words else ""
        
        return query_prefix.strip(), partial_word.strip()

    def _is_prefix_match(self, suggestion: str, partial_query: str) -> bool:
        """
        Check if suggestion matches the partial query with proper prefix matching
        and typo tolerance.
        
        Args:
            suggestion: The suggestion text to check
            partial_query: The user's partial query
            
        Returns:
            Boolean indicating if the suggestion is a valid prefix match
        """
        # Get the query prefix and the partial word being typed
        query_prefix, partial_word = self._extract_partial_word(partial_query)
        
        # If the partial word is empty, any suggestion is valid
        if not partial_word:
            return True
        
        # Extract words from the suggestion
        suggestion_words = suggestion.lower().split()
        
        # If query has a prefix (multiple words), make sure they match in the suggestion
        if query_prefix:
            # Create a prefix pattern to match
            prefix_pattern = query_prefix.lower()
            suggestion_prefix = " ".join(suggestion_words[:len(prefix_pattern.split())])
            
            # If the prefix doesn't match, reject the suggestion
            # Allow for typo tolerance in longer prefixes
            if len(prefix_pattern) > 5:
                # For longer prefixes, use edit distance
                if not self._fuzzy_match(suggestion_prefix, prefix_pattern, max_distance=1):
                    return False
            else:
                # For shorter prefixes, require exact match
                if not suggestion_prefix.startswith(prefix_pattern):
                    return False
        
        # Find the word in the suggestion that should complete the partial word
        # This handles cases where suggestion might have a different word structure
        for suggestion_word in suggestion_words:
            # Check exact prefix match first (most common case)
            if suggestion_word.startswith(partial_word.lower()):
                return True
                
            # For longer partial words (>3 chars), allow for typo tolerance
            if len(partial_word) >= 3 and self._fuzzy_match(suggestion_word, partial_word, max_distance=1):
                return True
        
        # No matching prefix found
        return False


    def _execute_faiss_search(self, query_embedding, top_k, partial_query):
        """
        Search the FAISS index for the top_k nearest neighbors.
        """
        if query_embedding.shape[-1] != self.index.d:
            raise ValueError(f"Embedding dimension mismatch: expected {self.index.d}, got {query_embedding.shape[-1]}")
        
        try:
            scores, indices = self.index.search(query_embedding[np.newaxis, :], top_k)
        except Exception as e:
            self.logger.error(f"[FAISS][Query='{partial_query[:30]}...'] Error in index search: {e}")
            return []

        return scores[0], indices[0]

    async def _execute_gemini_suggestion_request(
        self, partial_query: str, query_intent: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Execute the actual Gemini suggestion request with intent awareness.
        
        Args:
            partial_query: The partial query text
            query_intent: The detected query intent
            limit: Maximum number of suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        # Create intent-specific prompt
        prompt_templates = {
            "person": f"""Generate {limit} search suggestions for "{partial_query}" focused on people (researchers, experts, professors).
            
            IMPORTANT:
            - Each suggestion must be on its own line
            - No numbering, no arrows, no bullet points
            - Focus on researcher names and titles
            - Keep each suggestion under 10 words
            - No explanations
            
            Suggestions:""",
            
            "publication": f"""Generate {limit} search suggestions for "{partial_query}" focused on research publications.
            
            IMPORTANT:
            - Each suggestion must be on its own line
            - No numbering or bullet points
            - Focus on paper titles and research topics
            - Keep each suggestion under 10 words
            - No explanations
            
            Suggestions:""",
            
            "theme": f"""Generate {limit} search suggestions for "{partial_query}" focused on research themes and disciplines.
            
            IMPORTANT:
            - Each suggestion must be on its own line
            - No numbering or bullet points
            - Focus on research fields and topics
            - Keep each suggestion under 10 words
            - No explanations
            
            Suggestions:""",
            
            "designation": f"""Generate {limit} search suggestions for "{partial_query}" focused on academic and research positions.
            
            IMPORTANT:
            - Each suggestion must be on its own line
            - No numbering or bullet points
            - Focus on job titles and roles
            - Keep each suggestion under 10 words
            - No explanations
            
            Suggestions:""",
            
            "general": f"""Generate {limit} single-line search suggestions for "{partial_query}" in a research context.
            
            IMPORTANT:
            - Each suggestion must be on its own line
            - No numbering, no arrows, no bullet points
            - Keep each suggestion under 10 words
            - Include relevant research terms
            - No explanations, just the suggestions
            - Prioritize common completions
            
            Suggestions:"""
        }
        
        # Use the right prompt for the detected intent
        prompt = prompt_templates.get(query_intent, prompt_templates["general"])
        
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
                    "source": f"gemini_{query_intent}",
                    "score": 0.85 - (i * 0.03),
                    "intent": query_intent
                })
            
        return suggestions

    def _generate_prefix_completions(self, partial_word: str) -> List[str]:
        """
        Generate completions for a partial word.
        
        Args:
            partial_word: The partial word to complete
            
        Returns:
            List of possible completions
        """
        if not partial_word:
            return []
        
        # Common word completions based on research domain
        common_words = {
            "m": ["methods", "maternal", "mortality", "monitoring", "management"],
            "n": ["newborn", "neonatal", "nutrition", "nurses", "network"],
            "c": ["children", "care", "clinical", "community", "covid"],
            "h": ["health", "hospital", "healthcare", "hiv", "hygiene"],
            "d": ["data", "development", "disease", "determinants", "delivery"],
            "r": ["research", "rates", "risk", "reproductive", "resources"],
            "p": ["public", "policy", "prevention", "primary", "pandemic"],
            "a": ["analysis", "assessment", "adolescent", "antenatal", "approach"],
            "e": ["evaluation", "evidence", "education", "epidemiology", "equity"],
            "i": ["intervention", "impact", "infant", "implementation", "infection"],
            "s": ["study", "system", "services", "survey", "statistics"],
            "t": ["treatment", "training", "tools", "transmission", "testing"],
            "f": ["framework", "funding", "factors", "facilities", "food"],
            "w": ["women", "water", "workers", "wellbeing", "workflow"],
            "q": ["quality", "qualitative", "quantitative", "questionnaire"],
            "v": ["vaccine", "virus", "vulnerable", "validation", "vector"],
            "o": ["outcomes", "outreach", "organization", "outbreak", "observation"],
            "b": ["birth", "behavior", "barriers", "baseline", "burden"],
            "g": ["guidelines", "global", "gender", "genomics", "growth"],
            "j": ["journal", "joint", "justice"],
            "k": ["knowledge", "key"],
            "l": ["long-term", "leadership", "laboratory", "logistics"],
            "u": ["universal", "utilization", "urban", "undernutrition"],
            "x": ["x-ray"],
            "y": ["youth", "year"],
            "z": ["zoonotic", "zones"]
        }
        
        # First, check if we have predefined completions for this prefix
        first_char = partial_word[0].lower() if partial_word else ""
        completions = []
        
        if first_char in common_words:
            # Filter completions that start with our partial word
            completions = [word for word in common_words[first_char] 
                        if word.startswith(partial_word.lower())]
        
        # Add domain-specific completions for common partial words
        if partial_word.lower() == "new":
            completions = ["newborn", "new research", "new methods", "new approaches"]
        elif partial_word.lower() == "mat":
            completions = ["maternal", "maternity", "matching", "materials"]
        elif partial_word.lower() == "hea":
            completions = ["health", "healthcare", "healthy", "healing"]
        elif partial_word.lower() == "dat":
            completions = ["data", "data collection", "database", "dates"]
        elif partial_word.lower() == "res":
            completions = ["research", "resource", "results", "response"]
        elif partial_word.lower() == "ana":
            completions = ["analysis", "analytical", "anatomy", "anaphylaxis"]
        elif partial_word.lower() == "chi":
            completions = ["children", "child", "childhood", "childbirth"]
        
        return completions[:5]  # Limit to 5 completions
        
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
        """
        Generate simplified Gemini suggestions with query intent classification.
        
        Args:
            partial_query: Partial query to get suggestions for
            limit: Maximum number of suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        self.metrics["api_calls"] += 1
        
        try:
            # First detect query intent
            query_intent = await self._detect_query_intent(partial_query)
            
            # Use circuit breaker
            suggestions = await self.generation_circuit.execute(
                self._execute_gemini_suggestion_request,
                partial_query,
                query_intent,
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
        Generate search suggestions with Google-like experience using hybrid ranking.
        
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
            
            # New: Get historical popularity data for weighting
            popularity_weights = await self._get_query_popularity_weights(normalized_query)
            
            # Combine suggestions with hybrid ranking approach
            combined_suggestions = []
            seen_texts = set()
            
            # Process each suggestion with hybrid scoring
            all_suggestions = faiss_suggestions + gemini_suggestions
            for suggestion in all_suggestions:
                text = suggestion["text"]
                if text in seen_texts:
                    continue
                
                seen_texts.add(text)
                
                # Base score from original source
                base_score = suggestion.get("score", 0.5)
                
                # Calculate additional scores for hybrid ranking
                exact_match_score = 1.0 if normalized_query == self._normalize_text(text) else 0.0
                prefix_match_score = 0.8 if text.lower().startswith(normalized_query) else 0.0
                substring_match_score = 0.4 if normalized_query in text.lower() else 0.0
                
                # Semantic relevance already captured in base_score for FAISS results
                popularity_score = popularity_weights.get(self._normalize_text(text), 0.0)
                
                # Length normalization (slightly prefer shorter suggestions)
                length_score = max(0.0, 1.0 - (len(text.split()) / 20))
                
                # Combine scores with appropriate weights
                hybrid_score = (
                    base_score * 0.5 +              # Base relevance score (50%)
                    exact_match_score * 0.15 +      # Exact match bonus (15%)
                    prefix_match_score * 0.15 +     # Prefix match (15%)
                    substring_match_score * 0.05 +  # Contains query (5%)
                    popularity_score * 0.1 +        # Historical popularity (10%)
                    length_score * 0.05             # Length normalization (5%)
                )
                
                # Add to combined results with hybrid score
                combined_suggestions.append({
                    "text": text,
                    "source": suggestion.get("source", "hybrid"),
                    "score": min(1.0, hybrid_score),  # Cap at 1.0
                    "original_score": base_score
                })
            
            # Fallback to pattern suggestions if we still don't have enough
            if not combined_suggestions:
                self.metrics["fallbacks_used"] += 1
                combined_suggestions = self._generate_pattern_suggestions(normalized_query, limit)
            
            # Update cache with new results
            await self._update_cache(normalized_query, combined_suggestions)
            
            # Sort by hybrid score for best results first
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