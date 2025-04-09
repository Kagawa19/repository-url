import datetime
import hashlib
import os
import google.generativeai as genai
import logging
import numpy as np
import faiss
import pickle
import redis
from asyncio import to_thread
from ai_services_api.services.search.utils.trie import PrefixTrie
import math
import asyncio

import json
from ai_services_api.services.search.core.database_predictor import DatabaseSuggestionGenerator
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
    def __init__(self, redis_config: Dict = None):
        # Directly use os.getenv to get the API key from .env
        key = os.getenv('GEMINI_API_KEY')
        
        
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file. "
                "Please ensure the API key is set in your environment variables."
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
            
   
            from ai_services_api.services.message.core.database import get_connection_params
            
            # Get connection parameters
            connection_params = get_connection_params()
            
            # Initialize suggestion generator with connection params
            self.suggestion_generator = DatabaseSuggestionGenerator(connection_params)
                
        
            # Create thread pool for CPU-bound tasks
            self.cpu_executor = ThreadPoolExecutor(max_workers=4)
            
            # Create circuit breakers for external API calls
            self.embedding_circuit = CircuitBreaker("gemini-embedding")
            self.generation_circuit = CircuitBreaker("gemini-generation")
            
            # Load FAISS index and mapping
            self._load_faiss_index()
            
            # Initialize prediction trie
            self.suggestion_trie = PrefixTrie()
            self.trie_last_updated = 0
            self.trie_update_interval = 3600  # Update once per hour
            
            # Initialize trending suggestions tracker
            self.trending_suggestions = {}

            self.redis_binary = None
            if redis_config:
                try:
                    # Create a separate Redis client for binary operations
                    self.redis_binary = redis.Redis(
                        host=redis_config.get('host', 'localhost'),
                        port=redis_config.get('port', 6379),
                        db=redis_config.get('db', 0),
                        password=redis_config.get('password', None),
                        decode_responses=False  # Important for binary operations
                    )
                    # Test connection
                    self.redis_binary.ping()
                except Exception as e:
                    self.logger.error(f"Failed to connect to Redis binary client: {e}")
                    self.redis_binary = None
            
            # Create background task for trie updates if Redis available
            if self.redis_client:
                try:
                    asyncio.create_task(self._init_trie_from_redis())
                except Exception as e:
                    self.logger.warning(f"Could not initialize trie update task: {e}")
            
            # Initialize selection log for local tracking if database isn't available
            self._selection_log = []
            
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

    @retry(
    retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=1.5)
    )
    async def _generate_simplified_gemini_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate simplified Gemini suggestions with query intent classification.
        Adds fallback to FAISS before using pattern suggestions.
        
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
                self.logger.warning("Circuit breaker open, attempting FAISS fallback")
                # Try FAISS before pattern suggestions
                try:
                    faiss_suggestions = await self._generate_faiss_suggestions(partial_query, limit)
                    if faiss_suggestions and len(faiss_suggestions) > 0:
                        self.logger.info("Using FAISS fallback suggestions")
                        return faiss_suggestions
                except Exception as faiss_error:
                    self.logger.error(f"FAISS fallback error: {faiss_error}")
                    
                # Try Trie-based suggestions before using pattern suggestions
                if hasattr(self, 'suggestion_trie'):
                    try:
                        trie_results = self.suggestion_trie.search(partial_query, limit=limit*2)
                        if trie_results and len(trie_results) > 0:
                            trie_suggestions = []
                            for suggestion, frequency, timestamp in trie_results:
                                trie_suggestions.append({
                                    "text": suggestion,
                                    "source": "trie",
                                    "score": min(0.85, 0.6 + (frequency / 20))
                                })
                            self.logger.info(f"Using trie fallback suggestions: {len(trie_suggestions)} results")
                            return trie_suggestions
                    except Exception as trie_error:
                        self.logger.error(f"Trie fallback error: {trie_error}")
                
                # If both FAISS and Trie failed, fall back to pattern suggestions
                self.logger.warning("All fallbacks failed, using pattern suggestions")
                return await self._generate_pattern_suggestions(partial_query, limit)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini suggestions after retries: {e}")
            self.metrics["fallbacks_used"] += 1
            
            # Try FAISS first as a fallback
            try:
                faiss_suggestions = await self._generate_faiss_suggestions(partial_query, limit)
                if faiss_suggestions and len(faiss_suggestions) > 0:
                    self.logger.info("Using FAISS fallback after Gemini error")
                    return faiss_suggestions
            except Exception as faiss_error:
                self.logger.error(f"FAISS fallback also failed: {faiss_error}")
            
            # Try Trie-based suggestions as second fallback
            if hasattr(self, 'suggestion_trie'):
                try:
                    trie_results = self.suggestion_trie.search(partial_query, limit=limit*2)
                    if trie_results and len(trie_results) > 0:
                        trie_suggestions = []
                        current_time = time.time()
                        for suggestion, frequency, timestamp in trie_results:
                            # Apply recency boosting
                            recency_factor = 1.0
                            if timestamp:
                                age_hours = (current_time - timestamp) / 3600
                                recency_factor = math.exp(-age_hours / 72)  # 3-day half-life
                                
                            trie_suggestions.append({
                                "text": suggestion,
                                "source": "trie",
                                "score": min(0.85, (0.6 + (frequency / 20)) * recency_factor)
                            })
                        self.logger.info(f"Using trie fallback after Gemini & FAISS errors: {len(trie_suggestions)} results")
                        return trie_suggestions
                except Exception as trie_error:
                    self.logger.error(f"Trie fallback also failed: {trie_error}")
            
            # If all else fails, use pattern suggestions
            return await self._generate_pattern_suggestions(partial_query, limit)
    async def _generate_pattern_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate data-driven suggestions based on user behavior and trending queries.
        
        Args:
            partial_query: Partial query to get suggestions for
            limit: Maximum number of suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        try:
            suggestions = []
            current_time = time.time()
            
            # Try trie suggestions first (fastest and most accurate)
            if hasattr(self, 'suggestion_trie'):
                trie_results = self.suggestion_trie.search(partial_query, limit=limit*2)
                
                for suggestion, frequency, timestamp in trie_results:
                    # Apply recency boost: more recent = higher score
                    recency_factor = 1.0
                    if timestamp:
                        # Calculate age in hours and apply decay
                        age_hours = (current_time - timestamp) / 3600
                        recency_factor = math.exp(-age_hours / 72)  # 3-day half-life
                    
                    # Calculate score based on frequency and recency
                    score = min(0.9, (0.5 + (frequency / 20)) * recency_factor)
                    
                    suggestions.append({
                        "text": suggestion,
                        "source": "user_history",
                        "score": score
                    })
            
            # If we have Redis, try to get trending suggestions
            if self.redis_client and len(suggestions) < limit:
                remaining = limit - len(suggestions)
                
                try:
                    # Get current hour's trending suggestions
                    current_hour = int(time.time() / 3600)
                    trending_key = f"trending:{current_hour}"
                    
                    # Use SCAN for pattern matching instead of direct prefix matching
                    # This handles cases where the prefix might be in the middle of a word
                    pattern = f"*{partial_query.lower()}*"
                    cursor = 0
                    trending_matches = []
                    
                    while len(trending_matches) < remaining * 2:
                        cursor, matches = self.redis_client.zscan(trending_key, cursor, pattern, count=100)
                        trending_matches.extend(matches)
                        if cursor == 0:  # We've gone through the whole set
                            break
                    
                    # Sort by score and take top matches
                    trending_matches.sort(key=lambda x: x[1], reverse=True)
                    
                    for suggestion, score in trending_matches[:remaining]:
                        # Only add if not already in suggestions
                        if not any(s["text"].lower() == suggestion.lower() for s in suggestions):
                            suggestions.append({
                                "text": suggestion,
                                "source": "trending",
                                "score": min(0.85, 0.6 + (score / 20))
                            })
                except Exception as redis_error:
                    self.logger.warning(f"Redis trending lookup error: {redis_error}")
            
            # If we still need more suggestions, try database fallback
            if len(suggestions) < limit:
                remaining = limit - len(suggestions)
                db_suggestions = await self._get_database_suggestions(partial_query, remaining)
                
                # Add database suggestions that aren't already included
                for db_suggestion in db_suggestions:
                    if not any(s["text"].lower() == db_suggestion["text"].lower() for s in suggestions):
                        suggestions.append(db_suggestion)
            
            # If we still have no suggestions, provide minimal fallback
            if not suggestions:
                # Last resort - return just the query itself
                suggestions.append({
                    "text": partial_query,
                    "source": "fallback",
                    "score": 0.5
                })
            
            return suggestions[:limit]
        
        except Exception as e:
            self.logger.error(f"Error generating pattern suggestions: {e}")
            # Return minimal fallback
            return [{
                "text": partial_query,
                "source": "error_fallback",
                "score": 0.5
            }]
    async def _check_cache(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Enhanced cache checking with intelligent prefix and fuzzy matching.
        Checks both Redis (if available) and local cache with tiered strategy.
        """
        normalized_query = self._normalize_text(query)
        
        # First check Redis for distributed cache if available
        if self.redis_client:
            try:
                # Try exact match first
                redis_key = f"suggestions:{normalized_query}"
                cached_data = self.redis_client.get(redis_key)
                
                if cached_data:
                    self.metrics["cache_hits"] += 1
                    self.logger.debug(f"Redis exact cache hit for query: {normalized_query}")
                    return json.loads(cached_data)
                
                # Try prefix match in Redis with scan for better performance
                if len(normalized_query) >= 3:  # Only check prefixes for queries of reasonable length
                    prefix_matches = []
                    cursor = 0
                    while True:
                        cursor, keys = self.redis_client.scan(
                            cursor, 
                            match=f"suggestions:{normalized_query[:3]}*", 
                            count=20
                        )
                        for key in keys:
                            if isinstance(key, bytes):
                                key = key.decode('utf-8')
                            key_query = key.replace("suggestions:", "")
                            # Check if our query is a prefix of a cached query or vice versa
                            if (normalized_query.startswith(key_query) or 
                                key_query.startswith(normalized_query)):
                                cached_data = self.redis_client.get(key)
                                if cached_data:
                                    prefix_matches.append((key_query, json.loads(cached_data)))
                        
                        if cursor == 0:
                            break
                    
                    # Find best prefix match
                    if prefix_matches:
                        # Sort by prefix match quality (longest common prefix)
                        prefix_matches.sort(
                            key=lambda x: self._calculate_prefix_quality(normalized_query, x[0]),
                            reverse=True
                        )
                        self.metrics["cache_hits"] += 1
                        self.logger.debug(f"Redis prefix cache hit for {normalized_query} matched {prefix_matches[0][0]}")
                        return prefix_matches[0][1]
                        
            except Exception as e:
                self.logger.warning(f"Error checking Redis cache: {e}")
        
        # Then check local cache with improved matching
        try:
            # Exact match first
            if normalized_query in self.suggestion_cache:
                self.metrics["cache_hits"] += 1
                self.logger.debug(f"Local exact cache hit for query: {normalized_query}")
                return self.suggestion_cache[normalized_query]
            
            # Intelligent prefix matching
            best_match = None
            best_score = 0
            min_prefix_length = min(3, len(normalized_query))
            
            for cached_query in list(self.suggestion_cache.keys()):
                # Skip unlikely matches early
                if abs(len(cached_query) - len(normalized_query)) > 5:
                    continue
                    
                # Calculate prefix quality
                prefix_score = self._calculate_prefix_quality(normalized_query, cached_query)
                
                if prefix_score > best_score:
                    best_score = prefix_score
                    best_match = cached_query
            
            # Use the best prefix match if good enough
            if best_match and best_score > 0.7:  # Threshold for good match
                self.metrics["cache_hits"] += 1
                self.logger.debug(f"Local prefix cache hit for {normalized_query} matched {best_match}")
                return self.suggestion_cache[best_match]
                
        except Exception as e:
            self.logger.warning(f"Error checking local cache: {e}")
        
        return None

    def _calculate_prefix_quality(self, query1: str, query2: str) -> float:
        """
        Calculate the quality of prefix match between two queries.
        Returns a score between 0 and 1, where 1 is a perfect match.
        """
        # If one is a prefix of the other, calculate how much of the longer
        # string is matched by the shorter one
        if query1.startswith(query2):
            return len(query2) / len(query1)
        elif query2.startswith(query1):
            return len(query1) / len(query2)
        
        # Find common prefix length
        common_len = 0
        for c1, c2 in zip(query1, query2):
            if c1 != c2:
                break
            common_len += 1
        
        if common_len == 0:
            return 0
        
        # Prioritize matches at word boundaries
        max_len = max(len(query1), len(query2))
        return common_len / max_len

    
    async def _personalize_suggestions_async(self, suggestions, user_id, query):
        """Run personalization in a separate task without blocking the main response."""
        try:
            from ai_services_api.services.search.core.personalization import get_user_search_history, personalize_suggestions
            
            # Get user's search history
            history = await get_user_search_history(user_id, limit=50)
            
            # Extract search terms from history and create a weighted frequency map
            term_weights = {}
            
            # Get current time for temporal weighting
            current_time = datetime.now()
            
            # Process history items with temporal decay
            for item in history:
                query_text = item.get("query", "").lower()
                timestamp_str = item.get("timestamp")
                
                # Calculate recency weight based on timestamp
                recency_weight = 1.0  # Default weight
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        # Calculate days difference
                        days_diff = (current_time - timestamp).days
                        # Apply exponential decay: weight = exp(-days/30)
                        recency_weight = math.exp(-days_diff/30)
                    except (ValueError, TypeError):
                        pass
                
                # Split into words and apply weighted counting
                words = query_text.split()
                for word in words:
                    if len(word) >= 3:  # Only consider significant words
                        current_weight = term_weights.get(word, 0)
                        # Add recency-weighted value
                        term_weights[word] = current_weight + recency_weight
            
            # Apply personalization boosting to suggestions
            personalized_suggestions = []
            for suggestion in suggestions:
                text = suggestion.get("text", "").lower()
                base_score = suggestion.get("score", 0.5)
                
                # Calculate boost based on term frequency in user history
                history_boost = 0.0
                
                # Apply term frequency boosts
                for term, weight in term_weights.items():
                    if term in text:
                        # Apply diminishing returns
                        history_boost += min(0.25, math.sqrt(weight) * 0.05)
                
                # Apply boost
                personalized_score = min(1.0, base_score + history_boost)
                
                personalized_suggestion = suggestion.copy()
                personalized_suggestion["score"] = personalized_score
                personalized_suggestion["personalized"] = True
                
                personalized_suggestions.append(personalized_suggestion)
            
            # Sort by personalized score
            personalized_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return personalized_suggestions
            
        except Exception as e:
            self.logger.error(f"Error in async personalization: {e}")
            return suggestions  # Return original suggestions if personalization fails

    async def _apply_personalization_later(self, initial_suggestions, personalization_task, user_id, normalized_query, limit):
        """Apply personalization after initial results are returned and update cache."""
        try:
            # Wait for personalization task to complete
            personalized = await personalization_task
            
            if not personalized:
                return
            
            # Update user-specific cache with personalized results
            cache_key = f"suggestions:{user_id}:{normalized_query}"
            await self._update_cache(cache_key, personalized[:limit])
            
            # Also update the trending data in Redis
            if self.redis_client:
                try:
                    # Update popularity weights for this user and query
                    user_pref_key = f"user_pref:{user_id}:{normalized_query[:3]}"
                    
                    preference_data = {
                        "query": normalized_query,
                        "suggestions": [s.get("text") for s in personalized[:5]],
                        "timestamp": time.time()
                    }
                    
                    self.redis_client.setex(
                        user_pref_key,
                        86400 * 7,  # 7-day expiry
                        json.dumps(preference_data)
                    )
                except Exception as redis_error:
                    self.logger.warning(f"Redis update error in personalization: {redis_error}")
            
        except Exception as e:
            self.logger.error(f"Late personalization error: {e}")

    def _get_trie_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Get suggestions from the prefix trie with dynamic fuzzy matching."""
        if not hasattr(self, 'suggestion_trie'):
            return []
            
        try:
            # Dynamic fuzzy match distance based on query length
            # Longer queries can tolerate more typos while remaining specific
            max_distance = 1  # Default distance
            if len(partial_query) >= 6:
                max_distance = 2  # More forgiving for longer queries
                
            # First try exact search
            trie_results = self.suggestion_trie.search(partial_query, limit=limit*2)
            
            # If exact search doesn't yield enough results, try fuzzy search
            if len(trie_results) < limit and len(partial_query) >= 3:
                fuzzy_results = self.suggestion_trie.fuzzy_search(
                    partial_query, 
                    max_distance=max_distance, 
                    limit=limit
                )
                
                # Combine results, prioritizing exact matches
                seen = set(result[0] for result in trie_results)
                for result in fuzzy_results:
                    if result[0] not in seen:
                        trie_results.append(result)
            
            suggestions = []
            current_time = time.time()
            
            for suggestion, frequency, timestamp in trie_results:
                # Apply recency boosting
                recency_factor = 1.0
                if timestamp:
                    age_hours = (current_time - timestamp) / 3600
                    recency_factor = math.exp(-age_hours / 72)  # 3-day half-life
                    
                suggestions.append({
                    "text": suggestion,
                    "source": "trie",
                    "score": min(0.9, (0.6 + (frequency / 20)) * recency_factor)
                })
                
            return suggestions
        except Exception as e:
            self.logger.error(f"Error getting trie suggestions: {e}")
            return []

    async def _update_cache(self, query: str, suggestions: List[Dict[str, Any]]):
        """
        Enhanced update for both Redis and local cache with tiered expiration.
        Implements hierarchical caching for prefix matching.
        """
        normalized_query = self._normalize_text(query)
        
        # Update local cache with TTL tracking
        self.suggestion_cache[normalized_query] = suggestions
        
        # Update Redis if available with intelligent TTL and tag-based invalidation
        if self.redis_client:
            try:
                # Store full query results
                redis_key = f"suggestions:{normalized_query}"
                
                # Apply different TTL based on query specificity and result count
                if len(normalized_query) <= 2:
                    ttl = 1800  # 30 minutes for very short queries (high change frequency)
                elif len(normalized_query) <= 4:
                    ttl = 3600  # 1 hour for short queries
                else:
                    ttl = 7200  # 2 hours for longer specific queries
                    
                # Reduce TTL for queries with few results (might be incomplete)
                if len(suggestions) < 3:
                    ttl = min(ttl, 900)  # Max 15 minutes for queries with few results
                
                # Store in Redis with appropriate TTL
                self.redis_client.setex(
                    redis_key, 
                    ttl,
                    json.dumps(suggestions)
                )
                
                # Also store prefix-based keys for the first 2-3 characters
                # to enable prefix matching
                if len(normalized_query) >= 3:
                    prefix_key = f"prefix:{normalized_query[:3]}"
                    prefix_data = {normalized_query: time.time()}
                    
                    # Update the prefix index (or create if doesn't exist)
                    existing_data = self.redis_client.get(prefix_key)
                    if existing_data:
                        try:
                            existing_prefix_data = json.loads(existing_data)
                            existing_prefix_data.update(prefix_data)
                            # Limit size to avoid unbounded growth
                            if len(existing_prefix_data) > 100:
                                # Keep most recent entries
                                sorted_items = sorted(
                                    existing_prefix_data.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True
                                )
                                existing_prefix_data = dict(sorted_items[:100])
                            prefix_data = existing_prefix_data
                        except:
                            pass
                    
                    self.redis_client.setex(
                        prefix_key,
                        86400,  # 24 hours for prefix index
                        json.dumps(prefix_data)
                    )
                    
            except Exception as e:
                self.logger.warning(f"Error updating Redis cache: {e}")

    async def _generate_faiss_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Optimized FAISS-based suggestion generation with proper prefix filtering,
        semantic matching, and intelligent fallbacks.
        """
        if not self.index or not self.id_mapping:
            return []

        try:
            start_time = time.time()
            # Generate embedding for the partial query
            embedding_future = asyncio.create_task(self._get_optimized_embedding(partial_query))
            
            # While embedding is being generated, prepare backup suggestions in parallel
            backup_future = asyncio.create_task(self._prepare_backup_suggestions(partial_query, limit))
            
            # Wait for embedding with timeout
            try:
                query_embedding = await asyncio.wait_for(embedding_future, timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"Embedding generation timed out or failed: {e}")
                # Return backup suggestions if embedding failed
                backup_suggestions = await backup_future
                return backup_suggestions
                
            if query_embedding is None:
                self.logger.warning("Embedding generation failed, falling back to backups")
                backup_suggestions = await backup_future
                return backup_suggestions

            # Convert to numpy array if needed
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)

            # Ensure we have a 2D array for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Normalize the embedding for cosine similarity
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            # Execute FAISS search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            search_result = await loop.run_in_executor(
                self.cpu_executor, 
                self._execute_faiss_search_with_tuning,
                query_embedding, 
                limit * 3,  # Get more results for better filtering
                partial_query
            )
            
            if not search_result or len(search_result) == 0:
                self.logger.warning("FAISS search returned no results, using backup")
                backup_suggestions = await backup_future
                return backup_suggestions
            
            # Process search results into suggestions
            direct_completions = []  # Suggestions starting with query
            related_suggestions = []  # Related suggestions containing query
            seen_texts = set()  # For deduplication
            partial_query_lower = partial_query.lower()
            
            for idx, score in search_result:
                if idx < 0 or idx >= len(self.id_mapping):  # Invalid index
                    continue
                    
                expert_id = self.id_mapping[idx]
                
                try:
                    # Get expert data from Redis
                    expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                    
                    if not expert_data:
                        continue
                        
                    metadata = json.loads(expert_data[b'metadata'].decode())
                    
                    # Create search suggestion from expert data
                    if 'first_name' in metadata and 'last_name' in metadata:
                        full_name = f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip()
                        full_name_lower = full_name.lower()
                        
                        if full_name and full_name_lower not in seen_texts:
                            seen_texts.add(full_name_lower)
                            
                            # Convert FAISS distance to similarity score (1.0 is best)
                            normalized_score = 1.0 - min(1.0, float(score))
                            
                            # Apply additional boosting based on query overlap
                            boost = self._calculate_text_overlap_score(partial_query, full_name)
                            
                            # Categorize as direct completion or related suggestion
                            if full_name_lower.startswith(partial_query_lower):
                                direct_completions.append({
                                    "text": full_name,
                                    "source": "faiss",
                                    "score": min(0.95, normalized_score * (1.0 + boost)),
                                    "type": "completion"  # Mark as direct completion
                                })
                            elif partial_query_lower in full_name_lower:
                                related_suggestions.append({
                                    "text": full_name,
                                    "source": "faiss",
                                    "score": min(0.7, normalized_score * 0.8),  # Lower score for related
                                    "type": "related"  # Mark as related but not direct completion
                                })
                    
                    # Also add expertise areas if relevant
                    if 'knowledge_expertise' in metadata:
                        expertise_areas = metadata.get('knowledge_expertise', [])
                        if isinstance(expertise_areas, list):
                            for area in expertise_areas:
                                area_lower = area.lower() if isinstance(area, str) else ""
                                if not area_lower or area_lower in seen_texts:
                                    continue
                                    
                                seen_texts.add(area_lower)
                                
                                # Expertise areas get a slightly lower base score
                                normalized_score = 0.85 * (1.0 - min(1.0, float(score)))
                                
                                # Categorize expertise areas too
                                if area_lower.startswith(partial_query_lower):
                                    direct_completions.append({
                                        "text": area,
                                        "source": "faiss_expertise",
                                        "score": normalized_score,
                                        "type": "completion"
                                    })
                                elif partial_query_lower in area_lower:
                                    related_suggestions.append({
                                        "text": area,
                                        "source": "faiss_expertise", 
                                        "score": normalized_score * 0.8,
                                        "type": "related"
                                    })
                except Exception as e:
                    self.logger.error(f"Error processing FAISS result: {e}")
                    continue
            
            # Sort each category by score
            direct_completions.sort(key=lambda x: x.get("score", 0), reverse=True)
            related_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Prioritize direct completions but include some related suggestions
            suggestions = direct_completions[:int(limit * 0.7)]  # 70% direct completions
            remaining_slots = limit - len(suggestions)
            
            if remaining_slots > 0 and related_suggestions:
                suggestions.extend(related_suggestions[:remaining_slots])
                
            # If we still don't have enough suggestions, use backup
            if len(suggestions) < limit / 2:
                backup_suggestions = await backup_future
                
                # Merge with backup suggestions, prioritizing FAISS results
                backup_texts = set(s["text"].lower() for s in suggestions)
                for backup in backup_suggestions:
                    if backup["text"].lower() not in backup_texts:
                        suggestions.append(backup)
                        backup_texts.add(backup["text"].lower())
            
            # Log performance metrics
            self.logger.info(f"FAISS search completed in {time.time() - start_time:.3f}s: {len(suggestions)} suggestions")
            
            return suggestions[:limit]

        except Exception as e:
            self.logger.error(f"Error in FAISS suggestions: {e}", exc_info=True)
            # Return backup suggestions on error
            try:
                backup_suggestions = await backup_future
                return backup_suggestions
            except:
                return []

    async def predict(self, partial_query: str, limit: int = 10, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate search suggestions with optimized parallel processing, 
        database-driven suggestion generation, and adaptive response strategies.
        
        Args:
            partial_query: The partial query to get suggestions for
            limit: Maximum number of suggestions to return
            user_id: Optional user ID for personalized suggestions
            
        Returns:
            List of dictionaries containing suggestion text and metadata
        """
        if not partial_query:
            return []
        
        # Track performance metrics
        start_time = time.time()
        normalized_query = self._normalize_text(partial_query)
        self.metrics["total_requests"] += 1
        
        # For very short queries, limit processing to improve responsiveness
        is_short_query = len(normalized_query) <= 2
        
        try:
            # First check cache (optimized version)
            cache_key = f"suggestions:{normalized_query}"
            if user_id:
                cache_key = f"suggestions:{user_id}:{normalized_query}"
            
            # Create tasks for parallel execution
            cache_task = asyncio.create_task(self._check_cache(cache_key))
            
            # Simultaneously generate database-driven suggestions
            database_suggestions_task = asyncio.create_task(
                self.suggestion_generator.generate_suggestions(
                    normalized_query, 
                    limit=limit * 2,  # Generate more to allow filtering
                    context=None  # Adjust context if needed
                )
            )
            
            # First wait for cache - it's fastest
            cached_suggestions = await cache_task
            
            # Also get database suggestions
            database_suggestions = await database_suggestions_task
            
            # Combine all suggestions
            all_suggestions = []
            
            # Add cached suggestions first if available
            if cached_suggestions:
                all_suggestions.extend(cached_suggestions)
            
            # Add database suggestions
            all_suggestions.extend(database_suggestions)
            
            # Filter and prioritize suggestions like Google
            direct_completions = []
            related_suggestions = []
            
            for suggestion in all_suggestions:
                suggestion_text = suggestion.get("text", "").lower()
                suggestion_score = suggestion.get("score", 0.5)
                
                # Prefix match priority
                if suggestion_text.startswith(normalized_query.lower()):
                    suggestion["type"] = "completion"
                    direct_completions.append(suggestion)
                elif normalized_query.lower() in suggestion_text:
                    suggestion["type"] = "related"
                    related_suggestions.append(suggestion)
            
            # Sort each category by score
            direct_completions.sort(key=lambda x: x.get("score", 0), reverse=True)
            related_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Combine, prioritizing direct completions
            combined_suggestions = direct_completions[:int(limit * 0.7)]  # 70% direct completions
            remaining_slots = limit - len(combined_suggestions)
            
            if remaining_slots > 0 and related_suggestions:
                combined_suggestions.extend(related_suggestions[:remaining_slots])
            
            # Ensure we have at least some suggestions
            if not combined_suggestions and all_suggestions:
                combined_suggestions = all_suggestions[:limit]
            
            # Update cache in background
            try:
                asyncio.create_task(self._update_cache(cache_key, combined_suggestions[:limit]))
            except Exception as e:
                self.logger.error(f"Error in background cache update: {e}")
            
            # Apply personalization if user_id provided
            if user_id:
                try:
                    personalized_suggestions = await self._personalize_suggestions_async(
                        combined_suggestions, 
                        user_id, 
                        normalized_query
                    )
                    combined_suggestions = personalized_suggestions
                except Exception as e:
                    self.logger.error(f"Personalization error: {e}")
            
            processing_time = time.time() - start_time
            self._update_latency_metric(processing_time)
            self.logger.info(
                f"Generated {len(combined_suggestions)} suggestions for '{normalized_query}' ({processing_time:.3f}s)"
            )
            
            return combined_suggestions[:limit]
                    
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_latency_metric(processing_time)
            self.logger.error(f"Error predicting suggestions: {e}")
            
            # Fallback to database suggestions if everything else fails
            self.metrics["fallbacks_used"] += 1
            try:
                fallback_suggestions = await self.suggestion_generator.generate_suggestions(
                    normalized_query, 
                    limit=limit,
                    context=None
                )
                
                # Filter fallback suggestions to prioritize prefix matches
                direct_matches = [
                    s for s in fallback_suggestions 
                    if s.get("text", "").lower().startswith(normalized_query.lower())
                ]
                
                return direct_matches[:limit] or fallback_suggestions[:limit]
            
            except Exception as fallback_error:
                self.logger.error(f"Fallback suggestion generation failed: {fallback_error}")
                return []
    def _calculate_text_overlap_score(self, query: str, text: str) -> float:
        """Calculate text overlap score for boosting relevant matches."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Direct substring match is highest value
        if query_lower in text_lower:
            return 0.5
        
        # Word-level matching
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        if not query_words or not text_words:
            return 0
        
        # Count matching words
        matching_words = query_words.intersection(text_words)
        match_ratio = len(matching_words) / len(query_words)
        
        return match_ratio * 0.3  # Scale the boost

    def _execute_faiss_search_with_tuning(self, query_embedding, top_k, partial_query):
        """
        Execute FAISS search with performance tuning based on query characteristics.
        Returns list of (index, distance) tuples.
        """
        if query_embedding.shape[-1] != self.index.d:
            raise ValueError(f"Embedding dimension mismatch: expected {self.index.d}, got {query_embedding.shape[-1]}")
        
        try:
            # Use nprobe tuning based on query length and specificity
            if hasattr(self.index, 'nprobe'):
                query_specificity = min(32, max(8, len(partial_query) * 4))
                original_nprobe = self.index.nprobe
                
                # Adjust nprobe: longer queries = more focused search
                self.index.nprobe = int(min(64, 128 // query_specificity))
            
            # Perform the search
            distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Reset nprobe if we changed it
            if hasattr(self.index, 'nprobe') and 'original_nprobe' in locals():
                self.index.nprobe = original_nprobe
            
            # Convert to list of (index, distance) tuples for valid indices
            result = [(int(idx), float(distances[0][i])) 
                    for i, idx in enumerate(indices[0]) 
                    if idx >= 0]
            
            return result
            
        except Exception as e:
            self.logger.error(f"[FAISS][Query='{partial_query[:30]}...'] Error in index search: {e}")
            return []

    async def _get_optimized_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with optimized performance and caching."""
        # Check embedding cache first
        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        
        if self.redis_binary:
            try:
                cached_embedding = self.redis_binary.get(cache_key)
                if cached_embedding:
                    return np.frombuffer(cached_embedding, dtype=np.float32)
            except Exception as e:
                self.logger.debug(f"Embedding cache read error: {e}")
        
        # Try to get embedding from API with circuit breaker
        try:
            embedding = await self.embedding_circuit.execute(
                self._execute_embedding_request,
                text
            )
            
            if embedding is not None:
                # Cache the embedding
                if self.redis_binary:
                    try:
                        embedding_array = np.array(embedding, dtype=np.float32)
                        self.redis_binary.setex(
                            cache_key,
                            86400,  # 24-hour cache
                            embedding_array.tobytes()
                        )
                        return embedding_array
                    except Exception as e:
                        self.logger.debug(f"Embedding cache write error: {e}")
                
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            self.logger.warning(f"Error getting embedding: {e}")
        
        # Fall back to deterministic embedding generation
        return self._generate_fallback_embedding(text)

    async def _prepare_backup_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Prepare backup suggestions to use if FAISS search fails."""
        # Try trie-based suggestions first
        if hasattr(self, 'suggestion_trie'):
            try:
                trie_results = self.suggestion_trie.search(partial_query, limit=limit*2)
                
                if trie_results and len(trie_results) > 0:
                    suggestions = []
                    for suggestion, frequency, timestamp in trie_results:
                        suggestions.append({
                            "text": suggestion,
                            "source": "trie_backup",
                            "score": min(0.8, 0.5 + (frequency / 20))
                        })
                    return suggestions
            except Exception as trie_error:
                self.logger.debug(f"Trie backup error: {trie_error}")
        
        # Fall back to pattern-based suggestions
        return await self._generate_pattern_suggestions(partial_query, limit)
            
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
    

    async def track_selection(self, partial_query: str, selected_suggestion: str, user_id: str = None):
        """
        Track which suggestion was selected to improve future predictions.
        
        Args:
            partial_query: The partial query that was typed
            selected_suggestion: The suggestion that was selected
            user_id: Optional user identifier for personalization
        """
        try:
            # First, update local trie if it exists
            if hasattr(self, 'suggestion_trie'):
                self.suggestion_trie.insert(selected_suggestion, frequency=2, timestamp=time.time())
            
            # Then, update Redis statistics if available
            if self.redis_client:
                try:
                    # Increment selection counter
                    selection_key = f"selection:{selected_suggestion.lower()}"
                    self.redis_client.zincrby("suggestion_selections", 1, selection_key)
                    
                    # Update query-specific selections
                    if partial_query:
                        query_key = f"query_selections:{partial_query.lower()}"
                        self.redis_client.zincrby(query_key, 1, selected_suggestion.lower())
                        self.redis_client.expire(query_key, 86400 * 7)  # 1 week expiry
                    
                    # Add to trending suggestions with timestamp
                    current_hour = int(time.time() / 3600)
                    trending_key = f"trending:{current_hour}"
                    self.redis_client.zincrby(trending_key, 1, selected_suggestion.lower())
                    self.redis_client.expire(trending_key, 86400)  # 24 hour expiry
                except Exception as redis_error:
                    self.logger.warning(f"Redis tracking error: {redis_error}")
            
            # Record in database asynchronously if function exists
            try:
                from ai_services_api.services.search.core.personalization import track_selected_suggestion
                if user_id:
                    asyncio.create_task(track_selected_suggestion(user_id, partial_query, selected_suggestion))
            except ImportError:
                # If the function doesn't exist, log selection locally
                if not hasattr(self, '_selection_log'):
                    self._selection_log = []
                self._selection_log.append({
                    'query': partial_query,
                    'selection': selected_suggestion,
                    'timestamp': time.time()
                })
                # Limit log size
                if len(self._selection_log) > 1000:
                    self._selection_log = self._selection_log[-1000:]
            
            self.logger.info(f"Tracked selection: '{selected_suggestion}' for query '{partial_query}'")
        except Exception as e:
            self.logger.error(f"Error tracking selection: {e}")

    async def _get_database_suggestions(self, partial_query: str, limit: int) -> List[Dict[str, Any]]:
        """Get suggestions from database search history"""
        try:
            # Import database connection function
            from ai_services_api.services.message.core.database import get_db_connection
            
            conn = None
            suggestions = []
            
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    # Get common completions from actual past searches
                    cur.execute("""
                        SELECT query, COUNT(*) as frequency 
                        FROM search_analytics 
                        WHERE query ILIKE %s
                        GROUP BY query
                        ORDER BY frequency DESC, MAX(timestamp) DESC
                        LIMIT %s
                    """, (f"%{partial_query}%", limit*2))
                    
                    rows = cur.fetchall()
                    
                    for i, (query, frequency) in enumerate(rows):
                        # Calculate score based on database frequency
                        normalized_score = min(0.8, 0.5 + (frequency / 100))  # Cap at 0.8
                        
                        suggestions.append({
                            "text": query,
                            "source": "db_history",
                            "score": normalized_score - (i * 0.02)  # Slight penalty for lower ranks
                        })
                    
            except Exception as db_error:
                self.logger.warning(f"Database suggestion error: {db_error}")
            finally:
                if conn:
                    conn.close()
                    
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in database suggestions: {e}")
            return []

    async def _init_trie_from_redis(self):
        """Initialize the suggestion trie from Redis data"""
        try:
            # Get top 1000 selections from Redis
            selections = self.redis_client.zrevrange("suggestion_selections", 0, 999, withscores=True)
            
            # Initialize trie with these selections
            for selection_key, score in selections:
                if selection_key.startswith("selection:"):
                    suggestion = selection_key[10:]  # Remove "selection:" prefix
                    self.suggestion_trie.insert(suggestion, frequency=int(score))
            
            # Also add trending suggestions from the last 24 hours
            current_hour = int(time.time() / 3600)
            for hour in range(current_hour - 24, current_hour + 1):
                trending_key = f"trending:{hour}"
                trending = self.redis_client.zrevrange(trending_key, 0, 100, withscores=True)
                for suggestion, score in trending:
                    # Add with higher frequency to prioritize recent trends
                    self.suggestion_trie.insert(suggestion, frequency=int(score) * 2)
            
            self.logger.info("Successfully initialized suggestion trie from Redis")
            self.trie_last_updated = time.time()
            
            # Schedule periodic updates
            asyncio.create_task(self._schedule_trie_updates())
        except Exception as e:
            self.logger.error(f"Error initializing trie from Redis: {e}")

    async def _schedule_trie_updates(self):
        """Schedule periodic updates to the suggestion trie"""
        while True:
            try:
                # Wait for the update interval
                await asyncio.sleep(self.trie_update_interval)
                
                # Update the trie
                await self._update_trie_from_redis()
            except Exception as e:
                self.logger.error(f"Error in trie update schedule: {e}")
                await asyncio.sleep(60)  # Wait a minute and try again

    async def _update_trie_from_redis(self):
        """Update the suggestion trie with latest data from Redis"""
        try:
            # Get recent trending suggestions
            current_hour = int(time.time() / 3600)
            trending_key = f"trending:{current_hour}"
            trending = self.redis_client.zrevrange(trending_key, 0, 100, withscores=True)
            
            # Update trie with trending suggestions
            for suggestion, score in trending:
                self.suggestion_trie.insert(suggestion, frequency=int(score) * 2, timestamp=time.time())
            
            # Also update with recent selections since last update
            last_update = self.trie_last_updated
            selection_log = self.redis_client.zrangebyscore("suggestion_log", last_update, "+inf", withscores=True)
            
            for suggestion, timestamp in selection_log:
                self.suggestion_trie.insert(suggestion, frequency=2, timestamp=timestamp)
            
            self.trie_last_updated = time.time()
            self.logger.info("Successfully updated suggestion trie")
        except Exception as e:
            self.logger.error(f"Error updating trie: {e}")
    
    async def _hybrid_rank_suggestions(
        self, suggestions: List[Dict[str, Any]], 
        query: str, 
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid ranking to suggestions using multiple factors.
        
        Args:
            suggestions: List of suggestion dictionaries
            query: Original query for relevance calculation
            user_id: Optional user ID for personalization
            
        Returns:
            Ranked list of suggestion dictionaries
        """
        if not suggestions:
            return []
        
        # Deduplicate suggestions
        unique_suggestions = {}
        for suggestion in suggestions:
            text = suggestion.get("text", "").lower()
            if text not in unique_suggestions or suggestion.get("score", 0) > unique_suggestions[text].get("score", 0):
                unique_suggestions[text] = suggestion
        
        # Convert back to list
        deduplicated = list(unique_suggestions.values())
        
        # Apply ranking factors
        ranked_suggestions = []
        
        for suggestion in deduplicated:
            text = suggestion.get("text", "")
            base_score = suggestion.get("score", 0.5)
            source = suggestion.get("source", "unknown")
            
            # Calculate ranking factors
            prefix_match = 1.5 if text.lower().startswith(query.lower()) else 1.0
            exact_match = 1.3 if text.lower() == query.lower() else 1.0
            
            # Source priority factor
            source_priority = {
                "user_history": 1.3,
                "trending": 1.2,
                "gemini": 1.1,
                "faiss": 1.05,
                "db_history": 1.0,
                "pattern": 0.9,
                "fallback": 0.7
            }.get(source, 1.0)
            
            # Apply personalization boost if user_id provided
            personalization_boost = 1.0
            if user_id and hasattr(self, '_selection_log'):
                # Check if this suggestion was selected by this user before
                for log_entry in self._selection_log:
                    if log_entry.get('user_id') == user_id and log_entry.get('selection', '').lower() == text.lower():
                        personalization_boost = 1.3
                        break
            
            # Calculate final score
            final_score = min(1.0, base_score * prefix_match * exact_match * source_priority * personalization_boost)
            
            ranked_suggestions.append({
                "text": text,
                "source": source,
                "score": final_score,
                "original_score": base_score
            })
        
        # Sort by final score
        ranked_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Ensure diversity in top results (avoid having all results from same source)
        return self._ensure_diversity(ranked_suggestions)

    def _ensure_diversity(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in suggestions by source type"""
        if len(suggestions) <= 3:
            return suggestions
            
        # Group by source
        source_groups = {}
        for suggestion in suggestions:
            source = suggestion.get("source", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(suggestion)
        
        # If we have multiple sources, ensure diversity in top results
        if len(source_groups) > 1:
            diverse_results = []
            
            # Take top result from each source for the first few slots
            for source in source_groups:
                if source_groups[source]:
                    diverse_results.append(source_groups[source][0])
                    source_groups[source] = source_groups[source][1:]
            
            # Then fill remaining slots with top results by score
            remaining = []
            for source in source_groups:
                remaining.extend(source_groups[source])
            
            # Sort remaining by score
            remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Combine and return
            diverse_results.extend(remaining)
            return diverse_results
        
        # If only one source, just return original suggestions
        return suggestions

    async def _get_personalized_suggestions(
        self, partial_query: str, user_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get personalized suggestions based on user's search history"""
        try:
            from ai_services_api.services.search.core.personalization import get_user_search_history
            
            # Get user's search history
            history = await get_user_search_history(user_id, limit=50)
            
            # Filter and rank historical queries that match the current partial query
            personalized_suggestions = []
            
            for item in history:
                query = item.get("query", "")
                timestamp = item.get("timestamp")
                
                # Skip if query doesn't match partial query
                if not query or partial_query.lower() not in query.lower():
                    continue
                
                # Calculate recency score (more recent = higher score)
                recency_score = 0.7  # Base score
                if timestamp:
                    try:
                        # Parse timestamp
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        
                        # Calculate days ago
                        days_ago = (datetime.now() - timestamp).days
                        # Apply exponential decay
                        recency_score = 0.7 * math.exp(-days_ago / 30)  # 30-day half-life
                    except (ValueError, TypeError):
                        pass
                
                personalized_suggestions.append({
                    "text": query,
                    "source": "user_history",
                    "score": recency_score
                })
            
            # Sort by score and return
            personalized_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
            return personalized_suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting personalized suggestions: {e}")
            return []

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

    def _fuzzy_match(self, str1: str, str2: str, max_distance: int = 5) -> bool:
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
        if max_distance == 5:
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