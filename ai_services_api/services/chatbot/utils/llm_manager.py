import os
import asyncio
import json
import logging
import random
import re
import time
import google.generativeai as genai
import numpy as np
from sentence_transformers.util import cos_sim

from typing import Dict, Set, Tuple, Any, List, AsyncGenerator, Optional
from datetime import datetime
from enum import Enum
from ai_services_api.services.chatbot.utils.db_utils import DatabaseConnector
from langchain.schema.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import numpy as np
import json
from typing import List, Dict, Optional, Tuple


logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enum for different types of query intents."""
    NAVIGATION = "navigation"
    PUBLICATION = "publication"
    EXPERT = "expert"  # New intent type for expert queries
    GENERAL = "general"
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

class CustomAsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    def __init__(self):
        """Initialize the callback handler with a queue."""
        super().__init__()
        self.queue = asyncio.Queue()
        self.finished = False
        self.error = None

    async def on_llm_start(self, *args, **kwargs):
        """Handle LLM start."""
        self.finished = False
        self.error = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break

    async def on_chat_model_start(self, serialized, messages, *args, **kwargs):
        """Handle chat model start by delegating to on_llm_start."""
        try:
            # Extract content safely from messages which can be in various formats
            prompts = []
            for msg in messages:
                if isinstance(msg, list):
                    # If message is a list, process each item
                    for item in msg:
                        if hasattr(item, 'content'):
                            prompts.append(item.content)
                        elif isinstance(item, dict) and 'content' in item:
                            prompts.append(item['content'])
                        elif isinstance(item, str):
                            prompts.append(item)
                elif hasattr(msg, 'content'):
                    # If message has content attribute
                    prompts.append(msg.content)
                elif isinstance(msg, dict) and 'content' in msg:
                    # If message is a dict with content key
                    prompts.append(msg['content'])
                elif isinstance(msg, str):
                    # If message is a string
                    prompts.append(msg)
            
            # Call on_llm_start with extracted prompts
            await self.on_llm_start(serialized, prompts, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in on_chat_model_start: {e}")
            self.error = e

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        """Handle new token."""
        if token and not self.finished:
            try:
                await self.queue.put(token)
            except Exception as e:
                logger.error(f"Error putting token in queue: {e}")
                self.error = e

    async def on_llm_end(self, *args, **kwargs):
        """Handle LLM end."""
        try:
            self.finished = True
            await self.queue.put(None)
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
            self.error = e

    async def on_llm_error(self, error: Exception, *args, **kwargs):
        """Handle LLM error."""
        try:
            self.error = error
            self.finished = True
            await self.queue.put(f"Error: {str(error)}")
            await self.queue.put(None)
        except Exception as e:
            logger.error(f"Error in on_llm_error handler: {e}")

    def reset(self):
        """Reset the handler state."""
        self.finished = False
        self.error = None
        self.queue = asyncio.Queue()

class GeminiLLMManager:
    def __init__(self):
        """Initialize with enhanced rate limiting and fallback support"""
        try:
            # Original initialization
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            self.callback = CustomAsyncCallbackHandler()
            
            try:
                self.redis_manager = ExpertRedisIndexManager()
                logger.info("Redis manager initialized successfully")
            except Exception as redis_error:
                logger.error(f"Error initializing Redis manager: {redis_error}")
                self.redis_manager = None

            self.context_window = []
            self.embedding_model = None
            self._init_embeddings()
            self.max_context_items = 5
            self.context_expiry = 1800
            self.confidence_threshold = 0.6

            self.expert_list_cache = {}
            self.cache_expiry = 3600

            # New enhancements
            self._rate_limited = False
            self._last_rate_limit_time = 0
            self._rate_limit_backoff = 60
            self.circuit_breaker = self._CircuitBreaker()
            self.metrics = {
                'total_calls': 0,
                'failed_calls': 0,
                'rate_limited': 0,
                'last_success': None,
                'last_failure': None
            }

            # Enhanced embedding model loading
            self.embedding_model = self._load_embedding_model()
            if not self.embedding_model:
                logger.warning("No local embedding model available")
                self._setup_remote_embeddings()

            # Original intent patterns remain unchanged
            self.intent_patterns = {
                QueryIntent.NAVIGATION: {'patterns': [...]},
                QueryIntent.PUBLICATION: {'patterns': [...]},
                QueryIntent.EXPERT: {'patterns': [...]}
            }
            
            logger.info("GeminiLLMManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GeminiLLMManager: {e}", exc_info=True)
            raise

    class _CircuitBreaker:
        """Internal circuit breaker implementation"""
        def __init__(self, threshold=5, timeout=300):
            self.threshold = threshold
            self.timeout = timeout
            self.failures = 0
            self.last_failure = 0
            self.state = "closed"

        async def check(self):
            if self.state == "open":
                if time.time() - self.last_failure > self.timeout:
                    self.state = "half-open"
                    return True
                return False
            return True

        def record_failure(self):
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "open"
                self.failures = 0

        def record_success(self):
            if self.state == "half-open":
                self.state = "closed"
            self.failures = 0

    async def get_experts_with_caching(self, query: str, limit: int = 5) -> Tuple[List[Dict], Optional[str]]:
        """
        Get experts with caching for repeated queries, especially list queries.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            Tuple[List[Dict], Optional[str]]: Expert data and optional message
        """
        # Generate cache key from normalized query
        cache_key = query.lower().strip()
        
        # Check cache first
        now = time.time()
        if cache_key in self.expert_list_cache:
            cached_data, timestamp = self.expert_list_cache[cache_key]
            # If cache is still valid
            if now - timestamp < self.cache_expiry:
                logger.info(f"Using cached expert data for query: {query}")
                return cached_data[0][:limit], cached_data[1]
        
        # Get fresh results
        experts, message = await self.get_experts(query, limit)
        
        # Store in cache
        self.expert_list_cache[cache_key] = ((experts, message), now)
        
        # Clean up expired cache entries
        self._cleanup_expert_cache()
        
        return experts, message

    def _cleanup_expert_cache(self):
        """Clean up expired cache entries."""
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.expert_list_cache.items()
            if now - timestamp > self.cache_expiry
        ]
        for key in expired_keys:
            del self.expert_list_cache[key]


    def _safe_load_embedding(self, embedding_data):
        """
        Safely load embedding data with robust error handling for different formats.
        Handles binary data, JSON strings, and different encodings.
        
        Args:
            embedding_data: Raw embedding data that could be in various formats
            
        Returns:
            numpy.ndarray: The loaded embedding vector or None if loading fails
        """
        if embedding_data is None:
            return None
            
        try:
            # Case 1: Already a numpy array
            if isinstance(embedding_data, np.ndarray):
                return embedding_data
                
            # Case 2: Binary data (most likely case)
            if isinstance(embedding_data, bytes):
                try:
                    # First try: Interpret as binary float32 array directly
                    try:
                        # Try to interpret as float32 binary data
                        arr = np.frombuffer(embedding_data, dtype=np.float32)
                        if len(arr) > 0:
                            return arr
                    except Exception as binary_err:
                        logger.debug(f"Could not interpret as float32 binary: {binary_err}")
                    
                    # Second try: Interpret as float64 binary data
                    try:
                        arr = np.frombuffer(embedding_data, dtype=np.float64)
                        if len(arr) > 0:
                            return arr
                    except Exception as binary_err:
                        logger.debug(f"Could not interpret as float64 binary: {binary_err}")
                    
                    # Try as JSON with different encodings
                    for encoding in ['latin-1', 'cp1252', 'utf-8', 'ascii']:
                        try:
                            decoded = embedding_data.decode(encoding)
                            # Try to parse as JSON
                            if decoded.startswith('[') and decoded.endswith(']'):
                                embedding_json = json.loads(decoded)
                                return np.array(embedding_json, dtype=np.float32)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    
                    # As a last resort, try to interpret directly as raw floats
                    # This assumes the binary data is just a sequence of float values
                    embedding_size = len(embedding_data) // 4  # Assuming float32 (4 bytes)
                    if embedding_size > 0:
                        try:
                            # Just create a standard-sized embedding with zeros if all else fails
                            # Better to have an imperfect embedding than none at all for list requests
                            if embedding_size != 384 and embedding_size != 512 and embedding_size != 768 and embedding_size != 1024:
                                # Create a standard sized embedding (384 is common for small models)
                                return np.zeros(384, dtype=np.float32)
                            else:
                                return np.frombuffer(embedding_data[:embedding_size*4], dtype=np.float32)
                        except Exception as raw_err:
                            logger.debug(f"Failed to interpret as raw float data: {raw_err}")
                    
                    # If we truly can't parse it, return a zero vector rather than None
                    # This allows the expert to still be included in list results
                    return np.zeros(384, dtype=np.float32)  # Standard small embedding size
                    
                except Exception as binary_parse_err:
                    logger.warning(f"All binary parsing methods failed: {binary_parse_err}")
                    # Return zero vector as fallback
                    return np.zeros(384, dtype=np.float32)
            
            # Case 3: JSON string
            if isinstance(embedding_data, str):
                try:
                    embedding_json = json.loads(embedding_data)
                    return np.array(embedding_json, dtype=np.float32)
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Failed to parse embedding string as JSON: {json_err}")
                    # Return zero vector as fallback
                    return np.zeros(384, dtype=np.float32)
                    
            # Case 4: List or other iterable
            if hasattr(embedding_data, '__iter__'):
                return np.array(list(embedding_data), dtype=np.float32)
                
            # Unhandled case - return a zero vector rather than None
            logger.warning(f"Unhandled embedding data type: {type(embedding_data)}")
            return np.zeros(384, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error loading embedding: {e}")
            # Return a zero vector as fallback for ranking
            return np.zeros(384, dtype=np.float32)

    
    
    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Enhanced intent detection using multiple fallback strategies.
        Now includes improved list detection and better handling of expert requests.
        """
        logger.info(f"Detecting intent for message: {message}")
        
        # ADDED: Check for explicit list queries with improved detection
        is_list_request = bool(re.search(r'\b(list|show|all|many)\b.*?\b(expert|researcher|scientist|publication|paper|article|resource)', message.lower()))
        
        # ADDED: Extract the entity type being listed
        list_entity = None
        if is_list_request:
            if re.search(r'\b(expert|researcher|scientist)\b', message.lower()):
                list_entity = "expert"
                logger.info("Detected expert list request")
            elif re.search(r'\b(publication|paper|article|research)\b', message.lower()):
                list_entity = "publication"
                logger.info("Detected publication list request")
        
        try:
            # First try with embeddings if available
            if self.embedding_model:
                try:
                    intent_result = await self._detect_intent_with_embeddings(message)
                    logger.info(f"Embedding intent detection result: {intent_result['intent']} with confidence {intent_result['confidence']}")
                    
                    # ADDED: Adjust confidence for list requests
                    if is_list_request and list_entity:
                        if list_entity == "expert" and intent_result['intent'] != QueryIntent.EXPERT:
                            # Override with EXPERT intent for expert lists
                            logger.info(f"Overriding detected intent for expert list request")
                            intent_result['intent'] = QueryIntent.EXPERT
                            intent_result['confidence'] = max(intent_result['confidence'], 0.8)
                        elif list_entity == "publication" and intent_result['intent'] != QueryIntent.PUBLICATION:
                            # Override with PUBLICATION intent for publication lists
                            logger.info(f"Overriding detected intent for publication list request")
                            intent_result['intent'] = QueryIntent.PUBLICATION
                            intent_result['confidence'] = max(intent_result['confidence'], 0.8)
                        
                        # Add list_request flag
                        intent_result['is_list_request'] = True
                        
                    return intent_result
                except Exception as e:
                    logger.warning(f"Embedding intent detection failed: {e}")
                        
            # Fallback to keyword matching
            keyword_result = self._detect_intent_with_keywords(message)
            
            # ADDED: Adjust confidence for list requests in keyword detection
            if is_list_request and list_entity:
                if list_entity == "expert":
                    # Override with EXPERT intent for expert lists
                    keyword_result['intent'] = QueryIntent.EXPERT
                    keyword_result['confidence'] = max(keyword_result['confidence'], 0.8)
                    keyword_result['is_list_request'] = True
                elif list_entity == "publication":
                    # Override with PUBLICATION intent for publication lists
                    keyword_result['intent'] = QueryIntent.PUBLICATION
                    keyword_result['confidence'] = max(keyword_result['confidence'], 0.8)
                    keyword_result['is_list_request'] = True
            
            if keyword_result['confidence'] > 0.7:
                logger.info(f"Keyword intent detection result: {keyword_result['intent']} with confidence {keyword_result['confidence']}")
                return keyword_result
                    
            # Final fallback to Gemini API with rate limit protection
            try:
                if not await self.circuit_breaker.check():
                    logger.warning("Circuit breaker open, using default GENERAL intent")
                    
                    # ADDED: Even with circuit breaker, respect list detection
                    if is_list_request and list_entity:
                        if list_entity == "expert":
                            return {
                                'intent': QueryIntent.EXPERT,
                                'confidence': 0.8,
                                'clarification': None,
                                'is_list_request': True
                            }
                        elif list_entity == "publication":
                            return {
                                'intent': QueryIntent.PUBLICATION,
                                'confidence': 0.8,
                                'clarification': None,
                                'is_list_request': True
                            }
                    
                    return {
                        'intent': QueryIntent.GENERAL,
                        'confidence': 0.0,
                        'clarification': None
                    }

                model = self._setup_gemini()
                prompt = f"""
                Analyze this query and classify its intent:
                Query: "{message}"

                Options:
                - PUBLICATION (research papers, studies)
                - EXPERT (researchers, specialists)
                - NAVIGATION (website sections, resources)
                - GENERAL (other queries)

                If asking about publications BY someone, extract the name.

                Return ONLY JSON in this format:
                {{
                    "intent": "PUBLICATION|EXPERT|NAVIGATION|GENERAL",
                    "confidence": 0.0-1.0,
                    "clarification": "optional question",
                    "expert_name": "name if detected"
                }}
                """

                response = await model.generate_content(prompt)
                content = response.text.replace("```json", "").replace("```", "").strip()
                
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                    intent_mapping = {
                        'PUBLICATION': QueryIntent.PUBLICATION,
                        'EXPERT': QueryIntent.EXPERT,
                        'NAVIGATION': QueryIntent.NAVIGATION,
                        'GENERAL': QueryIntent.GENERAL
                    }

                    intent_result = {
                        'intent': intent_mapping.get(result.get('intent', 'GENERAL'), QueryIntent.GENERAL),
                        'confidence': min(1.0, max(0.0, float(result.get('confidence', 0.0)))),
                        'clarification': result.get('clarification')
                    }
                    
                    if 'expert_name' in result and result['expert_name']:
                        intent_result['expert_name'] = result['expert_name']
                    
                    # ADDED: Adjust for list requests detected by pattern
                    if is_list_request and list_entity:
                        if list_entity == "expert":
                            intent_result['intent'] = QueryIntent.EXPERT
                            intent_result['confidence'] = max(intent_result['confidence'], 0.8)
                        elif list_entity == "publication":
                            intent_result['intent'] = QueryIntent.PUBLICATION
                            intent_result['confidence'] = max(intent_result['confidence'], 0.8)
                        intent_result['is_list_request'] = True
                    
                    logger.info(f"Gemini intent detection result: {intent_result['intent']} with confidence {intent_result['confidence']}")
                    return intent_result
                else:
                    # Handle case where JSON could not be extracted properly
                    logger.warning(f"Could not extract valid JSON from response: {content}")
                    
                    # ADDED: Default to appropriate intent for list requests even with JSON parsing error
                    if is_list_request and list_entity:
                        if list_entity == "expert":
                            return {
                                'intent': QueryIntent.EXPERT,
                                'confidence': 0.8,
                                'clarification': None,
                                'is_list_request': True
                            }
                        elif list_entity == "publication":
                            return {
                                'intent': QueryIntent.PUBLICATION,
                                'confidence': 0.8,
                                'clarification': None,
                                'is_list_request': True
                            }
                            
                    return {
                        'intent': QueryIntent.GENERAL,
                        'confidence': 0.0,
                        'clarification': None
                    }

            except Exception as e:
                logger.error(f"Gemini intent detection failed: {e}")
                if "429" in str(e):
                    await self._handle_rate_limit()
                
                # ADDED: Try to maintain list intent even with API errors
                if is_list_request and list_entity:
                    if list_entity == "expert":
                        return {
                            'intent': QueryIntent.EXPERT,
                            'confidence': 0.8,
                            'clarification': None,
                            'is_list_request': True
                        }
                    elif list_entity == "publication":
                        return {
                            'intent': QueryIntent.PUBLICATION,
                            'confidence': 0.8,
                            'clarification': None,
                            'is_list_request': True
                        }
                
                # Ensure we have a return in this except block
                return {
                    'intent': QueryIntent.GENERAL,
                    'confidence': 0.0,
                    'clarification': None
                }

        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            # Make sure the outer try block also has a return
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }
        
    def _init_embeddings(self):
        """Initialize embedding model with verification and fallback."""
        try:
            self.embedding_model = self._load_embedding_model()
            if self.embedding_model:
                # Test the model
                test_embed = self.embedding_model.encode("test")
                if len(test_embed) > 0:
                    logger.info("Embedding model loaded and verified")
                else:
                    logger.warning("Embedding model returned empty vector")
                    self.embedding_model = None
            else:
                logger.warning("No local embedding model available")
            if not self.embedding_model:
                self._setup_remote_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embedding_model = None
        
    
            
    def semantic_search(self, query: str, embeddings: List[np.ndarray], texts: List[str], top_k: int = 3) -> List[str]:
        try:
            query_embedding = self.embedding_model.encode(query)
            similarities = cos_sim(query_embedding, embeddings).numpy().flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [texts[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
        
    async def get_experts(self, query: str, limit: int = 3) -> Tuple[List[Dict], Optional[str]]:
        """
        Enhanced method to fetch experts by name/query with improved matching and richer results.
        Includes expertise matching and publication information.
        """
        try:
            # Check if this is a "list experts" request
            is_list_request = bool(re.search(r'\b(list|show|all|many)\b.*?\b(expert|researcher|scientist|specialist)', query.lower()))
            
            # ADDED: Check for theme-specific queries like "theme HAW"
            theme_match = re.search(r'\b(theme|unit|department|field)\s+([A-Za-z0-9]+)\b', query.lower())
            search_field = None
            search_value = None
            if theme_match:
                search_field = theme_match.group(1)
                search_value = theme_match.group(2).upper()  # Convert to uppercase for matching abbreviations
                logger.info(f"Detected specific {search_field} search for '{search_value}'")
            
            # Adjust limit for list requests
            if is_list_request and limit < 5:
                limit = 5  # Increase limit for explicit list requests
                logger.info(f"Detected list request, increased limit to {limit}")

            if not self.redis_manager or not self.redis_manager.redis_text:
                return [], "Redis manager not available"
                
            # Check if we need to use async or sync Redis methods
            is_async_redis = hasattr(self.redis_manager.redis_text, 'ascan') or hasattr(self.redis_manager.redis_text, 'akeys')
            
            # Get expert keys using the appropriate Redis method
            keys = []
            if is_async_redis:
                # Use async methods if available
                if hasattr(self.redis_manager.redis_text, 'akeys'):
                    keys = await self.redis_manager.redis_text.akeys("meta:aphrc_expert:*")
                else:
                    # Use async scan
                    cursor = 0
                    while True:
                        cursor, batch = await self.redis_manager.redis_text.ascan(
                            cursor=cursor, 
                            match="meta:aphrc_expert:*", 
                            count=100
                        )
                        keys.extend(batch)
                        if cursor == 0:
                            break
            else:
                # Use synchronous methods
                if hasattr(self.redis_manager.redis_text, 'keys'):
                    keys = self.redis_manager.redis_text.keys("meta:aphrc_expert:*")
                else:
                    # Use synchronous scan
                    cursor = 0
                    while True:
                        cursor, batch = self.redis_manager.redis_text.scan(
                            cursor=cursor, 
                            match="meta:aphrc_expert:*", 
                            count=100
                        )
                        keys.extend(batch)
                        if cursor == 0:
                            break
            
            # Log the number of expert keys found
            logger.info(f"Found {len(keys)} expert keys in Redis")
            if not keys:
                return [], "No experts found in database"
            
            # Initialize collections
            experts = []
            query_embedding = None
            match_scores = {}  # Track multiple matching criteria

            # Encode the query if an embedding model is available
            if self.embedding_model:
                try:
                    query_embedding = self.embedding_model.encode(query.lower().strip())
                except Exception as emb_err:
                    logger.warning(f"Failed to create query embedding: {emb_err}")

            # Extract meaningful terms from the query
            query_terms = query.lower().split()
            # Extract name terms, expertise terms, and domain terms
            name_terms = []
            expertise_terms = []
            
            for term in query_terms:
                # Skip very short terms and common words
                if len(term) <= 2 or term in ['is', 'there', 'an', 'expert', 'who', 'the', 'a', 'of', 'and', 
                                            'what', 'where', 'when', 'how', 'why', 'which', 'this', 'that',
                                            'list', 'show', 'all', 'many', 'theme', 'unit', 'department', 'field']:
                    continue
                    
                # Categorize terms (could be in multiple categories)
                name_terms.append(term)
                
                # Terms longer than 4 chars could be expertise-related
                if len(term) > 4:
                    expertise_terms.append(term)
            
            logger.info(f"Query analysis - Name terms: {name_terms}, Expertise terms: {expertise_terms}")

            # Process each expert key
            for key in keys:
                try:
                    # Get expert data using appropriate method
                    expert = {}
                    if is_async_redis:
                        if hasattr(self.redis_manager.redis_text, 'ahgetall'):
                            expert = await self.redis_manager.redis_text.ahgetall(key)
                        else:
                            expert = await self.redis_manager.redis_text.hgetall(key)
                    else:
                        expert = self.redis_manager.redis_text.hgetall(key)
                    
                    if not expert:
                        continue
                        
                    expert_id = expert.get('id') or key.split(':')[-1]
                    match_score = 0.0
                    matched_criteria = []
                    
                    # For list requests, give all experts a base score
                    if is_list_request:
                        match_score = 0.3  # Base score for list requests
                        matched_criteria.append("Included in expert list")
                    
                    # ADDED: Handle specific theme/unit search
                    if search_field and search_value:
                        expert_value = None
                        if search_field == 'theme':
                            expert_value = expert.get('theme', '').upper()
                        elif search_field in ['unit', 'department']:
                            expert_value = expert.get('unit', '').upper()
                        elif search_field == 'field':
                            # Check fields JSON array
                            try:
                                fields = json.loads(expert.get('fields', '[]'))
                                expert_value = ' '.join([str(f).upper() for f in fields if f])
                            except:
                                expert_value = ''
                        
                        # Exact match gives high score
                        if expert_value and search_value in expert_value:
                            match_score += 0.9
                            matched_criteria.append(f"Matched {search_field} '{search_value}'")
                            # For exact theme match, add 0.5 to the score (total 1.4)
                            if search_field == 'theme' and expert_value == search_value:
                                match_score += 0.5
                                matched_criteria.append(f"Exact {search_field} match")
                            # For partial theme match like 'HAW' in 'HAWL', add 0.3 (total 1.2)
                            elif search_field == 'theme' and search_value in expert_value:
                                match_score += 0.3
                                matched_criteria.append(f"Partial {search_field} match")
                    
                    # 1. NAME MATCHING - check first name, last name, middle name
                    expert_first_name = expert.get('first_name', '').lower()
                    expert_last_name = expert.get('last_name', '').lower()
                    expert_middle_name = expert.get('middle_name', '').lower()
                    full_name = f"{expert_first_name} {expert_middle_name} {expert_last_name}".strip()
                    
                    # If no specific terms beyond "list experts", skip detailed matching
                    if name_terms or not is_list_request:
                        for term in name_terms:
                            if term in expert_first_name:
                                match_score += 0.8
                                matched_criteria.append(f"First name contains '{term}'")
                            elif term in expert_last_name:
                                match_score += 0.9
                                matched_criteria.append(f"Last name contains '{term}'")
                            elif term in expert_middle_name:
                                match_score += 0.7
                                matched_criteria.append(f"Middle name contains '{term}'")
                            elif term in full_name:
                                match_score += 0.6
                                matched_criteria.append(f"Full name contains '{term}'")
                    
                    # 2. EXPERTISE MATCHING - check expertise, domains, fields, skills
                    if expertise_terms:
                        # Parse JSON fields
                        try:
                            expert_expertise = json.loads(expert.get('expertise', '{}')) if expert.get('expertise') else {}
                            domains = json.loads(expert.get('domains', '[]')) if expert.get('domains') else []
                            fields = json.loads(expert.get('fields', '[]')) if expert.get('fields') else []
                            skills = json.loads(expert.get('normalized_skills', '[]')) if expert.get('normalized_skills') else []
                            keywords = json.loads(expert.get('keywords', '[]')) if expert.get('keywords') else []
                            # ADDED: Parse knowledge_expertise
                            knowledge_expertise = expert.get('knowledge_expertise', '')
                        except json.JSONDecodeError:
                            expert_expertise = {}
                            domains = []
                            fields = []
                            skills = []
                            keywords = []
                            knowledge_expertise = ''
                        
                        # Flatten expertise into a list of terms
                        expertise_values = []
                        if isinstance(expert_expertise, dict):
                            for values in expert_expertise.values():
                                if isinstance(values, list):
                                    expertise_values.extend([str(v).lower() for v in values if v])
                                elif values:
                                    expertise_values.append(str(values).lower())
                        
                        # ADDED: Add knowledge_expertise to expertise values
                        if knowledge_expertise:
                            if isinstance(knowledge_expertise, str):
                                expertise_values.append(knowledge_expertise.lower())
                            elif isinstance(knowledge_expertise, list):
                                expertise_values.extend([str(k).lower() for k in knowledge_expertise if k])
                        
                        # Check if any expertise term matches query terms
                        for term in expertise_terms:
                            # Check in expertise values
                            for value in expertise_values:
                                if term in value:
                                    match_score += 0.7
                                    matched_criteria.append(f"Expertise contains '{term}'")
                                    break
                            
                            # Check in domains
                            for domain in domains:
                                if term in str(domain).lower():
                                    match_score += 0.6
                                    matched_criteria.append(f"Domain contains '{term}'")
                                    break
                            
                            # Check in fields
                            for field in fields:
                                if term in str(field).lower():
                                    match_score += 0.5
                                    matched_criteria.append(f"Field contains '{term}'")
                                    break
                                    
                            # Check in skills and keywords
                            for skill in skills + keywords:
                                if term in str(skill).lower():
                                    match_score += 0.6
                                    matched_criteria.append(f"Skill/keyword contains '{term}'")
                                    break
                    
                    # 3. BIOGRAPHICAL MATCHING
                    if expert.get('bio'):
                        bio = expert.get('bio', '').lower()
                        for term in expertise_terms:
                            if term in bio:
                                match_score += 0.4
                                matched_criteria.append(f"Biography contains '{term}'")
                    
                    # ADDED: Check if "HAW" or other theme abbreviation appears anywhere
                    if theme_match and search_value:
                        # Check designation, bio and other text fields
                        for field in ['designation', 'bio', 'knowledge_expertise']:
                            field_text = str(expert.get(field, '')).upper()
                            if search_value in field_text:
                                match_score += 0.4
                                matched_criteria.append(f"{field.capitalize()} contains '{search_value}'")
                    
                    # Adjust threshold based on request type
                    match_threshold = 0.1 if is_list_request else 0.3  # Lower threshold for list requests
                    # Further lower threshold for specific theme/unit searches
                    if search_field and search_value:
                        match_threshold = 0.05
                    
                    if match_score > match_threshold:
                        logger.info(f"Expert {expert_id} matched with score {match_score:.2f}: {', '.join(matched_criteria[:3])}")
                        
                        # ADDED: Ensure knowledge_expertise is explicitly retrieved and included
                        if 'knowledge_expertise' not in expert or not expert['knowledge_expertise']:
                            # Try to get knowledge_expertise from a different field or attribute
                            if 'expertise' in expert:
                                try:
                                    # Try parsing expertise as JSON to extract knowledge areas
                                    expertise_data = json.loads(expert['expertise']) if isinstance(expert['expertise'], str) else expert['expertise']
                                    if isinstance(expertise_data, dict):
                                        # Extract first few values or a specific key as knowledge_expertise
                                        expertise_values = []
                                        for k, v in expertise_data.items():
                                            if isinstance(v, list):
                                                expertise_values.extend(v)
                                            else:
                                                expertise_values.append(v)
                                        
                                        if expertise_values:
                                            expert['knowledge_expertise'] = ", ".join(str(v) for v in expertise_values[:3])
                                    elif isinstance(expertise_data, list):
                                        expert['knowledge_expertise'] = ", ".join(str(v) for v in expertise_data[:3])
                                except (json.JSONDecodeError, TypeError):
                                    # If parsing fails, use the raw string
                                    if expert['expertise']:
                                        expert['knowledge_expertise'] = str(expert['expertise'])
                            
                            # Check additional fields if knowledge_expertise is still missing
                            if not expert.get('knowledge_expertise') and expert.get('normalized_skills'):
                                try:
                                    skills = json.loads(expert['normalized_skills']) if isinstance(expert['normalized_skills'], str) else expert['normalized_skills']
                                    if skills and isinstance(skills, list):
                                        expert['knowledge_expertise'] = ", ".join(str(s) for s in skills[:3])
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            
                            # If still no knowledge_expertise, use areas or domains
                            if not expert.get('knowledge_expertise'):
                                for field in ['research_areas', 'domains', 'fields']:
                                    if expert.get(field):
                                        try:
                                            data = json.loads(expert[field]) if isinstance(expert[field], str) else expert[field]
                                            if data and isinstance(data, list):
                                                expert['knowledge_expertise'] = ", ".join(str(d) for d in data[:3])
                                                break
                                        except (json.JSONDecodeError, TypeError):
                                            if expert[field]:
                                                expert['knowledge_expertise'] = str(expert[field])
                                                break
                            
                            # Fallback to a placeholder if still no knowledge_expertise
                            if not expert.get('knowledge_expertise'):
                                expert['knowledge_expertise'] = "Health research"
                        
                        # Store match score for ranking
                        expert['match_score'] = match_score
                        expert['matched_criteria'] = matched_criteria
                        
                        # Add to results
                        experts.append(expert)
                
                except Exception as expert_err:
                    logger.warning(f"Error processing expert from key {key}: {expert_err}")
                    continue

            logger.info(f"Found {len(experts)} matching experts for query: {query}")
            
            # For list requests with no results, return top experts instead of empty list
            if is_list_request and len(experts) == 0:
                logger.info("No matches for list request - returning top experts instead")
                # Process first 5-10 experts without filtering
                count = 0
                for key in keys[:10]:  # Try first 10 keys
                    if count >= limit:
                        break
                    
                    try:
                        expert = {}
                        if is_async_redis:
                            if hasattr(self.redis_manager.redis_text, 'ahgetall'):
                                expert = await self.redis_manager.redis_text.ahgetall(key)
                            else:
                                expert = await self.redis_manager.redis_text.hgetall(key)
                        else:
                            expert = self.redis_manager.redis_text.hgetall(key)
                        
                        if expert:
                            expert_id = expert.get('id') or key.split(':')[-1]
                            expert['match_score'] = 0.5  # Default score
                            expert['matched_criteria'] = ["Top expert"]
                            
                            # ADDED: Ensure knowledge_expertise is included for fallback experts too
                            if 'knowledge_expertise' not in expert or not expert['knowledge_expertise']:
                                # Try to extract from expertise or other fields
                                if expert.get('expertise'):
                                    try:
                                        expertise_data = json.loads(expert['expertise']) if isinstance(expert['expertise'], str) else expert['expertise']
                                        if isinstance(expertise_data, dict):
                                            # Extract values as knowledge_expertise
                                            values = []
                                            for v in expertise_data.values():
                                                if isinstance(v, list):
                                                    values.extend(v)
                                                else:
                                                    values.append(v)
                                            if values:
                                                expert['knowledge_expertise'] = ", ".join(str(v) for v in values[:3])
                                        elif isinstance(expertise_data, list):
                                            expert['knowledge_expertise'] = ", ".join(str(v) for v in expertise_data[:3])
                                    except (json.JSONDecodeError, TypeError):
                                        expert['knowledge_expertise'] = str(expert.get('expertise', 'Health research'))
                                else:
                                    expert['knowledge_expertise'] = "Health research"
                            
                            experts.append(expert)
                            count += 1
                    except Exception as e:
                        logger.warning(f"Error processing fallback expert: {e}")
                        continue
                    
                # Add a note that these are fallback experts if we had to use fallbacks
                if experts:
                    if search_field and search_value:
                        return experts[:limit], f"No experts found with exact {search_field} '{search_value}'. Showing top experts instead."
                    else:
                        return experts[:limit], "No experts matched your query exactly. Showing top experts instead."
            
            # If we have too many matches, filter and rank them
            if len(experts) > limit:
                # For list requests, skip complex embedding ranking and just use match score
                if is_list_request:
                    # Sort primarily by match score
                    ranked_experts = sorted(
                        experts,
                        key=lambda e: e.get('match_score', 0),
                        reverse=True
                    )
                    return ranked_experts[:limit], None
                    
                # For non-list requests, try semantic ranking if possible
                if query_embedding is not None:
                    try:
                        # For experts with embeddings, calculate semantic similarity
                        experts_with_embeddings = []
                        for expert in experts:
                            # Try to get embedding 
                            if 'embedding' not in expert or expert['embedding'] is None:
                                # Get embedding from Redis
                                expert_id = expert.get('id', '')
                                if expert_id:
                                    embedding_key = f"emb:aphrc_expert:{expert_id}"
                                    try:
                                        embedding_data = None
                                        if is_async_redis:
                                            if hasattr(self.redis_manager.redis_text, 'aget'):
                                                embedding_data = await self.redis_manager.redis_text.aget(embedding_key)
                                            else:
                                                embedding_data = await self.redis_manager.redis_text.get(embedding_key)
                                        else:
                                            embedding_data = self.redis_manager.redis_text.get(embedding_key)
                                        
                                        if embedding_data:
                                            # Use _safe_load_embedding helper for binary handling
                                            try:
                                                expert['embedding'] = self._safe_load_embedding(embedding_data)
                                            except Exception as safe_load_err:
                                                logger.warning(f"Failed to load embedding for {expert_id}: {safe_load_err}")
                                    except Exception as emb_err:
                                        logger.warning(f"Failed to load embedding for {expert_id}: {emb_err}")
                            
                            if 'embedding' in expert and expert['embedding'] is not None:
                                experts_with_embeddings.append(expert)
                        
                        if experts_with_embeddings:
                            # Calculate semantic similarity scores
                            for expert in experts_with_embeddings:
                                try:
                                    similarity = float(cos_sim(query_embedding, expert['embedding'])[0][0])
                                    # Boost match score with semantic similarity
                                    expert['match_score'] = 0.6 * expert['match_score'] + 0.4 * similarity
                                except Exception as sim_err:
                                    logger.warning(f"Error calculating similarity for expert: {sim_err}")
                        
                        # Sort by final match score
                        ranked_experts = sorted(
                            experts,
                            key=lambda e: e.get('match_score', 0),
                            reverse=True
                        )
                        
                        return ranked_experts[:limit], None
                    except Exception as ranking_err:
                        logger.warning(f"Error in semantic ranking: {ranking_err}")
                
                # Fallback to basic match score ranking
                ranked_experts = sorted(
                    experts,
                    key=lambda e: e.get('match_score', 0),
                    reverse=True
                )
                return ranked_experts[:limit], None
            
            return experts[:limit], None
            
        except Exception as e:
            logger.error(f"Error fetching experts: {e}")
            return [], str(e)
    
                
    def _create_tailored_prompt(self, intent, context_type: str, message_style: str) -> str:
        """
        Create a tailored system prompt based on detected intent, context type, and message style.
        
        Args:
            intent: The detected intent from QueryIntent enum
            context_type (str): The type of context being provided
            message_style (str): The communication style detected in the user's message
            
        Returns:
            str: A specialized system prompt for the LLM
        """
        # Base instructions for all responses
        base_instructions = """
        You are an assistant for the African Population and Health Research Center (APHRC).
        Your role is to provide helpful, accurate information about APHRC research, 
        publications, experts, and resources. Maintain a professional, supportive tone
        while being conversational and engaging.
        """
        
        # Style-specific tone guidance
        tone_guidance = {
            "technical": """
            Use precise language and academic terminology appropriate for researchers and professionals.
            Provide detailed information with proper citations when possible.
            Maintain clarity and accuracy while demonstrating deep subject matter expertise.
            """,
            
            "formal": """
            Use professional language with moderate formality.
            Be thorough and precise while remaining accessible to non-specialists.
            Maintain a respectful, informed tone appropriate for official communication.
            """,
            
            "conversational": """
            Use natural, engaging language that balances professionalism with approachability.
            Create a dialogue-like experience using conversational transitions.
            Be friendly and helpful while maintaining APHRC's professional authority.
            """
        }
        
        # Intent-specific response structures
        intent_guidance = {
            QueryIntent.PUBLICATION: """
            When discussing publications:
            - Highlight key findings and their significance
            - Explain the research context and relevance
            - Connect publications to broader themes in APHRC's work
            - Use natural transitions between publications
            - Suggest related publications when appropriate
            """,
            
            QueryIntent.EXPERT: """
            When discussing experts:
            - Emphasize their specific areas of expertise and contributions
            - Make connections between their work and the user's query
            - Provide context about their role at APHRC
            - Highlight notable publications or projects
            - Suggest how their expertise might be relevant to the user's interests
            """,
            
            QueryIntent.NAVIGATION: """
            When providing navigation assistance:
            - Give clear, step-by-step guidance
            - Explain what the user will find in each section
            - Provide context about why certain resources might be useful
            - Anticipate related information needs
            - Offer suggestions for exploring related content
            """,
            
            QueryIntent.GENERAL: """
            For general information about APHRC:
            - Provide balanced, comprehensive overviews
            - Highlight APHRC's mission and impact when relevant
            - Connect information to broader themes in public health and population research
            - Anticipate follow-up questions
            - Provide examples and context to make information accessible
            """
        }
        
        # Context-specific guidance for empty results
        context_specific = ""
        if context_type == "no_experts":
            context_specific = """
            Although no specific experts were found, provide helpful general information
            about APHRC's work in the relevant domain. Suggest broader research areas
            the user might explore and mention that you can help find information about
            specific experts if they provide more details.
            """
        elif context_type == "no_publications":
            context_specific = """
            Although no specific publications were found, provide helpful general information
            about APHRC's research in the relevant domain. Suggest broader topics
            the user might explore and mention that you can help find specific publications
            if they provide more details about their interests.
            """
        
        # Combine appropriate guidance elements
        selected_tone = tone_guidance.get(message_style, tone_guidance["conversational"])
        selected_intent = intent_guidance.get(intent, intent_guidance[QueryIntent.GENERAL])
        
        full_prompt = f"{base_instructions}\n\n{selected_tone}\n\n{selected_intent}\n\n{context_specific}"
        
        # Add guidance on structuring the response
        full_prompt += """
        Structure your response for clarity:
        - Begin with a direct, helpful answer to the query
        - Provide context and details in a logical flow
        - Use natural transitions between topics
        - Include specific examples when helpful
        - End with a courteous closing that invites further engagement
        """
        
        return full_prompt.strip()

    def _analyze_message_style(self, message: str) -> str:
        """
        Analyze the user's message to determine the appropriate response style.
        
        Args:
            message (str): The user's query
            
        Returns:
            str: The detected communication style
        """
        # Convert to lowercase for analysis
        message_lower = message.lower()
        
        # Check for technical/academic indicators
        technical_indicators = [
            'methodology', 'study design', 'statistical', 'analysis', 
            'literature review', 'theoretical', 'framework', 'evidence-based',
            'quantitative', 'qualitative', 'research', 'findings', 'publication',
            'citations', 'references', 'peer-reviewed', 'journal', 'paper'
        ]
        
        # Check for formal tone indicators
        formal_indicators = [
            'would you please', 'I would like to', 'could you provide',
            'I am interested in', 'I request', 'kindly', 'formal', 'official',
            'the organization', 'professionals', 'documentation'
        ]
        
        # Check for conversational tone indicators
        conversational_indicators = [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'appreciate',
            'can you help', 'tell me about', 'what\'s', 'how about',
            'wondering', 'curious', 'question for you', 'quick question'
        ]
        
        # Count indicators
        technical_score = sum(1 for term in technical_indicators if term in message_lower)
        formal_score = sum(1 for term in formal_indicators if term in message_lower)
        conversational_score = sum(1 for term in conversational_indicators if term in message_lower)
        
        # Add score for sentence structure formality
        sentences = re.split(r'[.!?]', message)
        avg_words_per_sentence = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
        
        # Longer sentences tend to be more formal
        if avg_words_per_sentence > 15:
            formal_score += 2
        elif avg_words_per_sentence > 10:
            formal_score += 1
        elif avg_words_per_sentence < 6:
            conversational_score += 1
        
        # Questions with question marks are often conversational
        if '?' in message:
            conversational_score += 1
        
        # Determine style based on scores
        if technical_score > max(formal_score, conversational_score):
            return "technical"
        elif formal_score > conversational_score:
            return "formal"
        else:
            return "conversational"

    def rich_expert_summary(self, experts: List[Dict[str, Any]], include_header: bool = True) -> str:
        """
        Create a rich, well-formatted summary of experts optimized for readability.
        Uses the requested format for expert listings.
        
        Args:
            experts (List[Dict]): List of expert dictionaries
            include_header (bool): Whether to include a header
            
        Returns:
            str: Richly formatted expert summary
        """
        if not experts:
            return "I couldn't find any experts matching your criteria. Would you like me to suggest some related experts instead?"
        
        # Begin with optional header
        lines = []
        if include_header:
            if len(experts) > 1:
                lines.append("# APHRC Experts\n")
            else:
                lines.append("# APHRC Expert Profile\n")
        
        # Format each expert according to the requested format
        for i, expert in enumerate(experts):
            # Get the expert's name
            first_name = expert.get('first_name', '').strip()
            last_name = expert.get('last_name', '').strip()
            full_name = f"{first_name} {last_name}".strip()
            
            # Start with the numbered list item
            lines.append(f"{i+1}. **{full_name}**")
            
            # Add designation
            designation = expert.get('designation', '')
            if designation:
                lines.append(f"   **Designation:** {designation}")
            
            # Add theme with expansion
            theme = expert.get('theme', '')
            if theme:
                theme_expanded = self.expand_abbreviation(theme, 'theme')
                lines.append(f"   **Theme:** {theme_expanded}")
            
            # Add unit with expansion
            unit = expert.get('unit', '')
            if unit:
                unit_expanded = self.expand_abbreviation(unit, 'unit')
                lines.append(f"   **Unit:** {unit_expanded}")
            
            # Add knowledge expertise
            expertise = self.format_knowledge_expertise(expert)
            lines.append(f"   **Knowledge & Expertise:** {expertise}")
            
            # Add separator between experts
            if i < len(experts) - 1:
                lines.append("")
        
        # Add suggestion for further exploration
        lines.append("\nYou can ask for more details about any specific expert or request information about their publications.")
        
        # Join all lines with newlines
        return "\n".join(lines)

    def extract_search_criteria(self, query: str) -> Dict[str, str]:
        """
        Extract search criteria from a query string.
        Detects theme, unit, and other specific search parameters.
        
        Args:
            query (str): The user's search query
            
        Returns:
            Dict[str, str]: Dictionary of extracted search criteria
        """
        criteria = {}
        
        # Extract theme criteria
        theme_match = re.search(r'\b(?:theme|in theme|with theme)\s+([A-Za-z0-9]+)\b', query.lower())
        if theme_match:
            criteria['theme'] = theme_match.group(1).upper()
        
        # Extract unit criteria
        unit_match = re.search(r'\b(?:unit|in unit|with unit|department)\s+([A-Za-z0-9]+)\b', query.lower())
        if unit_match:
            criteria['unit'] = unit_match.group(1).upper()
        
        # Extract field/expertise criteria
        field_match = re.search(r'\b(?:field|expertise|specializing in|working on)\s+([A-Za-z0-9\s]+?)\b(?:\s|$)', query.lower())
        if field_match:
            criteria['expertise'] = field_match.group(1).strip()
        
        # Extract location criteria
        location_match = re.search(r'\bin\s+([A-Za-z\s]+?)\b(?:\s|$)', query.lower())
        if location_match and 'theme' not in criteria and 'unit' not in criteria:
            location = location_match.group(1).strip()
            if location not in ['aphrc', 'the', 'health', 'research', 'center', 'centre']:
                criteria['location'] = location
        
        # Check for list request
        is_list_request = bool(re.search(r'\b(list|show|all|many)\b.*?\b(expert|researcher|scientist|specialist)', query.lower()))
        if is_list_request:
            criteria['is_list'] = 'true'
        
        return criteria

    def format_knowledge_expertise(self, expert: Dict[str, Any], max_items: int = 3) -> str:
        """
        Extract and format knowledge expertise from expert data with rich fallbacks.
        
        Args:
            expert (Dict[str, Any]): Expert dictionary with fields from database
            max_items (int): Maximum number of expertise items to include
            
        Returns:
            str: Formatted knowledge expertise string
        """
        # Try knowledge_expertise field first (preferred source)
        if expert.get('knowledge_expertise'):
            expertise = expert['knowledge_expertise']
            
            # If it's a string with commas, split and format
            if isinstance(expertise, str) and ',' in expertise:
                items = [item.strip() for item in expertise.split(',')]
                return ' | '.join(items[:max_items])
            
            # If it's a list, format it
            elif isinstance(expertise, list):
                items = [str(item).strip() for item in expertise if item]
                return ' | '.join(items[:max_items])
            
            # Otherwise, use as is
            return str(expertise)
        
        # Try expertise field next (second choice)
        if expert.get('expertise'):
            try:
                expertise = expert['expertise']
                
                # If it's a JSON string, parse it
                if isinstance(expertise, str) and (expertise.startswith('{') or expertise.startswith('[')):
                    expertise = json.loads(expertise)
                
                # Handle dict format
                if isinstance(expertise, dict):
                    # Extract values
                    values = []
                    for v in expertise.values():
                        if isinstance(v, list):
                            values.extend([str(item) for item in v if item])
                        elif v:
                            values.append(str(v))
                    return ' | '.join(values[:max_items])
                
                # Handle list format
                elif isinstance(expertise, list):
                    return ' | '.join([str(item) for item in expertise[:max_items] if item])
                
                # Otherwise, use as string
                return str(expertise)
            except:
                # If parsing fails, use as string
                return str(expertise)
        
        # Try normalized_skills field (third choice)
        if expert.get('normalized_skills'):
            try:
                skills = expert['normalized_skills']
                
                # If it's a JSON string, parse it
                if isinstance(skills, str) and (skills.startswith('[') or skills.startswith('{')):
                    skills = json.loads(skills)
                
                # Format as list
                if isinstance(skills, list):
                    return ' | '.join([str(item) for item in skills[:max_items] if item])
                
                return str(skills)
            except:
                return str(skills)
        
        # Try other fields as fallbacks
        for field in ['fields', 'domains', 'subfields', 'keywords']:
            if expert.get(field):
                try:
                    data = expert[field]
                    
                    # If it's a JSON string, parse it
                    if isinstance(data, str) and (data.startswith('[') or data.startswith('{')):
                        data = json.loads(data)
                    
                    # Format as list
                    if isinstance(data, list):
                        return ' | '.join([str(item) for item in data[:max_items] if item])
                    
                    return str(data)
                except:
                    continue
        
        # Last resort - check if bio exists and extract a relevant snippet
        if expert.get('bio'):
            bio = expert['bio']
            expertise_snippet = re.search(r'expertise in\s+([^.]+)', bio, re.IGNORECASE)
            if expertise_snippet:
                return expertise_snippet.group(1).strip()
        
        # If all else fails
        return "Health research"

    def expand_abbreviation(self, abbreviation: str, abbr_type: str = 'theme') -> str:
        """
        Expand common APHRC abbreviations for themes and units to their full names.
        
        Args:
            abbreviation (str): The abbreviation to expand
            abbr_type (str): Type of abbreviation ('theme' or 'unit')
            
        Returns:
            str: Expanded form or original string if not found
        """
        # Theme abbreviations
        theme_expansions = {
            'HAW': 'Health and Wellbeing',
            'SRMNCAH': 'Sexual, Reproductive, Maternal, Newborn, Child, and Adolescent Health',
            'UHP': 'Urban Health and Poverty',
            'ECD': 'Early Childhood Development',
            'PEC': 'Population, Environment, and Climate',
            'RSD': 'Research Systems Development',
            'HDSS': 'Health and Demographic Surveillance System',
            'MCH': 'Maternal and Child Health',
            'NCDA': 'Non-Communicable Diseases Alliance',
            'SRHR': 'Sexual and Reproductive Health and Rights'
        }
        
        # Unit abbreviations
        unit_expansions = {
            'SRMNCAH': 'Sexual, Reproductive, Maternal, Newborn, Child, and Adolescent Health',
            'RSD': 'Research Systems Development',
            'IDSSS': 'Innovations and Data Systems Support Services',
            'KDHS': 'Kenya Demographic Health Survey',
            'PMTCT': 'Prevention of Mother-to-Child Transmission',
            'SSA': 'Sub-Saharan Africa',
            'M&E': 'Monitoring and Evaluation',
            'HMIS': 'Health Management Information Systems',
            'HSS': 'Health Systems Strengthening'
        }
        
        # Select the appropriate expansion dictionary
        expansions = theme_expansions if abbr_type.lower() == 'theme' else unit_expansions
        
        # Try to find a match (case insensitive)
        for abbr, full_name in expansions.items():
            if abbreviation.upper() == abbr:
                return f"{abbreviation} ({full_name})"
        
        # No match found, return original
        return abbreviation

   

    def format_expert_context(self, experts: List[Dict[str, Any]]) -> str:
        """
        Format expert information into Markdown format for rendering in the frontend.
        
        Args:
            experts: List of expert dictionaries
            
        Returns:
            Formatted Markdown string with structured expert presentations
        """
        if not experts:
            return "I couldn't find any expert information on this topic. Would you like me to help you search for something else?"

        # Create header based on the number of experts
        markdown_text = "# Experts in Health Sciences at APHRC:\n\n" if len(experts) > 1 else "# Expert Profile:\n\n"

        for idx, expert in enumerate(experts):
            try:
                # Extract name components
                first_name = expert.get('first_name', '').strip()
                last_name = expert.get('last_name', '').strip()
                full_name = f"{first_name} {last_name}".strip()
                
                # Use numbered list with clear formatting
                markdown_text += f"{idx + 1}. **{full_name}**\n\n"

                # ADDED: Format and include designation (position/title)
                designation = expert.get('designation', '')
                if designation:
                    markdown_text += f"    - **Designation:** {designation}\n\n"
                
                # ADDED: Format and include theme with expansion
                theme = expert.get('theme', '')
                if theme:
                    # Expand common theme abbreviations
                    theme_expansion = {
                        'HAW': 'Health and Wellbeing',
                        'SRMNCAH': 'Sexual, Reproductive, Maternal, Newborn, Child, and Adolescent Health',
                        'UHP': 'Urban Health and Poverty',
                        'ECD': 'Early Childhood Development',
                        'PEC': 'Population, Environment, and Climate'
                    }
                    
                    theme_full = f"{theme} ({theme_expansion.get(theme, '')})" if theme in theme_expansion else theme
                    markdown_text += f"    - **Theme:** {theme_full}\n\n"
                
                # ADDED: Format and include unit with expansion
                unit = expert.get('unit', '')
                if unit:
                    # Expand common unit abbreviations
                    unit_expansion = {
                        'SRMNCAH': 'Sexual, Reproductive, Maternal, Newborn, Child, and Adolescent Health',
                        'RSD': 'Research Systems Development',
                        'IDSSS': 'Innovations and Data Systems Support Services',
                        'KDHS': 'Kenya Demographic Health Survey'
                    }
                    
                    unit_full = f"{unit} ({unit_expansion.get(unit, '')})" if unit in unit_expansion else unit
                    markdown_text += f"    - **Unit:** {unit_full}\n\n"

                # Display expertise with consistent formatting
                knowledge_expertise = expert.get('knowledge_expertise', '')
                if knowledge_expertise:
                    # Format knowledge expertise as a list if possible
                    if isinstance(knowledge_expertise, str) and ',' in knowledge_expertise:
                        expertise_items = [item.strip() for item in knowledge_expertise.split(',')][:3]  # Limit to top 3
                        markdown_text += f"    - **Knowledge & Expertise:** {' | '.join(expertise_items)}\n\n"
                    elif isinstance(knowledge_expertise, list):
                        expertise_items = [str(item).strip() for item in knowledge_expertise][:3]  # Limit to top 3
                        markdown_text += f"    - **Knowledge & Expertise:** {' | '.join(expertise_items)}\n\n"
                    else:
                        markdown_text += f"    - **Knowledge & Expertise:** {knowledge_expertise}\n\n"
                else:
                    # Try to get expertise from other fields if knowledge_expertise is not available
                    expertise_fields = expert.get('expertise', [])
                    if expertise_fields:
                        if isinstance(expertise_fields, list):
                            expertise_str = " | ".join(str(item) for item in expertise_fields[:3])  # Limit to top 3
                        elif isinstance(expertise_fields, dict):
                            expertise_str = " | ".join(str(v) for v in list(expertise_fields.values())[:3])  # Limit to top 3
                        else:
                            expertise_str = str(expertise_fields)
                        markdown_text += f"    - **Knowledge & Expertise:** {expertise_str}\n\n"

                # ADDED: Include bio if available (but truncated for readability)
                bio = expert.get('bio', '')
                if bio:
                    # Truncate long bio to about 100-150 words for readability
                    if len(bio) > 300:
                        bio_words = bio.split()
                        if len(bio_words) > 50:
                            bio = ' '.join(bio_words[:50]) + '...'
                    markdown_text += f"    - **Bio:** {bio}\n\n"

                # ADDED: Add notable publications with consistent indentation if available
                publications = expert.get('publications', [])
                if publications:
                    markdown_text += "    - **Notable publications:**\n\n"
                    for pub in publications[:2]:  # Limit to top 2 publications
                        pub_title = pub.get('title', 'Untitled')
                        pub_year = pub.get('publication_year', '')
                        year_text = f" ({pub_year})" if pub_year else ""
                        markdown_text += f"        - \"{pub_title}\"{year_text}\n\n"

                # Add an extra line break between experts for clear separation
                if idx < len(experts) - 1:
                    markdown_text += "\n"

            except Exception as e:
                logger.error(f"Error formatting expert {idx + 1}: {e}")
                continue

        # Add closing message
        markdown_text += "\nWould you like more detailed information about any of these experts? You can ask by name or area of expertise."
        return markdown_text
    def safely_decode_binary_data(self, binary_data, default_encoding='utf-8'):
        """
        Safely decode binary data using multiple encoding attempts.
        Addresses the 'utf-8' codec can't decode byte issues.
        
        Args:
            binary_data (bytes): The binary data to decode
            default_encoding (str): The default encoding to try first
            
        Returns:
            str: The decoded string or None if decoding fails
        """
        if not binary_data:
            return None
            
        if not isinstance(binary_data, bytes):
            return str(binary_data)
        
        # Try multiple encodings in order of likelihood
        encodings = [default_encoding, 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                return binary_data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all explicit encodings fail, use replace mode as last resort
        try:
            return binary_data.decode('latin-1', errors='replace')
        except Exception as e:
            logger.warning(f"All decoding attempts failed: {e}")
            return None
        
    async def generate_async_response(self, message: str, user_interests: str = "") -> AsyncGenerator[str, None]:
        """
        Generates a response with enhanced prompting strategies tailored to detected intent,
        improved context handling, tone guidance for more natural, engaging interactions,
        and personalization based on user interests.
        """
        try:
            # Import inspect if not already imported
            import inspect
            
            # Step 1: Intent Detection 
            intent_result = await self.detect_intent(message)
            intent = intent_result['intent']
            confidence = intent_result.get('confidence', 0.0)
            
            # Check if this is a list request
            is_list_request = intent_result.get('is_list_request', False)
            
            # Determine conversation style based on message analysis
            message_style = self._analyze_message_style(message)
            
            # Step 2: Context preparation based on intent
            context = ""
            context_type = "general"
            
            # Handle EXPERT Intent
            if intent == QueryIntent.EXPERT:
                # Adjust limit for list requests
                expert_limit = 5 if is_list_request else 3
                experts, error = await self.get_experts(message, limit=expert_limit)
                
                if experts:
                    # Check if we have a valid embedding model
                    if self.embedding_model:
                        try:
                            # Get query embedding
                            query_embedding = self.embedding_model.encode(message)
                            
                            # Only sort by similarity for non-list requests
                            if not is_list_request:
                                # Sort experts by similarity when both embeddings exist
                                ranked_experts = sorted(
                                    experts,
                                    key=lambda e: cos_sim(query_embedding, e['embedding'])[0][0] 
                                    if 'embedding' in e and e['embedding'] is not None else 0,
                                    reverse=True
                                )
                            else:
                                # For list requests, use original order or rank by match score
                                ranked_experts = sorted(
                                    experts,
                                    key=lambda e: e.get('match_score', 0),
                                    reverse=True
                                )
                        except Exception as e:
                            logger.warning(f"Error ranking experts: {e}")
                            ranked_experts = experts  # Fall back to original order
                    else:
                        # No embedding model available
                        ranked_experts = experts
                        
                    context = self.format_expert_context(ranked_experts)
                    context_type = "expert"
                else:
                    context = "No matching experts found, but I can still try to help with your query."
                    context_type = "no_experts"
            
            # Handle PUBLICATION Intent
            elif intent == QueryIntent.PUBLICATION:
                # Adjust limit for list requests
                pub_limit = 5 if is_list_request else 3
                publications, error = await self.get_publications(message, limit=pub_limit)
                if publications:
                    context = self.format_publication_context(publications)
                    context_type = "publication"
                else:
                    context = "No matching publications found, but I can still provide general information."
                    context_type = "no_publications"
            
            # Handle NAVIGATION Intent
            elif intent == QueryIntent.NAVIGATION:
                context = "I can help you navigate APHRC resources and website sections."
                context_type = "navigation"
            
            # Default context for GENERAL Intent
            else:
                context = "How can I assist you with APHRC research today?"
                context_type = "general"
            
            # Step 3: Stream Metadata
            yield json.dumps({'is_metadata': True, 'metadata': {'intent': intent.value, 'style': message_style, 'is_list_request': is_list_request}})
            
            # Step 4: Create tailored prompt with tone, style guidance, and user interests
            # FIXED: Check signature to determine correct method call
            # Get the signature of _create_tailored_prompt
            sig = inspect.signature(self._create_tailored_prompt)
            params = list(sig.parameters.keys())
            
            # Determine if the method accepts is_list_request
            if len(params) >= 5 and 'is_list_request' in params:
                # Method accepts is_list_request parameter
                system_message = self._create_tailored_prompt(intent, context_type, message_style, user_interests, is_list_request)
            else:
                # Use the version with only 4 parameters
                system_message = self._create_tailored_prompt(intent, context_type, message_style, user_interests)
            
            # Step 5: Prepare the full context with system guidance and user query
            full_prompt = f"{system_message}\n\nContext:\n{context}\n\nUser Query: {message}"
            
            # For list requests, add specific instructions
            if is_list_request:
                if intent == QueryIntent.EXPERT:
                    full_prompt += "\n\nThis is a request for a LIST of experts. Please format your response as a clear, numbered list with concise details about each expert."
                elif intent == QueryIntent.PUBLICATION:
                    full_prompt += "\n\nThis is a request for a LIST of publications. Please format your response as a clear, numbered list with concise details about each publication."
            
            # Step 6: Stream Enhanced Response from Gemini
            model = self._setup_gemini()
            response = model.generate_content(full_prompt, stream=True)
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'parts') and chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            yield part.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield "I apologize, but I encountered an issue while processing your request. Could you please rephrase your question or try again later?"

    def _create_tailored_prompt(self, intent, context_type: str, message_style: str, user_interests: str = "") -> str:
        """
        Create a tailored system prompt based on detected intent, context type, message style,
        and user interests when available.
        
        Args:
            intent: The detected intent from QueryIntent enum
            context_type (str): The type of context being provided
            message_style (str): The communication style detected in the user's message
            user_interests (str): Optional string describing user's previous interests
                
        Returns:
            str: A specialized system prompt for the LLM
        """
        # Base instructions for all responses (unchanged)
        base_instructions = """
        You are an assistant for the African Population and Health Research Center (APHRC).
        Your role is to provide helpful, accurate information about APHRC research, 
        publications, experts, and resources. Maintain a professional, supportive tone
        while being conversational and engaging.
        """
        
        # Style-specific tone guidance (unchanged)
        tone_guidance = {
            "technical": """
            Use precise language and academic terminology appropriate for researchers and professionals.
            Provide detailed information with proper citations when possible.
            Maintain clarity and accuracy while demonstrating deep subject matter expertise.
            """,
            
            "formal": """
            Use professional language with moderate formality.
            Be thorough and precise while remaining accessible to non-specialists.
            Maintain a respectful, informed tone appropriate for official communication.
            """,
            
            "conversational": """
            Use natural, engaging language that balances professionalism with approachability.
            Create a dialogue-like experience using conversational transitions.
            Be friendly and helpful while maintaining APHRC's professional authority.
            """
        }
        
        # Intent-specific response structures (unchanged)
        intent_guidance = {
            QueryIntent.PUBLICATION: """
            When discussing publications:
            - Highlight key findings and their significance
            - Explain the research context and relevance
            - Connect publications to broader themes in APHRC's work
            - Use natural transitions between publications
            - Suggest related publications when appropriate
            """,
            
            QueryIntent.EXPERT: """
            When discussing experts:
            - Emphasize their specific areas of expertise and contributions
            - Make connections between their work and the user's query
            - Provide context about their role at APHRC
            - Highlight notable publications or projects
            - Suggest how their expertise might be relevant to the user's interests
            """,
            
            QueryIntent.NAVIGATION: """
            When providing navigation assistance:
            - Give clear, step-by-step guidance
            - Explain what the user will find in each section
            - Provide context about why certain resources might be useful
            - Anticipate related information needs
            - Offer suggestions for exploring related content
            """,
            
            QueryIntent.GENERAL: """
            For general information about APHRC:
            - Provide balanced, comprehensive overviews
            - Highlight APHRC's mission and impact when relevant
            - Connect information to broader themes in public health and population research
            - Anticipate follow-up questions
            - Provide examples and context to make information accessible
            """
        }
        
        # Context-specific guidance for empty results (unchanged)
        context_specific = ""
        if context_type == "no_experts":
            context_specific = """
            Although no specific experts were found, provide helpful general information
            about APHRC's work in the relevant domain. Suggest broader research areas
            the user might explore and mention that you can help find information about
            specific experts if they provide more details.
            """
        elif context_type == "no_publications":
            context_specific = """
            Although no specific publications were found, provide helpful general information
            about APHRC's research in the relevant domain. Suggest broader topics
            the user might explore and mention that you can help find specific publications
            if they provide more details about their interests.
            """
        
        # NEW: Add user interest guidance if provided
        user_interest_guidance = ""
        if user_interests:
            user_interest_guidance = f"""
            This user has previously shown interest in:
            {user_interests}
            
            When appropriate, consider connecting your response to these interests.
            If you find related publications or experts that align with these interests,
            highlight those connections. However, always prioritize directly answering
            their current question.
            """
        
        # Combine appropriate guidance elements (modified to include user interests)
        selected_tone = tone_guidance.get(message_style, tone_guidance["conversational"])
        selected_intent = intent_guidance.get(intent, intent_guidance[QueryIntent.GENERAL])
        
        full_prompt = f"{base_instructions}\n\n{selected_tone}\n\n{selected_intent}\n\n{context_specific}\n\n{user_interest_guidance}"
        
        # Add guidance on structuring the response (unchanged)
        full_prompt += """
        Structure your response for clarity:
        - Begin with a direct, helpful answer to the query
        - Provide context and details in a logical flow
        - Use natural transitions between topics
        - Include specific examples when helpful
        - End with a courteous closing that invites further engagement
        """
        
        return full_prompt.strip()

    def _load_embedding_model(self):
        """Try loading embedding model from various locations."""
        model_paths = [
            '/app/models/sentence-transformers/all-MiniLM-L6-v2',
            './models/sentence-transformers/all-MiniLM-L6-v2',
            os.path.expanduser('~/models/sentence-transformers/all-MiniLM-L6-v2')
        ]
        for path in model_paths:
            try:
                logger.info(f"Attempting to load model from {path}")
                model = SentenceTransformer(path, device='cpu')
                # Test the model by encoding a small string
                test_embedding = model.encode("test")
                if len(test_embedding) > 0:
                    logger.info(f"Successfully loaded SentenceTransformer model from: {path}")
                    return model
                else:
                    logger.warning(f"Model at {path} returned empty vector")
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
        logger.warning("No local embedding model available")
        return None

    def _setup_remote_embeddings(self):
        """Setup fallback embedding options"""
        self.embedding_service = os.getenv('EMBEDDING_SERVICE_URL')
        if self.embedding_service:
            logger.info(f"Using remote embedding service at {self.embedding_service}")
        else:
            logger.warning("No embedding service available - some features will be limited")

    async def _check_rate_limit(self):
        """Check and handle rate limiting"""
        if self._rate_limited:
            elapsed = time.time() - self._last_rate_limit_time
            if elapsed < self._rate_limit_backoff:
                remaining = self._rate_limit_backoff - elapsed
                logger.warning(f"Rate limit active, waiting {remaining:.1f}s")
                await asyncio.sleep(remaining)
            self._rate_limited = False
            self._rate_limit_backoff = min(600, self._rate_limit_backoff * 2)

    async def _handle_rate_limit(self):
        """Handle rate limit response and set backoff"""
        self._rate_limited = True
        self._last_rate_limit_time = time.time()
        logger.warning(f"Rate limit encountered, backing off for {self._rate_limit_backoff}s")
        self.metrics['rate_limited'] += 1
    async def _safe_generate(self, message: str) -> AsyncGenerator[str, None]:
        """Protected generation with rate limit handling"""
        await self._check_rate_limit()
        try:
            model = self._setup_gemini()
            self.metrics['total_calls'] += 1
            
            response = model.generate_content(
                [{"role": "user", "parts": [{"text": message}]}],  # Fixed list syntax with closing bracket
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                )
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'parts') and chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            yield part.text
            
            self.metrics['last_success'] = time.time()
            
        except Exception as e:
            self.metrics['failed_calls'] += 1
            self.metrics['last_failure'] = time.time()
            if "429" in str(e) or "quota" in str(e).lower():
                await self._handle_rate_limit()
            raise


    async def _detect_intent_with_embeddings(self, message: str) -> Dict[str, Any]:
        try:
            if not self.embedding_model:
                raise ValueError("No embedding model available")
            
            # Clean and normalize the query
            cleaned_message = re.sub(r'[^\w\s]', ' ', message.lower()).strip()
            query_embedding = self.embedding_model.encode(cleaned_message)
            
            # Define example queries for each intent type
            intent_examples = {
                QueryIntent.PUBLICATION: [
                    "Show me publications about maternal health",
                    "What papers have been published on climate change",
                    "Research articles on education in Africa"
                ],
                QueryIntent.EXPERT: [
                    "Who are the experts in health policy",
                    "Find researchers working on climate change",
                    "Information about Dr. Johnson"
                ],
                QueryIntent.NAVIGATION: [
                    "How do I find the contact page",
                    "Where can I access research tools",
                    "Show me the about section"
                ]
            }
            
            # Compute similarity scores for each intent
            intent_scores = {}
            for intent, examples in intent_examples.items():
                example_embeddings = self.embedding_model.encode(examples)
                similarities = cos_sim(query_embedding, example_embeddings).numpy().flatten()
                max_similarity = max(similarities)
                intent_scores[intent] = float(max_similarity)
            
            # Determine the intent with the highest score
            max_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[max_intent]
            
            # Apply threshold to determine final intent
            if confidence < 0.6:
                max_intent = QueryIntent.GENERAL
            
            return {
                'intent': max_intent,
                'confidence': confidence,
                'clarification': None
            }
        except Exception as e:
            logger.error(f"Error in embedding intent detection: {e}")
            raise

    async def _detect_intent_with_gemini(self, message: str) -> Dict[str, Any]:
        """Detect intent using the Gemini model."""
        try:
            # Set up the Gemini model
            model = self._setup_gemini()
            
            # Create a structured prompt for intent detection
            prompt = f"""
            Analyze this query and classify its intent:
            Query: "{message}"
            Options:
            - PUBLICATION (research papers, studies)
            - EXPERT (researchers, specialists)
            - NAVIGATION (website sections, resources)
            - GENERAL (other queries)
            If asking about publications BY someone, extract the name.
            Return ONLY JSON in this format:
            {{
                "intent": "PUBLICATION|EXPERT|NAVIGATION|GENERAL",
                "confidence": 0.0-1.0,
                "clarification": "optional question",
                "expert_name": "name if detected"
            }}
            """
            
            # Generate content using Gemini
            response = await model.generate_content(prompt)  # Use generateContent for async
            content = response.text.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(content[json_start:json_end])
                
                # Map intent strings to QueryIntent enum
                intent_mapping = {
                    'PUBLICATION': QueryIntent.PUBLICATION,
                    'EXPERT': QueryIntent.EXPERT,
                    'NAVIGATION': QueryIntent.NAVIGATION,
                    'GENERAL': QueryIntent.GENERAL
                }
                intent_result = {
                    'intent': intent_mapping.get(result.get('intent', 'GENERAL'), QueryIntent.GENERAL),
                    'confidence': result.get('confidence', 0.0),
                    'clarification': result.get('clarification', None)
                }
                logger.info(f"Gemini intent detection result: {intent_result['intent']} with confidence {intent_result['confidence']}")
                return intent_result
            else:
                logger.warning(f"Could not extract valid JSON from response: {content}")
                return {'intent': QueryIntent.GENERAL, 'confidence': 0.0, 'clarification': None}
        except Exception as e:
            logger.error(f"Gemini intent detection failed: {e}")
            if "429" in str(e):  # Handle rate-limiting errors
                await self._handle_rate_limit()
            return {'intent': QueryIntent.GENERAL, 'confidence': 0.0, 'clarification': None}

    def _detect_intent_with_keywords(self, message: str) -> Dict[str, Any]:
        """
        Detect intent using keyword pattern matching instead of embedding models.
        
        Args:
            message: The user's query message
            
        Returns:
            Dictionary with intent type, confidence score, and optional clarification
        """
        try:
            # Clean and normalize the query
            cleaned_query = message.lower().strip()
            
            # Define patterns for different intent types with confidence weights
            patterns = {
                QueryIntent.PUBLICATION: {
                    'high_confidence': [
                        r'publication', r'paper', r'research paper', r'article', r'journal', 
                        r'study (on|about)', r'research (on|about)', r'published', r'doi'
                    ],
                    'medium_confidence': [
                        r'study', r'research', r'findings', r'results', r'conclusion', 
                        r'abstract', r'methodology', r'data', r'analysis'
                    ]
                },
                QueryIntent.EXPERT: {
                    'high_confidence': [
                        r'expert', r'researcher', r'scientist', r'professor', r'dr\.', 
                        r'specialist', r'author', r'lead', r'pi ', r'principal investigator'
                    ],
                    'medium_confidence': [
                        r'who (is|are|was|were)', r'person', r'people', r'team', r'group', 
                        r'department', r'faculty', r'staff', r'work(s|ed|ing) on'
                    ]
                },
                QueryIntent.NAVIGATION: {
                    'high_confidence': [
                        r'website', r'page', r'section', r'find', r'where (is|are|can)', 
                        r'how (do|can) i (find|get to|access)', r'link', r'url', r'navigate'
                    ],
                    'medium_confidence': [
                        r'resources', r'tools', r'services', r'information', r'contact', 
                        r'about', r'help', r'support', r'faq'
                    ]
                }
            }
            
            # Calculate confidence scores for each intent
            scores = {}
            for intent, pattern_groups in patterns.items():
                score = 0.0
                
                # Check high confidence patterns (match = 0.9)
                for pattern in pattern_groups.get('high_confidence', []):
                    if re.search(pattern, cleaned_query, re.IGNORECASE):
                        score = max(score, 0.9)
                        break
                        
                # If no high confidence match, check medium confidence patterns (match = 0.7)
                if score < 0.7:
                    for pattern in pattern_groups.get('medium_confidence', []):
                        if re.search(pattern, cleaned_query, re.IGNORECASE):
                            score = max(score, 0.7)
                            break
                
                scores[intent] = score
            
            # Determine the intent with the highest confidence
            max_score = 0.0
            max_intent = QueryIntent.GENERAL
            
            for intent, score in scores.items():
                if score > max_score:
                    max_score = score
                    max_intent = intent
            
            # Extract potential expert name if expert intent is detected
            expert_name = None
            if max_intent == QueryIntent.EXPERT and max_score > 0.7:
                # Try to extract the expert name using regex patterns
                name_patterns = [
                    r'(by|from|about) ([A-Z][a-z]+ [A-Z][a-z]+)',  # Match "by John Smith"
                    r'([A-Z][a-z]+ [A-Z][a-z]+)\'s (publications|research|papers|work)',  # Match "John Smith's publications"
                    r'(Dr\.|Professor) ([A-Z][a-z]+ [A-Z][a-z]+)'  # Match "Dr. John Smith"
                ]
                
                for pattern in name_patterns:
                    match = re.search(pattern, message, re.IGNORECASE)
                    if match:
                        # Extract the name from the appropriate match group
                        if len(match.groups()) > 1:
                            expert_name = match.group(2)
                        else:
                            expert_name = match.group(1)
                        break
            
            # Generate appropriate clarification if needed
            clarification = None
            if max_score < 0.6:
                if max_intent == QueryIntent.PUBLICATION:
                    clarification = "Could you specify which publication or research topic you're interested in?"
                elif max_intent == QueryIntent.EXPERT:
                    clarification = "Are you looking for information about a specific researcher or expert?"
                elif max_intent == QueryIntent.NAVIGATION:
                    clarification = "Which part of our website or resources are you trying to access?"
            
            # Build result dictionary
            result = {
                'intent': max_intent if max_score >= 0.5 else QueryIntent.GENERAL,
                'confidence': max_score,
                'clarification': clarification
            }
            
            # Add expert name if found
            if expert_name:
                result['expert_name'] = expert_name
                
            logger.debug(f"Keyword intent detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in keyword intent detection: {e}")
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }

    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """Enhanced intent detection with fallbacks"""
        # Try with embeddings first
        if self.embedding_model:
            try:
                return await self._detect_intent_with_embeddings(message)
            except Exception as e:
                logger.warning(f"Embedding intent detection failed: {e}")
        
        # Fallback to keywords
        keyword_result = self._detect_intent_with_keywords(message)
        if keyword_result['confidence'] > 0.7:
            return keyword_result
            
        # Final Gemini fallback
        try:
            return await self._detect_intent_with_gemini(message)
        except Exception as e:
            logger.error(f"All intent detection methods failed: {e}")
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }

    
    async def _get_all_publication_keys(self):
        """Helper method to get all publication keys from Redis with consistent patterns."""
        try:
            # Only use new pattern
            patterns = [
                'meta:expert_resource:*'  # New pattern for expert resources
            ]
            
            all_keys = []
            
            # For each pattern, scan Redis for matching keys
            for pattern in patterns:
                cursor = 0
                pattern_keys = []
                
                while cursor != 0 or len(pattern_keys) == 0:
                    try:
                        cursor, batch = self.redis_manager.redis_text.scan(
                            cursor=cursor, 
                            match=pattern, 
                            count=100
                        )
                        pattern_keys.extend(batch)
                        
                        if cursor == 0:
                            break
                    except Exception as scan_error:
                        logger.warning(f"Error scanning with pattern '{pattern}': {scan_error}")
                        break
                
                logger.info(f"Found {len(pattern_keys)} keys with pattern '{pattern}'")
                all_keys.extend(pattern_keys)
            
            # Remove any duplicates
            unique_keys = list(set(all_keys))
            
            logger.info(f"Found {len(unique_keys)} total unique publication keys in Redis")
            return unique_keys
            
        except Exception as e:
            logger.error(f"Error retrieving publication keys: {e}")
            return []

    # 3. Update get_relevant_experts method to better handle empty results


   
    
    def _setup_gemini(self):
        """Set up and configure the Gemini model."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            
            # Configure the Gemini API client
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')  # Use the correct model name
            
            logger.info("Gemini model setup completed")
            return model
        except Exception as e:
            logger.error(f"Error setting up Gemini model: {e}")
            raise

    
    

    
   
            
    async def _reset_rate_limited_after(self, seconds: int):
        """Reset rate limited flag after specified seconds."""
        await asyncio.sleep(seconds)
        self._rate_limited = False
        logger.info(f"Rate limit cooldown expired after {seconds} seconds")

    

    def format_expert_with_publications(self, expert: Dict[str, Any]) -> str:
        """
        Format a single expert with their publications for presentation.
        
        Args:
            expert: Expert dictionary with publications included
            
        Returns:
            Formatted string with expert information and their publications
        """
        if not expert:
            return "No expert information available."
        
        # Build the expert context
        full_name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
        
        context_parts = [f"Expert information for {full_name}:"]
        
        # Expert basic info
        expert_info = [f"**{full_name}**"]
        
        # Add position and department
        position = expert.get('position', '')
        if position:
            expert_info.append(f"* Position: {position}")
            
        department = expert.get('department', '')
        if department:
            expert_info.append(f"* Department: {department}")
        
        # MODIFIED: Add knowledge_expertise instead of email
        knowledge_expertise = expert.get('knowledge_expertise', '')
        if knowledge_expertise:
            expert_info.append(f"* Knowledge Expertise: {knowledge_expertise}")
        
        # Add expertise areas as bullet points
        expertise = expert.get('expertise', [])
        if expertise and isinstance(expertise, list):
            expert_info.append("* Areas of expertise:")
            for area in expertise:
                expert_info.append(f"  * {area}")
        elif expertise:
            expert_info.append(f"* Expertise: {expertise}")
        
        # Add research interests
        research_interests = expert.get('research_interests', [])
        if research_interests and isinstance(research_interests, list):
            expert_info.append("* Research interests:")
            for interest in research_interests:
                expert_info.append(f"  * {interest}")
        
        # REMOVED: Email contact information section
        
        # Add expert bio if available
        bio = expert.get('bio', '')
        if bio:
            # Truncate long bios
            if len(bio) > 300:
                bio = bio[:297] + "..."
            expert_info.append(f"* Bio: {bio}")
        
        # Add the expert info section
        context_parts.append("\n".join(expert_info))
        
        # Add publications section with clear formatting
        publications = expert.get('publications', [])
        if publications:
            pub_section = [f"\nPublications by {full_name}:"]
            
            for idx, pub in enumerate(publications):
                title = pub.get('title', 'Untitled')
                pub_info = [f"{idx+1}. **{title}**"]
                
                # Publication year
                year = pub.get('publication_year', '')
                if year:
                    pub_info.append(f"* Published: {year}")
                
                # DOI if available
                doi = pub.get('doi', '')
                if doi:
                    pub_info.append(f"* DOI: {doi}")
                
                # Authors
                authors = pub.get('authors', [])
                if authors:
                    if isinstance(authors, list):
                        # Format author list
                        if len(authors) > 3:
                            author_text = f"{', '.join(str(a) for a in authors[:3])} et al."
                        else:
                            author_text = f"{', '.join(str(a) for a in authors)}"
                        pub_info.append(f"* Authors: {author_text}")
                    else:
                        pub_info.append(f"* Authors: {authors}")
                
                # Abstract snippet
                abstract = pub.get('abstract', '')
                if abstract:
                    # Truncate long abstracts
                    if len(abstract) > 250:
                        abstract = abstract[:247] + "..."
                    pub_info.append(f"* Abstract: {abstract}")
                
                pub_section.append("\n".join(pub_info))
            
            # Add all publications
            context_parts.append("\n\n".join(pub_section))
        else:
            context_parts.append(f"\nNo publications found for {full_name}.")
        
        # Combine all parts
        return "\n\n".join(context_parts)

    
   
    async def analyze_quality(self, message: str, response: str = "") -> Dict:
        """
        Analyze the quality of a response with enhanced hallucination detection for publications.
        
        Args:
            message (str): The user's original query
            response (str): The chatbot's response to analyze (if available)
        
        Returns:
            Dict: Quality metrics including helpfulness, hallucination risk, and factual grounding
        """
        try:
            # Limit analysis during high load periods or if a rate limit was recently hit
            if hasattr(self, '_rate_limited') and self._rate_limited:
                logger.warning("Skipping quality analysis due to recent rate limit")
                return self._get_default_quality()
            
            # Check if query relates to publications
            is_publication_query = False
            publication_patterns = [
                r'publications?', r'papers?', r'research', r'articles?', 
                r'studies', r'doi', r'authors?', r'published'
            ]
            if any(re.search(pattern, message.lower()) for pattern in publication_patterns):
                is_publication_query = True
                logger.info("Detected publication-related query - applying enhanced quality checks")
            
            # If no response provided, analyze the query only
            if not response:
                prompt = f"""Analyze this query for an APHRC chatbot and return a JSON object with quality expectations.
                The chatbot helps users find publications and navigate APHRC resources.
                Return ONLY the JSON object with no markdown formatting, no code blocks, and no additional text.
                
                Required format:
                {{
                    "helpfulness_score": <float between 0 and 1, representing expected helpfulness>,
                    "hallucination_risk": <float between 0 and 1, representing risk based on query complexity>,
                    "factual_grounding_score": <float between 0 and 1, representing how much factual knowledge is needed>,
                    "unclear_elements": [<array of strings representing potential unclear aspects of the query>],
                    "potentially_fabricated_elements": []
                }}
                
                Query to analyze: {message}
                """
            else:
                # For publication queries, add specific checks for hallucination
                if is_publication_query:
                    prompt = f"""Analyze the quality of this chatbot response about APHRC publications and return a JSON object.
                    Focus especially on detecting potential fabrication of publication details.
                    Return ONLY the JSON object with no markdown formatting, no code blocks, and no additional text.
                    
                    Required format:
                    {{
                        "helpfulness_score": <float between 0 and 1>,
                        "hallucination_risk": <float between 0 and 1>,
                        "factual_grounding_score": <float between 0 and 1>,
                        "unclear_elements": [<array of strings representing unclear aspects of the response>],
                        "potentially_fabricated_elements": [<array of specific publication details that may be fabricated>]
                    }}
                    
                    Pay special attention to:
                    1. Publication titles that seem generic or made-up
                    2. Author names
                    3. DOIs that don't follow standard formats (e.g., missing "10." prefix)
                    4. Specific years or dates mentioned without context
                    5. Publication findings or conclusions stated without reference to actual documents
                    
                    Flag any specific sections of the response that appear fabricated or hallucinated.
                    
                    User query: {message}
                    
                    Chatbot response: {response}
                    """
                else:
                    # For non-publication queries, use standard analysis
                    prompt = f"""Analyze the quality of this chatbot response for the given query and return a JSON object.
                    The APHRC chatbot helps users find publications and navigate APHRC resources.
                    Evaluate helpfulness, factual accuracy, and potential hallucination.
                    Return ONLY the JSON object with no markdown formatting, no code blocks, and no additional text.
                    
                    Required format:
                    {{
                        "helpfulness_score": <float between 0 and 1>,
                        "hallucination_risk": <float between 0 and 1>,
                        "factual_grounding_score": <float between 0 and 1>,
                        "unclear_elements": [<array of strings representing unclear aspects of the response>],
                        "potentially_fabricated_elements": [<array of strings representing statements that may be hallucinated>]
                    }}
                    
                    User query: {message}
                    
                    Chatbot response: {response}
                    """
            
            # Use model with retry logic already built in from get_gemini_model()
            try:
                model = self.get_gemini_model()
                
                # Call the model and get response content safely
                try:
                    # Handle async or sync invoke results properly
                    model_response = await model.generate_content(prompt)
                    response_text = self._extract_content_safely(model_response)
                    
                    # Reset any rate limit flag if successful
                    if hasattr(self, '_rate_limited'):
                        self._rate_limited = False
                    
                    if not response_text:
                        logger.warning("Empty response from quality analysis model")
                        return self._get_default_quality()
                    
                    cleaned_response = response_text.strip()
                    cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
                    
                    try:
                        quality_data = json.loads(cleaned_response)
                        logger.info(f"Response quality analysis result: {quality_data}")
                        
                        # For publication queries, add an additional check for post-processing action
                        if is_publication_query and 'potentially_fabricated_elements' in quality_data:
                            if quality_data['potentially_fabricated_elements']:
                                logger.warning(f"Detected potentially fabricated publication elements: {quality_data['potentially_fabricated_elements']}")
                                # Flag this response for review or intervention
                                quality_data['requires_review'] = True
                                
                        return quality_data
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse quality analysis response: {cleaned_response}")
                        logger.error(f"JSON parse error: {e}")
                        return self._get_default_quality()
                        
                except Exception as api_error:
                    # Check if this was a rate limit error
                    if any(x in str(api_error).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
                        logger.warning("Rate limit detected in quality analysis, marking for cooldown")
                        self._rate_limited = True
                        # Set expiry for rate limit status
                        asyncio.create_task(self._reset_rate_limited_after(300))  # 5 minutes cooldown
                    
                    logger.error(f"API error in quality analysis: {api_error}")
                    return self._get_default_quality()
                        
            except Exception as e:
                logger.error(f"Error in quality analysis: {e}")
                return self._get_default_quality()

        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return self._get_default_quality()

    def _extract_content_safely(self, model_response):
        """
        Safely extract content from model response with multiple format support.
        
        Args:
            model_response: The response from the model which could be in various formats
            
        Returns:
            str: The extracted content text, or empty string if extraction fails
        """
        try:
            # Case 1: Direct string
            if isinstance(model_response, str):
                return model_response
                
            # Case 2: Object with content attribute (like AIMessage)
            if hasattr(model_response, 'content'):
                return model_response.content
                
            # Case 3: List of messages
            if isinstance(model_response, list):
                # Try to get the last message
                if model_response and hasattr(model_response[-1], 'content'):
                    return model_response[-1].content
                # Try to join all message contents
                contents = []
                for msg in model_response:
                    if hasattr(msg, 'content'):
                        contents.append(msg.content)
                if contents:
                    return ''.join(contents)
                    
            # Case 4: Dictionary with content key
            if isinstance(model_response, dict) and 'content' in model_response:
                return model_response['content']
                
            # Case 5: Generation object from LangChain
            if hasattr(model_response, 'generations'):
                generations = model_response.generations
                if generations and generations[0]:
                    if hasattr(generations[0][0], 'text'):
                        return generations[0][0].text
            
            # Case 6: Object with text attribute
            if hasattr(model_response, 'text'):
                return model_response.text
                
            # If we can't determine the format, convert to string safely
            return str(model_response)
            
        except Exception as e:
            logger.error(f"Error extracting content from model response: {e}")
            return ""

    def _get_default_quality(self) -> Dict:
        """Return default quality metric values with publication-specific flags."""
        return {
            'helpfulness_score': 0.5,
            'hallucination_risk': 0.5,
            'factual_grounding_score': 0.5,
            'unclear_elements': [],
            'potentially_fabricated_elements': [],
            'requires_review': False
        }
    
    # 

    # This maintains backwards compatibility with code still calling analyze_sentiment
    async def analyze_sentiment(self, message: str) -> Dict:
        """
        Legacy method maintained for backwards compatibility.
        Now redirects to analyze_quality.
        """
        logger.warning("analyze_sentiment is deprecated, using analyze_quality instead")
        quality_data = await self.analyze_quality(message)
        
        # Transform quality data to match the expected sentiment structure
        # This ensures old code expecting sentiment data continues to work
        return {
            'sentiment_score': quality_data.get('helpfulness_score', 0.5) * 2 - 1,  # Map 0-1 to -1-1
            'emotion_labels': [],
            'confidence': 1.0 - quality_data.get('hallucination_risk', 0.5),
            'aspects': {
                'satisfaction': quality_data.get('helpfulness_score', 0.5),
                'urgency': 0.5,
                'clarity': quality_data.get('factual_grounding_score', 0.5)
            }
        }
    
    

    def format_expert_with_publications(self, expert: Dict[str, Any]) -> str:
        """
        Format a single expert with their publications for presentation.
        
        Args:
            expert: Expert dictionary with publications included
            
        Returns:
            Formatted string with expert information and their publications
        """
        if not expert:
            return "No expert information available."
        
        # Build the expert context
        full_name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
        
        context_parts = [f"Expert information for {full_name}:"]
        
        # Expert basic info
        expert_info = [f"**{full_name}**"]
        
        # Add position and department
        position = expert.get('position', '')
        if position:
            expert_info.append(f"* Position: {position}")
            
        department = expert.get('department', '')
        if department:
            expert_info.append(f"* Department: {department}")
        
        # Add expertise areas as bullet points
        expertise = expert.get('expertise', [])
        if expertise and isinstance(expertise, list):
            expert_info.append("* Areas of expertise:")
            for area in expertise:
                expert_info.append(f"  * {area}")
        elif expertise:
            expert_info.append(f"* Expertise: {expertise}")
        
        # Add research interests
        research_interests = expert.get('research_interests', [])
        if research_interests and isinstance(research_interests, list):
            expert_info.append("* Research interests:")
            for interest in research_interests:
                expert_info.append(f"  * {interest}")
        
        # Add contact info
        email = expert.get('email', '')
        if email:
            expert_info.append(f"* Contact: {email}")
        
        # Add expert bio if available
        bio = expert.get('bio', '')
        if bio:
            # Truncate long bios
            if len(bio) > 300:
                bio = bio[:297] + "..."
            expert_info.append(f"* Bio: {bio}")
        
        # Add the expert info section
        context_parts.append("\n".join(expert_info))
        
        # Add publications section with clear formatting
        publications = expert.get('publications', [])
        if publications:
            pub_section = [f"\nPublications by {full_name}:"]
            
            for idx, pub in enumerate(publications):
                title = pub.get('title', 'Untitled')
                pub_info = [f"{idx+1}. **{title}**"]
                
                # Publication year
                year = pub.get('publication_year', '')
                if year:
                    pub_info.append(f"* Published: {year}")
                
                # DOI if available
                doi = pub.get('doi', '')
                if doi:
                    pub_info.append(f"* DOI: {doi}")
                
                # Authors
                authors = pub.get('authors', [])
                if authors:
                    if isinstance(authors, list):
                        # Format author list
                        if len(authors) > 3:
                            author_text = f"{', '.join(str(a) for a in authors[:3])} et al."
                        else:
                            author_text = f"{', '.join(str(a) for a in authors)}"
                        pub_info.append(f"* Authors: {author_text}")
                    else:
                        pub_info.append(f"* Authors: {authors}")
                
                # Abstract snippet
                abstract = pub.get('abstract', '')
                if abstract:
                    # Truncate long abstracts
                    if len(abstract) > 250:
                        abstract = abstract[:247] + "..."
                    pub_info.append(f"* Abstract: {abstract}")
                
                # Confidence score if available (for debugging, remove in production)
                # confidence = pub.get('confidence', None)
                # if confidence is not None:
                #     pub_info.append(f"* Confidence: {confidence:.2f}")
                
                pub_section.append("\n".join(pub_info))
            
            # Add all publications
            context_parts.append("\n\n".join(pub_section))
        else:
            context_parts.append(f"\nNo publications found for {full_name}.")
        
        # Combine all parts
        return "\n\n".join(context_parts)




    def create_context(self, relevant_data: List[Dict]) -> str:
        """
        Create a flowing context narrative from relevant content.
        """
        if not relevant_data:
            return ""
        
        navigation_content = []
        publication_content = []
        
        for item in relevant_data:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            content_type = metadata.get('type', 'unknown')
            
            if content_type == 'navigation':
                navigation_content.append(
                    f"The {metadata.get('title', 'section')} of our website ({metadata.get('url', '')}) "
                    f"provides information about {text[:300].strip()}..."
                )
            
            elif content_type == 'publication':
                authors = metadata.get('authors', 'our researchers')
                date = metadata.get('date', '')
                date_text = f" in {date}" if date else ""
                
                publication_content.append(
                    f"In a study published{date_text}, {authors} explored {metadata.get('title', 'research')}. "
                    f"Their work revealed that {text[:300].strip()}..."
                )
        
        context_parts = []
        
        if navigation_content:
            context_parts.append(
                "Regarding our online resources: " + 
                " ".join(navigation_content)
            )
        
        if publication_content:
            context_parts.append(
                "Our research has produced several relevant findings: " + 
                " ".join(publication_content)
            )
        
        return "\n\n".join(context_parts)

    def manage_context_window(self, new_context: Dict):
        """Manage sliding window of conversation context."""
        current_time = datetime.now().timestamp()
        
        # Remove expired contexts
        self.context_window = [
            ctx for ctx in self.context_window 
            if current_time - ctx.get('timestamp', 0) < self.context_expiry
        ]
        
        # Add new context
        new_context['timestamp'] = current_time
        self.context_window.append(new_context)
        
        # Maintain maximum window size
        if len(self.context_window) > self.max_context_items:
            self.context_window.pop(0)

    
    async def get_publications(self, query: str = None, expert_id: str = None, limit: int = 3) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieves publications (works) either by general search or for a specific expert (author) using pre-stored links.
        """
        try:
            if not self.redis_manager:
                return [], "Database unavailable"
                
            publications = []
            
            # Expert-specific publication retrieval
            if expert_id:
                # Using new key pattern for author-work links
                links_key = f"author:{expert_id}:works"
                
                if self.redis_manager.redis_text.exists(links_key):
                    # Get all works for this author
                    resource_ids = self.redis_manager.redis_text.smembers(links_key)
                    
                    for resource_id in resource_ids:
                        meta_key = f"meta:work:{resource_id}"
                        if self.redis_manager.redis_text.exists(meta_key):
                            pub = self.redis_manager.redis_text.hgetall(meta_key)
                            if pub:
                                publications.append(pub)
                                # Limit results if we have enough
                                if len(publications) >= limit:
                                    break
            
            # General search
            else:
                cursor = 0
                while len(publications) < limit and (cursor != 0 or not publications):
                    # Updated pattern to match the new key structure
                    cursor, keys = self.redis_manager.redis_text.scan(
                        cursor, match='meta:work:*', count=limit*3)
                    
                    for key in keys:
                        pub = self.redis_manager.redis_text.hgetall(key)
                        if not query or self._publication_matches_query(pub, query):
                            publications.append(pub)
                            
                    if cursor == 0 or len(publications) >= limit:
                        break
            
            return publications[:limit], None
            
        except Exception as e:
            logger.error(f"Error in get_publications: {e}")
            return [], str(e)
    def _publication_matches_query(self, publication: Dict, query: str) -> bool:
        """
        Check if a publication matches the given query.
        Simplified version that searches common fields.
        """
        if not query:
            return True  # Return all publications if no query specified
        
        # Convert query to lowercase for case-insensitive matching
        query = query.lower().strip()
        
        # Fields to search
        fields_to_search = ['title', 'abstract', 'doi']
        
        # Check each field
        for field in fields_to_search:
            value = publication.get(field, '')
            if value and query in value.lower():
                return True
        
        # Check authors
        authors_json = publication.get('authors', '[]')
        try:
            authors = json.loads(authors_json) if authors_json else []
            if any(query in str(author).lower() for author in authors):
                return True
        except:
            # If authors can't be parsed as JSON, try string match
            if query in str(authors_json).lower():
                return True
        
        return False

    

    def _format_expert_profile(self, expert: Dict) -> str:
        """Formats an expert profile WITHOUT publications."""
        # MODIFIED: Replace email with expertise
        expertise = expert.get('knowledge_expertise', '')
        if not expertise:
            # Fall back to expertise field if knowledge_expertise is not available
            expertise_list = expert.get('expertise', [])
            if expertise_list:
                if isinstance(expertise_list, list):
                    expertise = ", ".join(expertise_list)
                else:
                    expertise = str(expertise_list)
        
        return f"""
        **Expert Profile: {expert.get('first_name')} {expert.get('last_name')}**
        - Position: {expert.get('position', 'N/A')}
        - Expertise: {expertise}
        """


    

    async def _enrich_experts_with_publications(self, experts: List[Dict[str, Any]], limit_per_expert: int = 2) -> List[Dict[str, Any]]:
        """
        Enhanced method to enrich experts with publications from targeted indexing.
        
        Improvements:
        - Prioritizes publications from expert-resource links
        - Maintains existing enrichment logic
        - Ensures efficient publication retrieval
        - ADDED: Ensures knowledge_expertise is included
        """
        if not experts:
            return experts
        
        try:
            for expert in experts:
                expert_id = expert.get('id')
                if not expert_id:
                    continue
                
                # ADDED: Ensure knowledge_expertise is present
                if 'knowledge_expertise' not in expert or not expert['knowledge_expertise']:
                    # Try to extract from expertise or other fields
                    if expert.get('expertise'):
                        try:
                            expertise_data = json.loads(expert['expertise']) if isinstance(expert['expertise'], str) else expert['expertise']
                            if isinstance(expertise_data, dict):
                                # Extract values as knowledge_expertise
                                values = []
                                for v in expertise_data.values():
                                    if isinstance(v, list):
                                        values.extend(v)
                                    else:
                                        values.append(v)
                                if values:
                                    expert['knowledge_expertise'] = ", ".join(str(v) for v in values[:3])
                            elif isinstance(expertise_data, list):
                                expert['knowledge_expertise'] = ", ".join(str(v) for v in expertise_data[:3])
                        except (json.JSONDecodeError, TypeError):
                            expert['knowledge_expertise'] = str(expert.get('expertise', 'Health research'))
                    else:
                        # Check other potential sources of expertise information
                        for field in ['research_areas', 'normalized_skills', 'domains', 'fields']:
                            if expert.get(field):
                                try:
                                    field_data = json.loads(expert[field]) if isinstance(expert[field], str) else expert[field]
                                    if isinstance(field_data, list) and field_data:
                                        expert['knowledge_expertise'] = ", ".join(str(item) for item in field_data[:3])
                                        break
                                except (json.JSONDecodeError, TypeError):
                                    if expert[field]:
                                        expert['knowledge_expertise'] = str(expert[field])
                                        break
                        
                        # If still nothing found, set a default
                        if 'knowledge_expertise' not in expert or not expert['knowledge_expertise']:
                            expert['knowledge_expertise'] = "Health research"
                
                # Skip if already has sufficient publications
                if expert.get('publications') and len(expert.get('publications', [])) >= limit_per_expert:
                    continue
                
                # Look for publications through expert-resource links
                links_key = f"links:expert:{expert_id}:resources"
                
                if self.redis_manager.redis_text.exists(links_key):
                    # Get resource IDs sorted by confidence
                    resource_items = self.redis_manager.redis_text.zrevrange(
                        links_key, 0, limit_per_expert-1, withscores=True
                    )
                    
                    publications = []
                    for resource_id, confidence in resource_items:
                        try:
                            # Retrieve publication metadata
                            meta_key = f"meta:resource:{resource_id}"
                            if self.redis_manager.redis_text.exists(meta_key):
                                meta = self.redis_manager.redis_text.hgetall(meta_key)
                                
                                if meta:
                                    publication = {
                                        'id': meta.get('id', ''),
                                        'title': meta.get('title', ''),
                                        'doi': meta.get('doi', ''),
                                        'abstract': meta.get('abstract', ''),
                                        'publication_year': meta.get('publication_year', ''),
                                        'confidence': float(confidence)
                                    }
                                    
                                    # Parse authors safely
                                    try:
                                        authors_json = meta.get('authors', '[]')
                                        publication['authors'] = json.loads(authors_json) if authors_json else []
                                    except json.JSONDecodeError:
                                        publication['authors'] = [authors_json] if authors_json else []
                                    
                                    publications.append(publication)
                        except Exception as pub_error:
                            logger.error(f"Error retrieving publication {resource_id}: {pub_error}")
                    
                    # Update expert publications
                    if publications:
                        publications.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                        expert['publications'] = publications[:limit_per_expert]
            
            return experts
        
        except Exception as e:
            logger.error(f"Error enriching experts with publications: {e}")
            return experts
    async def _enrich_publications_with_experts(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced method to enrich publications with expert information from targeted indexing.
        
        Improvements:
        - Prioritizes experts from publication-expert links
        - Maintains existing enrichment logic
        - Ensures efficient expert retrieval
        """
        if not publications:
            return publications
        
        try:
            for publication in publications:
                pub_id = publication.get('id')
                if not pub_id:
                    continue
                
                # Look for experts through resource-expert links
                links_key = f"links:resource:{pub_id}:experts"
                
                if self.redis_manager.redis_text.exists(links_key):
                    # Get expert IDs sorted by confidence
                    expert_items = self.redis_manager.redis_text.zrevrange(
                        links_key, 0, 5, withscores=True  # Top 5 experts
                    )
                    
                    experts = []
                    for expert_id, confidence in expert_items:
                        try:
                            # Retrieve expert metadata
                            meta_key = f"meta:expert:{expert_id}"
                            if self.redis_manager.redis_text.exists(meta_key):
                                raw_data = self.redis_manager.redis_text.hgetall(meta_key)
                                
                                if raw_data:
                                    expert_info = {
                                        'id': expert_id,
                                        'first_name': raw_data.get('first_name', ''),
                                        'last_name': raw_data.get('last_name', ''),
                                        'position': raw_data.get('position', ''),
                                        'department': raw_data.get('department', ''),
                                        'confidence': float(confidence)
                                    }
                                    
                                    # Parse JSON fields
                                    for field in ['expertise', 'research_interests']:
                                        try:
                                            if raw_data.get(field):
                                                expert_info[field] = json.loads(raw_data.get(field, '[]'))
                                            else:
                                                expert_info[field] = []
                                        except json.JSONDecodeError:
                                            expert_info[field] = []
                                    
                                    experts.append(expert_info)
                        except Exception as expert_error:
                            logger.error(f"Error retrieving expert {expert_id}: {expert_error}")
                    
                    # Enrich publication with expert information
                    if experts:
                        experts.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                        
                        # Add expert names for display
                        publication['aphrc_experts'] = [
                            f"{e.get('first_name', '')} {e.get('last_name', '')}".strip() 
                            for e in experts
                        ]
                        
                        # Add detailed expert information
                        publication['expert_details'] = experts[:3]  # Limit to top 3
            
            return publications
        
        except Exception as e:
            logger.error(f"Error enriching publications with experts: {e}")
            return publications

    def _safe_json_load(self, json_str: str) -> Any:
        """Safely load JSON string to Python object."""
        try:
            if not json_str:
                return {} if json_str.startswith('{') else []
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {} if json_str.startswith('{') else []
        except Exception:
            return {} if json_str.startswith('{') else []

    

    
    async def _throttle_request(self):
        """Apply throttling to incoming requests to help prevent rate limits."""
        await self.new_method()

    async def new_method(self):
        """Apply throttling to incoming requests to help prevent rate limits."""
        # Get global throttling status from class variable
        if not hasattr(self.__class__, '_last_request_time'):
            self.__class__._last_request_time = 0
            self.__class__._request_count = 0
            self.__class__._throttle_lock = asyncio.Lock()
            
        async with self.__class__._throttle_lock:
            current_time = time.time()
            time_since_last = current_time - self.__class__._last_request_time
                
            # Reset counter if more than 1 minute has passed
            if time_since_last > 60:
                self.__class__._request_count = 0
                    
            # Increment request counter
            self.__class__._request_count += 1
                
            # Calculate delay based on request count within the minute
            # As we approach Gemini's limits, add increasingly longer delays
            if self.__class__._request_count > 50:  # Getting close to limit
                delay = 2.0  # 2 seconds
            elif self.__class__._request_count > 30:
                delay = 1.0  # 1 second
            elif self.__class__._request_count > 20:
                delay = 0.5  # 0.5 seconds
            elif self.__class__._request_count > 10:
                delay = 0.2  # 0.2 seconds
            else:
                delay = 0
                    
            # Add randomization to prevent request bunching
            if delay > 0:
                jitter = delay * 0.2 * (random.random() * 2 - 1)  # ±20% jitter
                delay += jitter
                logger.debug(f"Adding throttling delay of {delay:.2f}s (request {self.__class__._request_count})")
                await asyncio.sleep(delay)
                    
            # Update last request time
            self.__class__._last_request_time = time.time()

class ConversationHelper:
    """New helper class for conversation enhancements"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.summary_cache = {}
        
    async def generate_summary(self, conversation_id: str, turns: List[Dict]) -> str:
        """Generate conversation summary"""
        if conversation_id in self.summary_cache:
            return self.summary_cache[conversation_id]
            
        key_points = []
        for turn in turns[-5:]:  # Last 5 turns
            if turn.get('intent') == QueryIntent.PUBLICATION:
                key_points.append(f"Discussed publications about {turn.get('topics', 'various topics')}")
            elif turn.get('intent') == QueryIntent.NAVIGATION:
                key_points.append(f"Asked about {turn.get('section', 'website sections')}")
        
        summary = "Our conversation so far:\n- " + "\n- ".join(key_points[-3:])  # Last 3 points
        self.summary_cache[conversation_id] = summary
        return summary
        
  