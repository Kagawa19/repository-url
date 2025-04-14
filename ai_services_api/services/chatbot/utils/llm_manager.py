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
                    # First try: Interpret as numpy binary data
                    return np.frombuffer(embedding_data, dtype=np.float32)
                except Exception as binary_err:
                    logger.debug(f"Could not interpret as binary float array: {binary_err}")
                    
                    # Second try: Decode as JSON
                    try:
                        # Try different encodings for JSON decoding
                        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                            try:
                                decoded = embedding_data.decode(encoding)
                                embedding_json = json.loads(decoded)
                                return np.array(embedding_json)
                            except UnicodeDecodeError:
                                # Try next encoding
                                continue
                            except json.JSONDecodeError:
                                # Valid encoding but not valid JSON, try next
                                continue
                    except Exception as json_err:
                        logger.debug(f"Failed to decode as JSON: {json_err}")
                    
                    # Last resort: Force decode with errors='replace'
                    try:
                        decoded = embedding_data.decode('utf-8', errors='replace')
                        cleaned = re.sub(r'[^\[\]\d\s,.-]', '', decoded)  # Keep only valid characters
                        if cleaned.startswith('[') and cleaned.endswith(']'):
                            array_data = json.loads(cleaned)
                            return np.array(array_data)
                    except Exception as err:
                        logger.warning(f"All embedding decoding methods failed: {err}")
                        return None
            
            # Case 3: JSON string
            if isinstance(embedding_data, str):
                # Try to parse as JSON
                try:
                    embedding_json = json.loads(embedding_data)
                    return np.array(embedding_json)
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Failed to parse embedding string as JSON: {json_err}")
                    return None
                    
            # Case 4: List or other iterable
            if hasattr(embedding_data, '__iter__'):
                return np.array(embedding_data)
                
            # Unhandled case
            logger.warning(f"Unhandled embedding data type: {type(embedding_data)}")
            return None
            
        except Exception as e:
            logger.warning(f"Error loading embedding: {e}")
            return None

    
    
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
            # Check if this is a "list experts" request - ADDED
            is_list_request = bool(re.search(r'\b(list|show|all|many)\b.*?\b(expert|researcher|scientist|specialist)', query.lower()))
            
            # Adjust limit for list requests - ADDED
            if is_list_request and limit < O:
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
                                            'list', 'show', 'all', 'many']:  # ADDED: Skip request indicators
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
                    
                    # MODIFIED: For list requests, give all experts a base score
                    if is_list_request:
                        match_score = 0.3  # Base score for list requests
                        matched_criteria.append("Included in expert list")
                    
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
                        except json.JSONDecodeError:
                            expert_expertise = {}
                            domains = []
                            fields = []
                            skills = []
                            keywords = []
                        
                        # Flatten expertise into a list of terms
                        expertise_values = []
                        if isinstance(expert_expertise, dict):
                            for values in expert_expertise.values():
                                if isinstance(values, list):
                                    expertise_values.extend([str(v).lower() for v in values if v])
                                elif values:
                                    expertise_values.append(str(values).lower())
                        
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
                    
                    # MODIFIED: Adjust threshold based on request type
                    # If we have any match OR this is a list request, prepare the expert data
                    match_threshold = 0.1 if is_list_request else 0.3  # Lower threshold for list requests
                    
                    if match_score > match_threshold:
                        logger.info(f"Expert {expert_id} matched with score {match_score:.2f}: {', '.join(matched_criteria[:3])}")
                        
                        # Get embedding from Redis
                        embedding_key = f"emb:aphrc_expert:{expert_id}"
                        embedding_data = None
                        
                        try:
                            if is_async_redis:
                                if hasattr(self.redis_manager.redis_text, 'aget'):
                                    embedding_data = await self.redis_manager.redis_text.aget(embedding_key)
                                else:
                                    embedding_data = await self.redis_manager.redis_text.get(embedding_key)
                            else:
                                embedding_data = self.redis_manager.redis_text.get(embedding_key)
                            
                            if embedding_data:
                                # Load binary embedding data
                                if isinstance(embedding_data, bytes):
                                    # Handle binary data
                                    arr = np.frombuffer(embedding_data, dtype=np.float32)
                                    expert['embedding'] = arr
                                else:
                                    # Handle JSON data
                                    expert['embedding'] = np.array(json.loads(embedding_data))
                        except Exception as emb_err:
                            logger.warning(f"Failed to load embedding for {expert_id}: {emb_err}")
                        
                        # Retrieve and add linked publications
                        links_key = f"links:expert:{expert_id}:resources"
                        if is_async_redis:
                            if hasattr(self.redis_manager.redis_text, 'azrevrange'):
                                resource_items = await self.redis_manager.redis_text.azrevrange(
                                    links_key, 0, 2, withscores=True
                                )
                            else:
                                resource_items = await self.redis_manager.redis_text.zrevrange(
                                    links_key, 0, 2, withscores=True
                                )
                        else:
                            resource_items = self.redis_manager.redis_text.zrevrange(
                                links_key, 0, 2, withscores=True
                            )
                        
                        publications = []
                        if resource_items:
                            for resource_id, confidence in resource_items:
                                resource_key = f"meta:resource:{resource_id}"
                                
                                # Fetch resource metadata
                                resource_meta = {}
                                try:
                                    if is_async_redis:
                                        if hasattr(self.redis_manager.redis_text, 'ahgetall'):
                                            resource_meta = await self.redis_manager.redis_text.ahgetall(resource_key)
                                        else:
                                            resource_meta = await self.redis_manager.redis_text.hgetall(resource_key)
                                    else:
                                        resource_meta = self.redis_manager.redis_text.hgetall(resource_key)
                                except Exception as res_err:
                                    logger.warning(f"Error fetching resource {resource_id}: {res_err}")
                                
                                if resource_meta:
                                    # Only include essential publication info
                                    publication = {
                                        'id': resource_id,
                                        'title': resource_meta.get('title', 'Untitled Publication'),
                                        'publication_year': resource_meta.get('publication_year', ''),
                                        'doi': resource_meta.get('doi', ''),
                                        'confidence': confidence
                                    }
                                    publications.append(publication)
                        
                        # Add publications to expert data
                        expert['publications'] = publications
                        
                        # Store match score for ranking
                        expert['match_score'] = match_score
                        expert['matched_criteria'] = matched_criteria
                        
                        # Add to results
                        experts.append(expert)
                
                except Exception as expert_err:
                    logger.warning(f"Error processing expert from key {key}: {expert_err}")
                    continue

            logger.info(f"Found {len(experts)} matching experts for query: {query}")
            
            # MODIFIED: For list requests with no results, return top experts instead of empty list
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
                            expert['publications'] = []
                            experts.append(expert)
                            count += 1
                    except Exception as e:
                        logger.warning(f"Error processing fallback expert: {e}")
                        continue
            
            # If we have too many matches, filter and rank them
            if len(experts) > limit:
                # Rank experts by multiple criteria
                if query_embedding is not None and not is_list_request:  # Skip complex ranking for list requests
                    try:
                        # For experts with embeddings, calculate semantic similarity
                        experts_with_embeddings = [e for e in experts if 'embedding' in e and e['embedding'] is not None]
                        
                        if experts_with_embeddings:
                            # Calculate semantic similarity scores
                            for expert in experts_with_embeddings:
                                similarity = float(cos_sim(query_embedding, expert['embedding'])[0][0])
                                # Boost match score with semantic similarity
                                expert['match_score'] = 0.6 * expert['match_score'] + 0.4 * similarity
                        
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
            # Step 1: Intent Detection 
            intent_result = await self.detect_intent(message)
            intent = intent_result['intent']
            confidence = intent_result.get('confidence', 0.0)
            
            # ADDED: Check if this is a list request
            is_list_request = intent_result.get('is_list_request', False)
            
            # Determine conversation style based on message analysis
            message_style = self._analyze_message_style(message)
            
            # Step 2: Context preparation based on intent
            context = ""
            context_type = "general"
            
            # Handle EXPERT Intent
            if intent == QueryIntent.EXPERT:
                # MODIFIED: Adjust limit for list requests
                expert_limit = 5 if is_list_request else 3
                experts, error = await self.get_experts(message, limit=expert_limit)
                
                if experts:
                    # Check if we have a valid embedding model
                    if self.embedding_model:
                        try:
                            # Get query embedding
                            query_embedding = self.embedding_model.encode(message)
                            
                            # ADDED: Only sort by similarity for non-list requests
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
            
            # Handle PUBLICATION Intent (unchanged)
            elif intent == QueryIntent.PUBLICATION:
                # MODIFIED: Adjust limit for list requests
                pub_limit = 5 if is_list_request else 3
                publications, error = await self.get_publications(message, limit=pub_limit)
                if publications:
                    context = self.format_publication_context(publications)
                    context_type = "publication"
                else:
                    context = "No matching publications found, but I can still provide general information."
                    context_type = "no_publications"
            
            # Handle NAVIGATION Intent (unchanged)
            elif intent == QueryIntent.NAVIGATION:
                context = "I can help you navigate APHRC resources and website sections."
                context_type = "navigation"
            
            # Default context for GENERAL Intent (unchanged)
            else:
                context = "How can I assist you with APHRC research today?"
                context_type = "general"
            
            # Step 3: Stream Metadata
            yield json.dumps({'is_metadata': True, 'metadata': {'intent': intent.value, 'style': message_style, 'is_list_request': is_list_request}})
            
            # Step 4: Create tailored prompt with tone, style guidance, and user interests
            # MODIFIED: Pass is_list_request to the prompt creation method
            system_message = self._create_tailored_prompt(intent, context_type, message_style, user_interests, is_list_request)
            
            # Step 5: Prepare the full context with system guidance and user query
            full_prompt = f"{system_message}\n\nContext:\n{context}\n\nUser Query: {message}"
            
            # ADDED: For list requests, add specific instructions
            if is_list_request:
                if intent == QueryIntent.EXPERT:
                    full_prompt += "\n\nThis is a request for a LIST of experts. Please format your response as a clear, numbered list with concise details about each expert."
                elif intent == QueryIntent.PUBLICATION:
                    full_prompt += "\n\nThis is a request for a LIST of publications. Please format your response as a clear, numbered list with concise details about each publication."
            
            # Step 6: Stream Enhanced Response from Gemini (unchanged)
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

    
        
    def format_expert_context(self, experts: List[Dict[str, Any]]) -> str:
        """
        Format expert information into a rich, conversational presentation with 
        natural language introductions and transitions between experts.
        
        Args:
            experts: List of expert dictionaries
            
        Returns:
            Formatted context string with engaging expert presentations
        """
        # UNCHANGED: Handle empty experts case
        if not experts:
            return "I couldn't find any expert information on this topic. Would you like me to help you search for something else?"
        
        # MODIFIED: Detect if this is likely a list request based on number of experts
        is_list_request = len(experts) >= 3
        
        # Create engaging introduction based on number of experts found
        # MODIFIED: More robust introduction handling
        if len(experts) == 1:
            context_parts = [f"I found an APHRC expert who specializes in this area:"]
        elif len(experts) == 2:
            context_parts = [f"I found two APHRC experts who can provide insights on this topic:"]
        elif len(experts) <= 5:
            context_parts = [f"Here are {len(experts)} APHRC experts who specialize in this field:"]
        else:
            context_parts = [f"I've identified several APHRC experts in this area. Here are the most relevant ones:"]
        
        # UNCHANGED: Add transitions between experts
        transitions = [
            "",  # No transition for first expert
            "Another expert in this field is ",
            "You might also be interested in the work of ",
            "Additionally, ",
            "I should also mention ",
            "Another researcher with relevant expertise is ",
            "The research team also includes "
        ]
        
        # MODIFIED: Improved expert information processing with better error handling
        for idx, expert in enumerate(experts):
            try:
                # Extract name components with fallbacks for missing data
                first_name = expert.get('first_name', '').strip()
                last_name = expert.get('last_name', '').strip()
                
                # Skip if no name available at all
                if not first_name and not last_name:
                    logger.warning(f"Skipping expert at index {idx} due to missing name")
                    continue
                    
                full_name = f"{first_name} {last_name}".strip()
                
                # Create expert entry with appropriate transition for position in list
                if idx == 0:
                    # First expert gets number and bold name
                    expert_info = [f"{idx+1}. **{full_name}**"]
                else:
                    # Subsequent experts get a transition phrase
                    transition = transitions[min(idx, len(transitions)-1)]
                    expert_info = [f"{idx+1}. {transition}**{full_name}**"]
                
                # MODIFIED: Better handling of expertise areas with improved error handling
                try:
                    expertise = expert.get('expertise', [])
                    if expertise:
                        # Handle both string and JSON formats
                        if isinstance(expertise, str):
                            try:
                                expertise = json.loads(expertise)
                            except json.JSONDecodeError:
                                # If not valid JSON, use as-is
                                expertise = [expertise]
                        
                        # Format expertise based on type
                        if isinstance(expertise, dict):
                            # Extract values from dictionary
                            expertise_values = []
                            for key, values in expertise.items():
                                if isinstance(values, list):
                                    expertise_values.extend(values)
                                else:
                                    expertise_values.append(values)
                            expertise = expertise_values
                        
                        # Ensure expertise is a list
                        if not isinstance(expertise, list):
                            expertise = [expertise]
                        
                        # Format expertise list
                        if len(expertise) == 1:
                            expert_info.append(f"* Specializes in {expertise[0]}")
                        elif len(expertise) > 1:
                            expertise_text = ", ".join(str(e) for e in expertise[:-1])
                            expert_info.append(f"* Areas of expertise include {expertise_text}, and {expertise[-1]}")
                except Exception as exp_err:
                    logger.warning(f"Error formatting expertise for expert {full_name}: {exp_err}")
                    # Fallback: Use generic expertise line if available
                    if expert.get('expertise'):
                        expert_info.append(f"* Has expertise in various research areas")
                
                # MODIFIED: Improved research interests handling
                try:
                    research_interests = expert.get('research_interests', [])
                    if research_interests:
                        # Handle string format
                        if isinstance(research_interests, str):
                            try:
                                research_interests = json.loads(research_interests)
                            except json.JSONDecodeError:
                                research_interests = [research_interests]
                        
                        # Ensure it's a list
                        if not isinstance(research_interests, list):
                            research_interests = [research_interests]
                        
                        # Format research interests
                        if research_interests and len(research_interests) > 0:
                            if len(research_interests) == 1:
                                expert_info.append(f"* Current research focuses on {research_interests[0]}")
                            else:
                                interests_text = ", ".join(str(ri) for ri in research_interests[:-1])
                                expert_info.append(f"* Research interests span {interests_text}, and {research_interests[-1]}")
                except Exception as ri_err:
                    logger.warning(f"Error formatting research interests for expert {full_name}: {ri_err}")
                
                # UNCHANGED: Add position and department with natural phrasing
                position = expert.get('position', '')
                department = expert.get('department', '')
                
                if position and department:
                    expert_info.append(f"* Serves as {position} in the {department}")
                elif position:
                    expert_info.append(f"* Current role: {position}")
                elif department:
                    expert_info.append(f"* Works in the {department}")
                
                # UNCHANGED: Add contact info with more helpful framing
                email = expert.get('email', '')
                if email:
                    expert_info.append(f"* You can reach them at {email}")
                
                # MODIFIED: Improved publication handling with better error checks
                try:
                    publications = expert.get('publications', [])
                    if publications:
                        if len(publications) == 1:
                            pub = publications[0]
                            pub_title = pub.get('title', 'Untitled')
                            expert_info.append(f"* Notable publication: \"{pub_title}\"")
                        elif len(publications) > 1:
                            expert_info.append("* Notable publications include:")
                            for i, pub in enumerate(publications[:2]):  # Limit to 2 publications
                                pub_title = pub.get('title', 'Untitled')
                                pub_year = pub.get('publication_year', '')
                                year_text = f" ({pub_year})" if pub_year else ""
                                expert_info.append(f"  * \"{pub_title}\"{year_text}")
                except Exception as pub_err:
                    logger.warning(f"Error formatting publications for expert {full_name}: {pub_err}")
                
                # Combine all information about this expert with proper line breaks
                context_parts.append("\n".join(expert_info))
            
            except Exception as expert_err:
                logger.error(f"Error processing expert at index {idx}: {expert_err}")
                # Skip this expert but continue processing others
                continue
        
        # MODIFIED: Add appropriate conclusion based on expert count
        if len(experts) > 1:
            # For list presentations, add more specific follow-up options
            if is_list_request:
                context_parts.append("Would you like more detailed information about any specific expert from this list? You can ask by name or area of expertise.")
            else:
                context_parts.append("Would you like more detailed information about any of these experts or their research areas?")
        else:
            context_parts.append("Would you like to know more about this expert's research or publications?")
        
        # Join with double newlines for better readability
        return "\n\n".join(context_parts)

    def format_publication_context(self, publications: List[Dict[str, Any]]) -> str:
        """
        Format publication information into a rich, conversational presentation with
        natural language introductions, transitions, and context.
        
        Args:
            publications: List of publication dictionaries
            
        Returns:
            Formatted context string with engaging publication presentations
        """
        logger.info(f"Starting format_publication_context with {len(publications)} publications")
        
        if not publications:
            logger.info("No publications provided, returning helpful message")
            return "I couldn't find any publications matching your query. Would you like me to help you search for related topics instead?"
        
        try:
            # Create engaging introduction based on relevance and number of publications
            if len(publications) == 1:
                context_parts = ["I found a relevant APHRC publication that addresses your query:"]
            elif len(publications) <= 3:
                context_parts = [f"Here are {len(publications)} relevant APHRC publications on this topic:"]
            else:
                context_parts = ["I've found several APHRC publications related to your query. Here are the most relevant ones:"]
            
            # Add transitions between publications
            transitions = [
                "",  # No transition for first publication
                "Another important study is ",
                "Related research includes ",
                "You might also be interested in ",
                "Additionally, APHRC researchers published "
            ]
            
            for idx, pub in enumerate(publications):
                logger.debug(f"Processing publication {idx+1}")
                
                # Check publication data
                if not isinstance(pub, dict):
                    logger.warning(f"Publication {idx+1} is not a dictionary but a {type(pub).__name__}")
                    continue
                    
                # Extract title
                title = pub.get('title', 'Untitled')
                if not title:
                    title = "Untitled Publication"
                
                logger.debug(f"Publication {idx+1} title: {title[:50]}...")
                
                # Create numbered publication with narrative transition
                if idx == 0:
                    # First publication gets number and bold title
                    pub_info = [f"{idx+1}. **{title}**"]
                else:
                    # Subsequent publications get a transition phrase
                    transition = transitions[min(idx, len(transitions)-1)]
                    pub_info = [f"{idx+1}. {transition}**{title}**"]
                
                # Add year as bullet point with better phrasing
                pub_year = pub.get('publication_year', '')
                if pub_year:
                    logger.debug(f"Adding year: {pub_year}")
                    pub_info.append(f"* Published in {pub_year}")
                
                # Add authors with careful handling and better formatting
                try:
                    authors = pub.get('authors', [])
                    logger.debug(f"Authors type: {type(authors).__name__}")
                    
                    if authors:
                        if isinstance(authors, list):
                            # Format author list with error handling
                            try:
                                # Format author list with proper Oxford comma and "et al."
                                if len(authors) > 3:
                                    author_text = f"{', '.join(str(a) for a in authors[:2])} et al."
                                    pub_info.append(f"* Written by {author_text}")
                                elif len(authors) == 2:
                                    author_text = f"{authors[0]} and {authors[1]}"
                                    pub_info.append(f"* Authors: {author_text}")
                                elif len(authors) == 1:
                                    pub_info.append(f"* Author: {authors[0]}")
                                else:
                                    author_text = f"{', '.join(str(a) for a in authors[:-1])}, and {authors[-1]}"
                                    pub_info.append(f"* Authors: {author_text}")
                            except Exception as author_error:
                                logger.warning(f"Error formatting authors: {author_error}")
                                pub_info.append(f"* Authors: {len(authors)} contributors")
                        else:
                            logger.debug(f"Authors not a list: {authors}")
                            pub_info.append(f"* Author(s): {authors}")
                except Exception as authors_error:
                    logger.error(f"Error processing authors: {authors_error}", exc_info=True)
                
                # Add APHRC experts with better phrasing if available
                try:
                    aphrc_experts = pub.get('aphrc_experts', [])
                    if aphrc_experts:
                        if isinstance(aphrc_experts, list):
                            if len(aphrc_experts) == 1:
                                pub_info.append(f"* APHRC Expert: {aphrc_experts[0]}")
                            else:
                                experts_text = f"{', '.join(str(e) for e in aphrc_experts[:2])}"
                                if len(aphrc_experts) > 2:
                                    experts_text += " and others"
                                pub_info.append(f"* APHRC Experts: {experts_text}")
                        else:
                            pub_info.append(f"* APHRC Expert: {aphrc_experts}")
                except Exception as experts_error:
                    logger.error(f"Error processing APHRC experts: {experts_error}", exc_info=True)
                
                # Add abstract snippet with better framing
                try:
                    abstract = pub.get('abstract', '')
                    if abstract:
                        # Add introductory language and format the abstract better
                        if len(abstract) > 300:
                            abstract_intro = "* Key findings: "
                            trimmed_abstract = abstract[:297] + "..."
                            
                            # Try to end at a sentence boundary
                            last_period = trimmed_abstract.rfind('.')
                            if last_period > 150:  # Only trim to sentence if we don't lose too much
                                trimmed_abstract = abstract[:last_period+1]
                            
                            pub_info.append(f"{abstract_intro}{trimmed_abstract}")
                        else:
                            pub_info.append(f"* Abstract: {abstract}")
                except Exception as abstract_error:
                    logger.error(f"Error processing abstract: {abstract_error}", exc_info=True)
                
                # Add DOI if available with better explanation
                doi = pub.get('doi', '')
                if doi:
                    pub_info.append(f"* Access via DOI: {doi}")
                
                # Combine all information about this publication
                logger.debug(f"Completed processing publication {idx+1}, {len(pub_info)} info parts")
                try:
                    context_parts.append("\n".join(pub_info))
                except Exception as join_error:
                    logger.error(f"Error joining pub_info: {join_error}", exc_info=True)
                    context_parts.append(f"Publication {idx+1}: {title}")
            
            # Add helpful conclusion with follow-up suggestion
            if len(publications) > 1:
                context_parts.append("Would you like more detailed information about any of these publications or related research areas?")
            else:
                context_parts.append("Would you like to know more about this research or related publications?")
            
            # Join all context parts
            logger.info(f"Joining {len(context_parts)} context parts")
            try:
                result = "\n\n".join(context_parts)
                logger.info(f"Successfully created context, length: {len(result)}")
                return result
            except Exception as final_join_error:
                logger.error(f"Error in final context join: {final_join_error}", exc_info=True)
                return "Error formatting publications. Please try a more specific query."
                
        except Exception as e:
            logger.error(f"Unhandled error in format_publication_context: {e}", exc_info=True)
            return f"I encountered an issue while preparing publication information. Would you like to try a different search term?"
                
                                                        
   
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
        return f"""
        **Expert Profile: {expert.get('first_name')} {expert.get('last_name')}**
        - Position: {expert.get('position', 'N/A')}
        - Expertise: {', '.join(expert.get('expertise', []))}
        - Email: {expert.get('email', 'N/A')}
        """


    

    async def _enrich_experts_with_publications(self, experts: List[Dict[str, Any]], limit_per_expert: int = 2) -> List[Dict[str, Any]]:
        """
        Enhanced method to enrich experts with publications from targeted indexing.
        
        Improvements:
        - Prioritizes publications from expert-resource links
        - Maintains existing enrichment logic
        - Ensures efficient publication retrieval
        """
        if not experts:
            return experts
        
        try:
            for expert in experts:
                expert_id = expert.get('id')
                if not expert_id:
                    continue
                
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
        
  