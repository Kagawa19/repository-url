import os
import asyncio
import json
import logging
import random
import re
import time
import numpy as np
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
        """Initialize the LLM manager with required components."""
        try:
            # Load API key
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            # Initialize callback handler
            self.callback = CustomAsyncCallbackHandler()
            
            # Initialize Redis manager for accessing real data
            try:
                self.redis_manager = ExpertRedisIndexManager()
                logger.info("Redis manager initialized successfully")
            except Exception as redis_error:
                logger.error(f"Error initializing Redis manager: {redis_error}")
                self.redis_manager = None
                logger.warning("Continuing without Redis integration - will use generative responses")
            
            # Initialize context management
            self.context_window = []
            self.max_context_items = 5
            self.context_expiry = 1800  # 30 minutes
            self.confidence_threshold = 0.6
             # Explicitly set model path to pre-downloaded location
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            model_cache_dir = '/app/models'  # Use the pre-downloaded model directory
            
            try:
                logger.info(f"Attempting to load model from {model_cache_dir}")
                self.embedding_model = SentenceTransformer(
                    model_name, 
                    cache_folder=model_cache_dir,
                    local_files_only=True  # Force local files
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Falling back to None and using manual embedding")
                self.embedding_model = None
            
            
            # Initialize intent patterns
            self.intent_patterns = {
                QueryIntent.NAVIGATION: {
                    'patterns': [
                        (r'website', 1.0),
                        (r'page', 0.9),
                        (r'find', 0.8),
                        (r'where', 0.8),
                        (r'how to', 0.7),
                        (r'navigate', 0.9),
                        (r'section', 0.8),
                        (r'content', 0.7),
                        (r'information about', 0.7)
                    ],
                    'threshold': 0.6
                },
                
                QueryIntent.PUBLICATION: {
                    'patterns': [
                        (r'research', 1.0),
                        (r'paper', 1.0),
                        (r'publication', 1.0),
                        (r'study', 0.9),
                        (r'article', 0.9),
                        (r'journal', 0.8),
                        (r'doi', 0.9),
                        (r'published', 0.8),
                        (r'authors', 0.8),
                        (r'findings', 0.7)
                    ],
                    'threshold': 0.6
                },
                QueryIntent.EXPERT: {
                    'patterns': [
                        (r'expert', 1.0),
                        (r'researcher', 1.0),
                        (r'author', 0.9),
                        (r'scientist', 0.9),
                        (r'specialist', 0.8),
                        (r'who studies', 0.9),
                        (r'who researches', 0.9),
                        (r'who works on', 0.8),
                        (r'expertise in', 0.9),
                        (r'find people', 0.8),
                        (r'profile', 0.8)
                    ],
                    'threshold': 0.6
                }

            }
            
            logger.info("GeminiLLMManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GeminiLLMManager: {e}", exc_info=True)
            raise

    async def _get_expert_publications(self, expert_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get publications associated with an expert."""
        try:
            # Create placeholders for SQL query
            async with DatabaseConnector.get_connection() as conn:
                query = f"""
                    SELECT r.id, r.title, r.publication_year, r.doi, l.confidence_score
                    FROM expert_resource_links l
                    JOIN resources_resource r ON l.resource_id = r.id
                    WHERE l.expert_id = $1
                    AND l.confidence_score >= 0.7
                    ORDER BY l.confidence_score DESC, r.publication_year DESC
                    LIMIT $2
                """
                
                rows = await conn.fetch(query, expert_id, limit)
                
                # Format results
                publications = []
                for row in rows:
                    publications.append({
                        'id': row['id'],
                        'title': row['title'],
                        'publication_year': row['publication_year'],
                        'doi': row['doi'],
                        'confidence': row['confidence_score']
                    })
                
                return publications
                
        except Exception as e:
            logger.error(f"Error fetching publications for expert {expert_id}: {e}")
            return []

    # 2. Update _get_all_expert_keys method in GeminiLLMManager

    async def _get_all_expert_keys(self):
        """Helper method to get all expert keys from Redis with improved error handling."""
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve expert keys")
                return []
                
            patterns = [
                'meta:expert:*'  # Primary pattern for expert data
            ]
            
            all_keys = []
            
            # For each pattern, scan Redis for matching keys
            for pattern in patterns:
                cursor = 0
                pattern_keys = []
                
                while True:
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
            
            logger.info(f"Found {len(unique_keys)} total unique expert keys in Redis")
            return unique_keys
            
        except Exception as e:
            logger.error(f"Error retrieving expert keys: {e}")
            return []

    # 3. Update get_relevant_experts method to better handle empty results


   
    def get_gemini_model(self):
        """Initialize and return the Gemini model with built-in retry logic."""
        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            stream=True,
            model="gemini-2.0-flash-thinking-exp-01-21",
            convert_system_message_to_human=True,
            callbacks=[self.callback],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_retries=5,  # Use built-in retry mechanism
            timeout=30  # Set a reasonable timeout
        )

    
    

    
    async def _get_all_publication_keys(self):
        """Helper method to get all publication keys from Redis"""
        try:
            # Use scan to find all publication keys
            cursor = 0
            publication_keys = []
            
            while cursor != 0 or len(publication_keys) == 0:
                cursor, batch = self.redis_manager.redis_text.scan(
                    cursor=cursor, 
                    match='meta:resource:*', 
                    count=100
                )
                publication_keys.extend(batch)
                
                if cursor == 0:
                    break
            
            logger.info(f"Found {len(publication_keys)} total publications in Redis")
            return publication_keys
        
        except Exception as e:
            logger.error(f"Error retrieving publication keys: {e}")
            return []

    

   
    async def _reset_rate_limited_after(self, seconds: int):
        """Reset rate limited flag after specified seconds."""
        await asyncio.sleep(seconds)
        self._rate_limited = False
        logger.info(f"Rate limit cooldown expired after {seconds} seconds")

    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """Advanced intent detection using Gemini"""
        try:
            prompt = f"""
            Analyze this query and classify its intent:
            Query: "{message}"
            
            Options:
            - PUBLICATION (research papers, studies)
            - EXPERT (researchers, specialists)
            - NAVIGATION (website sections, resources)
            - GENERAL (other queries)
            
            Return JSON format:
            {{
                "intent": "PUBLICATION|EXPERT|NAVIGATION|GENERAL",
                "confidence": 0.0-1.0,
                "clarification": "optional clarification question"
            }}
            """
            
            model = self.get_gemini_model()
            response = await model.ainvoke(prompt)
            content = response.content
            
            try:
                result = json.loads(content)
                intent_mapping = {
                    'PUBLICATION': QueryIntent.PUBLICATION,
                    'EXPERT': QueryIntent.EXPERT,
                    'NAVIGATION': QueryIntent.NAVIGATION,
                    'GENERAL': QueryIntent.GENERAL
                }
                
                return {
                    'intent': intent_mapping.get(result.get('intent', 'GENERAL'), 
                    'confidence': min(1.0, max(0.0, float(result.get('confidence', 0.0))),
                    'clarification': result.get('clarification')
                }
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse intent detection response")
                return {
                    'intent': QueryIntent.GENERAL,
                    'confidence': 0.0,
                    'clarification': None
                }
                
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }

    async def generate_async_response(self, message: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response with metadata.
        
        Args:
            message: User input message
            
        Yields:
            Response chunks or metadata dictionaries
        """
        try:
            # Detect intent
            intent, confidence = self.detect_intent(message)
            
            # Get relevant content based on intent
            context = ""
            metadata = {
                'intent': {'type': intent.value, 'confidence': confidence},
                'content_matches': [],
                'timestamp': datetime.now().isoformat()
            }

            if intent == QueryIntent.PUBLICATION:
                publications, _ = await self.get_relevant_publications(message)
                if publications:
                    context = self.format_publication_context(publications)
                    metadata['content_matches'] = [p['id'] for p in publications]
                    metadata['content_types'] = {'publication': len(publications)}
            
            elif intent == QueryIntent.EXPERT:
                experts, _ = await self.get_relevant_experts(message)
                if experts:
                    context = self.format_expert_context(experts)
                    metadata['content_matches'] = [e['id'] for e in experts]
                    metadata['content_types'] = {'expert': len(experts)}

            # Prepare messages for LLM
            messages = [
                SystemMessage(content="You are a helpful research assistant."),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message}")
            ]

            # Yield metadata first
            yield {'is_metadata': True, 'metadata': metadata}

            # Generate streaming response
            model = self.get_gemini_model()
            response = await model.agenerate([messages])
            
            # Stream tokens
            async for token in self.callback.aiter():
                if token is not None:
                    yield token

        except Exception as e:
            logger.error(f"Error in generate_async_response: {e}")
            yield f"Error: {str(e)}"

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
                    model_response = await model.ainvoke(prompt)
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
    
    # 2. Updated _get_all_publication_keys method in GeminiLLMManager
    async def _get_all_publication_keys(self):
        """Helper method to get all publication keys from Redis with consistent patterns."""
        try:
            # Use the exact same patterns as used in the _store_resource_data method
            patterns = [
                'meta:resource:*',  # Primary pattern used by _store_resource_data
                'meta::*'  # Alternative pattern
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
    
    

    

   

  

    
    async def get_relevant_publications(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieve publications from Redis with advanced matching capabilities
        """
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve publications")
                return [], "Our publication database is currently unavailable. Please try again later."
            
            # Get all publication keys
            publication_keys = await self._get_all_publication_keys()
            
            if not publication_keys:
                return [], "No publications found in the database."
            
            # Parse specific requests for publication counts
            count_pattern = r'(\d+)\s+publications?'
            count_match = re.search(count_pattern, query, re.IGNORECASE)
            requested_count = int(count_match.group(1)) if count_match else limit
            
            # Comprehensive publication retrieval
            matched_publications = []
            query_terms = set(query.lower().split())
            
            # Extract potential exact title match
            title_match = self._extract_title_from_query(query)
            
            for key in publication_keys:
                try:
                    # Retrieve full publication metadata
                    raw_metadata = self.redis_manager.redis_text.hgetall(key)
                    
                    # Skip entries without essential metadata
                    if not raw_metadata or 'title' not in raw_metadata:
                        continue
                    
                    # Reconstruct metadata exactly as it was stored
                    metadata = {
                        'id': raw_metadata.get('id', ''),
                        'doi': raw_metadata.get('doi', ''),
                        'title': raw_metadata.get('title', ''),
                        'abstract': raw_metadata.get('abstract', ''),
                        'summary': raw_metadata.get('summary', ''),
                        'domains': json.loads(raw_metadata.get('domains', '[]')),
                        'topics': json.loads(raw_metadata.get('topics', '{}')),
                        'description': raw_metadata.get('description', ''),
                        'expert_id': raw_metadata.get('expert_id', ''),
                        'type': raw_metadata.get('type', 'publication'),
                        'subtitles': json.loads(raw_metadata.get('subtitles', '{}')),
                        'publishers': json.loads(raw_metadata.get('publishers', '{}')),
                        'collection': raw_metadata.get('collection', ''),
                        'date_issue': raw_metadata.get('date_issue', ''),
                        'citation': raw_metadata.get('citation', ''),
                        'language': raw_metadata.get('language', ''),
                        'identifiers': json.loads(raw_metadata.get('identifiers', '{}')),
                        'created_at': raw_metadata.get('created_at', ''),
                        'updated_at': raw_metadata.get('updated_at', ''),
                        'source': raw_metadata.get('source', 'unknown'),
                        'authors': json.loads(raw_metadata.get('authors', '[]')),
                        'publication_year': raw_metadata.get('publication_year', '')
                    }
                    
                    # Prepare comprehensive search text
                    search_text = ' '.join([
                        str(metadata.get('title', '')).lower(),
                        str(metadata.get('description', '')).lower(),
                        str(metadata.get('abstract', '')).lower(),
                        str(metadata.get('summary', '')).lower(),
                        ' '.join([str(a).lower() for a in metadata.get('authors', []) if a]),
                    ])
                    
                    # Advanced matching logic
                    match_score = self._calculate_publication_match_score(
                        metadata, 
                        query_terms, 
                        title_match
                    )
                    
                    if match_score > 0:
                        # Attach match score for later sorting
                        metadata['_match_score'] = match_score
                        matched_publications.append(metadata)
                
                except Exception as e:
                    logger.error(f"Error processing publication {key}: {e}")
            
            # Sort publications by match score and year
            sorted_publications = sorted(
                matched_publications,
                key=lambda x: (x.get('_match_score', 0), x.get('publication_year', '0000')),
                reverse=True
            )
            
            # Limit to requested count
            top_publications = sorted_publications[:requested_count]
            
            # Remove internal match score before returning
            for pub in top_publications:
                pub.pop('_match_score', None)
            
            logger.info(f"Found {len(top_publications)} publications matching query")
            return top_publications, None
        
        except Exception as e:
            logger.error(f"Error retrieving publications: {e}")
            return [], "We encountered an error searching publications. Try simplifying your query."

   


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

    async def _enrich_publications_with_experts(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich publication data with information about associated experts.
        
        Args:
            publications: List of publication dictionaries
            
        Returns:
            Enriched publication list with expert information
        """
        if not publications:
            return publications
            
        try:
            # Extract publication IDs
            pub_ids = [str(pub.get('id', '')) for pub in publications if pub.get('id')]
            if not pub_ids:
                return publications
                
            # Get expert information for these publications
            pub_expert_map = await self._get_expert_for_publications(pub_ids)
            
            # Enrich each publication with expert information
            for pub in publications:
                pub_id = str(pub.get('id', ''))
                if pub_id in pub_expert_map and pub_expert_map[pub_id]:
                    # Add expert information to publication
                    experts = pub_expert_map[pub_id]
                    
                    # Format expert names consistently
                    expert_names = []
                    for expert in experts:
                        name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
                        if name:
                            expert_names.append(name)
                    
                    if expert_names:
                        # Add to publication metadata
                        pub['aphrc_experts'] = expert_names
                        
                        # Also add expert details for better context
                        pub['expert_details'] = [
                            {
                                'name': f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip(),
                                'id': expert.get('id'),
                                'confidence': expert.get('confidence', 0.0)
                            }
                            for expert in experts
                        ]
            
            return publications
            
        except Exception as e:
            logger.error(f"Error enriching publications with expert information: {e}")
            # Return original publications if enrichment fails
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
        
  