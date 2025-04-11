import os
import asyncio
import json
import logging
import random
import re
import time
import google.generativeai as genai
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

    def _load_embedding_model(self):
        """Try loading embedding model from various locations"""
        model_paths = [
            '/app/models/all-MiniLM-L6-v2',
            './models/all-MiniLM-L6-v2',
            os.path.expanduser('~/models/all-MiniLM-L6-v2')
        ]
        
        for path in model_paths:
            try:
                logger.info(f"Attempting to load model from {path}")
                return SentenceTransformer(path, device='cpu')
            except Exception as e:
                logger.debug(f"Could not load model from {path}: {e}")
        return None
    
    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Enhanced intent detection using multiple fallback strategies.
        Now includes embedding model fallback and better error handling.
        """
        logger.info(f"Detecting intent for message: {message}")
        
        try:
            # First try with embeddings if available
            if self.embedding_model:
                try:
                    intent_result = await self._detect_intent_with_embeddings(message)
                    logger.info(f"Embedding intent detection result: {intent_result['intent']} with confidence {intent_result['confidence']}")
                    return intent_result
                except Exception as e:
                    logger.warning(f"Embedding intent detection failed: {e}")
                        
            # Fallback to keyword matching
            keyword_result = self._detect_intent_with_keywords(message)
            if keyword_result['confidence'] > 0.7:
                logger.info(f"Keyword intent detection result: {keyword_result['intent']} with confidence {keyword_result['confidence']}")
                return keyword_result
                    
            # Final fallback to Gemini API with rate limit protection
            try:
                if not await self.circuit_breaker.check():
                    logger.warning("Circuit breaker open, using default GENERAL intent")
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

                response = await model.ainvoke(prompt)
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
                    
                    logger.info(f"Gemini intent detection result: {intent_result['intent']} with confidence {intent_result['confidence']}")
                    return intent_result
                else:
                    # Handle case where JSON could not be extracted properly
                    logger.warning(f"Could not extract valid JSON from response: {content}")
                    return {
                        'intent': QueryIntent.GENERAL,
                        'confidence': 0.0,
                        'clarification': None
                    }

            except Exception as e:
                logger.error(f"Gemini intent detection failed: {e}")
                if "429" in str(e):
                    await self._handle_rate_limit()
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
    async def generate_async_response(self, message: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response with metadata, using unified semantic search methods.
        Now includes circuit breaking and rate limit protection.
        """
        try:
            # Check circuit breaker first
            if not await self.circuit_breaker.check():
                yield "Our systems are currently busy. Please try again in a moment."
                return

            # Detect intent with fallback protection
            intent_result = await self.detect_intent(message)
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            
            # Get relevant content based on intent
            context = ""
            metadata = {
                'intent': {'type': intent.value, 'confidence': confidence},
                'content_matches': [],
                'timestamp': datetime.now().isoformat()
            }

            # Handle specialized expert-publication queries
            expert_name = intent_result.get('expert_name')
            found_expert = None
            
            if intent == QueryIntent.EXPERT and expert_name:
                logger.info(f"Handling expert-publication query for: {expert_name}")
                experts, _ = await self.get_experts(expert_name, limit=1)
                if experts:
                    found_expert = experts[0]
                    expert_id = found_expert.get('id')
                    
                    if expert_id:
                        expert_publications = await self.get_publications(expert_id=expert_id, limit=5)
                        if expert_publications and expert_publications[0]:
                            expert_with_pubs = {**found_expert, 'publications': expert_publications[0]}
                            context = self.format_expert_with_publications(expert_with_pubs)
                            metadata['content_matches'] = [p.get('id') for p in expert_publications[0]]
                            metadata['content_types'] = {'expert_publications': len(expert_publications[0])}
                        else:
                            context = f"Found expert {expert_name} but no associated publications."
                            metadata['content_matches'] = [found_expert.get('id')]
                            metadata['content_types'] = {'expert': 1, 'publications': 0}
            
            # Standard intent handling
            elif intent == QueryIntent.PUBLICATION:
                publications, _ = await self.get_publications(message, limit=3)
                if publications:
                    context = self.format_publication_context(publications)
                    metadata['content_matches'] = [p['id'] for p in publications]
                    metadata['content_types'] = {'publication': len(publications)}
            
            elif intent == QueryIntent.EXPERT and not found_expert:
                experts, _ = await self.get_experts(message, limit=3)
                if experts:
                    context = self.format_expert_context(experts)
                    metadata['content_matches'] = [e['id'] for e in experts]
                    metadata['content_types'] = {'expert': len(experts)}

            # Yield metadata first
            yield {'is_metadata': True, 'metadata': metadata}

            # Prepare system prompt and content prompt
            system_prompt = "You are a helpful research assistant."
            
            if intent == QueryIntent.EXPERT and expert_name and found_expert:
                content_prompt = f"""
                Context:
                {context}

                Question: {message}
                
                Please format your response about the publications by {expert_name} as a numbered list,
                with each publication title in bold and details as bullet points.
                """
            else:
                content_prompt = f"Context:\n{context}\n\nQuestion: {message}"
            
            # Generate response with protected streaming
            try:
                async for chunk in self._safe_generate(content_prompt):
                    yield chunk
                self.circuit_breaker.record_success()
                
            except Exception as e:
                self.circuit_breaker.record_failure()
                yield f"Error: {str(e)}"

        except Exception as e:
            logger.error(f"Error in generate_async_response: {e}")
            yield f"Error: {str(e)}"

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
        """
        Detect intent using embedding models to compute similarity scores.
        
        Args:
            message: The user's query message
            
        Returns:
            Dictionary with intent type, confidence score, and optional clarification
        """
        try:
            if not self.embedding_model:
                raise ValueError("No embedding model available")
                
            # Prepare cleaned message
            cleaned_message = re.sub(r'[^\w\s]', ' ', message.lower()).strip()
            
            # Define example queries for each intent type
            intent_examples = {
                QueryIntent.PUBLICATION: [
                    "Show me publications about maternal health",
                    "What papers have been published on climate change",
                    "Research articles on education in Africa",
                    "Find studies on urban development",
                    "Recent publications by Dr. Smith"
                ],
                QueryIntent.EXPERT: [
                    "Who are the experts in health policy",
                    "Find researchers working on climate change",
                    "Information about Dr. Johnson",
                    "Which scientists study education outcomes",
                    "Tell me about specialists in urban planning"
                ],
                QueryIntent.NAVIGATION: [
                    "How do I find the contact page",
                    "Where can I access research tools",
                    "Show me the about section",
                    "Where is the publications list",
                    "How to navigate to resources"
                ]
            }
            
            # Get embeddings of the query
            query_embedding = self.embedding_model.encode(cleaned_message)
            
            # Calculate similarity scores for each intent
            intent_scores = {}
            expert_name = None
            
            for intent, examples in intent_examples.items():
                # Encode all examples for this intent
                example_embeddings = self.embedding_model.encode(examples)
                
                # Calculate cosine similarity between query and examples
                similarities = np.dot(example_embeddings, query_embedding) / (
                    np.linalg.norm(example_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Use max similarity as the score for this intent
                max_similarity = np.max(similarities)
                intent_scores[intent] = float(max_similarity)  # Convert numpy float to Python float
                
                # If this is an expert intent with high similarity, try to extract expert name
                if intent == QueryIntent.EXPERT and max_similarity > 0.7:
                    # Use regex patterns to extract potential expert names
                    name_patterns = [
                        r'(by|from|about) ([A-Z][a-z]+ [A-Z][a-z]+)',
                        r'([A-Z][a-z]+ [A-Z][a-z]+)\'s',
                        r'(Dr\.|Professor) ([A-Z][a-z]+ [A-Z][a-z]+)'
                    ]
                    
                    for pattern in name_patterns:
                        match = re.search(pattern, message)
                        if match:
                            if len(match.groups()) > 1:
                                expert_name = match.group(2)
                            else:
                                expert_name = match.group(1)
                            break
            
            # Determine the intent with the highest score
            max_score = 0.0
            max_intent = QueryIntent.GENERAL
            
            for intent, score in intent_scores.items():
                if score > max_score:
                    max_score = score
                    max_intent = intent
            
            # Apply threshold to determine final intent
            confidence = max_score
            
            # If confidence is too low, default to GENERAL intent
            if confidence < 0.6:
                max_intent = QueryIntent.GENERAL
                
            # Prepare appropriate clarification if confidence is moderate
            clarification = None
            if 0.6 <= confidence < 0.75:
                if max_intent == QueryIntent.PUBLICATION:
                    clarification = "Which specific publication topic are you interested in?"
                elif max_intent == QueryIntent.EXPERT:
                    clarification = "Can you specify which expert or research area you're looking for?"
                elif max_intent == QueryIntent.NAVIGATION:
                    clarification = "Which specific section of our resources are you trying to find?"
            
            # Build result
            result = {
                'intent': max_intent,
                'confidence': confidence,
                'clarification': clarification
            }
            
            # Add expert name if found
            if expert_name:
                result['expert_name'] = expert_name
                
            logger.info(f"Embedding intent detection result: {max_intent.value} with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in embedding intent detection: {e}", exc_info=True)
            raise

    async def _detect_intent_with_gemini(self, message: str) -> Dict[str, Any]:
        """
        Detect intent using Gemini API as a fallback.
        
        Args:
            message: The user's query message
            
        Returns:
            Dictionary with intent type, confidence score, and optional clarification
        """
        try:
            if not await self.circuit_breaker.check():
                logger.warning("Circuit breaker open, skipping Gemini intent detection")
                return {
                    'intent': QueryIntent.GENERAL,
                    'confidence': 0.0,
                    'clarification': None
                }

            # Get Gemini model
            model = self._setup_gemini()
            
            # Create structured prompt for intent detection
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

            # Call Gemini API with rate limit protection
            try:
                await self._throttle_request()
                response = await model.ainvoke(prompt)
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
                    
                    # Add expert name if detected
                    if 'expert_name' in result and result['expert_name']:
                        intent_result['expert_name'] = result['expert_name']
                    
                    logger.info(f"Gemini intent detection result: {intent_result}")
                    return intent_result
                    
                else:
                    logger.warning(f"Failed to extract JSON from Gemini response: {content}")
                    return {
                        'intent': QueryIntent.GENERAL,
                        'confidence': 0.0,
                        'clarification': None
                    }
                    
            except Exception as api_error:
                logger.error(f"Gemini API error in intent detection: {api_error}")
                if "429" in str(api_error) or "quota" in str(api_error).lower():
                    await self._handle_rate_limit()
                return {
                    'intent': QueryIntent.GENERAL,
                    'confidence': 0.0,
                    'clarification': None
                }
                
        except Exception as e:
            logger.error(f"Error in Gemini intent detection: {e}")
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }

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

    async def generate_async_response(self, message: str) -> AsyncGenerator[str, None]:
        """Generate response with circuit breaker protection"""
        if not await self.circuit_breaker.check():
            yield "System is currently busy. Please try again shortly."
            return
            
        try:
            async for chunk in self._safe_generate(message):
                yield chunk
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            yield f"Error: {str(e)}"

    # All original methods below remain exactly the same (format_expert_context, 
    # format_publication_context, analyze_quality, etc.) with no changes to their
    # implementation or signatures to maintain full backward compatibility

    
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


   
    
    def _setup_gemini(self):
        """Set up and configure the Gemini model."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            
            # Simple configuration without specifying API version
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            logger.info("Gemini model setup completed")
            return model
            
        except Exception as e:
            logger.error(f"Error setting up Gemini model: {e}")
            raise

    
    

    
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

 

    

    


    
        
    def format_expert_context(self, experts: List[Dict[str, Any]]) -> str:
        """
        Format expert information into a numbered list with bulleted expertise areas.
        
        Args:
            experts: List of expert dictionaries
            
        Returns:
            Formatted context string with numbered experts and bulleted lists
        """
        if not experts:
            return "No expert information available."
        
        context_parts = ["Here is information about relevant APHRC experts:"]
        
        for idx, expert in enumerate(experts):
            full_name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
            
            # Create numbered expert entry with name in bold
            expert_info = [f"{idx+1}. **{full_name}**"]
            
            # Add expertise areas as bullet points
            expertise = expert.get('expertise', [])
            if expertise and isinstance(expertise, list):
                for area in expertise:
                    expert_info.append(f"* {area}")
            elif expertise:
                expert_info.append(f"* {expertise}")
            
            # Add research interests as bullet points if available
            research_interests = expert.get('research_interests', [])
            if research_interests and isinstance(research_interests, list):
                for interest in research_interests:
                    expert_info.append(f"* {interest}")
            
            # Add position and department as bullet points if available
            position = expert.get('position', '')
            if position:
                expert_info.append(f"* Position: {position}")
                
            department = expert.get('department', '')
            if department:
                expert_info.append(f"* Department: {department}")
            
            # Add contact info as a bullet point if available
            email = expert.get('email', '')
            if email:
                expert_info.append(f"* Contact: {email}")
            
            # Add publications as bullet points if available
            publications = expert.get('publications', [])
            if publications:
                expert_info.append("* Notable publications:")
                for pub in publications[:2]:  # Limit to 2 publications
                    pub_title = pub.get('title', 'Untitled')
                    expert_info.append(f"  * {pub_title}")
            
            # Combine all information about this expert with proper line breaks
            context_parts.append("\n".join(expert_info))
        
        return "\n\n".join(context_parts)
    def format_publication_context(self, publications: List[Dict[str, Any]]) -> str:
        """
        Format publication information into a numbered list with bulleted details.
        
        Args:
            publications: List of publication dictionaries
            
        Returns:
            Formatted context string with numbered publications and bulleted lists
        """
        logger.info(f"Starting format_publication_context with {len(publications)} publications")
        
        if not publications:
            logger.info("No publications provided, returning default message")
            return "No publication information available."
        
        try:
            context_parts = ["Here is information about relevant APHRC publications:"]
            
            for idx, pub in enumerate(publications):
                logger.debug(f"Processing publication {idx+1}")
                
                # Check publication data
                if not isinstance(pub, dict):
                    logger.warning(f"Publication {idx+1} is not a dictionary but a {type(pub).__name__}")
                    continue
                    
                # Extract title
                title = pub.get('title', 'Untitled')
                logger.debug(f"Publication {idx+1} title: {title[:50]}...")
                
                # Create numbered publication with title in bold
                pub_info = [f"{idx+1}. **{title}**"]
                
                # Add year as bullet point
                pub_year = pub.get('publication_year', '')
                if pub_year:
                    logger.debug(f"Adding year: {pub_year}")
                    pub_info.append(f"* Published: {pub_year}")
                
                # Add authors with careful handling
                try:
                    authors = pub.get('authors', [])
                    logger.debug(f"Authors type: {type(authors).__name__}")
                    
                    if authors:
                        if isinstance(authors, list):
                            # Format author list with error handling
                            try:
                                # Format author list, limit to first 3 with "et al." if more
                                if len(authors) > 3:
                                    author_text = f"{', '.join(str(a) for a in authors[:3])} et al."
                                else:
                                    author_text = f"{', '.join(str(a) for a in authors)}"
                                pub_info.append(f"* Authors: {author_text}")
                            except Exception as author_error:
                                logger.warning(f"Error formatting authors: {author_error}")
                                pub_info.append(f"* Authors: {len(authors)} contributors")
                        else:
                            logger.debug(f"Authors not a list: {authors}")
                            pub_info.append(f"* Authors: {authors}")
                except Exception as authors_error:
                    logger.error(f"Error processing authors: {authors_error}", exc_info=True)
                
                # Add APHRC experts if available
                try:
                    aphrc_experts = pub.get('aphrc_experts', [])
                    if aphrc_experts:
                        if isinstance(aphrc_experts, list):
                            pub_info.append(f"* APHRC Experts: {', '.join(str(e) for e in aphrc_experts[:3])}")
                        else:
                            pub_info.append(f"* APHRC Experts: {aphrc_experts}")
                except Exception as experts_error:
                    logger.error(f"Error processing APHRC experts: {experts_error}", exc_info=True)
                
                # Add abstract snippet as bullet point
                try:
                    abstract = pub.get('abstract', '')
                    if abstract:
                        # Truncate long abstracts
                        if len(abstract) > 300:
                            abstract = abstract[:297] + "..."
                        pub_info.append(f"* Abstract: {abstract}")
                except Exception as abstract_error:
                    logger.error(f"Error processing abstract: {abstract_error}", exc_info=True)
                
                # Add DOI if available
                doi = pub.get('doi', '')
                if doi:
                    pub_info.append(f"* DOI: {doi}")
                
                # Combine all information about this publication
                logger.debug(f"Completed processing publication {idx+1}, {len(pub_info)} info parts")
                try:
                    context_parts.append("\n".join(pub_info))
                except Exception as join_error:
                    logger.error(f"Error joining pub_info: {join_error}", exc_info=True)
                    context_parts.append(f"Publication {idx+1}: {title}")
            
            # Join all context parts
            logger.info(f"Joining {len(context_parts)} context parts")
            try:
                result = "\n\n".join(context_parts)
                logger.info(f"Successfully created context, length: {len(result)}")
                return result
            except Exception as final_join_error:
                logger.error(f"Error in final context join: {final_join_error}", exc_info=True)
                return "Error formatting publications."
                
        except Exception as e:
            logger.error(f"Unhandled error in format_publication_context: {e}", exc_info=True)
            return f"Error formatting publications: {str(e)}"

   
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

    def update_expert_resource_links(self):
        """Update the expert-resource relationships in Redis without full reindexing."""
        try:
            # Connect to database to get expert-resource links
            conn = self.db.get_connection()
            with conn.cursor() as cur:
                # Get links with confidence scores
                cur.execute("""
                    SELECT expert_id, resource_id, confidence_score 
                    FROM expert_resource_links
                    WHERE confidence_score >= 0.7
                    ORDER BY expert_id, confidence_score DESC
                """)
                
                links = cur.fetchall()
                logger.info(f"Retrieved {len(links)} expert-resource links to update")
                
                # Store links in Redis
                pipeline = self.redis_text.pipeline()
                
                # Group by expert_id for efficiency
                expert_resources = {}
                for expert_id, resource_id, confidence in links:
                    if expert_id not in expert_resources:
                        expert_resources[expert_id] = []
                    expert_resources[expert_id].append((resource_id, float(confidence)))
                
                # Store in Redis using sets and sorted sets
                for expert_id, resources in expert_resources.items():
                    # Create a sorted set key for each expert
                    key = f"links:expert:{expert_id}:resources"
                    
                    # Clear existing set
                    pipeline.delete(key)
                    
                    # Add all resources with confidence as score
                    for resource_id, confidence in resources:
                        pipeline.zadd(key, {str(resource_id): confidence})
                    
                    # Add inverse lookups for resources to experts
                    for resource_id, confidence in resources:
                        res_key = f"links:resource:{resource_id}:experts"
                        pipeline.zadd(res_key, {str(expert_id): confidence})
                
                # Execute pipeline
                pipeline.execute()
                logger.info(f"Successfully updated {len(expert_resources)} expert-resource relationships")
                
                return True
        
        except Exception as e:
            logger.error(f"Error updating expert-resource links: {e}")
            return False

    

    async def get_experts(self, query: str, limit: int = 3) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Enhanced expert retrieval prioritizing experts indexed during create_expert_redis_index.
        
        Improvements:
        - Prioritizes experts stored during targeted indexing
        - Maintains existing semantic search capabilities
        - Preserves fallback mechanisms
        - Improves retrieval efficiency
        """
        logger.info(f"Retrieving experts with query: {query}")
        
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve experts")
                return [], "Our expert database is currently unavailable. Please try again later."
            
            # Detect gender-specific queries
            gender_query = False
            if any(term in query.lower() for term in ["female", "women", "woman"]):
                gender_query = True
                logger.info("Detected gender-specific query for female experts")
            
            # Clean query for processing
            cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
            logger.info(f"Cleaned query: '{cleaned_query}'")
            
            # 1. Direct Targeted Indexing Lookup
            targeted_matches = []
            
            # Scan through expert metadata keys from targeted indexing
            cursor = 0
            keys_found = False  # Track if any keys were found
            
            while cursor != 0 or not keys_found:
                cursor, keys = self.redis_manager.redis_text.scan(
                    cursor, 
                    match='meta:expert:*', 
                    count=100
                )
                
                keys_found = keys_found or bool(keys)
                logger.debug(f"Redis scan returned {len(keys)} expert keys with cursor {cursor}")
                
                for key in keys:
                    try:
                        # Get expert metadata
                        raw_data = self.redis_manager.redis_text.hgetall(key)
                        if not raw_data:
                            continue
                        
                        # Extract expert details with targeted indexing focus
                        first_name = raw_data.get('first_name', '').lower()
                        last_name = raw_data.get('last_name', '').lower()
                        full_name = f"{first_name} {last_name}".strip()
                        
                        # For gender-specific queries
                        if gender_query:
                            # Try to determine gender from metadata
                            gender = raw_data.get('gender', '').lower()
                            
                            # If gender is specified and not female, skip
                            if gender and gender not in ['female', 'woman', 'women', 'f']:
                                continue
                            
                            # If no gender in metadata, we could use a name-based inference
                            # but for now include all experts if no gender filter
                            
                            # Add to matches for gender query with reasonable confidence
                            match_score = 0.75
                            expert = {
                                'id': key.split(':')[-1],
                                'first_name': first_name.title(),
                                'last_name': last_name.title(),
                                'expertise': self._safe_json_load(raw_data.get('expertise', '[]')),
                                'research_interests': self._safe_json_load(raw_data.get('research_interests', '[]')),
                                'position': raw_data.get('position', ''),
                                'department': raw_data.get('department', ''),
                                'email': raw_data.get('email', ''),
                                'confidence': match_score
                            }
                            
                            targeted_matches.append(expert)
                            continue
                        
                        # Non-gender specific matching
                        match_score = 0.0
                        # Exact match in name
                        if cleaned_query in full_name:
                            match_score = 1.0
                            logger.debug(f"Exact name match for expert '{full_name}'")
                        # Partial match in name
                        elif any(part in full_name for part in cleaned_query.split()):
                            match_score = 0.8
                            logger.debug(f"Partial name match for expert '{full_name}'")
                        
                        # Check expertise and research interests for additional matching
                        expertise = self._safe_json_load(raw_data.get('expertise', '[]'))
                        research_interests = self._safe_json_load(raw_data.get('research_interests', '[]'))
                        
                        # Check if query terms match expertise or research interests
                        expertise_match = any(
                            cleaned_query in str(exp).lower() 
                            for exp in expertise + research_interests
                        )
                        
                        if expertise_match:
                            match_score = max(match_score, 0.7)
                            logger.debug(f"Expertise match for expert '{full_name}'")
                        
                        # If we have a reasonable match
                        if match_score > 0.5:
                            expert = {
                                'id': key.split(':')[-1],
                                'first_name': first_name.title(),
                                'last_name': last_name.title(),
                                'expertise': expertise,
                                'research_interests': research_interests,
                                'position': raw_data.get('position', ''),
                                'department': raw_data.get('department', ''),
                                'email': raw_data.get('email', ''),
                                'confidence': match_score
                            }
                            
                            targeted_matches.append(expert)
                            logger.debug(f"Added expert '{full_name}' with score {match_score}")
                            
                            # Stop if we've found enough matches
                            if len(targeted_matches) >= limit:
                                break
                    except Exception as expert_error:
                        logger.warning(f"Error processing expert key {key}: {expert_error}")
                
                if cursor == 0 or len(targeted_matches) >= limit:
                    break
            
            # Sort targeted matches by confidence
            targeted_matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            logger.info(f"Found {len(targeted_matches)} experts through targeted indexing")
            
            # If we found matches in targeted indexing, return them
            if targeted_matches:
                # Log the specific experts found
                for expert in targeted_matches[:limit]:
                    name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
                    logger.info(f"Returning expert: {name} with confidence {expert.get('confidence', 0)}")
                return targeted_matches[:limit], None
            
            # 2. If no targeted matches and we have a gender query, do a scan for all experts
            if gender_query and not targeted_matches:
                logger.info("No targeted matches for gender query, scanning all experts")
                all_experts = []
                cursor = 0
                while cursor != 0 or not all_experts:
                    cursor, keys = self.redis_manager.redis_text.scan(
                        cursor, 
                        match='meta:expert:*', 
                        count=100
                    )
                    
                    for key in keys:
                        try:
                            raw_data = self.redis_manager.redis_text.hgetall(key)
                            if not raw_data:
                                continue
                            
                            first_name = raw_data.get('first_name', '').title()
                            last_name = raw_data.get('last_name', '').title()
                            
                            expert = {
                                'id': key.split(':')[-1],
                                'first_name': first_name,
                                'last_name': last_name,
                                'expertise': self._safe_json_load(raw_data.get('expertise', '[]')),
                                'research_interests': self._safe_json_load(raw_data.get('research_interests', '[]')),
                                'position': raw_data.get('position', ''),
                                'department': raw_data.get('department', ''),
                                'email': raw_data.get('email', ''),
                                'confidence': 0.6  # Default confidence for gender-only queries
                            }
                            
                            all_experts.append(expert)
                            
                            if len(all_experts) >= limit:
                                break
                        except Exception as e:
                            logger.warning(f"Error processing expert key {key}: {e}")
                    
                    if cursor == 0 or len(all_experts) >= limit:
                        break
                
                if all_experts:
                    logger.info(f"Found {len(all_experts)} experts for gender query")
                    return all_experts[:limit], None
            
            # 3. If still no results, check if we have a fallback method
            # Try to import the fallback method if it exists
            if hasattr(self, '_fallback_expert_semantic_search'):
                logger.info("No experts found through targeted indexing, falling back to semantic search")
                return await self._fallback_expert_semantic_search(cleaned_query, limit)
            else:
                logger.warning("No fallback semantic search method available")
                # If no results and no fallback, return empty with a message
                return [], "No matching experts found in our database."
            
        except Exception as e:
            logger.error(f"Unhandled error in get_experts: {e}")
            return [], f"An unexpected error occurred while searching for experts: {str(e)}"

    async def get_publications(self, query: str = None, expert_id: str = None, limit: int = 3) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Enhanced publication retrieval prioritizing publications linked during expert indexing.
        
        Improvements:
        - Prioritizes publications from expert-resource links
        - Maintains semantic search capabilities
        - Preserves existing fallback mechanisms
        """
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve publications")
                return [], "Our publication database is currently unavailable. Please try again later."
            
            logger.info(f"Retrieving publications for query: {query}, expert_id: {expert_id}")
            
            # 1. Expert-Specific Publication Retrieval
            if expert_id:
                # First, look for publications through expert-resource links from targeted indexing
                publications = []
                links_key = f"links:expert:{expert_id}:resources"
                
                if self.redis_manager.redis_text.exists(links_key):
                    # Get resource IDs sorted by confidence from expert indexing
                    resource_items = self.redis_manager.redis_text.zrevrange(
                        links_key, 0, limit-1, withscores=True
                    )
                    
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
                    
                    # If we found publications through expert links, return them
                    if publications:
                        publications.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                        return publications[:limit], None
            
            # 2. Semantic Search for Publications
            # (Existing semantic search implementation remains unchanged)
            # Fallback to existing robust publication search
            return await self._fallback_publication_semantic_search(query, expert_id, limit)
        
        except Exception as e:
            logger.error(f"Unhandled error in get_publications: {e}")
            return [], f"An unexpected error occurred while searching for publications: {str(e)}"

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
        
  