from enum import Enum
import asyncio
import json
import logging
import random
import re
import os
import time
from typing import Dict, Tuple, Any, List, AsyncGenerator
from datetime import datetime
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import AsyncIteratorCallbackHandler

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enum for different types of query intents."""
    NAVIGATION = "navigation"
    PUBLICATION = "publication"
    GENERAL = "general"

class CustomAsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    async def on_llm_start(self, *args, **kwargs):
        """Handle LLM start."""
        pass

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        """Handle new token."""
        if token:
            self.queue.put_nowait(token)

    async def on_llm_end(self, *args, **kwargs):
        """Handle LLM end."""
        self.queue.put_nowait(None)

    async def on_llm_error(self, error: Exception, *args, **kwargs):
        """Handle LLM error."""
        self.queue.put_nowait(f"Error: {str(error)}")

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
            
            # Initialize context management
            self.context_window = []
            self.max_context_items = 5
            self.context_expiry = 1800  # 30 minutes
            self.confidence_threshold = 0.6
            
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
                }
            }
            
            logger.info("GeminiLLMManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GeminiLLMManager: {e}", exc_info=True)
            raise

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
    async def analyze_quality(self, message: str, response: str = "") -> Dict:
        """
        Analyze the quality of a response in terms of helpfulness, factual accuracy, and potential hallucination.
        Includes improved rate limit handling.
        
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
            
            # If no response provided, we can only analyze the query
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
                # If we have both query and response, analyze the response quality
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
                response = await model.invoke(prompt)
                
                # Reset any rate limit flag if successful
                if hasattr(self, '_rate_limited'):
                    self._rate_limited = False
                
                cleaned_response = response.content.strip()
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
                
                try:
                    quality_data = json.loads(cleaned_response)
                    logger.info(f"Response quality analysis result: {quality_data}")
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

    async def _reset_rate_limited_after(self, seconds: int):
        """Reset rate limited flag after specified seconds."""
        await asyncio.sleep(seconds)
        self._rate_limited = False
        logger.info(f"Rate limit cooldown expired after {seconds} seconds")

    def _get_default_quality(self) -> Dict:
        """Return default quality metric values."""
        return {
            'helpfulness_score': 0.5,
            'hallucination_risk': 0.5,
            'factual_grounding_score': 0.5,
            'unclear_elements': [],
            'potentially_fabricated_elements': []
        }

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
    
    async def detect_intent(self, message: str) -> Tuple[QueryIntent, float]:
        """Detect intent of the message with confidence scoring."""
        try:
            message = message.lower()
            intent_scores = {intent: 0.0 for intent in QueryIntent}
            
            for intent, config in self.intent_patterns.items():
                score = 0.0
                matches = 0
                
                for pattern, weight in config['patterns']:
                    if re.search(pattern, message):
                        score += weight
                        matches += 1
                
                if matches > 0:
                    intent_scores[intent] = score / matches
            
            max_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            if max_intent[1] >= self.intent_patterns.get(max_intent[0], {}).get('threshold', 0.6):
                return max_intent[0], max_intent[1]
            
            return QueryIntent.GENERAL, 0.0
            
        except Exception as e:
            logger.error(f"Error in intent detection: {e}", exc_info=True)
            return QueryIntent.GENERAL, 0.0

    def _create_system_message(self, intent: QueryIntent) -> str:
        """
        Create appropriate system message based on intent with concise, focused responses.
        """
        base_prompts = {
            "common": (
                "You are APHRC's AI assistant. Provide concise, clear responses. "
                "Limit responses to 2-3 paragraphs. Include only essential details. "
                "Focus on accuracy and brevity."
            ),
            QueryIntent.NAVIGATION: (
                "Guide website navigation concisely. Include direct URLs. "
                "Format: [Section Name](URL) - Brief description. "
                "Prioritize most relevant sections first."
            ),
            QueryIntent.PUBLICATION: (
                "Summarize research publications briefly. Format citations as: "
                "Author et al. (Year) - Key finding. Include DOIs when available. "
                "Focus on main conclusions and practical implications."
            ),
            QueryIntent.GENERAL: (
                "Provide focused overview of APHRC's work. "
                "Balance between navigation help and research insights. "
                "Direct users to specific resources when applicable."
            )
        }

        response_guidelines = (
            "Structure responses as:\n"
            "1. Direct answer first (1 sentence)\n"
            "2. Supporting details (1-2 sentences)\n"
            "3. Relevant links/references (if applicable)\n"
            "Avoid repetition. Use natural language."
        )

        return f"{base_prompts['common']} {base_prompts[intent]} {response_guidelines}"

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

    async def generate_async_response(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate async response with parallel processing and enhanced logging."""
        start_time = time.time()
        logger.info(f"Starting async response generation for message: {message}")
        
        try:
            # Log message preprocessing
            logger.debug("Preprocessing message")
            processed_message = message
            
            # Detect intent
            logger.info("Creating parallel tasks for intent and quality")
            try:
                intent_result = await self.detect_intent(processed_message)
                # Skip quality analysis to avoid extra API calls
                initial_quality_data = self._get_default_quality()
            except Exception as task_error:
                logger.error(f"Error in task processing: {task_error}", exc_info=True)
                intent_result = (QueryIntent.GENERAL, 0.0)
                initial_quality_data = self._get_default_quality()
            
            # Log intent results
            logger.info(f"Intent detected: {intent_result[0]} (Confidence: {intent_result[1]})")
            
            # Unpack intent result
            intent, confidence = intent_result
            
            # Create context and manage window
            context = "I'll help you find information about APHRC's publications."
            logger.debug("Managing context window")
            self.manage_context_window({'text': context, 'query': processed_message})
            
            # Prepare messages for model
            logger.debug("Preparing system and human messages")
            system_message = self._create_system_message(intent)
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Context: {context}\n\nQuery: {processed_message}")
            ]
            
            # Initialize response tracking
            response_chunks = []
            buffer = ""
            
            try:
                logger.info("Initializing model streaming")
                model = self.get_gemini_model()
                
                try:
                    # Get response with proper error handling
                    response_data = await model.agenerate([messages])
                    
                    if not response_data.generations or not response_data.generations[0]:
                        logger.warning("Empty response received from model")
                        yield {
                            'chunk': "I'm sorry, but I couldn't generate a response at this time.",
                            'is_metadata': False
                        }
                        return
                    
                    # Process the main response content
                    content = response_data.generations[0][0].text
                    
                    if not content:
                        logger.warning("Empty content received from model")
                        yield {
                            'chunk': "I'm sorry, but I couldn't generate a response at this time.",
                            'is_metadata': False
                        }
                        return
                    
                    # Process the content in chunks for streaming-like behavior
                    remaining_content = content
                    while remaining_content:
                        # Find a good breaking point
                        end_pos = min(100, len(remaining_content))
                        if end_pos < len(remaining_content):
                            # Try to break at a sentence or paragraph
                            for break_char in ['. ', '! ', '? ', '\n']:
                                pos = remaining_content[:end_pos].rfind(break_char)
                                if pos > 0:
                                    end_pos = pos + len(break_char)
                                    break
                        
                        # Extract current chunk
                        current_chunk = remaining_content[:end_pos]
                        remaining_content = remaining_content[end_pos:]
                        
                        # Save and yield chunk
                        response_chunks.append(current_chunk)
                        logger.debug(f"Yielding chunk (length: {len(current_chunk)})")
                        yield {
                            'chunk': current_chunk,
                            'is_metadata': False
                        }
                    
                    # Prepare complete response
                    complete_response = ''.join(response_chunks)
                    logger.info(f"Complete response generated. Total length: {len(complete_response)}")
                    
                    # Skip second quality analysis to avoid rate limits
                    quality_data = self._get_default_quality()
                    
                    # Yield metadata
                    logger.debug("Preparing and yielding metadata")
                    yield {
                        'is_metadata': True,
                        'metadata': {
                            'response': complete_response,
                            'timestamp': datetime.now().isoformat(),
                            'metrics': {
                                'response_time': time.time() - start_time,
                                'intent': {
                                    'type': intent.value,
                                    'confidence': confidence
                                },
                                'quality': quality_data  # Quality metrics
                            },
                            'error_occurred': False
                        }
                    }
                except Exception as stream_error:
                    logger.error(f"Error in response processing: {stream_error}", exc_info=True)
                    error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
                    yield {
                        'chunk': error_message,
                        'is_metadata': False
                    }
            except Exception as e:
                logger.error(f"Critical error generating response: {e}", exc_info=True)
                error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
                
                # Yield error chunk
                yield {
                    'chunk': error_message,
                    'is_metadata': False
                }
                
                # Yield error metadata
                yield {
                    'is_metadata': True,
                    'metadata': {
                        'response': error_message,
                        'timestamp': datetime.now().isoformat(),
                        'metrics': {
                            'response_time': time.time() - start_time,
                            'intent': {'type': 'error', 'confidence': 0.0},
                            'quality': self._get_default_quality()  # Use default quality metrics
                        },
                        'error_occurred': True
                    }
                }
        except Exception as e:
            logger.error(f"Critical error in generate_async_response: {e}", exc_info=True)
            error_message = "I apologize for the inconvenience. Could you please rephrase your question."
            
            # Yield error chunk
            yield {
                'chunk': error_message,
                'is_metadata': False
            }
            
            # Yield error metadata
            yield {
                'is_metadata': True,
                'metadata': {
                    'response': error_message,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'response_time': time.time() - start_time,
                        'intent': {'type': 'error', 'confidence': 0.0},
                        'quality': self._get_default_quality()
                    },
                    'error_occurred': True
                }
            }
        
        finally:
            logger.info(f"Async response generation completed. Total time: {time.time() - start_time:.2f} seconds")

    async def _throttle_request(self):
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