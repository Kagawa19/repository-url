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


    async def generate_async_response(self, message: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response with metadata, using unified semantic search methods.
        
        Args:
            message: User input message
                    
        Yields:
            Response chunks or metadata dictionaries
        """
        try:
            # Detect intent
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

            # Handle specialized expert-publication queries using unified methods
            expert_name = intent_result.get('expert_name')
            found_expert = None
            
            if intent == QueryIntent.EXPERT and expert_name:
                logger.info(f"Handling expert-publication query for: {expert_name}")
                
                # First, find the expert by name
                experts, _ = await self.get_experts(expert_name, limit=1)  # Use unified method
                if experts:
                    found_expert = experts[0]
                    expert_id = found_expert.get('id')
                    
                    # Get publications for this expert using unified method
                    if expert_id:
                        expert_publications = await self.get_publications(expert_id=expert_id, limit=5)
                        
                        if expert_publications and expert_publications[0]:
                            # Format the found expert with their publications
                            expert_with_pubs = {**found_expert, 'publications': expert_publications[0]}
                            context = self.format_expert_with_publications(expert_with_pubs)
                            metadata['content_matches'] = [p.get('id') for p in expert_publications[0]]
                            metadata['content_types'] = {'expert_publications': len(expert_publications[0])}
                        else:
                            # Found expert but no publications
                            context = f"Found expert {expert_name} but they have no associated publications."
                            metadata['content_matches'] = [found_expert.get('id')]
                            metadata['content_types'] = {'expert': 1, 'publications': 0}
            
            # Standard intent handling if not handled by expert-publication special case
            elif intent == QueryIntent.PUBLICATION:
                publications, _ = await self.get_publications(message, limit=3)  # Use unified method
                if publications:
                    context = self.format_publication_context(publications)
                    metadata['content_matches'] = [p['id'] for p in publications]
                    metadata['content_types'] = {'publication': len(publications)}
            
            elif intent == QueryIntent.EXPERT and not found_expert:
                experts, _ = await self.get_experts(message, limit=3)  # Use unified method
                if experts:
                    context = self.format_expert_context(experts)
                    metadata['content_matches'] = [e['id'] for e in experts]
                    metadata['content_types'] = {'expert': len(experts)}

            # Yield metadata first
            yield {'is_metadata': True, 'metadata': metadata}

            # Prepare system prompt and content prompt
            system_prompt = "You are a helpful research assistant."
            
            # Create a more specific content prompt for expert-publication queries
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
            
            # Set up model
            model = self._setup_gemini()
            
            # Create content parts
            content = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": content_prompt}]}
            ]
            
            # Generate response with streaming
            response = model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                ),
                stream=True
            )
            
            # Stream the response chunks
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'parts') and chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            yield part.text

        except Exception as e:
            logger.error(f"Error in generate_async_response: {e}")
            yield f"Error: {str(e)}"

    async def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Enhanced intent detection using only embeddings with no keyword matching.
        
        Args:
            message: User query to analyze
        
        Returns:
            Dictionary with intent classification and related metadata
        """
        try:
            # Check if embedding model is available
            if not self.embedding_model:
                logger.warning("Embedding model not available, using Gemini model for intent detection")
                model = self._setup_gemini()
                
                prompt = f"""
                Analyze this query and classify its intent:
                Query: "{message}"

                Options:
                - PUBLICATION (research papers, studies)
                - EXPERT (researchers, specialists)
                - NAVIGATION (website sections, resources)
                - GENERAL (other queries)

                If the query is asking about publications BY a specific person, extract the person's name.

                Return ONLY the JSON in this format with no other text:
                {{
                    "intent": "PUBLICATION|EXPERT|NAVIGATION|GENERAL",
                    "confidence": 0.0-1.0,
                    "clarification": "optional clarification question",
                    "expert_name": "name of the expert if detected, otherwise null"
                }}
                """

                response = model.generate_content(prompt)
                content = response.text.replace("```json", "").replace("```", "").strip()
                
                # Find JSON content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    try:
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
                        
                        # Add expert name if available
                        if 'expert_name' in result and result['expert_name']:
                            intent_result['expert_name'] = result['expert_name']
                        
                        return intent_result
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse intent detection response")
                        
                return {
                    'intent': QueryIntent.GENERAL,
                    'confidence': 0.0,
                    'clarification': None
                }
                
            # Using pure embedding approach with no keyword matching
            # Clean message for embedding
            cleaned_message = re.sub(r'[^\w\s]', ' ', message.lower()).strip()
            
            # Generate message embedding
            message_embedding = self.embedding_model.encode(cleaned_message)
            
            # Define prototype examples for each intent type
            intent_examples = {
                QueryIntent.PUBLICATION: [
                    "Can you show me the latest research papers?",
                    "I need information about studies on climate change",
                    "What publications are available about maternal health?",
                    "Are there any articles on education impact?",
                    "Show me papers published in 2022"
                ],
                QueryIntent.EXPERT: [
                    "Who are the researchers studying health policy?",
                    "Which experts work on urban development?",
                    "Tell me about the scientists at your organization",
                    "I want to know about specialists in data science",
                    "Which researcher is leading the climate initiative?",
                    "Publications by Sarah Johnson",
                    "Papers authored by Dr. Smith",
                    "Research by Professor Williams"
                ],
                QueryIntent.NAVIGATION: [
                    "How do I navigate to the resources page?",
                    "Where can I find information about your programs?",
                    "Is there a section about funding opportunities?",
                    "Can you help me find the contact page?",
                    "I'm looking for the about section of your website"
                ]
            }
            
            # Get or create intent embeddings (with caching logic)
            intent_embeddings = {}
            for intent, examples in intent_examples.items():
                # Check for cached embeddings in Redis
                cache_key = f"embedding:intent:{intent.value}"
                cached_embedding = None
                
                # Try to get from Redis if available
                if self.redis_manager and self.redis_manager.redis_text.exists(cache_key):
                    try:
                        embedding_bytes = self.redis_manager.redis_text.get(cache_key)
                        if embedding_bytes:
                            cached_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    except Exception as cache_error:
                        logger.warning(f"Error retrieving cached intent embedding: {cache_error}")
                
                # If not cached, compute embedding from examples
                if cached_embedding is not None:
                    intent_embeddings[intent] = cached_embedding
                else:
                    # Encode each example and average them
                    example_embeddings = [self.embedding_model.encode(ex) for ex in examples]
                    intent_embedding = np.mean(example_embeddings, axis=0)
                    intent_embeddings[intent] = intent_embedding
                    
                    # Cache in Redis if available
                    if self.redis_manager:
                        try:
                            self.redis_manager.redis_text.set(
                                cache_key, 
                                intent_embedding.astype(np.float32).tobytes()
                            )
                        except Exception as cache_error:
                            logger.warning(f"Error caching intent embedding: {cache_error}")
            
            # Calculate similarity scores for each intent
            similarity_scores = {}
            for intent, embedding in intent_embeddings.items():
                similarity = np.dot(message_embedding, embedding) / (
                    np.linalg.norm(message_embedding) * np.linalg.norm(embedding)
                )
                # Convert to 0-1 scale
                similarity = max(0, min(1, (similarity + 1) / 2))
                similarity_scores[intent] = similarity
            
            # Find the most likely intent
            max_intent = max(similarity_scores.items(), key=lambda x: x[1])
            best_intent, confidence = max_intent
            
            # Check if we need clarification (scores are close)
            clarification = None
            second_best = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[1]
            
            if second_best[1] > 0.7 and (confidence - second_best[1]) < 0.15:
                # Scores are close, might need clarification
                if best_intent == QueryIntent.PUBLICATION and second_best[0] == QueryIntent.EXPERT:
                    clarification = "Are you looking for specific research papers or information about the researchers?"
                elif best_intent == QueryIntent.EXPERT and second_best[0] == QueryIntent.PUBLICATION:
                    clarification = "Are you interested in the experts themselves or their publications?"
            
            # If confidence is too low, default to GENERAL
            if confidence < 0.5:
                return {
                    'intent': QueryIntent.GENERAL,
                    'confidence': confidence,
                    'clarification': clarification
                }
            
            # Prepare result
            result = {
                'intent': best_intent,
                'confidence': confidence,
                'clarification': clarification
            }
            
            # If intent is EXPERT, try to extract expert name using embedding-based approach
            if best_intent == QueryIntent.EXPERT:
                # Use a separate embedding model call to extract possible names
                expert_extraction_prompt = f"""
                Extract any person names from this text: "{message}"
                Only return the names, without any explanation or additional text.
                If there are no names, return "None".
                """
                
                try:
                    model = self._setup_gemini()
                    name_response = model.generate_content(expert_extraction_prompt)
                    possible_name = name_response.text.strip()
                    
                    if possible_name and possible_name.lower() != "none":
                        # Clean up potential formatting
                        possible_name = possible_name.replace('"', '').replace('*', '').strip()
                        result['expert_name'] = possible_name
                except Exception as name_error:
                    logger.warning(f"Error extracting expert name: {name_error}")
            
            return result

        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }
    
   


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
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve experts")
                return [], "Our expert database is currently unavailable. Please try again later."
            
            logger.info(f"Retrieving experts relevant to: {query}")
            
            # Clean query for processing
            cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
            
            # 1. Direct Targeted Indexing Lookup
            targeted_matches = []
            
            # Scan through expert metadata keys from targeted indexing
            cursor = 0
            while cursor != 0 or not targeted_matches:
                cursor, keys = self.redis_manager.redis_text.scan(
                    cursor, 
                    match='meta:expert:*', 
                    count=100
                )
                
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
                        
                        # Enhanced matching for targeted indexing
                        match_score = 0.0
                        if cleaned_query in full_name:
                            match_score = 1.0
                        elif any(part in full_name for part in cleaned_query.split()):
                            match_score = 0.8
                        
                        # Check expertise and research interests for additional matching
                        expertise = self._safe_json_load(raw_data.get('expertise', '[]'))
                        research_interests = self._safe_json_load(raw_data.get('research_interests', '[]'))
                        
                        expertise_match = any(
                            cleaned_query in str(exp).lower() 
                            for exp in expertise + research_interests
                        )
                        
                        if expertise_match:
                            match_score = max(match_score, 0.7)
                        
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
                            
                            # Stop if we've found enough matches
                            if len(targeted_matches) >= limit:
                                break
                    except Exception as expert_error:
                        logger.warning(f"Error processing expert key {key}: {expert_error}")
                
                if cursor == 0 or len(targeted_matches) >= limit:
                    break
            
            # Sort targeted matches by confidence
            targeted_matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # If we found matches in targeted indexing, return them
            if targeted_matches:
                logger.info(f"Found {len(targeted_matches)} experts through targeted indexing")
                return targeted_matches[:limit], None
            
            # 2. Fallback to existing semantic search methods
            # (Existing semantic search implementation remains unchanged)
            # This ensures we maintain the previous robust search capabilities
            return await self._fallback_expert_semantic_search(cleaned_query, limit)
        
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
        
  