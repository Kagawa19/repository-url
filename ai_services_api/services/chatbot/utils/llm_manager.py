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

    async def get_relevant_experts(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieve experts from Redis based on the query using embedding similarity.
        
        Args:
            query: The search query
            limit: Maximum number of experts to return
            
        Returns:
            Tuple containing list of expert dictionaries and optional error message
        """
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve experts")
                return [], "Our expert database is currently unavailable. Please try again later."
            
            # Get all expert keys
            expert_keys = await self._get_all_expert_keys()
            
            if not expert_keys:
                logger.warning("No expert keys found in Redis")
                return [], "No experts found in the database."
            
            # Get query embedding if model is available
            query_embedding = None
            if self.embedding_model:
                try:
                    query_embedding = self.embedding_model.encode(query)
                except Exception as e:
                    logger.error(f"Error creating query embedding: {e}")
            
            # Collect expert data
            experts = []
            for key in expert_keys:
                try:
                    # Get raw data from Redis
                    raw_data = self.redis_manager.redis_text.hgetall(key)
                    if not raw_data:
                        continue
                    
                    # Extract expert ID from key
                    expert_id = key.split(':')[-1]
                    
                    # Parse expert data
                    expert = {
                        'id': expert_id,
                        'first_name': raw_data.get('first_name', ''),
                        'last_name': raw_data.get('last_name', ''),
                        'position': raw_data.get('position', ''),
                        'department': raw_data.get('department', ''),
                        'bio': raw_data.get('bio', ''),
                        'email': raw_data.get('email', '')
                    }
                    
                    # Parse JSON fields
                    for field in ['expertise', 'research_interests']:
                        try:
                            if raw_data.get(field):
                                expert[field] = json.loads(raw_data.get(field, '[]'))
                            else:
                                expert[field] = []
                        except json.JSONDecodeError:
                            expert[field] = []
                    
                    # Calculate a match score
                    if query_embedding is not None and 'embedding' in raw_data:
                        try:
                            # Use embedding similarity if available
                            expert_embedding = json.loads(raw_data.get('embedding'))
                            similarity = np.dot(query_embedding, expert_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(expert_embedding)
                            )
                            expert['similarity'] = float(similarity)
                        except (json.JSONDecodeError, TypeError):
                            # Fallback to text matching if embedding is invalid
                            expert['similarity'] = self._calculate_text_similarity(query, expert)
                    else:
                        # Fallback to text matching
                        expert['similarity'] = self._calculate_text_similarity(query, expert)
                    
                    experts.append(expert)
                except Exception as e:
                    logger.error(f"Error processing expert {key}: {e}")
            
            # Sort by similarity score and limit results
            experts.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            top_experts = experts[:limit]
            
            # Remove similarity score from response
            for expert in top_experts:
                expert.pop('similarity', None)
                
            # Get publications for each expert and enrich them
            try:
                top_experts = await self._enrich_experts_with_publications(top_experts, limit_per_expert=2)
            except Exception as pub_error:
                logger.error(f"Error enriching experts with publications: {pub_error}")
            
            logger.info(f"Found {len(top_experts)} relevant experts")
            return top_experts, None
            
        except Exception as e:
            logger.error(f"Error in get_relevant_experts: {e}")
            return [], "We encountered an error finding experts. Please try again with a simpler query."
        
    async def _enrich_experts_with_publications(self, experts: List[Dict[str, Any]], limit_per_expert: int = 2) -> List[Dict[str, Any]]:
        """
        Enrich expert data with their associated publications.
        
        Args:
            experts: List of expert dictionaries
            limit_per_expert: Maximum number of publications per expert
            
        Returns:
            List of expert dictionaries enriched with publication information
        """
        if not experts:
            return experts
        
        try:
            # Process each expert
            for expert in experts:
                expert_id = expert.get('id')
                if not expert_id:
                    continue
                    
                # Skip if already has publications
                if expert.get('publications') and len(expert.get('publications', [])) >= limit_per_expert:
                    continue
                    
                # Get publications for this expert
                publications = await self._get_expert_publications(expert_id, limit=limit_per_expert)
                
                # Add or merge with existing publications
                if 'publications' not in expert or not expert['publications']:
                    expert['publications'] = publications
                else:
                    # Avoid duplicates when merging
                    existing_ids = {pub.get('id') for pub in expert['publications'] if pub.get('id')}
                    for pub in publications:
                        if pub.get('id') and pub['id'] not in existing_ids:
                            expert['publications'].append(pub)
                            existing_ids.add(pub['id'])
                    
                    # Sort by confidence and limit
                    expert['publications'].sort(key=lambda p: p.get('confidence', 0), reverse=True)
                    expert['publications'] = expert['publications'][:limit_per_expert]
                    
            return experts
            
        except Exception as e:
            logger.error(f"Error enriching experts with publications: {e}")
            return experts

    def _calculate_text_similarity(self, query: str, expert: Dict) -> float:
        """
        Calculate text-based similarity between query and expert.
        Used as a fallback when embeddings are not available.
        """
        query_terms = set(query.lower().split())
        expert_text = ' '.join([
            expert.get('first_name', ''),
            expert.get('last_name', ''),
            expert.get('position', ''),
            expert.get('department', ''),
            expert.get('bio', ''),
            ' '.join(str(item) for item in expert.get('expertise', [])),
            ' '.join(str(item) for item in expert.get('research_interests', []))
        ]).lower()
        
        # Count matching terms
        matches = sum(1 for term in query_terms if term in expert_text)
        
        # Simple score based on percentage of query terms matched
        return matches / max(1, len(query_terms))

    

    

    async def _get_expert_publications(self, expert_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get publications associated with an expert from both database and Redis."""
        try:
            publications = []
            
            # 1. First try Redis for faster retrieval
            if self.redis_manager:
                try:
                    redis_publications = self.redis_manager.get_publications_by_expert_id(expert_id, limit)
                    if redis_publications:
                        publications.extend(redis_publications)
                        logger.info(f"Retrieved {len(redis_publications)} publications for expert {expert_id} from Redis")
                except Exception as redis_error:
                    logger.error(f"Error retrieving publications from Redis: {redis_error}")
            
            # 2. If we don't have enough publications, try database
            if len(publications) < limit:
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
                        
                        rows = await conn.fetch(query, expert_id, limit - len(publications))
                        
                        # Format results
                        db_publications = []
                        for row in rows:
                            db_publications.append({
                                'id': row['id'],
                                'title': row['title'],
                                'publication_year': row['publication_year'],
                                'doi': row['doi'],
                                'confidence': row['confidence_score']
                            })
                        
                        # Add database publications
                        publications.extend(db_publications)
                        logger.info(f"Retrieved {len(db_publications)} publications for expert {expert_id} from database")
                except Exception as db_error:
                    logger.error(f"Error fetching publications from database: {db_error}")
            
            # 3. Sort by confidence score
            publications.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # 4. Apply final limit
            return publications[:limit]
                    
        except Exception as e:
            logger.error(f"Error fetching publications for expert {expert_id}: {e}")
            return []
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

            # Yield metadata first
            yield {'is_metadata': True, 'metadata': metadata}

            # Prepare system prompt and content prompt
            system_prompt = "You are a helpful research assistant."
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
        """Advanced intent detection using Gemini"""
        try:
            model = self._setup_gemini()
            
            prompt = f"""
            Analyze this query and classify its intent:
            Query: "{message}"

            Options:
            - PUBLICATION (research papers, studies)
            - EXPERT (researchers, specialists)
            - NAVIGATION (website sections, resources)
            - GENERAL (other queries)

            Return ONLY the JSON in this format with no other text:
            {{
                "intent": "PUBLICATION|EXPERT|NAVIGATION|GENERAL",
                "confidence": 0.0-1.0,
                "clarification": "optional clarification question"
            }}
            """

            response = model.generate_content(prompt)
            content = response.text
            
            # Clean up the response to extract just the JSON part
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Find JSON content - look for opening and closing braces
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                # Extract just the JSON part
                json_content = content[json_start:json_end]
                
                try:
                    result = json.loads(json_content)
                    intent_mapping = {
                        'PUBLICATION': QueryIntent.PUBLICATION,
                        'EXPERT': QueryIntent.EXPERT,
                        'NAVIGATION': QueryIntent.NAVIGATION,
                        'GENERAL': QueryIntent.GENERAL
                    }

                    return {
                        'intent': intent_mapping.get(result.get('intent', 'GENERAL'), QueryIntent.GENERAL),
                        'confidence': min(1.0, max(0.0, float(result.get('confidence', 0.0)))),
                        'clarification': result.get('clarification')
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse intent detection response: {e}")
            else:
                logger.warning("Could not find valid JSON in the response")

            # Default fallback
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

    def _extract_title_from_query(self, query: str) -> Optional[str]:
        """Extract potential publication title from query."""
        title_patterns = [
            r'title[:\s]*["\'](.+?)["\']',
            r'called ["\'](.+?)["\']',
            r'named ["\'](.+?)["\']',
            r'about ["\'](.+?)["\']'
        ]
        for pattern in title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _calculate_publication_match_score(self, metadata: Dict, query_terms: Set[str], title_match: Optional[str]) -> float:
        """Calculate match score between publication and query."""
        score = 0.0
        
        # Title match boost
        if title_match and title_match.lower() in metadata.get('title', '').lower():
            score += 1.0
        
        # Term frequency in text fields
        text_fields = [
            metadata.get('title', ''),
            metadata.get('abstract', ''),
            metadata.get('description', '')
        ]
        text = ' '.join(text_fields).lower()
        
        for term in query_terms:
            if term in text:
                score += 0.2  # Small boost for each term match
        
        return min(score, 1.0)  # Cap at 1.0

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
                    expert_details = []
                    
                    for expert in experts:
                        name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
                        if name:
                            expert_names.append(name)
                            
                            # Add detailed entry
                            expert_details.append({
                                'name': name,
                                'id': expert.get('id'),
                                'confidence': expert.get('confidence', 0.0),
                                # Include additional expert details if available
                                'position': expert.get('position', ''),
                                'department': expert.get('department', ''),
                                'expertise': expert.get('expertise', [])
                            })
                    
                    if expert_names:
                        # Add to publication metadata
                        pub['aphrc_experts'] = expert_names
                        
                        # Add enhanced expert details for better context
                        pub['expert_details'] = expert_details
                        
                        # Add a confidence score for the publication based on expert links
                        if 'confidence' not in pub:
                            # Use the highest expert confidence as publication confidence
                            pub['confidence'] = max([exp.get('confidence', 0.0) for exp in experts])
                
                # Ensure we have a confidence score for sorting
                if 'confidence' not in pub:
                    pub['confidence'] = 0.5  # Default mid-range confidence
                
            # Sort by confidence for better results
            publications.sort(key=lambda p: p.get('confidence', 0), reverse=True)
            
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
        
  