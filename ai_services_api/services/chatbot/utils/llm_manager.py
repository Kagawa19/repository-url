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

    def format_expert_context(self, experts: List[Dict[str, Any]]) -> str:
        """Format expert data into a structured context for LLM responses."""
        if not experts:
            return "No experts found matching the query."
        
        # Determine formatting based on number of experts
        is_list_format = len(experts) > 1
        
        # Start with an appropriate header
        if is_list_format:
            context_parts = [f"Found {len(experts)} relevant experts:"]
        else:
            context_parts = ["Found the following relevant expert:"]
        
        # Process each expert
        for idx, expert in enumerate(experts, 1):
            # Create a formatted expert entry
            expert_parts = []
            
            # Add index for list format
            prefix = f"{idx}. " if is_list_format else ""
            
            # NAME - Always include as the main identifier
            name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
            if name:
                expert_parts.append(f"{prefix}Name: {name}")
                prefix = "" if not is_list_format else prefix
            
            # EXPERTISE - Format areas of expertise
            expertise = expert.get('expertise', {})
            if expertise:
                expertise_sections = []
                
                if isinstance(expertise, dict):
                    for area, details in expertise.items():
                        if isinstance(details, list) and details:
                            expertise_sections.append(f"{area.title()}: {', '.join(str(d) for d in details if d)}")
                        elif isinstance(details, str) and details.strip():
                            expertise_sections.append(f"{area.title()}: {details}")
                
                if expertise_sections:
                    expert_parts.append(f"{prefix}Areas of Expertise:")
                    for section in expertise_sections:
                        expert_parts.append(f"{prefix}  • {section}")
            
            # Add any PUBLICATIONS linked to this expert
            expert_id = expert.get('id')
            if expert_id:
                try:
                    # Get linked publications (this would be implemented in a separate method)
                    linked_pubs = await self._get_expert_publications(expert_id, limit=3)
                    
                    if linked_pubs:
                        expert_parts.append(f"{prefix}Selected Publications:")
                        for pub in linked_pubs:
                            pub_title = pub.get('title', 'Untitled')
                            pub_year = pub.get('publication_year', '')
                            year_text = f" ({pub_year})" if pub_year else ""
                            expert_parts.append(f"{prefix}  • {pub_title}{year_text}")
                except Exception as e:
                    logger.error(f"Error retrieving publications for expert {expert_id}: {e}")
            
            # Join all expert parts with explicit newlines
            formatted_expert = "\n".join(expert_parts)
            
            # Add a separator line between experts for clearer formatting
            if is_list_format and idx < len(experts):
                formatted_expert += "\n" + "-" * 40
            
            context_parts.append(formatted_expert)
        
        # Add instructions for the model about how to use this data
        usage_instructions = """
    When discussing these experts in your response:
    - Reference their full names and areas of expertise
    - Mention their research focus areas when relevant
    - If publications are listed, you can reference them as examples of their work
    - IMPORTANT: Only discuss information explicitly provided in the expert profiles
    """

        # Join everything with double newlines for better separation
        return "\n\n".join(context_parts) + "\n\n" + usage_instructions

    def _extract_name_from_query(self, query: str) -> Optional[str]:
        """Extract potential expert name from query."""
        name_patterns = [
            r'expert (?:named|called) ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'researcher (?:named|called) ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'find ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'profile of ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'about ([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip().lower()
        
        # Try to find capitalized names that might be experts
        words = query.split()
        for i in range(len(words) - 1):
            if (i+1 < len(words) and 
                words[i][0].isupper() and words[i+1][0].isupper() and
                len(words[i]) > 1 and len(words[i+1]) > 1):
                return f"{words[i]} {words[i+1]}".lower()
        
        return None

    def _extract_expertise_terms(self, query: str) -> List[str]:
        """Extract expertise-related terms from query."""
        expertise_patterns = [
            r'expertise in (.+)',
            r'specializing in (.+)',
            r'who studies (.+)',
            r'who researches (.+)',
            r'expert in (.+)',
            r'working on (.+)'
        ]
        
        for pattern in expertise_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                expertise_text = match.group(1).strip().lower()
                # Split into terms and remove common stopwords
                terms = [term.strip() for term in re.split(r'[,;]|\band\b', expertise_text) if term.strip()]
                return terms
        
        return []

    def _calculate_expert_keyword_score(
        self, 
        metadata: Dict[str, Any], 
        query_terms: Set[str],
        name_match: Optional[str],
        expertise_terms: List[str]
    ) -> float:
        """Calculate weighted keyword match score for an expert."""
        score = 0.0
        
        # 1. NAME MATCH (high weight)
        expert_name = f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".lower()
        if name_match and name_match in expert_name:
            score += 3.0  # Significant boost for name match
        
        # 2. EXPERTISE MATCH
        expertise_data = metadata.get('expertise', {})
        expertise_text = ""
        
        # Extract text from different expertise fields
        if isinstance(expertise_data, dict):
            for field, values in expertise_data.items():
                if isinstance(values, list):
                    expertise_text += " ".join([str(v).lower() for v in values if v])
                elif isinstance(values, str):
                    expertise_text += values.lower()
        
        # 3. Score expertise terms
        for term in expertise_terms:
            if term in expertise_text:
                score += 2.5  # Strong boost for expertise match
        
        # 4. General query term matching
        for term in query_terms:
            if len(term) <= 2:  # Skip very short terms
                continue
                
            # Check name
            if term in expert_name:
                score += 1.5
                
            # Check expertise
            if term in expertise_text:
                score += 2.0
        
        return score

    async def _get_all_expert_keys(self):
        """Helper method to get all expert keys from Redis."""
        try:
            patterns = [
                'meta:expert:*'  # Pattern used for storing expert data
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
            
            logger.info(f"Found {len(unique_keys)} total unique expert keys in Redis")
            return unique_keys
            
        except Exception as e:
            logger.error(f"Error retrieving expert keys: {e}")
            return []

    async def get_relevant_experts(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Retrieve experts from Redis with hybrid search combining keyword and semantic matching."""
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve experts")
                return [], "Our expert database is currently unavailable. Please try again later."
            
            # Get all expert keys
            expert_keys = await self._get_all_expert_keys()
            
            if not expert_keys:
                return [], "No expert profiles found in the database."
            
            # Parse specific requests for expert counts
            count_pattern = r'(\d+)\s+experts?'
            count_match = re.search(count_pattern, query, re.IGNORECASE)
            requested_count = int(count_match.group(1)) if count_match else limit
            
            # Extract potential name from query
            name_match = self._extract_name_from_query(query)
            expertise_terms = self._extract_expertise_terms(query)
            
            # Create query embedding for semantic search if model available
            query_embedding = None
            if self.redis_manager.embedding_model is not None:
                try:
                    query_embedding = self.redis_manager.embedding_model.encode(query)
                except Exception as e:
                    logger.error(f"Error creating query embedding: {e}")
            
            # Retrieve and score experts
            matched_experts = []
            
            # Remove stopwords from query
            stopwords = self._get_stopwords()
            query_terms = {word.lower() for word in query.split() if word.lower() not in stopwords}
            
            for key in expert_keys:
                try:
                    # Retrieve expert metadata
                    raw_metadata = self.redis_manager.redis_text.hgetall(key)
                    
                    # Skip entries without essential metadata
                    if not raw_metadata or 'name' not in raw_metadata:
                        continue
                    
                    # Extract expert ID
                    expert_id = raw_metadata.get('id', '')
                    
                    # Reconstruct metadata
                    metadata = {
                        'id': expert_id,
                        'name': raw_metadata.get('name', ''),
                        'first_name': raw_metadata.get('first_name', ''),
                        'last_name': raw_metadata.get('last_name', ''),
                        'expertise': self._safe_json_load(raw_metadata.get('expertise', '{}')),
                    }
                    
                    # Calculate match score
                    keyword_score = self._calculate_expert_keyword_score(
                        metadata, query_terms, name_match, expertise_terms
                    )
                    
                    # Add semantic scoring if available
                    semantic_score = 0.0
                    if query_embedding is not None:
                        expert_embedding_bytes = self.redis_manager.redis_binary.get(f"emb:expert:{expert_id}")
                        if expert_embedding_bytes:
                            expert_embedding = np.frombuffer(expert_embedding_bytes, dtype=np.float32)
                            # Calculate cosine similarity
                            semantic_score = np.dot(query_embedding, expert_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(expert_embedding)
                            )
                    
                    # Combine scores (70% keyword, 30% semantic when available)
                    if semantic_score > 0:
                        combined_score = 0.7 * keyword_score + 0.3 * semantic_score
                    else:
                        combined_score = keyword_score
                    
                    # Add to results if score is positive
                    if combined_score > 0:
                        # Attach match score for later sorting
                        metadata['_match_score'] = combined_score
                        matched_experts.append(metadata)
                
                except Exception as e:
                    logger.error(f"Error processing expert {key}: {e}")
            
            # Sort experts by match score
            sorted_experts = sorted(
                matched_experts,
                key=lambda x: x.get('_match_score', 0),
                reverse=True
            )
            
            # Limit to requested count
            top_experts = sorted_experts[:requested_count]
            
            # Remove internal match score before returning
            for expert in top_experts:
                expert.pop('_match_score', None)
            
            logger.info(f"Found {len(top_experts)} experts matching query")
            
            return top_experts, None
        
        except Exception as e:
            logger.error(f"Error retrieving experts: {e}")
            return [], "We encountered an error searching for experts. Try simplifying your query."

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
            
    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model is not available."""
        logger.info("Creating fallback embedding")
        # Create a deterministic embedding based on character values
        embedding = np.zeros(384)  # Standard dimension for simple embeddings
        for i, char in enumerate(text):
            embedding[i % len(embedding)] += ord(char) / 1000
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    

    
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

    def _get_stopwords(self) -> Set[str]:
        """Return common stopwords to ignore in queries"""
        return {
            'what', 'when', 'where', 'which', 'who', 'why', 'how', 
            'the', 'and', 'for', 'that', 'this', 'are', 'with',
            'list', 'show', 'find', 'get', 'give', 'me', 'about',
            'publications', 'publication', 'papers', 'paper', 'research',
            'from', 'during', 'between'
        }

    def _calculate_publication_relevance(self, metadata: Dict[str, Any], query_terms: Set[str]) -> float:
        """Calculate relevance score for a publication with improved weighting"""
        score = 0
        
        # Extract relevant fields for matching
        title = metadata.get('title', '').lower()
        summary = metadata.get('summary_snippet', '').lower()
        field = metadata.get('field', '').lower()
        
        # Process authors with better error handling
        authors = []
        domains = []
        if metadata.get('authors'):
            try:
                authors_data = json.loads(metadata.get('authors', '[]'))
                if isinstance(authors_data, list):
                    authors = [str(a).lower() for a in authors_data if a]
                elif isinstance(authors_data, dict):
                    authors = [str(v).lower() for v in authors_data.values() if v]
            except json.JSONDecodeError:
                authors = []
                
        if metadata.get('domains'):
            try:
                domains_data = json.loads(metadata.get('domains', '[]'))
                domains = [str(d).lower() for d in domains_data if d]
            except json.JSONDecodeError:
                domains = []
        
        # Apply semantic weights to different fields
        for keyword in query_terms:
            if keyword in title:
                score += 4  # Higher weight for title matches
            if keyword in summary:
                score += 2  # Medium weight for summary matches
            if keyword in field:
                score += 2.5  # Increased weight for field matches
            if any(keyword in author for author in authors):
                score += 3  # Higher weight for author matches
            if any(keyword in domain for domain in domains):
                score += 2  # Medium weight for domain matches
        
        # Apply multiplier for publications with DOIs for higher quality
        if metadata.get('doi') and metadata.get('doi').strip() and metadata.get('doi') != 'None':
            score *= 1.5
        
        return score
    
    async def debug_publication_data(self, query: str = "nutrition", limit: int = 5):
        """Debug method to check what publication data is actually being retrieved from Redis"""
        try:
            publications, suggestion = await self.get_relevant_publications(query, limit)
            
            if not publications:
                logger.info(f"No publications found for query '{query}'")
                if suggestion:
                    logger.info(f"Suggestion: {suggestion}")
                return
                
            logger.info(f"Found {len(publications)} publications for query '{query}'")
            
            for idx, pub in enumerate(publications):
                logger.info(f"Publication {idx+1}:")
                logger.info(f"  Title: {pub.get('title', 'N/A')}")
                logger.info(f"  Year: {pub.get('publication_year', 'N/A')}")
                logger.info(f"  DOI: {pub.get('doi', 'N/A')}")
                
                # Check if DOI is properly formatted
                if pub.get('doi'):
                    doi = pub.get('doi')
                    if not doi.startswith('https://doi.org/') and not doi.startswith('10.'):
                        logger.warning(f"  Unusual DOI format: {doi}")
                
                # Log author data to verify format
                authors = pub.get('authors', [])
                if authors:
                    if isinstance(authors, list):
                        logger.info(f"  Authors (list): {authors}")
                    elif isinstance(authors, str):
                        logger.info(f"  Authors (string): {authors}")
                    elif isinstance(authors, dict):
                        logger.info(f"  Authors (dict): {authors}")
                    else:
                        logger.info(f"  Authors (unknown type): {type(authors)}")
                
                # Log full publication data for reference
                logger.debug(f"  Full data: {pub}")
                
        except Exception as e:
            logger.error(f"Error in debug_publication_data: {e}")

    def _prepare_publication_metadata(self, metadata: Dict[str, Any]) -> None:
        """Prepare publication metadata for response formatting"""
        # Process authors data
        if metadata.get('authors'):
            try:
                metadata['authors'] = json.loads(metadata.get('authors', '[]'))
            except:
                metadata['authors'] = []
                
        # Process domains data
        if metadata.get('domains'):
            try:
                metadata['domains'] = json.loads(metadata.get('domains', '[]'))
            except:
                metadata['domains'] = []

    async def _check_for_expert_publications(self, query: str) -> Dict[str, float]:
        """Check if query mentions experts and find their linked publications with direct matcher integration"""
        publication_confidence = {}
        
        try:
            # Extract potential name fragments from query
            name_fragments = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
            
            if not name_fragments:
                return publication_confidence
                
            # Use EnhancedMatcher directly when possible
            try:
                from ai_services_api.services.centralized_repository.expert_matching.models import EnhancedMatcher
                matcher = EnhancedMatcher()
                
                # First try to get experts by name from the matcher
                expert_ids = []
                for name in name_fragments:
                    # Use the normalized name functionality from matcher
                    normalized_name = matcher._normalize_name(name)
                    if normalized_name:
                        # Connect to database to search for matching experts
                        async with DatabaseConnector.get_connection() as conn:
                            query_text = """
                                SELECT id FROM experts_expert
                                WHERE first_name ILIKE $1 OR last_name ILIKE $1
                            """
                            rows = await conn.fetch(query_text, f"%{name}%")
                            expert_ids.extend([row['id'] for row in rows])
                
                if expert_ids:
                    # Get linked publications with confidence scores from expert_resource_links
                    expert_placeholders = ",".join([f"${i+1}" for i in range(len(expert_ids))])
                    query_text = f"""
                        SELECT resource_id, confidence_score 
                        FROM expert_resource_links
                        WHERE expert_id IN ({expert_placeholders})
                        ORDER BY confidence_score DESC
                    """
                    
                    async with DatabaseConnector.get_connection() as conn:
                        rows = await conn.fetch(query_text, *expert_ids)
                        
                        for row in rows:
                            resource_id = str(row['resource_id'])
                            confidence = float(row['confidence_score'])
                            
                            # Use highest confidence score if multiple experts match
                            if resource_id not in publication_confidence or confidence > publication_confidence[resource_id]:
                                publication_confidence[resource_id] = confidence
                                
                    logger.info(f"Found {len(publication_confidence)} publications linked to experts mentioned in query (EnhancedMatcher)")
                    
            except ImportError:
                # Fall back to direct database query if EnhancedMatcher not available
                logger.info("EnhancedMatcher not available, using direct database query")
                
                # Look for experts whose names might be mentioned
                async with DatabaseConnector.get_connection() as conn:
                    placeholders = ",".join([f"${i+1}" for i in range(len(name_fragments))])
                    query_text = f"""
                        SELECT e.id, e.first_name, e.last_name, l.resource_id, l.confidence_score
                        FROM experts_expert e
                        JOIN expert_resource_links l ON e.id = l.expert_id
                        WHERE e.first_name IN ({placeholders}) OR e.last_name IN ({placeholders})
                        ORDER BY l.confidence_score DESC
                    """
                    
                    # Double the name fragments list for both first_name and last_name checks
                    query_params = name_fragments + name_fragments
                    
                    # Execute query and process results
                    rows = await conn.fetch(query_text, *query_params)
                    
                    for row in rows:
                        resource_id = str(row['resource_id'])
                        confidence = float(row['confidence_score'])
                        
                        # Use highest confidence score if multiple experts match
                        if resource_id not in publication_confidence or confidence > publication_confidence[resource_id]:
                            publication_confidence[resource_id] = confidence
                            
                    logger.info(f"Found {len(publication_confidence)} publications linked to experts mentioned in query (direct query)")
                
        except Exception as e:
            logger.error(f"Error checking for expert publications: {e}")
            
        return publication_confidence

        
    def _generate_search_suggestions(self, query: str) -> str:
        """Generate search suggestions for empty results"""
        suggestions = [
            "Try different keywords or more general terms",
            "Search by publication year (e.g., 'studies from 2020-2022')",
            "Include author names if known",
            "Browse our publications by topic at [APHRC Publications](https://aphrc.org/publications)"
        ]
        
        # Simple keyword analysis
        if len(query.split()) > 4:
            suggestions.insert(0, "Try a shorter, more focused query")
        if any(word in query.lower() for word in ["recent", "new", "latest"]):
            suggestions.append("For recent work, try 'publications from the last 3 years'")
        
        return "No exact matches found. Suggestions:\n- " + "\n- ".join(suggestions)

    
    async def _get_expert_for_publications(self, publication_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get experts associated with publications through expert-resource links"""
        publication_experts = {}
        
        if not publication_ids:
            return publication_experts
            
        try:
            # Convert all IDs to strings for consistent comparison
            publication_ids = [str(pub_id) for pub_id in publication_ids]
            
            # Create placeholders for SQL query
            placeholders = ",".join([f"${i+1}" for i in range(len(publication_ids))])
            
            async with DatabaseConnector.get_connection() as conn:
                query = f"""
                    SELECT l.resource_id, e.id, e.first_name, e.last_name, l.confidence_score
                    FROM expert_resource_links l
                    JOIN experts_expert e ON l.expert_id = e.id
                    WHERE l.resource_id IN ({placeholders})
                    AND l.confidence_score >= 0.7
                    ORDER BY l.confidence_score DESC
                """
                
                rows = await conn.fetch(query, *publication_ids)
                
                # Group experts by publication
                for row in rows:
                    resource_id = str(row['resource_id'])
                    
                    if resource_id not in publication_experts:
                        publication_experts[resource_id] = []
                        
                    expert = {
                        'id': row['id'],
                        'first_name': row['first_name'],
                        'last_name': row['last_name'],
                        'confidence': row['confidence_score']
                    }
                    
                    publication_experts[resource_id].append(expert)
                    
            logger.info(f"Found experts for {len(publication_experts)} publications")
            
        except Exception as e:
            logger.error(f"Error fetching experts for publications: {e}")
            
        return publication_experts

    

   
    async def _reset_rate_limited_after(self, seconds: int):
        """Reset rate limited flag after specified seconds."""
        await asyncio.sleep(seconds)
        self._rate_limited = False
        logger.info(f"Rate limit cooldown expired after {seconds} seconds")

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
    
    async def detect_intent(self, message: str, is_followup: bool = False) -> Dict[str, Any]:
        """Enhanced intent detection with improved pattern recognition and clarification support."""
        try:
            original_message = message
            message = message.lower()
            
            # Initialize default results
            result = {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': None
            }
            
            # First check for specific publication title requests - these should be highest priority
            specific_pub_patterns = [
                r'summarize the publication (.+)',
                r'publication titled (.+)',
                r'paper titled (.+)',
                r'article about (.+)',
                r'research on (.+)',
                r'study about (.+)',
                r'summarize (.+)',
                r'publication (.+)'
            ]
            
            # Check if the query contains a specific publication title
            for pattern in specific_pub_patterns:
                match = re.search(pattern, original_message, re.IGNORECASE)
                if match:
                    # If we find a specific publication title, this is a high-confidence publication intent
                    logger.info(f"Detected specific publication title request: {match.group(1)}")
                    result['intent'] = QueryIntent.PUBLICATION
                    result['confidence'] = 0.95  # High confidence for specific publication requests
                    return result
            
            # Check for publication list requests as a high-priority pattern
            list_patterns = [
                r'(\d+)\s+publications',
                r'list\s+(\d+)',
                r'top\s+(\d+)\s+papers',
                r'show\s+(\d+)\s+research',
                r'find\s+(\d+)\s+studies'
            ]
            
            for pattern in list_patterns:
                if re.search(pattern, message):
                    logger.info(f"Detected publication list request: {message}")
                    result['intent'] = QueryIntent.PUBLICATION
                    result['confidence'] = 0.95
                    return result
            
            # Check for specific publication requests
            publication_patterns = [
                r'doi', 
                r'10\.\d+\/', 
                r'publication',
                r'paper', 
                r'article',
                r'research',
                r'study',
                r'journal'
            ]
            
            # Count how many publication-related keywords match
            pub_keyword_count = sum(1 for pattern in publication_patterns if re.search(f'\b{pattern}\b', message))
            
            # If multiple publication keywords are found, this is likely a publication intent
            if pub_keyword_count >= 2:
                logger.info(f"Detected publication request with {pub_keyword_count} keywords: {message}")
                result['intent'] = QueryIntent.PUBLICATION
                result['confidence'] = 0.8 + (min(pub_keyword_count, 5) - 2) * 0.05  # Increase confidence with more keywords
                return result
            
            # Initialize scores for general intent detection
            intent_scores = {intent: 0.0 for intent in QueryIntent}
            
            # Enhanced pattern matching
            intent_patterns = {
                QueryIntent.PUBLICATION: [
                    (r'research paper', 1.0),
                    (r'papers from', 0.9),
                    (r'articles', 0.9),
                    (r'publications from (\d{4})', 1.0),
                    (r'published in (\d{4})', 1.0),
                    (r'recent studies', 0.9),
                    (r'bibliography', 0.8),
                    (r'works cited', 0.8),
                    (r'references', 0.7),
                    (r'latest research', 0.9),
                    (r'summary of', 0.9),
                    (r'summarize', 0.9)
                ],
                QueryIntent.NAVIGATION: [
                    (r'how do i get to', 1.0),
                    (r'link to', 0.9),
                    (r'url for', 1.0),
                    (r'go to', 0.8),
                    (r'access the', 0.7),
                    (r'browse', 0.7),
                    (r'site map', 0.9),
                    (r'homepage', 0.9),
                    (r'menu', 0.8)
                ]
            }
            
            # Score patterns
            for intent, patterns in intent_patterns.items():
                score = 0.0
                matches = 0
                
                for pattern, weight in patterns:
                    pattern_matches = re.findall(pattern, message)
                    if pattern_matches:
                        score += weight
                        matches += 1
                
                if matches > 0:
                    intent_scores[intent] = score / matches
            
            # Determine top intent
            max_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[max_intent]
            
            # For follow-up questions, we should be more lenient with confidence requirements
            if is_followup:
                # Lower threshold for follow-ups to maintain conversation flow
                threshold = 0.1
            else:
                threshold = 0.1
            
            # If confidence is too low and this isn't a follow-up, consider clarification
            if max_score < threshold and not is_followup:
                # Check for any publication-related terms to bias toward publication intent
                # This helps with ambiguous queries that might contain publication titles
                for pub_term in ["publication", "paper", "research", "study", "article", "journal"]:
                    if pub_term in message:
                        logger.info(f"Low confidence but found publication term '{pub_term}', defaulting to publication intent")
                        result['intent'] = QueryIntent.PUBLICATION
                        result['confidence'] = 0.7  # Reasonable confidence for publication-related terms
                        return result
                
                # If no publication terms and confidence is truly low, ask for clarification
                logger.info(f"Low confidence ({max_score}) for intent detection, requesting clarification")
                result['clarification'] = (
                    "Could you clarify what you're looking for? For example:\n"
                    "- For publications: 'Find papers about maternal health' or 'List 5 publications from 2020'\n"
                    "- For website navigation: 'Where can I find research datasets?' or 'How do I access reports?'"
                )
            
            # Set final result
            result['intent'] = max_intent
            result['confidence'] = max_score
            
            return result
        
        except Exception as e:
            logger.error(f"Intent detection failed: {e}", exc_info=True)
            return {
                'intent': QueryIntent.GENERAL,
                'confidence': 0.0,
                'clarification': "I'm sorry, I couldn't understand your request. Could you rephrase it?"
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

    def _extract_title_from_query(self, query: str) -> Optional[str]:
        """
        Extract potential publication title from query
        """
        title_patterns = [
            r'publication (?:titled|called|named) ["\']([^"\']+)["\']',
            r'publication ["\']([^"\']+)["\']',
            r'paper (?:titled|called|named) ["\']([^"\']+)["\']',
            r'paper ["\']([^"\']+)["\']',
            r'article (?:titled|called|named) ["\']([^"\']+)["\']',
            r'article ["\']([^"\']+)["\']',
            r'summarize (?:the publication|the paper|the article|) ["\']([^"\']+)["\']',
            r'summarize (?:the publication|the paper|the article|) (.+?)(?:\.|$)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()
        
        return None

    def _calculate_publication_match_score(
        self, 
        metadata: Dict[str, Any], 
        query_terms: Set[str], 
        title_match: Optional[str]
    ) -> float:
        """
        Calculate a comprehensive match score for a publication
        """
        score = 0.0
        
        # Exact title match
        pub_title = str(metadata.get('title', '')).lower()
        if title_match and title_match in pub_title:
            score += 2.0  # Significant boost for exact title match
        
        # Prepare search text
        search_text = ' '.join([
            pub_title,
            str(metadata.get('description', '')).lower(),
            str(metadata.get('abstract', '')).lower(),
            str(metadata.get('summary', '')).lower(),
            ' '.join([str(a).lower() for a in metadata.get('authors', []) if a]),
        ])
        
        # Term matching
        for term in query_terms:
            if term in search_text:
                score += 1.0
        
        # Similarity check for non-matching terms
        for term in query_terms:
            if term not in search_text:
                # Check if any word in the search text is similar to the term
                similar_match = any(
                    self._similar_strings(term, word) 
                    for word in search_text.split()
                )
                if similar_match:
                    score += 0.5
        
        return score


    def _similar_strings(self, s1: str, s2: str) -> bool:
            """Check if two strings are similar (for fuzzy title matching)."""
            if not s1 or not s2:
                return False
                
            # Quick exact match check
            if s1 == s2:
                return True
                
            # Get words from each string
            words1 = set(s1.split())
            words2 = set(s2.split())
            
            # Remove very short words
            words1 = {w for w in words1 if len(w) > 2}
            words2 = {w for w in words2 if len(w) > 2}
            
            if not words1 or not words2:
                return False
            
            # Check word overlap
            common_words = words1.intersection(words2)
            if len(common_words) >= 2:  # At least 2 significant words in common
                return True
                
            # For shorter titles, check percentage overlap
            overlap_ratio = len(common_words) / min(len(words1), len(words2))
            return overlap_ratio > 0.7 # Over 50% word overlap


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

    def _extract_author_from_query(self, query: str) -> Optional[str]:
        """Extract potential author name from query."""
        author_patterns = [
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'author\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'authored by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'written by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip().lower()
        
        # Try to find capitalized names that might be authors
        words = query.split()
        for i in range(len(words) - 1):
            if (i+1 < len(words) and 
                words[i][0].isupper() and words[i+1][0].isupper() and
                len(words[i]) > 1 and len(words[i+1]) > 1):
                return f"{words[i]} {words[i+1]}".lower()
        
        return None

    def _extract_year_from_query(self, query: str) -> Optional[str]:
        """Extract publication year from query."""
        # Look for 4-digit years between 1900 and current year
        current_year = datetime.now().year
        year_patterns = [
            r'in\s+(\d{4})',
            r'from\s+(\d{4})',
            r'year\s+(\d{4})',
            r'published\s+in\s+(\d{4})',
            r'(\d{4})'  # General pattern - will be validated after
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, query)
            if match:
                year = match.group(1)
                if 1900 <= int(year) <= current_year:
                    return year
        
        return None

    def _reconstruct_publication_metadata(self, raw_metadata: Dict[str, str]) -> Dict[str, Any]:
        """Reconstruct structured publication metadata from Redis."""
        metadata = {
            'id': raw_metadata.get('id', ''),
            'doi': raw_metadata.get('doi', ''),
            'title': raw_metadata.get('title', ''),
            'abstract': raw_metadata.get('abstract', ''),
            'summary': raw_metadata.get('summary', ''),
            'domains': self._safe_json_load(raw_metadata.get('domains', '[]')),
            'topics': self._safe_json_load(raw_metadata.get('topics', '{}')),
            'description': raw_metadata.get('description', ''),
            'expert_id': raw_metadata.get('expert_id', ''),
            'type': raw_metadata.get('type', 'publication'),
            'subtitles': self._safe_json_load(raw_metadata.get('subtitles', '{}')),
            'publishers': self._safe_json_load(raw_metadata.get('publishers', '{}')),
            'collection': raw_metadata.get('collection', ''),
            'date_issue': raw_metadata.get('date_issue', ''),
            'citation': raw_metadata.get('citation', ''),
            'language': raw_metadata.get('language', ''),
            'identifiers': self._safe_json_load(raw_metadata.get('identifiers', '{}')),
            'created_at': raw_metadata.get('created_at', ''),
            'updated_at': raw_metadata.get('updated_at', ''),
            'source': raw_metadata.get('source', 'unknown'),
            'authors': self._safe_json_load(raw_metadata.get('authors', '[]')),
            'publication_year': raw_metadata.get('publication_year', '')
        }
        return metadata

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

    def _calculate_publication_keyword_score(
        self, 
        metadata: Dict[str, Any], 
        query_terms: Set[str], 
        title_match: Optional[str],
        author_match: Optional[str],
        year_match: Optional[str]
    ) -> float:
        """
        Calculate weighted keyword match score for a publication based on multiple parameters.
        """
        score = 0.0
        
        # 1. EXACT TITLE MATCH (highest weight)
        pub_title = str(metadata.get('title', '')).lower()
        if title_match and (title_match in pub_title or self._fuzzy_title_match(title_match, pub_title)):
            score += 3.0  # Significant boost for exact or fuzzy title match
        
        # 2. AUTHOR MATCH
        pub_authors = metadata.get('authors', [])
        author_text = ""
        
        if isinstance(pub_authors, list):
            author_text = " ".join([str(a).lower() for a in pub_authors if a])
        elif isinstance(pub_authors, dict):
            author_text = " ".join([str(v).lower() for v in pub_authors.values() if v])
        elif isinstance(pub_authors, str):
            author_text = pub_authors.lower()
        
        if author_match and author_match in author_text:
            score += 2.5  # Strong boost for author match
        
        # 3. YEAR MATCH
        pub_year = str(metadata.get('publication_year', '')).strip()
        if year_match and year_match == pub_year:
            score += 2.0  # Good boost for year match
        
        # 4. PREPARE SEARCH TEXT FOR TERM MATCHING
        search_text = ' '.join([
            pub_title,
            str(metadata.get('abstract', '')).lower(),
            str(metadata.get('summary', '')).lower(),
            str(metadata.get('description', '')).lower(),
            author_text,
            # Add domains and topics for better matching
            " ".join([str(d).lower() for d in metadata.get('domains', []) if d]),
            " ".join([f"{k} {v}".lower() for k, v in metadata.get('topics', {}).items() if v])
        ])
        
        # 5. WEIGHTED TERM MATCHING
        # Count term occurrences for higher relevance
        term_counts = {}
        for term in query_terms:
            if len(term) <= 2:  # Skip very short terms
                continue
                
            # Count occurrences with word boundaries for better precision
            pattern = r'\b' + re.escape(term) + r'\b'
            occurrences = len(re.findall(pattern, search_text))
            
            if occurrences > 0:
                term_counts[term] = occurrences
                # Base score per occurrence with diminishing returns
                term_score = min(occurrences, 3) * 0.5
                score += term_score
        
        # 6. FIELD-SPECIFIC BONUSES
        # Boost if the publication has a DOI (more likely to be significant)
        if metadata.get('doi') and metadata.get('doi').strip() and metadata.get('doi') != 'None':
            score *= 1.2
        
        # Boost if it has an abstract/summary (more complete information)
        if metadata.get('abstract') or metadata.get('summary'):
            score *= 1.1
        
        # Apply a small recency bias for recent publications
        if pub_year and pub_year.isdigit():
            current_year = datetime.now().year
            years_old = current_year - int(pub_year)
            if years_old <= 3:  # Published in last 3 years
                score *= 1.15
        
        return score

    def _fuzzy_title_match(self, search_title: str, actual_title: str) -> bool:
        """Implement fuzzy matching for publication titles."""
        # 1. Direct substring match
        if search_title in actual_title:
            return True
            
        # 2. Word-by-word matching (80% of words must match)
        search_words = set(search_title.lower().split())
        title_words = set(actual_title.lower().split())
        
        if len(search_words) == 0:
            return False
            
        matching_words = search_words.intersection(title_words)
        if len(matching_words) / len(search_words) >= 0.8:
            return True
        
        # 3. Character-level Levenshtein similarity for short titles
        if len(search_title) < 30 and len(actual_title) < 100:
            try:
                from Levenshtein import ratio
                similarity = ratio(search_title, actual_title)
                if similarity > 0.7:  # 70% similarity threshold
                    return True
            except ImportError:
                # Fallback if Levenshtein library not available
                common_chars = sum(1 for c in search_title if c in actual_title)
                similarity = common_chars / len(search_title)
                if similarity > 0.8:  # 80% character overlap threshold
                    return True
        
        return False

    async def get_relevant_publications(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieve publications from Redis with hybrid search combining keyword and semantic matching.
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
            
            # Extract potential title, author, and year parameters from query
            title_match = self._extract_title_from_query(query)
            author_match = self._extract_author_from_query(query)
            year_match = self._extract_year_from_query(query)
            
            # Create query embedding for semantic search if model available
            query_embedding = None
            if self.redis_manager.embedding_model is not None:
                try:
                    query_embedding = self.redis_manager.embedding_model.encode(query)
                except Exception as e:
                    logger.error(f"Error creating query embedding: {e}")
            
            # Comprehensive publication retrieval with hybrid approach
            matched_publications = []
            keyword_scores = {}
            semantic_scores = {}
            
            # Clean query terms (remove stopwords)
            stopwords = self._get_stopwords()
            query_terms = {word.lower() for word in query.split() if word.lower() not in stopwords}
            
            for key in publication_keys:
                try:
                    # Retrieve full publication metadata
                    raw_metadata = self.redis_manager.redis_text.hgetall(key)
                    
                    # Skip entries without essential metadata
                    if not raw_metadata or 'title' not in raw_metadata:
                        continue
                    
                    # Extract publication ID for tracking
                    pub_id = raw_metadata.get('id', '')
                    
                    # Reconstruct metadata exactly as it was stored
                    metadata = self._reconstruct_publication_metadata(raw_metadata)
                    
                    # 1. KEYWORD MATCHING SCORE
                    keyword_score = self._calculate_publication_keyword_score(
                        metadata, query_terms, title_match, author_match, year_match
                    )
                    
                    # 2. SEMANTIC SIMILARITY SCORE (if embedding available)
                    semantic_score = 0.0
                    if query_embedding is not None:
                        pub_embedding_bytes = self.redis_manager.redis_binary.get(f"emb:resource:{pub_id}")
                        if pub_embedding_bytes:
                            pub_embedding = np.frombuffer(pub_embedding_bytes, dtype=np.float32)
                            # Calculate cosine similarity
                            semantic_score = np.dot(query_embedding, pub_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(pub_embedding)
                            )
                    
                    # 3. COMBINE SCORES (70% keyword, 30% semantic when available)
                    if semantic_score > 0:
                        combined_score = 0.7 * keyword_score + 0.3 * semantic_score
                    else:
                        combined_score = keyword_score
                    
                    # Save scores for logging/debugging
                    keyword_scores[pub_id] = keyword_score
                    semantic_scores[pub_id] = semantic_score
                    
                    # Add to results if score is positive
                    if combined_score > 0:
                        # Attach match score for later sorting
                        metadata['_match_score'] = combined_score
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
            
            # Log the search results with scores for debugging
            if len(matched_publications) > 0:
                top_scores = sorted(
                    [(pub_id, keyword_scores.get(pub_id, 0), semantic_scores.get(pub_id, 0)) 
                    for pub_id in keyword_scores.keys()],
                    key=lambda x: x[1] + x[2],
                    reverse=True
                )[:5]
                logger.debug(f"Top publication scores: {top_scores}")
            
            # If we're returning zero publications but found some potential matches, suggest query refinement
            if not top_publications and matched_publications:
                return [], "No strong matches found. Try refining your query with more specific terms like author names, publication year, or keywords from the title."
            
            return top_publications, None
        
        except Exception as e:
            logger.error(f"Error retrieving publications: {e}")
            return [], "We encountered an error searching publications. Try simplifying your query."

    def format_publication_context(self, publications: List[Dict[str, Any]]) -> str:
        """
        Format publication data into a comprehensive, structured context for better LLM responses.
        
        Args:
            publications (List[Dict[str, Any]]): List of publication dictionaries to format
        
        Returns:
            str: Formatted publication context optimized for LLM consumption
        """
        if not publications:
            return "No publications found matching the query."
        
        # Determine formatting based on number of publications
        is_list_format = len(publications) > 1
        
        # Start with an appropriate header
        if is_list_format:
            context_parts = [f"Found {len(publications)} relevant publications:"]
        else:
            context_parts = ["Found the following relevant publication:"]
        
        # Process each publication with comprehensive formatting
        for idx, pub in enumerate(publications, 1):
            # Create a formatted publication entry
            pub_parts = []
            
            # Add index for list format
            prefix = f"{idx}. " if is_list_format else ""
            
            # TITLE - Always include as the main identifier
            if pub.get('title'):
                pub_parts.append(f"{prefix}Title: {pub.get('title')}")
                prefix = "" if not is_list_format else prefix  # Reset prefix after first use in non-list format
            
            # AUTHORS - Format consistently regardless of data structure
            authors = pub.get('authors', [])
            if authors:
                author_text = ""
                if isinstance(authors, list):
                    author_text = ", ".join(str(author) for author in authors if author)
                elif isinstance(authors, dict):
                    author_text = ", ".join(str(v) for k, v in authors.items() if v)
                elif isinstance(authors, str):
                    author_text = authors
                    
                if author_text:
                    pub_parts.append(f"{prefix}Authors: {author_text}")
            
            # PUBLICATION YEAR - Important for contextualizing the research
            if pub.get('publication_year'):
                pub_parts.append(f"{prefix}Publication Year: {pub.get('publication_year')}")
            
            # DOI - Critical for findability and citation
            doi = pub.get('doi', '').strip()
            if doi and doi.lower() != 'none':
                # Format DOI consistently as a URL if possible
                if not doi.startswith('https://doi.org/') and doi.startswith('10.'):
                    doi = f"https://doi.org/{doi}"
                pub_parts.append(f"{prefix}DOI: {doi}")
            
            # SOURCE/JOURNAL - Adds credibility information
            if pub.get('source') and pub.get('source') != 'unknown':
                pub_parts.append(f"{prefix}Source: {pub.get('source')}")
            
            # ABSTRACT/SUMMARY - Include for content understanding
            if pub.get('abstract'):
                # Truncate very long abstracts to keep context manageable
                abstract = pub.get('abstract')
                if len(abstract) > 800:
                    abstract = abstract[:800] + "..."
                pub_parts.append(f"{prefix}Abstract: {abstract}")
            elif pub.get('summary'):
                summary = pub.get('summary')
                if len(summary) > 800:
                    summary = summary[:800] + "..."
                pub_parts.append(f"{prefix}Summary: {summary}")
            
            # DOMAINS/TOPICS - Include for better topical understanding
            domains = pub.get('domains', [])
            if domains and isinstance(domains, list) and len(domains) > 0:
                domains_text = ", ".join(str(d) for d in domains if d)
                if domains_text:
                    pub_parts.append(f"{prefix}Research Domains: {domains_text}")
            
            # Add CITATION if available for scholarly context
            if pub.get('citation') and str(pub.get('citation')).strip():
                pub_parts.append(f"{prefix}Citation: {pub.get('citation')}")
                
            # Add any EXPERT connections if applicable
            expert_id = pub.get('expert_id')
            if expert_id:
                pub_parts.append(f"{prefix}Associated with APHRC Expert ID: {expert_id}")
            
            # Join all publication parts with explicit newlines and add extra spacing between fields
            formatted_pub = "\n".join(pub_parts) + "\n"  # Extra newline after each publication
            
            # Add a separator line between publications for clearer formatting
            if is_list_format and idx < len(publications):
                formatted_pub += "\n" + "-" * 40 + "\n"
            
            context_parts.append(formatted_pub)
        
        # Add instructions for the model about how to use this data
        usage_instructions = """
    When discussing these publications in your response:
    - Reference specific details from the publications
    - Use the exact titles, authors, and years provided
    - Include DOI links when relevant
    - Mention research domains to provide context
    - IMPORTANT: Maintain line-by-line formatting for readability
    """

        # Join everything with double newlines for better separation
        return "\n\n".join(context_parts) + "\n\n" + usage_instructions

    def _create_system_message(self, intent: QueryIntent) -> str:
        """
        Create detailed system prompt for better response quality.
        """
        base_prompts = {
            "common": (
                "You are APHRC's AI assistant. Provide helpful, accurate, and well-structured responses. "
                "Include relevant details while maintaining clarity and conciseness. "
                "Focus on being informative and useful to the user."
            ),
            QueryIntent.NAVIGATION: (
                "Guide website navigation concisely. Include direct URLs. "
                "Format: [Section Name](URL) - Brief description. "
                "Prioritize most relevant sections first."
            ),
            QueryIntent.PUBLICATION: (
                "When discussing APHRC publications, follow these guidelines:\n\n"
                "1. FORMAT RESPONSES PROPERLY:\n"
                "   - For multiple publications: Use clear headings and maintain separate sections for each publication\n"
                "   - For single publications: Provide a comprehensive overview with all relevant details\n"
                "   - Use markdown formatting for better readability (bold for titles, etc.)\n\n"
                "2. INCLUDE ESSENTIAL INFORMATION:\n"
                "   - Publication title (always in full)\n"
                "   - Complete author list (never abbreviate or truncate)\n"
                "   - Publication year\n"
                "   - DOI link, formatted as clickable when possible\n"
                "   - Key findings or summary when available\n\n"
                "3. CONTENT FIDELITY RULES:\n"
                "   - ONLY reference publications explicitly provided in the context\n"
                "   - NEVER create, invent, or embellish publication details\n"
                "   - Present publication information exactly as provided\n"
                "   - If specific information is not provided, do not make assumptions\n\n"
                "4. QUANTITY HANDLING:\n"
                "   - If asked for a specific number of publications (e.g., '5 publications') and fewer are available,\n"
                "     explicitly state 'I found X publications out of the Y requested'\n"
                "   - If no publications match the criteria, clearly state this and suggest alternatives\n\n"
                "5. CITATION SUPPORT:\n"
                "   - If asked about citations, provide the full citation in a standard academic format\n"
                "   - Always include DOI links when available for easy access\n\n"
                "6. EXPERT CONNECTIONS:\n"
                "   - When APHRC experts are associated with publications, highlight their connection to the research\n"
                "   - Mention any institutional affiliations or research groups connected to the publications\n\n"
                "7. CONTEXTUAL INFORMATION:\n"
                "   - Place publications in their broader research context when possible\n"
                "   - Mention research domains or fields to provide better understanding"
            ),
            QueryIntent.EXPERT: (
                "When discussing APHRC experts, follow these guidelines:\n\n"
                "1. FORMAT RESPONSES PROPERLY:\n"
                "   - For multiple experts: Use clear headings and maintain separate sections for each expert\n"
                "   - For single experts: Provide a comprehensive overview of their expertise and work\n"
                "   - Use markdown formatting for better readability\n\n"
                "2. INCLUDE ESSENTIAL INFORMATION:\n"
                "   - Full name of the expert\n"
                "   - Areas of expertise and research focus\n"
                "   - Key publications when available\n"
                "   - Research domains they work in\n\n"
                "3. CONTENT FIDELITY RULES:\n"
                "   - ONLY reference experts explicitly provided in the context\n"
                "   - NEVER create, invent, or embellish expert details\n"
                "   - Present expert information exactly as provided\n"
                "   - If specific information is not provided, do not make assumptions\n\n"
                "4. EXPERT-PUBLICATION CONNECTIONS:\n"
                "   - When publications are associated with experts, highlight this connection\n"
                "   - Contextualize how their research relates to their expertise\n\n"
                "5. CONTEXTUAL INFORMATION:\n"
                "   - Place experts in their broader research context when possible\n"
                "   - Mention research domains or fields to provide better understanding"
            ),
            QueryIntent.GENERAL: (
                "Provide a balanced overview of APHRC's work, combining navigation help with research insights. "
                "Direct users to specific resources when applicable and offer clear paths to more information."
            )
        }
        
        response_guidelines = (
            "Structure your responses thoughtfully:\n"
            "1. Begin with a direct answer to the query\n"
            "2. Provide supporting details organized in a logical flow\n"
            "3. Use natural, conversational language without unnecessary jargon\n"
            "4. End with relevant suggestions or next steps when appropriate\n"
            "5. For complex information, use formatting to improve readability"
        )
        
        # Combined prompt based on intent
        if intent == QueryIntent.PUBLICATION or intent == QueryIntent.EXPERT:
            return f"{base_prompts['common']}\n\n{base_prompts[intent]}\n\n{response_guidelines}"
        else:
            return f"{base_prompts['common']}\n\n{base_prompts[intent]}\n\n{response_guidelines}"
  
    async def generate_async_response(self, message: str, conversation_history: Optional[List[Dict]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Enhanced response generation with improved publication and expert profile handling"""
        start_time = time.time()
        logger.info(f"Starting async response generation for message: {message}")
        
        try:
            # Detect intent with conversation context
            is_followup = conversation_history is not None and len(conversation_history) > 0
            logger.info(f"Follow-up detected: {is_followup}")
            
            # Detect intent with conversation context, handling follow-ups
            try:
                intent_data = await self.detect_intent(message, is_followup)
                intent = intent_data['intent']
                clarification = intent_data.get('clarification')
                confidence = intent_data.get('confidence', 0.0)
                
                # Handle clarification questions first
                if clarification:
                    logger.info(f"Clarification needed: {clarification}")
                    yield {'chunk': clarification, 'is_metadata': False}
                    return
                
            except Exception as task_error:
                logger.error(f"Error in task processing: {task_error}", exc_info=True)
                intent = QueryIntent.GENERAL
                confidence = 0.0
                clarification = None
            
            # Log intent results
            logger.info(f"Intent detected: {intent} (Confidence: {confidence})")
            
            # Initialize default context
            context = "I'll help you find information about APHRC's work."
            
            # If publication intent, try to retrieve real publications with enhanced handling
            if intent == QueryIntent.PUBLICATION and self.redis_manager:
                logger.info("Publication intent detected, retrieving real publication data")
                try:
                    # Get relevant publications from Redis
                    publications, suggestion = await self.get_relevant_publications(message)
                    
                    # Handle no publications found, but offer suggestion if available
                    if not publications and suggestion:
                        logger.info("No publications found, but suggestion available")
                        yield {'chunk': suggestion, 'is_metadata': False}
                        return
                    
                    if publications:
                        # Enrich publications with expert information
                        enriched_publications = await self._enrich_publications_with_experts(publications)
                        
                        # Format publications into readable context
                        context = self.format_publication_context(enriched_publications)
                        logger.info(f"Retrieved and formatted {len(publications)} publications for context")
                    else:
                        context = "I don't have specific publications on this topic. I can provide general information about APHRC's research areas instead."
                        logger.info("No relevant publications found in Redis")
                except Exception as pub_error:
                    logger.error(f"Error retrieving publications: {pub_error}")
                    context = "I can help you find information about APHRC's publications, though I'm having trouble accessing specific details right now."
            
            # If expert intent, retrieve expert profiles
            elif intent == QueryIntent.EXPERT and self.redis_manager:
                logger.info("Expert intent detected, retrieving expert profiles")
                try:
                    # Get relevant experts from Redis
                    experts, suggestion = await self.get_relevant_experts(message)
                    
                    # Handle no experts found, but offer suggestion if available
                    if not experts and suggestion:
                        logger.info("No experts found, but suggestion available")
                        yield {'chunk': suggestion, 'is_metadata': False}
                        return
                    
                    if experts:
                        # Format experts into readable context
                        context = self.format_expert_context(experts)
                        logger.info(f"Retrieved and formatted {len(experts)} expert profiles for context")
                    else:
                        context = "I don't have specific expert profiles on this topic. I can provide general information about APHRC's research areas instead."
                        logger.info("No relevant expert profiles found in Redis")
                except Exception as expert_error:
                    logger.error(f"Error retrieving expert profiles: {expert_error}")
                    context = "I can help you find information about APHRC's experts, though I'm having trouble accessing specific details right now."
            
            # Manage context window
            logger.debug("Managing context window")
            self.manage_context_window({'text': context, 'query': message})
            
            # Prepare messages for model with enhanced system message
            logger.debug("Preparing system and human messages")
            system_message = self._create_system_message(intent)
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Context: {context}\n\nQuery: {message}")
            ]
            
            # Initialize response tracking
            response_chunks = []
            buffer = ""
            
            try:
                logger.info("Initializing model streaming")
                model = self.get_gemini_model()
                
                try:
                    # Properly handle both streaming and non-streaming responses
                    response_iterator = None
                    
                    try:
                        # First attempt with streaming (agenerate)
                        self.callback.queue = asyncio.Queue()  # Reset queue
                        response_data = await model.agenerate([messages])
                        
                        # Check if we have valid generations
                        if not response_data.generations or not response_data.generations[0]:
                            logger.warning("Empty response received from model")
                            yield {'chunk': "I'm sorry, but I couldn't generate a response at this time.", 'is_metadata': False}
                            return
                        
                        # Process the main response content
                        content = response_data.generations[0][0].text
                        
                        if not content:
                            logger.warning("Empty content received from model")
                            yield {'chunk': "I'm sorry, but I couldn't generate a response at this time.", 'is_metadata': False}
                            return
                        
                        # Process the content in smaller chunks for streaming-like behavior
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
                            yield {'chunk': current_chunk, 'is_metadata': False}
                            
                            # Small delay to simulate streaming
                            await asyncio.sleep(0.05)
                        
                    except Exception as streaming_error:
                        # If streaming fails, try non-streaming invoke
                        logger.warning(f"Streaming generation failed, falling back to standard invoke: {streaming_error}")
                        
                        # Use regular invoke with proper content extraction
                        model.streaming = False  # Turn off streaming
                        response = await model.ainvoke(messages)
                        content = self._extract_content_safely(response)
                        
                        if not content:
                            logger.warning("Empty content received from fallback invoke")
                            yield {'chunk': "I'm sorry, but I couldn't generate a response at this time.", 'is_metadata': False}
                            return
                        
                        # Process the content in chunks for consistent behavior
                        remaining_content = content
                        while remaining_content:
                            end_pos = min(100, len(remaining_content))
                            if end_pos < len(remaining_content):
                                for break_char in ['. ', '! ', '? ', '\n']:
                                    pos = remaining_content[:end_pos].rfind(break_char)
                                    if pos > 0:
                                        end_pos = pos + len(break_char)
                                        break
                            
                            current_chunk = remaining_content[:end_pos]
                            remaining_content = remaining_content[end_pos:]
                            
                            response_chunks.append(current_chunk)
                            logger.debug(f"Yielding chunk from fallback (length: {len(current_chunk)})")
                            yield {'chunk': current_chunk, 'is_metadata': False}
                            
                            # Small delay to simulate streaming
                            await asyncio.sleep(0.05)
                        
                    # Prepare complete response
                    complete_response = ''.join(response_chunks)
                    logger.info(f"Complete response generated. Total length: {len(complete_response)}")
                    
                    # Analyze response quality
                    quality_metrics = await self.analyze_quality(message, complete_response)
                    
                    # Yield metadata
                    logger.debug("Preparing and yielding metadata")
                    yield {
                        'is_metadata': True,
                        'metadata': {
                            'response': complete_response,
                            'timestamp': datetime.now().isoformat(),
                            'metrics': {
                                'response_time': time.time() - start_time,
                                'intent': {'type': intent.value, 'confidence': confidence},
                                'quality': quality_metrics
                            },
                            'error_occurred': False
                        }
                    }
                except Exception as stream_error:
                    logger.error(f"Error in response processing: {stream_error}", exc_info=True)
                    error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
                    yield {'chunk': error_message, 'is_metadata': False}
            
            except Exception as e:
                logger.error(f"Critical error generating response: {e}", exc_info=True)
                error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
                
                # Yield error chunk
                yield {'chunk': error_message, 'is_metadata': False}
                
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
            error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
            
            # Yield error chunk
            yield {'chunk': error_message, 'is_metadata': False}
            
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
        
    def get_suggested_questions(self, last_intent: QueryIntent) -> List[str]:
        """Get context-aware follow-up questions"""
        suggestions = {
            QueryIntent.PUBLICATION: [
                "Find more publications by the same authors",
                "See related publications from recent years",
                "Get citation formats for these papers"
            ],
            QueryIntent.NAVIGATION: [
                "Show more sections in this category",
                "Find contact information for this department",
                "Open this page in my browser"
            ],
            QueryIntent.GENERAL: [
                "What other programs does APHRC offer?",
                "Tell me about upcoming events",
                "Who are the key researchers in this field?"
            ]
        }
        return suggestions.get(last_intent, [])