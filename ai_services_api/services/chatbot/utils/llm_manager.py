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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enum for different types of query intents."""
    NAVIGATION = "navigation"
    PUBLICATION = "publication"
    EXPERT = "expert"
    GENERAL = "general"

class CustomAsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.finished = False
        self.error = None

    async def on_llm_start(self, *args, **kwargs):
        self.finished = False
        self.error = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break

    async def on_chat_model_start(self, serialized, messages, *args, **kwargs):
        try:
            prompts = []
            for msg in messages:
                if isinstance(msg, list):
                    for item in msg:
                        if hasattr(item, 'content'):
                            prompts.append(item.content)
                        elif isinstance(item, dict) and 'content' in item:
                            prompts.append(item['content'])
                        elif isinstance(item, str):
                            prompts.append(item)
                elif hasattr(msg, 'content'):
                    prompts.append(msg.content)
                elif isinstance(msg, dict) and 'content' in msg:
                    prompts.append(msg['content'])
                elif isinstance(msg, str):
                    prompts.append(msg)
            
            await self.on_llm_start(serialized, prompts, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in on_chat_model_start: {e}")
            self.error = e

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        if token and not self.finished:
            try:
                await self.queue.put(token)
            except Exception as e:
                logger.error(f"Error putting token in queue: {e}")
                self.error = e

    async def on_llm_end(self, *args, **kwargs):
        try:
            self.finished = True
            await self.queue.put(None)
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
            self.error = e

    async def on_llm_error(self, error: Exception, *args, **kwargs):
        try:
            self.error = error
            self.finished = True
            await self.queue.put(f"Error: {str(error)}")
            await self.queue.put(None)
        except Exception as e:
            logger.error(f"Error in on_llm_error handler: {e}")

    def reset(self):
        self.finished = False
        self.error = None
        self.queue = asyncio.Queue()

class GeminiLLMManager:
    def __init__(self):
        """Initialize the LLM manager with semantic search capabilities."""
        try:
            # Load API key
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            # Initialize callback handler
            self.callback = CustomAsyncCallbackHandler()
            
            # Initialize Redis manager
            try:
                self.redis_manager = ExpertRedisIndexManager()
                logger.info("Redis manager initialized successfully")
            except Exception as redis_error:
                logger.error(f"Error initializing Redis manager: {redis_error}")
                self.redis_manager = None
            
            # Initialize semantic search model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
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

    # Core Methods
    def get_gemini_model(self):
        """Initialize and return the Gemini model."""
        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            stream=True,
            model="gemini-2.0-flash-thinking-exp-01-21",
            convert_system_message_to_human=True,
            callbacks=[self.callback],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_retries=5,
            timeout=30
        )

    def _extract_content_safely(self, model_response):
        """Safely extract content from model response."""
        try:
            if isinstance(model_response, str):
                return model_response
            if hasattr(model_response, 'content'):
                return model_response.content
            if isinstance(model_response, list):
                if model_response and hasattr(model_response[-1], 'content'):
                    return model_response[-1].content
                contents = []
                for msg in model_response:
                    if hasattr(msg, 'content'):
                        contents.append(msg.content)
                if contents:
                    return ''.join(contents)
            if isinstance(model_response, dict) and 'content' in model_response:
                return model_response['content']
            if hasattr(model_response, 'generations'):
                generations = model_response.generations
                if generations and generations[0]:
                    if hasattr(generations[0][0], 'text'):
                        return generations[0][0].text
            if hasattr(model_response, 'text'):
                return model_response.text
            return str(model_response)
        except Exception as e:
            logger.error(f"Error extracting content from model response: {e}")
            return ""

    def _get_default_quality(self) -> Dict:
        """Return default quality metric values."""
        return {
            'helpfulness_score': 0.5,
            'hallucination_risk': 0.5,
            'factual_grounding_score': 0.5,
            'unclear_elements': [],
            'potentially_fabricated_elements': [],
            'requires_review': False
        }

    # Intent Detection
    def detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect query intent using semantic similarity and pattern matching."""
        try:
            # First check for exact matches in patterns
            for intent, config in self.intent_patterns.items():
                for pattern, weight in config['patterns']:
                    if re.search(pattern, query, re.IGNORECASE):
                        return intent, weight
            
            # Fall back to semantic similarity if no pattern matches
            query_embedding = self.embedding_model.encode(query)
            
            intent_scores = {}
            for intent, config in self.intent_patterns.items():
                # Create embedding for intent description
                intent_description = f"Find information about {intent.value}"
                intent_embedding = self.embedding_model.encode(intent_description)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [query_embedding],
                    [intent_embedding]
                )[0][0]
                
                intent_scores[intent] = similarity
            
            # Get the highest scoring intent
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            if best_intent[1] >= self.confidence_threshold:
                return best_intent[0], best_intent[1]
            else:
                return QueryIntent.GENERAL, best_intent[1]
                
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return QueryIntent.GENERAL, 0.0

    # Publication Retrieval
    async def get_relevant_publications(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Retrieve publications using semantic similarity."""
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available")
                return [], "Our publication database is currently unavailable."
            
            # Get all publication keys
            publication_keys = await self._get_all_publication_keys()
            
            if not publication_keys:
                return [], "No publications found in the database."
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            matched_publications = []
            
            # Process publications in batches for efficiency
            batch_size = 50
            for i in range(0, len(publication_keys), batch_size):
                batch_keys = publication_keys[i:i+batch_size]
                
                # Get batch metadata
                batch_metadata = []
                for key in batch_keys:
                    try:
                        raw_metadata = self.redis_manager.redis_text.hgetall(key)
                        if raw_metadata:
                            metadata = self._reconstruct_publication_metadata(raw_metadata)
                            batch_metadata.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing publication {key}: {e}")
                
                # Skip if no valid metadata in batch
                if not batch_metadata:
                    continue
                
                # Generate embeddings for batch
                search_texts = [
                    self._prepare_publication_metadata(pub) 
                    for pub in batch_metadata
                ]
                batch_embeddings = self.embedding_model.encode(search_texts)
                
                # Calculate similarities
                similarities = cosine_similarity(
                    [query_embedding],
                    batch_embeddings
                )[0]
                
                # Add to matched publications with scores
                for idx, similarity in enumerate(similarities):
                    if similarity > 0.3:  # Minimum similarity threshold
                        pub = batch_metadata[idx]
                        pub['_similarity'] = similarity
                        matched_publications.append(pub)
            
            # Sort by similarity and limit results
            matched_publications.sort(key=lambda x: x.get('_similarity', 0), reverse=True)
            top_publications = matched_publications[:limit]
            
            # Remove internal similarity score
            for pub in top_publications:
                pub.pop('_similarity', None)
            
            logger.info(f"Found {len(top_publications)} publications matching query")
            return top_publications, None
            
        except Exception as e:
            logger.error(f"Error retrieving publications: {e}")
            return [], "We encountered an error searching publications."

    def _prepare_publication_metadata(self, pub: Dict) -> str:
        """Prepare publication metadata text for embedding."""
        text_parts = [
            pub.get('title', ''),
            pub.get('abstract', ''),
            pub.get('summary', ''),
            ' '.join(pub.get('authors', [])),
            ' '.join(pub.get('domains', [])),
            pub.get('description', '')
        ]
        return ' '.join([p for p in text_parts if p])

    def _reconstruct_publication_metadata(self, raw_metadata: Dict) -> Dict:
        """Reconstruct publication metadata from Redis hash."""
        return {
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

    async def _get_all_publication_keys(self):
        """Get all publication keys from Redis."""
        try:
            if not self.redis_manager:
                return []
                
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
            
            return list(set(publication_keys))
        except Exception as e:
            logger.error(f"Error retrieving publication keys: {e}")
            return []

    # Expert Retrieval
    async def get_relevant_experts(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Retrieve experts using semantic similarity."""
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available")
                return [], "Our expert database is currently unavailable."
            
            # Get all expert keys
            expert_keys = await self._get_all_expert_keys()
            
            if not expert_keys:
                return [], "No experts found in the database."
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            matched_experts = []
            
            # Process experts in batches
            batch_size = 50
            for i in range(0, len(expert_keys), batch_size):
                batch_keys = expert_keys[i:i+batch_size]
                
                # Get batch metadata
                batch_metadata = []
                for key in batch_keys:
                    try:
                        raw_metadata = self.redis_manager.redis_text.hgetall(key)
                        if raw_metadata:
                            metadata = self._reconstruct_expert_metadata(raw_metadata)
                            batch_metadata.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing expert {key}: {e}")
                
                if not batch_metadata:
                    continue
                
                # Generate embeddings for batch
                search_texts = [
                    self._prepare_expert_metadata(expert) 
                    for expert in batch_metadata
                ]
                batch_embeddings = self.embedding_model.encode(search_texts)
                
                # Calculate similarities
                similarities = cosine_similarity(
                    [query_embedding],
                    batch_embeddings
                )[0]
                
                # Add to matched experts with scores
                for idx, similarity in enumerate(similarities):
                    if similarity > 0.3:  # Minimum similarity threshold
                        expert = batch_metadata[idx]
                        expert['_similarity'] = similarity
                        matched_experts.append(expert)
            
            # Sort by similarity and limit results
            matched_experts.sort(key=lambda x: x.get('_similarity', 0), reverse=True)
            top_experts = matched_experts[:limit]
            
            # Remove internal similarity score
            for expert in top_experts:
                expert.pop('_similarity', None)
            
            logger.info(f"Found {len(top_experts)} experts matching query")
            return top_experts, None
            
        except Exception as e:
            logger.error(f"Error retrieving experts: {e}")
            return [], "We encountered an error searching experts."

    def _prepare_expert_metadata(self, expert: Dict) -> str:
        """Prepare expert metadata text for embedding."""
        text_parts = [
            expert.get('first_name', ''),
            expert.get('last_name', ''),
            expert.get('title', ''),
            expert.get('bio', ''),
            ' '.join(expert.get('expertise', [])),
            ' '.join(expert.get('domains', [])),
            expert.get('institution', '')
        ]
        return ' '.join([p for p in text_parts if p])

    def _reconstruct_expert_metadata(self, raw_metadata: Dict) -> Dict:
        """Reconstruct expert metadata from Redis hash."""
        return {
            'id': raw_metadata.get('id', ''),
            'first_name': raw_metadata.get('first_name', ''),
            'last_name': raw_metadata.get('last_name', ''),
            'title': raw_metadata.get('title', ''),
            'bio': raw_metadata.get('bio', ''),
            'expertise': self._safe_json_load(raw_metadata.get('expertise', '[]')),
            'domains': self._safe_json_load(raw_metadata.get('domains', '[]')),
            'institution': raw_metadata.get('institution', ''),
            'profile_url': raw_metadata.get('profile_url', ''),
            'created_at': raw_metadata.get('created_at', ''),
            'updated_at': raw_metadata.get('updated_at', '')
        }

    async def _get_all_expert_keys(self):
        """Get all expert keys from Redis."""
        try:
            if not self.redis_manager:
                return []
                
            cursor = 0
            expert_keys = []
            
            while cursor != 0 or len(expert_keys) == 0:
                cursor, batch = self.redis_manager.redis_text.scan(
                    cursor=cursor, 
                    match='meta:expert:*', 
                    count=100
                )
                expert_keys.extend(batch)
                
                if cursor == 0:
                    break
            
            return list(set(expert_keys))
        except Exception as e:
            logger.error(f"Error retrieving expert keys: {e}")
            return []

    # Response Generation
    async def generate_async_response(self, query: str) -> AsyncGenerator[str, None]:
        """Generate response with semantic context."""
        try:
            # Detect intent
            intent, confidence = self.detect_intent(query)
            
            # Get relevant content based on intent
            context = ""
            metadata = {
                'intent': {'type': intent.value, 'confidence': confidence},
                'content_matches': []
            }
            
            if intent == QueryIntent.PUBLICATION:
                publications, error = await self.get_relevant_publications(query)
                if publications:
                    context = self.format_publication_context(publications)
                    metadata['content_matches'] = [p['id'] for p in publications]
                    metadata['content_types'] = {'publication': len(publications)}
            
            elif intent == QueryIntent.EXPERT:
                experts, error = await self.get_relevant_experts(query)
                if experts:
                    context = self.format_expert_context(experts)
                    metadata['content_matches'] = [e['id'] for e in experts]
                    metadata['content_types'] = {'expert': len(experts)}
            
            # Prepare messages for LLM
            messages = [
                SystemMessage(content="You are a helpful research assistant."),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
            ]
            
            # Generate response
            model = self.get_gemini_model()
            response = await model.agenerate([messages])
            
            # Add metadata to response
            yield {'is_metadata': True, 'metadata': metadata}
            
            # Stream response content
            async for token in self.callback.aiter():
                if token is not None:
                    yield token
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error: {str(e)}"

    def format_publication_context(self, publications: List[Dict]) -> str:
        """Format publication data for context."""
        if not publications:
            return ""
            
        context = "Relevant Publications:\n"
        for i, pub in enumerate(publications, 1):
            context += f"{i}. Title: {pub.get('title', 'N/A')}\n"
            context += f"   Authors: {', '.join(pub.get('authors', ['Unknown']))}\n"
            context += f"   Year: {pub.get('publication_year', 'Unknown')}\n"
            if pub.get('doi'):
                context += f"   DOI: {pub.get('doi')}\n"
            if pub.get('abstract'):
                context += f"   Abstract: {pub.get('abstract')[:200]}...\n"
            context += "\n"
        
        return context

    def format_expert_context(self, experts: List[Dict]) -> str:
        """Format expert data for context."""
        if not experts:
            return ""
            
        context = "Relevant Experts:\n"
        for i, expert in enumerate(experts, 1):
            context += f"{i}. Name: {expert.get('first_name', '')} {expert.get('last_name', '')}\n"
            context += f"   Title: {expert.get('title', 'N/A')}\n"
            if expert.get('expertise'):
                context += f"   Expertise: {', '.join(expert.get('expertise'))}\n"
            if expert.get('bio'):
                context += f"   Bio: {expert.get('bio')[:200]}...\n"
            context += "\n"
        
        return context

    def create_context(self, relevant_data: List[Dict]) -> str:
        """Create context from relevant data."""
        if not relevant_data:
            return ""
            
        context_parts = []
        for item in relevant_data:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            
            if metadata.get('type') == 'publication':
                context_parts.append(
                    f"Publication: {metadata.get('title', 'N/A')}\n"
                    f"Authors: {', '.join(metadata.get('authors', ['Unknown']))}\n"
                    f"Content: {text[:300]}..."
                )
            elif metadata.get('type') == 'expert':
                context_parts.append(
                    f"Expert: {metadata.get('first_name', '')} {metadata.get('last_name', '')}\n"
                    f"Expertise: {', '.join(metadata.get('expertise', ['Unknown']))}\n"
                    f"Bio: {text[:300]}..."
                )
        
        return "\n\n".join(context_parts)

    def manage_context_window(self, new_context: Dict):
        """Manage conversation context window."""
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

    # Database Interactions
    async def _get_expert_for_publications(self, publication_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get experts associated with publications."""
        try:
            async with DatabaseConnector.get_connection() as conn:
                query = """
                    SELECT l.resource_id, e.id, e.first_name, e.last_name, 
                           e.title, l.confidence_score
                    FROM expert_resource_links l
                    JOIN experts_expert e ON l.expert_id = e.id
                    WHERE l.resource_id = ANY($1)
                    AND l.confidence_score >= 0.7
                    ORDER BY l.confidence_score DESC
                """
                
                rows = await conn.fetch(query, publication_ids)
                
                # Map publications to experts
                pub_expert_map = {}
                for row in rows:
                    pub_id = row['resource_id']
                    if pub_id not in pub_expert_map:
                        pub_expert_map[pub_id] = []
                    
                    pub_expert_map[pub_id].append({
                        'id': row['id'],
                        'first_name': row['first_name'],
                        'last_name': row['last_name'],
                        'title': row['title'],
                        'confidence': row['confidence_score']
                    })
                
                return pub_expert_map
                
        except Exception as e:
            logger.error(f"Error fetching experts for publications: {e}")
            return {}

    async def _get_expert_publications(self, expert_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get publications associated with an expert."""
        try:
            async with DatabaseConnector.get_connection() as conn:
                query = """
                    SELECT r.id, r.title, r.publication_year, r.doi, l.confidence_score
                    FROM expert_resource_links l
                    JOIN resources_resource r ON l.resource_id = r.id
                    WHERE l.expert_id = $1
                    AND l.confidence_score >= 0.7
                    ORDER BY l.confidence_score DESC, r.publication_year DESC
                    LIMIT $2
                """
                
                rows = await conn.fetch(query, expert_id, limit)
                
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

    async def _get_expertise_summary(self, expert_id: str) -> str:
        """Get summary of expert's expertise."""
        try:
            async with DatabaseConnector.get_connection() as conn:
                query = """
                    SELECT expertise FROM experts_expert
                    WHERE id = $1
                """
                
                row = await conn.fetchrow(query, expert_id)
                if row and row['expertise']:
                    return ', '.join(row['expertise'])
                
                return ""
                
        except Exception as e:
            logger.error(f"Error fetching expertise for expert {expert_id}: {e}")
            return ""

    # Quality Analysis
    async def analyze_quality(self, message: str, response: str = "") -> Dict:
        """Analyze response quality with hallucination detection."""
        try:
            prompt = f"""Analyze this query and response for quality metrics.
            Return ONLY a JSON object with these fields:
            {{
                "helpfulness_score": <0-1>,
                "hallucination_risk": <0-1>,
                "factual_grounding_score": <0-1>,
                "unclear_elements": [<strings>],
                "potentially_fabricated_elements": [<strings>]
            }}
            
            Query: {message}
            Response: {response if response else 'N/A'}
            """
            
            model = self.get_gemini_model()
            model_response = await model.ainvoke(prompt)
            response_text = self._extract_content_safely(model_response)
            
            if not response_text:
                return self._get_default_quality()
            
            try:
                return json.loads(response_text.strip())
            except json.JSONDecodeError:
                return self._get_default_quality()
                
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return self._get_default_quality()

    async def analyze_sentiment(self, message: str) -> Dict:
        """Legacy sentiment analysis (redirects to quality analysis)."""
        quality_data = await self.analyze_quality(message)
        return {
            'sentiment_score': quality_data.get('helpfulness_score', 0.5) * 2 - 1,
            'emotion_labels': [],
            'confidence': 1.0 - quality_data.get('hallucination_risk', 0.5),
            'aspects': {
                'satisfaction': quality_data.get('helpfulness_score', 0.5),
                'urgency': 0.5,
                'clarity': quality_data.get('factual_grounding_score', 0.5)
            }
        }

    # Utility Methods
    def _safe_json_load(self, json_str: str) -> Any:
        """Safely load JSON string."""
        try:
            if not json_str:
                return {} if json_str.startswith('{') else []
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {} if json_str.startswith('{') else []
        except Exception:
            return {} if json_str.startswith('{') else []

    async def _throttle_request(self):
        """Apply request throttling."""
        if not hasattr(self.__class__, '_last_request_time'):
            self.__class__._last_request_time = 0
            self.__class__._request_count = 0
            self.__class__._throttle_lock = asyncio.Lock()
            
        async with self.__class__._throttle_lock:
            current_time = time.time()
            time_since_last = current_time - self.__class__._last_request_time
                
            if time_since_last > 60:
                self.__class__._request_count = 0
                    
            self.__class__._request_count += 1
                
            if self.__class__._request_count > 50:
                delay = 2.0
            elif self.__class__._request_count > 30:
                delay = 1.0
            elif self.__class__._request_count > 20:
                delay = 0.5
            elif self.__class__._request_count > 10:
                delay = 0.2
            else:
                delay = 0
                    
            if delay > 0:
                jitter = delay * 0.2 * (random.random() * 2 - 1)
                delay += jitter
                await asyncio.sleep(delay)
                    
            self.__class__._last_request_time = time.time()

    async def _reset_rate_limited_after(self, seconds: int):
        """Reset rate limited flag."""
        await asyncio.sleep(seconds)
        self._rate_limited = False
        logger.info(f"Rate limit cooldown expired after {seconds} seconds")