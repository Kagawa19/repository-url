from fastapi import APIRouter, HTTPException, Request, Depends, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any, Set
from pydantic import BaseModel, Field, constr
import logging
from datetime import datetime, timezone, timedelta
import json
import pandas as pd
from redis.asyncio import Redis
import uuid
import time
import asyncio
import os
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.ml.ml_predictor import MLPredictor
from ai_services_api.services.message.core.database import get_db_connection

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
TEST_USER_ID = "123"
CACHE_TTL = 3600  # Default cache TTL - 1 hour
POPULAR_QUERY_TTL = 7200  # TTL for popular queries - 2 hours
RATE_LIMIT = 50  # Default rate limit per minute
MAX_SUGGESTIONS = 8  # Maximum number of suggestions to show
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize ML Predictor and Search Manager
ml_predictor = MLPredictor()
search_manager = ExpertSearchIndexManager()
logger.info("Services initialized successfully")

# Initialize Gemini
def get_gemini_model():
    """Initialize and return the Gemini model."""
    return ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        model="gemini-pro",
        temperature=0.3
    )

# Enhanced Response Models
class ExpertSearchResult(BaseModel):
    id: str
    first_name: str
    last_name: str
    designation: str = ""
    theme: str = ""
    unit: str = ""
    contact: str = ""
    is_active: bool = True
    score: Optional[float] = None
    bio: Optional[str] = None  
    knowledge_expertise: List[str] = []
    research_areas: List[str] = []

class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total_pages: int
    total_items: int
    has_next: bool
    has_prev: bool

class SearchResponse(BaseModel):
    total_results: int
    experts: List[ExpertSearchResult]
    user_id: str
    session_id: str
    query: str
    pagination: Optional[PaginationInfo] = None
    filters_applied: Dict[str, Any] = {}
    response_time: float = 0.0

class SuggestionSource(str, Enum):
    """Source of the suggestion for tracking."""
    HISTORY = "history"
    POPULAR = "popular"
    COLLABORATIVE = "collaborative"
    SEMANTIC = "semantic"
    GEMINI = "gemini"
    ML = "ml_predictor"

class SearchSuggestion(BaseModel):
    """Model for search suggestions."""
    text: str
    type: SuggestionSource
    confidence: float = 0.0
    display_text: Optional[str] = None  # For highlighting partial matches
    metadata: Dict[str, Any] = {}

class CombinedSuggestions(BaseModel):
    """Combined response with all types of suggestions."""
    query: str
    suggestions: List[SearchSuggestion]
    history_suggestions: List[SearchSuggestion] = []
    popular_suggestions: List[SearchSuggestion] = []
    semantic_suggestions: List[SearchSuggestion] = []
    collaborative_suggestions: List[SearchSuggestion] = []
    total_found: int
    personalized: bool = False
    processing_time: float = 0.0

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int

async def get_redis():
    logger.debug("Initializing Redis connection")
    try:
        redis_client = Redis(host='redis', port=6379, db=3, decode_responses=True)
        await redis_client.ping()  # Test connection
        logger.info("Redis connection established")
        return redis_client
    except Exception as e:
        logger.error(f"Redis connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Redis service unavailable")

async def get_user_id(request: Request) -> str:
    logger.debug("Extracting user ID from request headers")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Missing required X-User-ID header in request")
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    logger.info(f"User ID extracted successfully: {user_id}")
    return user_id

async def get_test_user_id(request: Request) -> str:
    logger.debug("Using test user ID")
    return TEST_USER_ID

async def check_rate_limit(user_id: str, redis_client: Redis) -> bool:
    """Check if user has exceeded rate limit."""
    now = int(time.time())
    window_key = f"ratelimit:{user_id}:{now // 60}"
    
    try:
        current = await redis_client.incr(window_key)
        if current == 1:
            # Set expiry for new keys
            await redis_client.expire(window_key, 120)  # 2 minute expiry to account for clock drift
            
        return current <= RATE_LIMIT
    except Exception as e:
        logger.error(f"Rate limit check error: {str(e)}")
        return True  # Allow on Redis error

async def get_or_create_session(conn, user_id: str) -> str:
    """Get existing active session or create a new one."""
    logger.info(f"Getting or creating session for user: {user_id}")
    cur = conn.cursor()
    try:
        # Check for existing active session
        cur.execute("""
            SELECT session_id FROM search_sessions 
            WHERE user_id = %s AND is_active = true
            ORDER BY start_timestamp DESC LIMIT 1
        """, (user_id,))
        
        existing = cur.fetchone()
        if existing:
            session_id = existing[0]
            logger.debug(f"Found existing session: {session_id}")
            return str(session_id)
        
        # Create new session if none exists
        session_id = int(str(int(datetime.utcnow().timestamp()))[-8:])
        logger.debug(f"Generating new session ID: {session_id}")
        
        cur.execute("""
            INSERT INTO search_sessions 
                (session_id, user_id, start_timestamp, is_active)
            VALUES (%s, %s, CURRENT_TIMESTAMP, true)
            RETURNING session_id
        """, (session_id, user_id))
        
        conn.commit()
        logger.debug(f"Session created with ID: {session_id}")
        return str(session_id)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating session: {str(e)}", exc_info=True)
        raise
    finally:
        cur.close()

async def record_suggestion_impression(conn, session_id: str, user_id: str, 
                                     partial_query: str, suggestions: List[SearchSuggestion]):
    """Record that suggestions were shown to the user."""
    logger.info(f"Recording suggestion impressions - User: {user_id}, Query: '{partial_query}'")
    
    cur = conn.cursor()
    try:
        # Insert impression record
        cur.execute("""
            INSERT INTO suggestion_impressions
                (session_id, user_id, partial_query, timestamp)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (session_id, user_id, partial_query))
        
        impression_id = cur.fetchone()[0]
        
        # Record each suggestion
        for rank, suggestion in enumerate(suggestions, 1):
            cur.execute("""
                INSERT INTO suggestion_details
                    (impression_id, suggestion_text, suggestion_type, 
                     confidence, rank_position, clicked)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                impression_id,
                suggestion.text,
                suggestion.type,
                suggestion.confidence,
                rank,
                False  # Not clicked yet
            ))
        
        conn.commit()
        logger.info(f"Recorded {len(suggestions)} suggestion impressions")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording suggestion impressions: {str(e)}", exc_info=True)
    finally:
        cur.close()

async def record_suggestion_click(conn, user_id: str, suggestion_text: str, 
                                search_query: str, source_type: str = None):
    """Record that a suggestion was clicked."""
    logger.info(f"Recording suggestion click - User: {user_id}, Suggestion: '{suggestion_text}'")
    
    cur = conn.cursor()
    try:
        # Find the suggestion in recent impressions
        cur.execute("""
            UPDATE suggestion_details
            SET clicked = true,
                click_timestamp = CURRENT_TIMESTAMP,
                resulting_query = %s
            WHERE suggestion_text = %s
            AND impression_id IN (
                SELECT id FROM suggestion_impressions
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 10
            )
            RETURNING id
        """, (search_query, suggestion_text, user_id))
        
        updated = cur.rowcount
        
        if updated == 0 and source_type:
            # If not found in recent impressions, create a new direct click record
            cur.execute("""
                INSERT INTO suggestion_clicks
                    (user_id, suggestion_text, resulting_query, 
                     suggestion_type, timestamp)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (user_id, suggestion_text, search_query, source_type))
        
        conn.commit()
        logger.info(f"Recorded suggestion click success: {updated > 0}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording suggestion click: {str(e)}", exc_info=True)
    finally:
        cur.close()

async def get_history_suggestions(conn, user_id: str, partial_query: str) -> List[SearchSuggestion]:
    """Get suggestion from user's search history."""
    logger.info(f"Finding history suggestions for '{partial_query}'")
    
    try:
        cur = conn.cursor()
        
        # Get recent searches from this user that match the partial query
        cur.execute("""
            SELECT query, COUNT(*) as frequency, MAX(timestamp) as last_used
            FROM search_analytics
            WHERE user_id = %s AND query ILIKE %s
            GROUP BY query
            ORDER BY frequency DESC, last_used DESC
            LIMIT 5
        """, (user_id, f"{partial_query}%"))
        
        history_results = cur.fetchall()
        
        if not history_results:
            logger.info("No history suggestions found")
            return []
            
        logger.info(f"Found {len(history_results)} history suggestions")
        suggestions = []
        
        max_freq = max(row[1] for row in history_results) if history_results else 1
        
        for query, freq, last_used in history_results:
            # Calculate confidence based on frequency and recency
            freq_score = freq / max_freq * 0.7
            
            # Recency score (higher for more recent)
            days_ago = (datetime.now(timezone.utc) - last_used).days
            recency_score = 0.3 * (1.0 / (1.0 + days_ago))
            
            confidence = min(0.99, freq_score + recency_score)
            
            # Create display text with highlighting
            display_text = query
            if partial_query and partial_query.lower() in query.lower():
                start = query.lower().find(partial_query.lower())
                end = start + len(partial_query)
                display_text = f"{query[:start]}<b>{query[start:end]}</b>{query[end:]}"
            
            suggestions.append(SearchSuggestion(
                text=query,
                type=SuggestionSource.HISTORY,
                confidence=confidence,
                display_text=display_text,
                metadata={
                    "frequency": freq,
                    "days_ago": days_ago
                }
            ))
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting history suggestions: {str(e)}", exc_info=True)
        return []

async def get_popular_suggestions(conn, partial_query: str) -> List[SearchSuggestion]:
    """Get popular search suggestions."""
    logger.info(f"Finding popular suggestions for '{partial_query}'")
    
    try:
        cur = conn.cursor()
        
        # Get popular searches across all users
        cur.execute("""
            SELECT query, COUNT(*) as frequency, 
                   COUNT(DISTINCT user_id) as user_count
            FROM search_analytics
            WHERE query ILIKE %s
            GROUP BY query
            HAVING COUNT(*) >= 3  -- Minimum popularity threshold
            ORDER BY frequency DESC, user_count DESC
            LIMIT 5
        """, (f"{partial_query}%",))
        
        popular_results = cur.fetchall()
        
        if not popular_results:
            logger.info("No popular suggestions found")
            return []
            
        logger.info(f"Found {len(popular_results)} popular suggestions")
        suggestions = []
        
        max_freq = max(row[1] for row in popular_results) if popular_results else 1
        
        for query, freq, user_count in popular_results:
            # Calculate confidence based on popularity and spread
            popularity = freq / max_freq * 0.6
            spread = min(1.0, user_count / 10) * 0.3  # Normalize by assuming 10+ users is max diversity
            
            confidence = min(0.95, popularity + spread)
            
            # Create display text with highlighting
            display_text = query
            if partial_query and partial_query.lower() in query.lower():
                start = query.lower().find(partial_query.lower())
                end = start + len(partial_query)
                display_text = f"{query[:start]}<b>{query[start:end]}</b>{query[end:]}"
            
            suggestions.append(SearchSuggestion(
                text=query,
                type=SuggestionSource.POPULAR,
                confidence=confidence,
                display_text=display_text,
                metadata={
                    "frequency": freq,
                    "user_count": user_count
                }
            ))
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting popular suggestions: {str(e)}", exc_info=True)
        return []

async def get_collaborative_suggestions(conn, user_id: str, partial_query: str) -> List[SearchSuggestion]:
    """Get suggestions from similar users."""
    logger.info(f"Finding collaborative suggestions for '{partial_query}'")
    
    try:
        cur = conn.cursor()
        
        # Find similar users based on past search patterns
        similar_users_query = """
            WITH user_searches AS (
                -- Get this user's searches
                SELECT DISTINCT query
                FROM search_analytics
                WHERE user_id = %s
                AND timestamp > NOW() - INTERVAL '90 days'
            ),
            user_matches AS (
                -- Find users who searched for similar queries
                SELECT 
                    sa.user_id,
                    COUNT(DISTINCT sa.query) AS shared_queries
                FROM search_analytics sa
                JOIN user_searches us ON sa.query = us.query
                WHERE sa.user_id != %s
                GROUP BY sa.user_id
                HAVING COUNT(DISTINCT sa.query) >= 2  -- At least 2 shared queries
                ORDER BY shared_queries DESC
                LIMIT 10
            )
            SELECT user_id FROM user_matches
        """
        
        cur.execute(similar_users_query, (user_id, user_id))
        similar_users = [row[0] for row in cur.fetchall()]
        
        if not similar_users:
            logger.info("No similar users found")
            return []
            
        logger.info(f"Found {len(similar_users)} similar users")
        
        # Get queries from similar users
        collab_query = """
            SELECT 
                query, 
                COUNT(*) as frequency,
                COUNT(DISTINCT user_id) as user_count,
                MAX(timestamp) as last_used
            FROM search_analytics
            WHERE user_id IN %s
            AND query ILIKE %s
            AND query NOT IN (
                -- Exclude queries this user has already made
                SELECT DISTINCT query
                FROM search_analytics
                WHERE user_id = %s
            )
            GROUP BY query
            ORDER BY user_count DESC, frequency DESC
            LIMIT 5
        """
        
        cur.execute(collab_query, (tuple(similar_users), f"{partial_query}%", user_id))
        collab_results = cur.fetchall()
        
        if not collab_results:
            logger.info("No collaborative suggestions found")
            return []
            
        logger.info(f"Found {len(collab_results)} collaborative suggestions")
        suggestions = []
        
        # Calculate maximum values for normalization
        max_freq = max(row[1] for row in collab_results) if collab_results else 1
        max_users = max(row[2] for row in collab_results) if collab_results else 1
        
        for query, freq, user_count, last_used in collab_results:
            # Confidence based on frequency and user diversity
            freq_score = (freq / max_freq) * 0.4
            diversity_score = (user_count / max_users) * 0.6
            
            confidence = min(0.9, freq_score + diversity_score)
            
            # Create display text with highlighting
            display_text = query
            if partial_query and partial_query.lower() in query.lower():
                start = query.lower().find(partial_query.lower())
                end = start + len(partial_query)
                display_text = f"{query[:start]}<b>{query[start:end]}</b>{query[end:]}"
            
            days_ago = (datetime.now(timezone.utc) - last_used).days
            
            suggestions.append(SearchSuggestion(
                text=query,
                type=SuggestionSource.COLLABORATIVE,
                confidence=confidence,
                display_text=display_text,
                metadata={
                    "frequency": freq,
                    "user_count": user_count,
                    "days_ago": days_ago,
                    "similar_users": len(similar_users)
                }
            ))
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting collaborative suggestions: {str(e)}", exc_info=True)
        return []

async def get_semantic_suggestions(partial_query: str) -> List[SearchSuggestion]:
    """Get semantically similar suggestions using the search index."""
    logger.info(f"Finding semantic suggestions for '{partial_query}'")
    
    if len(partial_query) < 3:
        logger.info("Query too short for semantic suggestions")
        return []
    
    try:
        # Use the faiss index to find similar queries
        results = search_manager.get_similar_queries(partial_query, k=5)
        
        if not results:
            logger.info("No semantic suggestions found")
            return []
            
        logger.info(f"Found {len(results)} semantic suggestions")
        suggestions = []
        
        for query, score in results:
            # Normalize the similarity score to a confidence value
            confidence = min(0.9, max(0.5, score))
            
            # Create display text with highlighting
            display_text = query
            if partial_query and partial_query.lower() in query.lower():
                start = query.lower().find(partial_query.lower())
                end = start + len(partial_query)
                display_text = f"{query[:start]}<b>{query[start:end]}</b>{query[end:]}"
            
            suggestions.append(SearchSuggestion(
                text=query,
                type=SuggestionSource.SEMANTIC,
                confidence=confidence,
                display_text=display_text,
                metadata={
                    "similarity_score": score
                }
            ))
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting semantic suggestions: {str(e)}", exc_info=True)
        return []

async def get_gemini_suggestions(partial_query: str) -> List[SearchSuggestion]:
    """Get suggestions using Gemini AI."""
    logger.info(f"Generating Gemini suggestions for '{partial_query}'")
    
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set, skipping Gemini suggestions")
        return []
        
    if len(partial_query) < 3:
        logger.info("Query too short for Gemini suggestions")
        return []
    
    try:
        # Create a model instance
        model = get_gemini_model()
        
        # Create the prompt
        prompt = f"""
        Generate 3-5 search query suggestions for someone who started typing: "{partial_query}"
        
        Context: These are for searching expert profiles at a research institution focused on health, population, and development in Africa.
        
        Format: Return only the suggestions, one per line, with no additional text or numbering.
        Make sure the suggestions are:
        1. Relevant to African research topics
        2. Related to health, demographics, or development
        3. Start with the partial query: "{partial_query}"
        
        Examples of good suggestions:
        - "{partial_query} in urban slums"
        - "{partial_query} among adolescents"
        - "{partial_query} intervention programs"
        """
        
        # Generate suggestions
        response = model.invoke(prompt)
        suggestions_text = response.content.strip().split('\n')
        
        # Clean up suggestions
        cleaned_suggestions = []
        for suggestion in suggestions_text:
            # Remove any numbering, quotes, or dashes
            clean = suggestion.strip().strip('"-•').strip()
            
            # Ensure it starts with the partial query
            if clean and clean.lower().startswith(partial_query.lower()):
                cleaned_suggestions.append(clean)
        
        if not cleaned_suggestions:
            logger.info("No valid Gemini suggestions generated")
            return []
            
        logger.info(f"Generated {len(cleaned_suggestions)} Gemini suggestions")
        
        # Convert to suggestion objects
        suggestions = []
        base_confidence = 0.8
        
        for i, query in enumerate(cleaned_suggestions):
            # Decrease confidence slightly for each successive suggestion
            confidence = base_confidence - (i * 0.05)
            
            # Create display text with highlighting
            display_text = query
            if partial_query and partial_query.lower() in query.lower():
                start = query.lower().find(partial_query.lower())
                end = start + len(partial_query)
                display_text = f"{query[:start]}<b>{query[start:end]}</b>{query[end:]}"
            
            suggestions.append(SearchSuggestion(
                text=query,
                type=SuggestionSource.GEMINI,
                confidence=confidence,
                display_text=display_text,
                metadata={
                    "rank": i + 1
                }
            ))
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting Gemini suggestions: {str(e)}", exc_info=True)
        return []

async def get_ml_suggestions(partial_query: str, user_id: str) -> List[SearchSuggestion]:
    """Get suggestions from ML predictor fallback."""
    logger.info(f"Getting ML suggestions for '{partial_query}'")
    
    try:
        # Get predictions from ML model
        predictions = ml_predictor.predict(partial_query, user_id=user_id, max_suggestions=5)
        
        if not predictions:
            logger.info("No ML suggestions found")
            return []
            
        logger.info(f"Found {len(predictions)} ML suggestions")
        
        # Convert to suggestion objects
        suggestions = []
        
        for i, query in enumerate(predictions):
            # Decrease confidence for each successive suggestion
            confidence = 0.7 - (i * 0.1)
            
            # Create display text with highlighting
            display_text = query
            if partial_query and partial_query.lower() in query.lower():
                start = query.lower().find(partial_query.lower())
                end = start + len(partial_query)
                display_text = f"{query[:start]}<b>{query[start:end]}</b>{query[end:]}"
            
            suggestions.append(SearchSuggestion(
                text=query,
                type=SuggestionSource.ML,
                confidence=confidence,
                display_text=display_text,
                metadata={
                    "rank": i + 1
                }
            ))
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting ML suggestions: {str(e)}", exc_info=True)
        return []

async def combine_and_rank_suggestions(suggestions_lists: List[List[SearchSuggestion]], 
                                     max_results: int = MAX_SUGGESTIONS) -> List[SearchSuggestion]:
    """Combine and rank suggestions from multiple sources."""
    # Flatten all suggestions into one list
    all_suggestions = []
    for suggestions in suggestions_lists:
        all_suggestions.extend(suggestions)
    
    # Remove duplicates, keeping the highest confidence version
    unique_suggestions = {}
    for suggestion in all_suggestions:
        text = suggestion.text.lower()
        if text not in unique_suggestions or suggestion.confidence > unique_suggestions[text].confidence:
            unique_suggestions[text] = suggestion
    
    # Sort by confidence
    ranked_suggestions = sorted(
        unique_suggestions.values(),
        key=lambda x: x.confidence,
        reverse=True
    )
    
    # Return top results
    return ranked_suggestions[:max_results]

async def process_search_suggestions(partial_query: str, user_id: str, redis_client: Redis = None) -> CombinedSuggestions:
    """
    Process search suggestions with caching and personalization.
    """
    logger.info(f"Processing search suggestions - Query: '{partial_query}', User: {user_id}'")
    start_time = time.time()
    
    conn = None
    personalized = False
    
    try:
        if not partial_query:
            return CombinedSuggestions(
                query=partial_query,
                suggestions=[],
                total_found=0,
                personalized=False,
                processing_time=time.time() - start_time
            )
            
        if redis_client:
            cache_key = f"search_suggestions:{user_id}:{partial_query}"
            cached_response = await redis_client.get(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for suggestions: {cache_key}")
                return CombinedSuggestions(**json.loads(cached_response))

        conn = get_db_connection()
        session_id = await get_or_create_session(conn, user_id)
        
        # Get suggestions from all sources asynchronously
        history_task = asyncio.create_task(get_history_suggestions(conn, user_id, partial_query))
        popular_task = asyncio.create_task(get_popular_suggestions(conn, partial_query))
        collaborative_task = asyncio.create_task(get_collaborative_suggestions(conn, user_id, partial_query))
        semantic_task = asyncio.create_task(get_semantic_suggestions(partial_query))
        gemini_task = asyncio.create_task(get_gemini_suggestions(partial_query))
        ml_task = asyncio.create_task(get_ml_suggestions(partial_query, user_id))
        
        # Wait for all tasks
        history_suggestions = await history_task
        popular_suggestions = await popular_task
        collaborative_suggestions = await collaborative_task
        semantic_suggestions = await semantic_task
        gemini_suggestions = await gemini_task
        ml_suggestions = await ml_task
        
        # Check if we have personalized results
        personalized = len(history_suggestions) > 0 or len(collaborative_suggestions) > 0
        
        # Combine and rank suggestions
        all_sources = [
            history_suggestions,      # User's own history first
            collaborative_suggestions, # Similar users' queries next
            popular_suggestions,       # Popular queries
            semantic_suggestions,      # Semantically similar queries
            gemini_suggestions,        # AI-generated suggestions
            ml_suggestions             # ML predictor as fallback
        ]
        
        # Get final ranked suggestions
        ranked_suggestions = await combine_and_rank_suggestions(all_sources)
        
        # Record impressions
        await record_suggestion_impression(
            conn, 
            session_id, 
            user_id, 
            partial_query, 
            ranked_suggestions
        )
        
        # Create response
        response = CombinedSuggestions(
            query=partial_query,
            suggestions=ranked_suggestions,
            history_suggestions=history_suggestions[:3],
            popular_suggestions=popular_suggestions[:3],
            semantic_suggestions=semantic_suggestions[:3],
            collaborative_suggestions=collaborative_suggestions[:3],
            total_found=len(ranked_suggestions),
            personalized=personalized,
            processing_time=time.time() - start_time
        )

        # Cache the response
        if redis_client:
            # Shorter TTL for personalized results
            ttl = 600 if personalized else 1800  # 10 or 30 minutes
            await redis_client.setex(
                cache_key,
                ttl,
                json.dumps(response.dict())
            )
            logger.debug(f"Cached search suggestions with key: {cache_key}, TTL: {ttl}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing search suggestions: {str(e)}", exc_info=True)
        
        # Return empty response on error
        return CombinedSuggestions(
            query=partial_query,
            suggestions=[],
            total_found=0,
            personalized=False,
            processing_time=time.time() - start_time
        )
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")

# API Endpoints
@router.get("/experts/suggest/{partial_query}", response_model=CombinedSuggestions, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Server error"}
})
async def get_search_suggestions(
    partial_query: str = Path(..., description="Partial query to get suggestions for", example="hea"),
    request: Request = None,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """
    Get search suggestions as user types.
    
    This endpoint provides Google-style search suggestions from multiple sources:
    - User's own search history
    - Queries from similar users (collaborative filtering)
    - Popular searches across all users
    - Semantically similar queries
    - AI-generated suggestions
    
    Results are combined and ranked by relevance and confidence.
    """
    logger.info(f"Received search suggestion request - Query: '{partial_query}', User: {user_id}")
    
    # Rate limiting (less strict for suggestions)
    if redis_client:
        rate_limited = not await check_rate_limit(user_id, redis_client)
        if rate_limited:
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error="Rate limit exceeded",
                    detail="Too many requests. Please try again later.",
                    status_code=429
                ).dict()
            )
    
    try:
        return await process_search_suggestions(partial_query, user_id, redis_client)
    except Exception as e:
        logger.error(f"Error in suggestion endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Suggestion generation failed",
                detail=str(e),
                status_code=500
            ).dict()
        )

@router.post("/experts/suggestion/click", response_model=Dict[str, Any])
async def record_suggestion_selection(
    request: Request,
    user_id: str = Depends(get_user_id),
    suggestion: str = Query(..., description="The selected suggestion"),
    resulting_query: str = Query(..., description="The final search query"),
    source_type: Optional[str] = Query(None, description="Source of the suggestion")
):
    """
    Record that a user clicked on a search suggestion.
    
    This endpoint should be called when a user selects a suggestion,
    before redirecting to the actual search results page.
    """
    logger.info(f"Received suggestion selection - Suggestion: '{suggestion}', User: {user_id}")
    
    conn = None
    try:
        conn = get_db_connection()
        await record_suggestion_click(conn, user_id, suggestion, resulting_query, source_type)
        
        return {
            "success": True,
            "message": "Selection recorded successfully",
            "suggestion": suggestion,
            "query": resulting_query
        }
    except Exception as e:
        logger.error(f"Error recording suggestion click: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Failed to record selection",
                detail=str(e),
                status_code=500
            ).dict()
        )
    finally:
        if conn:
            conn.close()

@router.get("/test/experts/suggest/{partial_query}", response_model=CombinedSuggestions)
async def test_search_suggestions(
    partial_query: str = Path(..., description="Partial query to get suggestions for", example="hea"),
    request: Request = None,
    user_id: str = Depends(get_test_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Test endpoint for search suggestions that uses a fixed test user ID."""
    logger.info(f"Received test suggestion request - Query: '{partial_query}', Test User: {user_id}")
    return await process_search_suggestions(partial_query, user_id, redis_client)

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for the search service."""
    try:
        # Check database connection
        conn = get_db_connection()
        db_status = "up" if conn else "down"
        if conn:
            conn.close()
        
        # Check Redis connection
        redis_status = "unknown"
        try:
            redis_client = await get_redis()
            await redis_client.ping()
            redis_status = "up"
        except:
            redis_status = "down"
        
        # Check ML predictor
        ml_status = "up" if ml_predictor else "down"
        
        # Check search index
        search_status = "up" if search_manager else "down"
        
        return {
            "status": "healthy" if all(s == "up" for s in [db_status, redis_status, ml_status, search_status]) else "degraded",
            "dependencies": {
                "database": db_status,
                "redis": redis_status,
                "ml_predictor": ml_status,
                "search_index": search_status
            },
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }