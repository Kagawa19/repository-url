import os
import redis
from fastapi import HTTPException
from typing import Any, List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json
import uuid
import asyncio
from redis import asyncio as aioredis
import traceback

from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.search.core.models import PredictionResponse
from ai_services_api.services.search.core.personalization import get_selected_suggestions, get_trending_suggestions, get_user_search_history, personalize_suggestions
from ai_services_api.services.message.core.db_pool import get_pooled_connection, return_connection, DatabaseConnection
from ai_services_api.services.message.core.db_pool import (
    DatabaseConnection, get_pooled_connection, return_connection, log_pool_status, safe_background_task
)

# Configure logger
logger = logging.getLogger(__name__)

# Global predictor instance (singleton)
_predictor = None

# Global Redis instance
_redis = None





async def process_query_prediction(
    partial_query: str, 
    user_id: str, 
    context: Optional[str] = None
) -> PredictionResponse:
    """Process query prediction with improved connection pooling."""
    logger.info(f"Processing query prediction - Query: '{partial_query}', User: {user_id}, Context: {context}")
    
    valid_contexts = ["name", "theme", "designation", None]
    if context and context not in valid_contexts[:-1]:
        raise ValueError(f"Invalid context. Must be one of {', '.join(valid_contexts[:-1])}")
    
    # Initial connection values
    conn = None
    pool = None
    using_pool = False
    conn_id = None
    session_id = str(uuid.uuid4())[:8]
    
    # Log initial pool status
    log_pool_status()
    
    try:
        # First check if we can get results from cache
        redis = await get_redis()
        cache_key = f"google_autocomplete:{context or ''}:{partial_query}"
        
        # Try to get cached result first to avoid database connections if possible
        cache_hit = False
        if redis:
            try:
                cached_result = await redis.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache hit for query: {partial_query}")
                    cached_data = json.loads(cached_result)
                    suggestion_objects = cached_data.get("suggestions", [])
                    confidence_scores = cached_data.get("confidence_scores", [])
                    predictions = [s["text"] for s in suggestion_objects]
                    
                    if context:
                        predictions, confidence_scores = filter_predictions_by_context(
                            predictions, confidence_scores, partial_query, context
                        )
                    
                    # We'll need a DB connection for personalization
                    logger.info("Getting connection for personalization after cache hit")
                    conn, pool, using_pool, conn_id = get_pooled_connection()
                    
                    try:
                        # Get session ID
                        session_id = await get_or_create_session(conn, user_id)
                        logger.debug(f"Created session: {session_id}")
                        
                        # Personalize suggestions
                        personalized_suggestions = await personalize_suggestions(
                            [{"text": pred, "score": score} for pred, score in zip(predictions, confidence_scores)],
                            user_id, 
                            partial_query
                        )
                        predictions = [s["text"] for s in personalized_suggestions]
                        confidence_scores = [s.get("score", 0.5) for s in personalized_suggestions]
                        
                        # Record in database
                        if predictions:
                            await record_prediction(
                                conn,
                                session_id,
                                user_id,
                                partial_query,
                                predictions,
                                confidence_scores
                            )
                        
                        cache_hit = True
                    except Exception as personalize_error:
                        logger.error(f"Personalization error for cached results: {personalize_error}")
                        logger.error(f"Stacktrace: {traceback.format_exc()}")
                    finally:
                        # Always return the connection when done
                        if conn:
                            logger.info(f"Returning connection {conn_id} after cache personalization")
                            return_connection(conn, pool, using_pool, conn_id)
                            conn = None  # Prevent duplicate returns
                    
                    # Generate refinements
                    refinements_dict = generate_predictive_refinements(partial_query, predictions)
                    refinements_list = extract_refinements_list(refinements_dict)
                    
                    # Return cache results
                    if cache_hit:
                        logger.info(f"Returning cached results for '{partial_query}' ({len(predictions)} predictions)")
                        return PredictionResponse(
                            predictions=predictions,
                            confidence_scores=confidence_scores,
                            user_id=user_id,
                            refinements=refinements_list,
                            total_suggestions=len(predictions)
                        )

            except Exception as cache_error:
                logger.error(f"Cache retrieval error: {cache_error}")
                logger.error(f"Cache error stacktrace: {traceback.format_exc()}")
        
        # No cache hit, need to get fresh data
        logger.info("No cache hit, getting fresh predictions")
        
        # Use the context manager for database connection
        with DatabaseConnection() as conn:
            # The context manager will automatically return the connection when done
            logger.info("Using DatabaseConnection context manager")
            
            try:
                # Get session ID
                session_id = await get_or_create_session(conn, user_id)
                logger.debug(f"Created session: {session_id}")
                
                # Get predictions from API
                predictor = await get_predictor()
                suggestion_objects = await predictor.predict(partial_query, limit=10)
                predictions = [s["text"] for s in suggestion_objects]
                confidence_scores = [s.get("score", 0.5) for s in suggestion_objects]
                
                # Apply context filtering if needed
                if context:
                    predictions, confidence_scores = filter_predictions_by_context(
                        predictions, confidence_scores, partial_query, context
                    )
                
                # Personalize suggestions
                try:
                    personalized_suggestions = await personalize_suggestions(
                        [{"text": pred, "score": score} for pred, score in zip(predictions, confidence_scores)],
                        user_id, 
                        partial_query
                    )
                    predictions = [s["text"] for s in personalized_suggestions]
                    confidence_scores = [s.get("score", 0.5) for s in personalized_suggestions]
                except Exception as personalize_error:
                    logger.error(f"Personalization error for fresh results: {personalize_error}")
                    logger.error(f"Personalization stacktrace: {traceback.format_exc()}")
                
                logger.debug(f"Generated {len(predictions)} predictions: {predictions}")
                
                # Cache the results
                if redis and predictions:
                    try:
                        cache_data = {
                            "suggestions": [
                                {"text": pred, "score": score} 
                                for pred, score in zip(predictions, confidence_scores)
                            ],
                            "confidence_scores": confidence_scores
                        }
                        await redis.setex(cache_key, 900, json.dumps(cache_data))
                        logger.debug(f"Cached predictions for: {partial_query}")
                    except Exception as cache_error:
                        logger.error(f"Cache storage error: {cache_error}")
                
                # Generate refinements
                refinements_dict = generate_predictive_refinements(partial_query, predictions)
                refinements_list = extract_refinements_list(refinements_dict)
                
                # Record prediction in the database
                if predictions:
                    await record_prediction(
                        conn,
                        session_id,
                        user_id,
                        partial_query,
                        predictions,
                        confidence_scores
                    )
                
                logger.info(f"Returning fresh results for '{partial_query}' ({len(predictions)} predictions)")
                return PredictionResponse(
                    predictions=predictions,
                    confidence_scores=confidence_scores,
                    user_id=user_id,
                    refinements=refinements_list,
                    total_suggestions=len(predictions)
                )
            
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                logger.error(f"Query processing stacktrace: {traceback.format_exc()}")
                raise
    
    except Exception as e:
        logger.exception(f"Unexpected error in process_query_prediction: {e}")
        
        # Initialize empty lists for fallback response
        predictions = []
        confidence_scores = []
        refinements_list = []
        
        # Log final pool status
        log_pool_status()
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions)
        )
    finally:
        # Log final pool status
        log_pool_status()

# Helper function to extract refinements list
def extract_refinements_list(refinements_dict):
    """Extract a flat list of refinements from the refinements dictionary."""
    refinements_list = []
    if isinstance(refinements_dict, dict):
        # Extract related queries if they exist
        if "related_queries" in refinements_dict and isinstance(refinements_dict["related_queries"], list):
            refinements_list.extend(refinements_dict["related_queries"])
        
        # Extract expertise areas if they exist
        if "expertise_areas" in refinements_dict and isinstance(refinements_dict["expertise_areas"], list):
            refinements_list.extend(refinements_dict["expertise_areas"])
        
        # Extract values from filters if they exist
        if "filters" in refinements_dict and isinstance(refinements_dict["filters"], list):
            for filter_item in refinements_dict["filters"]:
                if isinstance(filter_item, dict) and "values" in filter_item:
                    if isinstance(filter_item["values"], list):
                        refinements_list.extend(filter_item["values"])
                    else:
                        refinements_list.append(str(filter_item["values"]))
    
    return refinements_list

async def get_context_popular_queries(context: str, limit: int = 5) -> List[str]:
    """
    Get popular queries specific to a context type.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        context: Context type (name, theme, designation, publication)
        limit: Maximum number of items to return
        
    Returns:
        List of popular queries for this context
    """
    try:
        popular_queries = []
        
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Get popular searches filtered by context
            if context == "name":
                # For name searches, look for patterns like "John Smith" or queries with "Name:"
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        (query LIKE '% %' AND length(query) < 30) OR
                        query LIKE 'name:%'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            elif context == "theme":
                # For theme searches, look for queries that were used in theme search
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        query LIKE 'theme:%' OR
                        search_type = 'theme'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            elif context == "designation":
                # For designation searches
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        query LIKE 'designation:%' OR
                        search_type = 'designation'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            elif context == "publication":
                # For publication searches
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        query LIKE 'publication:%' OR
                        search_type = 'publication'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            else:
                # For general searches
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE query NOT LIKE '%:%'  -- Exclude prefixed queries
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            # Process results
            for row in cur.fetchall():
                query = row[0]
                
                # Clean up query - remove context prefix if present
                if context and query.lower().startswith(f"{context}:"):
                    query = query[len(context)+1:].strip()
                
                if query and query.strip():
                    popular_queries.append(query.strip())
                    
            cur.close()
            return popular_queries
                
    except Exception as db_error:
        logger.error(f"Database error in context popular queries: {db_error}")
        
        # Fall back to Redis if available
        try:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_TRENDING_DB', 2)),
                decode_responses=True
            )
            
            # Try to get context-specific trending from Redis
            context_key = f"trending_context:{context or 'general'}"
            trending = redis_client.zrevrange(context_key, 0, limit-1)
            
            if trending:
                return trending
        except Exception as redis_error:
            logger.error(f"Redis error in context popular queries: {redis_error}")
        
        # If all else fails, return hardcoded defaults based on context
        if context == "name":
            return ["John Smith", "Jane Doe", "Michael Johnson", "David Williams", "Sara Chen"]
        elif context == "theme":
            return ["Neuroscience", "Data Science", "Artificial Intelligence", "Psychology", "Biochemistry"]
        elif context == "designation":
            return ["Professor", "Researcher", "Director", "Scientist", "Student"]
        elif context == "publication":
            return ["Journal of Neuroscience", "Nature", "Science", "IEEE Transactions", "PLOS ONE"]
        else:
            return ["Research", "Science", "Technology", "Medicine", "Engineering"]

async def get_redis():
    """Get or create Redis connection."""
    global _redis
    if _redis is None:
        try:
            # Create Redis connection
            _redis = await aioredis.Redis.from_url("redis://redis:6379/0", decode_responses=True)
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}", exc_info=True)
            _redis = None
    return _redis

async def get_predictor():
    """Get or create predictor instance (singleton pattern)."""
    global _predictor
    if _predictor is None:
        _predictor = GoogleAutocompletePredictor()
    return _predictor

# Record session and prediction in database for analytics
async def get_or_create_session(conn, user_id: str) -> str:
    """Create a session for tracking user interactions."""
    logger.info(f"Getting or creating session for user: {user_id}")
    cur = conn.cursor()
    try:
        session_id = int(str(int(datetime.utcnow().timestamp()))[-8:])
        logger.debug(f"Generated session ID: {session_id}")
        
        cur.execute("""
            INSERT INTO search_sessions 
                (session_id, user_id, start_timestamp, is_active)
            VALUES (%s, %s, CURRENT_TIMESTAMP, true)
            RETURNING session_id
        """, (session_id, user_id))
        
        conn.commit()
        logger.debug(f"Session created successfully with ID: {session_id}")
        return str(session_id)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating session: {str(e)}", exc_info=True)
        logger.debug(f"Session creation failed: {str(e)}")
        raise
    finally:
        cur.close()

async def _record_prediction_async(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """Asynchronous helper for recording predictions in the database."""
    logger.info(f"Recording predictions for user {user_id}, session {session_id}")
    
    # Create a new cursor for this async task to avoid connection issues
    try:
        cur = conn.cursor()
        try:
            for pred, conf in zip(predictions, confidence_scores):
                cur.execute("""
                    INSERT INTO query_predictions
                        (partial_query, predicted_query, confidence_score, 
                        user_id, timestamp)
                    VALUES 
                        (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (partial_query, pred, conf, user_id))
            
            conn.commit()
            logger.info("Successfully recorded all predictions")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error recording prediction: {str(e)}", exc_info=True)
        finally:
            cur.close()
    except Exception as conn_error:
        logger.error(f"Connection error in record_prediction: {conn_error}")
    # Note: Don't close the connection here as it might be used elsewhere



async def record_prediction(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """
    Record prediction in database for analytics.
    Uses an existing connection instead of creating a new one.
    """
    if not conn:
        logger.error("Cannot record prediction: No database connection provided")
        return
        
    try:
        # Create a cursor for this operation using the existing connection
        cur = conn.cursor()
        try:
            for pred, conf in zip(predictions, confidence_scores):
                cur.execute("""
                    INSERT INTO query_predictions
                        (partial_query, predicted_query, confidence_score, 
                        user_id, timestamp)
                    VALUES 
                        (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (partial_query, pred, conf, user_id))
            
            conn.commit()
            logger.info("Successfully recorded all predictions")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error recording prediction: {str(e)}", exc_info=True)
        finally:
            cur.close()
    except Exception as conn_error:
        logger.error(f"Connection error in record_prediction: {conn_error}")
    # Note: We don't close the connection here since it was passed in

def generate_predictive_refinements(partial_query: str, predictions: List[str]) -> Dict[str, Any]:
    """Generate refinement suggestions based on partial query and predictions."""
    try:
        # Generate related filters and expertise areas
        related_filters = []
        
        # Add a filter for query type if we have predictions
        if predictions:
            related_filters.append({
                "type": "query_type",
                "label": "Query Suggestions",
                "values": predictions[:3]
            })
        
        # Extract potential expertise areas from predictions
        expertise_areas = set()
        for prediction in predictions:
            words = prediction.lower().split()
            for word in words:
                if len(word) > 3 and word != partial_query.lower():
                    expertise_areas.add(word.capitalize())
        
        # Make sure we have some expertise areas even if extraction failed
        if len(expertise_areas) < 3 and partial_query:
            expertise_areas.add(partial_query.capitalize())
        
        # Remove duplicates while preserving order
        expertise_areas = list(dict.fromkeys(expertise_areas))[:5]
        
        # Construct refinements
        refinements = {
            "filters": related_filters,
            "related_queries": predictions[:5],
            "expertise_areas": expertise_areas
        }
        
        logger.info(f"Generated predictive refinements for query: {partial_query}")
        return refinements
    
    except Exception as e:
        logger.error(f"Error in predictive refinements: {e}")
        # Return a minimal valid structure even on error
        return {
            "filters": [],
            "related_queries": predictions[:5] if predictions else [],
            "expertise_areas": []
        }
    
from typing import Any, List, Dict, Optional
import logging
import uuid
import json
import traceback
from ai_services_api.services.message.core.db_pool import (
    DatabaseConnection, get_pooled_connection, return_connection, log_pool_status
)
from ai_services_api.services.search.core.models import PredictionResponse

# Configure logger
logger = logging.getLogger(__name__)



def is_publication_title(suggestion: str, partial_query: str) -> bool:
    """
    Check if suggestion appears to be a publication title.
    Publication titles tend to:
    - Be longer phrases
    - Often contain research-related terms
    - Have proper capitalization patterns
    
    Args:
        suggestion: The suggestion text to check
        partial_query: The partial query being typed
        
    Returns:
        Boolean indicating if the suggestion looks like a publication title
    """
    # Convert partial query and suggestion to lowercase for comparison
    partial_query_lower = partial_query.lower()
    suggestion_lower = suggestion.lower()
    
    # First, check if partial query is in the suggestion
    if partial_query_lower not in suggestion_lower:
        return False
    
    # Publication indicators - common in academic paper titles
    pub_indicators = [
        "study", "research", "analysis", "review", "survey", 
        "approach", "framework", "method", "evaluation", "assessment",
        "impact", "effect", "comparison", "investigation", "development"
    ]
    
    # Check for publication indicators
    has_indicator = any(indicator in suggestion_lower for indicator in pub_indicators)
    
    # Check length - publications titles are usually longer
    word_count = len(suggestion.split())
    good_length = word_count >= 3 and word_count <= 20
    
    # Check for patterns like "Title: Subtitle" or quotes
    has_title_pattern = ":" in suggestion or '"' in suggestion or "'" in suggestion
    
    # Publications often have proper nouns (capitalized words not at start)
    words = suggestion.split()
    has_mid_capitals = any(word[0].isupper() for word in words[1:] if len(word) > 2)
    
    # Check for common paper title ending patterns
    has_paper_ending = any(suggestion_lower.endswith(end) for end in 
                          [" study", " analysis", " review", " approach", " framework"])
    
    # Combine factors - need at least two positive indicators
    indicators = [has_indicator, good_length, has_title_pattern, has_mid_capitals, has_paper_ending]
    score = sum(1 for indicator in indicators if indicator)
    
    return score >= 2


async def process_advanced_query_prediction(
    partial_query: str, 
    user_id: str, 
    context: Optional[str] = None,
    limit: int = 10
) -> PredictionResponse:
    """
    Process advanced query prediction with context-specific suggestions using FAISS indexes.
    Enhanced to support dynamic filtering and initial suggestions.
    
    Args:
        partial_query: Partial query to predict (may be empty string for initial suggestions)
        user_id: User identifier
        context: Optional context for prediction (name, theme, designation, publication)
        limit: Maximum number of predictions
    
    Returns:
        PredictionResponse with dynamically filtered predictions
    """
    logger.info(f"Processing advanced query prediction - Query: '{partial_query}', Context: {context}")
    
    try:
        # Validate context
        valid_contexts = ["name", "theme", "designation", "publication", None]
        if context and context not in valid_contexts[:-1]:
            raise ValueError(f"Invalid context. Must be one of {', '.join(valid_contexts[:-1])}")
        
        conn = None
        session_id = str(uuid.uuid4())[:8]
        redis = await get_redis()
        
        # Initialize empty prediction lists
        predictions = []
        confidence_scores = []
        
        # Different handling based on whether we have input or not
        if not partial_query:
            # Case: No input yet - get initial suggestions based on history and trending
            try:
                # Get user's search history
                user_history = await get_user_search_history(user_id, limit=5)
                history_queries = [item.get("query", "") for item in user_history]
                
                # Add user's previously selected suggestions
                selected_suggestions = await get_selected_suggestions(user_id, limit=3)
                
                # Get trending queries
                trending_suggestions = await get_trending_suggestions("", limit=3)
                trending_queries = [item.get("text", "") for item in trending_suggestions]
                
                # Combine all sources
                all_suggestions = []
                
                # Add history with source tag
                for query in history_queries:
                    if query and query.strip():
                        all_suggestions.append({
                            "text": query,
                            "source": "history",
                            "score": 0.9  # High score for history items
                        })
                
                # Add selected suggestions with source tag
                for suggestion in selected_suggestions:
                    if suggestion and suggestion.strip() and suggestion not in history_queries:
                        all_suggestions.append({
                            "text": suggestion,
                            "source": "selected",
                            "score": 0.95  # Higher score for explicitly selected items
                        })
                
                # Add trending with source tag
                for query in trending_queries:
                    if query and query.strip() and query not in history_queries and query not in selected_suggestions:
                        all_suggestions.append({
                            "text": query,
                            "source": "trending",
                            "score": 0.8  # Slightly lower score for trending
                        })
                
                # Add context-specific popular queries if context is provided
                if context:
                    context_suggestions = await get_context_popular_queries(context, limit=3)
                    for query in context_suggestions:
                        if query and query.strip():
                            all_suggestions.append({
                                "text": query,
                                "source": f"{context}_popular",
                                "score": 0.85
                            })
                
                # Sort by score and deduplicate
                seen = set()
                filtered_suggestions = []
                for item in sorted(all_suggestions, key=lambda x: x.get("score", 0), reverse=True):
                    text = item.get("text", "").lower()
                    if text and text not in seen:
                        seen.add(text)
                        filtered_suggestions.append(item)
                
                # Convert to prediction lists
                predictions = [item.get("text", "") for item in filtered_suggestions[:limit]]
                confidence_scores = [item.get("score", 0.5) for item in filtered_suggestions[:limit]]
                
            except Exception as initial_error:
                logger.error(f"Error getting initial suggestions: {initial_error}", exc_info=True)
        else:
            # Case: User is typing - get filtered suggestions
            # Try to get from cache first
            cache_key = f"advanced_idx_predict:{context or ''}:{partial_query}"
            error_occurred = False
            
            if redis:
                try:
                    cached_result = await redis.get(cache_key)
                    
                    if cached_result:
                        logger.info(f"Cache hit for advanced query: {partial_query}")
                        cached_data = json.loads(cached_result)
                        predictions = cached_data.get("predictions", [])
                        confidence_scores = cached_data.get("confidence_scores", [])
                except Exception as cache_error:
                    logger.error(f"Cache retrieval error: {cache_error}", exc_info=True)
            
            # If not in cache, use the ExpertSearchIndexManager to get predictions
            if not predictions:
                try:
                    # Create search manager
                    from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
                    search_manager = ExpertSearchIndexManager()
                    
                    # Use the appropriate index-based search method based on context
                    if context == "name":
                        # Use the name index search
                        results = search_manager.search_experts_by_name(
                            partial_query, 
                            k=limit, 
                            active_only=True, 
                            min_score=0.1
                        )
                        
                        # Extract names and scores and ensure they start with the partial query
                        for result in results:
                            full_name = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
                            if full_name.lower().startswith(partial_query.lower()):
                                predictions.append(full_name)
                                confidence_scores.append(float(result.get('score', 0.5)))
                    
                    elif context == "designation":
                        # Use the designation index search
                        results = search_manager.search_experts_by_designation(
                            partial_query, 
                            k=limit*3, 
                            active_only=True, 
                            min_score=0.1
                        )
                        
                        # Extract unique designations with scores and filter by prefix
                        unique_designations = {}
                        for result in results:
                            designation = result.get('designation', '')
                            if designation and designation.lower().startswith(partial_query.lower()):
                                unique_designations[designation] = float(result.get('score', 0.5))
                        
                        # Convert to lists
                        predictions = list(unique_designations.keys())
                        confidence_scores = list(unique_designations.values())
                        
                        # Limit to top results
                        predictions = predictions[:limit]
                        confidence_scores = confidence_scores[:limit]
                    
                    elif context == "theme":
                        # Use the theme index search
                        results = search_manager.search_experts_by_theme(
                            partial_query, 
                            k=limit*3, 
                            active_only=True, 
                            min_score=0.1
                        )
                        
                        # Extract unique themes with scores and filter by prefix
                        unique_themes = {}
                        for result in results:
                            theme = result.get('theme', '')
                            if theme and theme.lower().startswith(partial_query.lower()):
                                unique_themes[theme] = float(result.get('score', 0.5))
                        
                        # Convert to lists
                        predictions = list(unique_themes.keys())
                        confidence_scores = list(unique_themes.values())
                        
                        # Limit to top results
                        predictions = predictions[:limit]
                        confidence_scores = confidence_scores[:limit]
                    
                    elif context == "publication":
                        # Use publication-specific search
                        try:
                            # Use the publication search if available
                            results = search_manager.search_experts_by_publication(
                                partial_query,
                                k=limit*3,
                                min_score=0.1
                            )
                            
                            # Extract publication titles from matches and filter by prefix
                            unique_publications = {}
                            for result in results:
                                if 'publication_match' in result and result['publication_match']:
                                    title = result['publication_match'].get('title', '')
                                    if title and title.lower().startswith(partial_query.lower()):
                                        unique_publications[title] = float(result.get('score', 0.5))
                            
                            # Convert to lists
                            predictions = list(unique_publications.keys())
                            confidence_scores = list(unique_publications.values())
                            
                            # Limit to top results
                            predictions = predictions[:limit]
                            confidence_scores = confidence_scores[:limit]
                        
                        except Exception as pub_error:
                            logger.error(f"Publication search error: {pub_error}", exc_info=True)
                            error_occurred = True
                        
                    else:
                        # No specific context, use general expert search and filter with prefix
                        results = search_manager.search_experts(
                            partial_query, 
                            k=limit*3, 
                            active_only=True, 
                            min_score=0.1
                        )
                        
                        # Collect all types of results
                        names = {}
                        designations = {}
                        themes = {}
                        
                        for result in results:
                            # Add name
                            full_name = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
                            if full_name and full_name.lower().startswith(partial_query.lower()):
                                names[full_name] = float(result.get('score', 0.5))
                                
                            # Add designation
                            designation = result.get('designation', '')
                            if designation and designation.lower().startswith(partial_query.lower()):
                                designations[designation] = float(result.get('score', 0.5)) * 0.9  # Slightly lower priority
                                
                            # Add theme
                            theme = result.get('theme', '')
                            if theme and theme.lower().startswith(partial_query.lower()):
                                themes[theme] = float(result.get('score', 0.5)) * 0.8  # Lower priority
                        
                        # Combine all results
                        combined_items = list(names.items()) + list(designations.items()) + list(themes.items())
                        
                        # Sort by score (descending)
                        combined_items.sort(key=lambda x: x[1], reverse=True)
                        
                        # Extract predictions and scores
                        predictions = [item[0] for item in combined_items[:limit]]
                        confidence_scores = [item[1] for item in combined_items[:limit]]
                    
                    # If not enough prefix matches, try to get Google suggestions and filter them
                    if len(predictions) < limit // 2:
                        try:
                            predictor = await get_predictor()
                            google_suggestions = await predictor.predict(partial_query, limit=limit*2)
                            
                            # Filter Google suggestions by prefix
                            for suggestion in google_suggestions:
                                suggestion_text = suggestion.get("text", "")
                                if suggestion_text and suggestion_text.lower().startswith(partial_query.lower()):
                                    if suggestion_text not in predictions:
                                        predictions.append(suggestion_text)
                                        confidence_scores.append(suggestion.get("score", 0.5) * 0.9)  # Lower confidence for Google
                        except Exception as google_error:
                            logger.warning(f"Error getting Google suggestions: {google_error}")
                    
                except Exception as idx_error:
                    logger.error(f"Index search error: {idx_error}", exc_info=True)
                    error_occurred = True
                    
                    # Generic fallback if everything fails - use Google but filter by prefix
                    predictor = await get_predictor()
                    suggestion_objects = await predictor.predict(partial_query, limit=limit*2)
                    
                    # Only include suggestions that start with the partial query
                    predictions = []
                    confidence_scores = []
                    for suggestion in suggestion_objects:
                        suggestion_text = suggestion.get("text", "")
                        if suggestion_text and suggestion_text.lower().startswith(partial_query.lower()):
                            predictions.append(suggestion_text)
                            confidence_scores.append(suggestion.get("score", 0.5) * 0.6)  # Very low confidence
        
        # Generate refinements that match the current query prefix
        refinements_dict = None
        try:
            # Get context-appropriate refinements
            refinements_dict = await generate_dynamic_refinements(
                partial_query, 
                context, 
                predictions,
                limit
            )
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements_dict = {
                "filters": [],
                "related_queries": predictions[:5] if predictions else [],
                "expertise_areas": []
            }
        
        # Convert refinements to a flat list of strings as expected by PredictionResponse
        refinements_list = []
        if isinstance(refinements_dict, dict):
            # Extract related queries if they exist
            if "related_queries" in refinements_dict and isinstance(refinements_dict["related_queries"], list):
                refinements_list.extend(refinements_dict["related_queries"])
            
            # Extract expertise areas if they exist
            if "expertise_areas" in refinements_dict and isinstance(refinements_dict["expertise_areas"], list):
                refinements_list.extend(refinements_dict["expertise_areas"])
            
            # Extract values from filters if they exist
            if "filters" in refinements_dict and isinstance(refinements_dict["filters"], list):
                for filter_item in refinements_dict["filters"]:
                    if isinstance(filter_item, dict) and "values" in filter_item:
                        if isinstance(filter_item["values"], list):
                            refinements_list.extend(filter_item["values"])
                        else:
                            refinements_list.append(str(filter_item["values"]))
        
        # Record analytics if we have a connection
        try:
            conn = get_db_connection()
            if conn and predictions:
                session_id = await get_or_create_session(conn, user_id)
                await record_prediction(
                    conn,
                    session_id,
                    user_id,
                    partial_query,
                    predictions,
                    confidence_scores
                )
        except Exception as record_error:
            logger.error(f"Error recording prediction: {record_error}", exc_info=True)
        finally:
            if conn:
                conn.close()
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions),
            search_context=context
        )
    
    except Exception as e:
        logger.error(f"Error in advanced query prediction: {e}", exc_info=True)
        # Initialize empty lists for fallback response
        predictions = []
        confidence_scores = []
        refinements_list = []
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions)
        )
def filter_predictions_by_context(
    predictions: List[str], 
    confidence_scores: List[float], 
    partial_query: str, 
    context: str
) -> Tuple[List[str], List[float]]:
    """
    Strictly filter predictions based on context type.
    For 'name' context, only return person name suggestions.
    For 'publication' context, prioritize publication-like suggestions.
    
    Args:
        predictions: List of predicted query completions
        confidence_scores: Corresponding confidence scores for each prediction
        partial_query: The partial query being typed
        context: Context type ('name', 'theme', 'designation', 'publication')
        
    Returns:
        Tuple of filtered predictions and their confidence scores
    """
    if context != "name" and context != "publication":
        # For other contexts, return original predictions
        return predictions, confidence_scores
    
    filtered_predictions = []
    filtered_scores = []
    
    if context == "name":
        for i, pred in enumerate(predictions):
            original_score = confidence_scores[i] if i < len(confidence_scores) else 0.5
            
            if is_person_name(pred, partial_query):
                filtered_predictions.append(pred)
                filtered_scores.append(original_score * 1.2)  # Boost name matches
        
        # Fallback suggestions when no perfect matches exist
        if not filtered_predictions:
            return generate_name_fallback_suggestions(partial_query)
    
    elif context == "publication":
        # Publication filtering logic
        for i, pred in enumerate(predictions):
            original_score = confidence_scores[i] if i < len(confidence_scores) else 0.5
            
            if is_publication_title(pred, partial_query):
                filtered_predictions.append(pred)
                filtered_scores.append(original_score * 1.2)  # Boost publication matches
        
        # If no matches, return original predictions for publication context
        if not filtered_predictions:
            return predictions, confidence_scores
    
    return filtered_predictions, filtered_scores

def is_person_name(suggestion: str, partial_query: str) -> bool:
    """
    Strict check if suggestion appears to be a person name.
    Enhanced for prefix matching.
    
    Names should:
    - Start with the partial query (case-insensitive)
    - Be 1-3 words 
    - Have each word capitalized
    - Not contain research or technical terms
    """
    # Research and technical terms to exclude
    topic_indicators = [
        "research", "study", "analysis", "method", 
        "framework", "data", "model", "system", 
        "pathway", "protein", "drug", "discovery", 
        "resistance", "structure", "function", 
        "signaling", "antimicrobial", "cancer"
    ]
    
    # Convert partial query and suggestion to lowercase for comparison
    partial_query_lower = partial_query.lower()
    suggestion_lower = suggestion.lower()
    
    # First, check if suggestion starts with the partial query
    if not suggestion_lower.startswith(partial_query_lower):
        return False
    
    words = suggestion.split()
    
    # Check length (allow single word for partial names)
    if not (1 <= len(words) <= 3):
        return False
    
    # Check capitalization (except for partial single words)
    if len(words) > 1 and not all(word.istitle() for word in words):
        return False
    
    # Check for topic indicators
    if any(indicator in suggestion_lower for indicator in topic_indicators):
        return False
    
    # Additional checks to prevent non-name suggestions
    if any(
        last_word.endswith(suffix) 
        for last_word in [words[-1].lower()] 
        for suffix in ['ing', 'tion', 'ism', 'ogy', 'path', 'ence']
    ):
        return False
    
    return True


async def generate_dynamic_refinements(
    partial_query: str,
    context: Optional[str],
    current_predictions: List[str],
    limit: int = 5
) -> Dict[str, Any]:
    """
    Generate refinement suggestions that match the current query prefix.
    
    Args:
        partial_query: The partial query being typed (may be empty)
        context: Optional context type (name, theme, designation, publication)
        current_predictions: Current predictions for reference
        limit: Maximum number of suggestions per category
        
    Returns:
        Dictionary with categorized refinement suggestions
    """
    try:
        # Set up empty refinements structure
        refinements = {
            "filters": [],
            "related_queries": [],
            "expertise_areas": []
        }
        
        # If no query, provide general refinements based on context
        if not partial_query:
            # Add context-specific filter categories
            if context == "name":
                refinements["filters"].append({
                    "type": "name_filter",
                    "label": "Name Search",
                    "values": ["First Name", "Last Name", "Full Name"]
                })
            elif context == "theme":
                refinements["filters"].append({
                    "type": "theme_filter",
                    "label": "Research Areas",
                    "values": ["Neuroscience", "Data Science", "Psychology", "Biochemistry", "Engineering"]
                })
            elif context == "designation":
                refinements["filters"].append({
                    "type": "role_filter",
                    "label": "Roles",
                    "values": ["Professor", "Researcher", "Director", "Scientist", "Student"]
                })
            elif context == "publication":
                refinements["filters"].append({
                    "type": "publication_filter",
                    "label": "Publication Types",
                    "values": ["Journal Articles", "Conference Papers", "Books", "Reviews", "Preprints"]
                })
            else:
                refinements["filters"].append({
                    "type": "search_type",
                    "label": "Search Types",
                    "values": ["People", "Research", "Publications", "Departments", "Expertise"]
                })
            
            # Add some trending topics as expertise areas
            trend_suggestions = await get_trending_suggestions("", limit)
            trend_texts = [s.get("text", "") for s in trend_suggestions]
            refinements["expertise_areas"] = trend_texts
            
            # Return early for empty query case
            return refinements
        
        # For active queries, filter refinements to match the prefix
        
        # First get existing refinements from Gemini or other sources
        try:
            # Use GoogleAutocompletePredictor to get related queries
            predictor = await get_predictor()
            suggestion_objects = await predictor.predict(partial_query, limit=limit*2)
            related_queries = [s["text"] for s in suggestion_objects]
            
            # Filter to only queries that start with the partial query
            prefix_queries = [q for q in related_queries if q.lower().startswith(partial_query.lower())]
            
            # If not enough prefix matches, also include queries that contain the partial query
            if len(prefix_queries) < limit:
                contains_queries = [q for q in related_queries if partial_query.lower() in q.lower() and q not in prefix_queries]
                related_queries = prefix_queries + contains_queries[:limit - len(prefix_queries)]
            else:
                related_queries = prefix_queries[:limit]
            
            refinements["related_queries"] = related_queries
        except Exception as query_error:
            logger.error(f"Error getting related queries: {query_error}")
        
        # Generate expertise areas relevant to the partial query
        try:
            # Extract potential expertise areas from predictions and related queries
            expertise_areas = set()
            
            # Use our current predictions
            for prediction in current_predictions:
                if prediction.lower().startswith(partial_query.lower()):
                    expertise_areas.add(prediction.capitalize())
            
            # Use related queries too
            for query in refinements["related_queries"]:
                words = query.lower().split()
                for word in words:
                    # Only add significant words that match our prefix
                    if len(word) > 3 and word.startswith(partial_query.lower()):
                        expertise_areas.add(word.capitalize())
            
            # Ensure we have at least one expertise area
            if not expertise_areas and partial_query:
                expertise_areas.add(partial_query.capitalize())
            
            # Convert to list and limit
            refinements["expertise_areas"] = list(expertise_areas)[:limit]
        except Exception as expertise_error:
            logger.error(f"Error generating expertise areas: {expertise_error}")
        
        # Add context-specific filters that match the prefix
        if context and partial_query:
            try:
                if context == "name":
                    # For name search, suggest common name prefixes
                    name_prefix = partial_query.capitalize()
                    refinements["filters"].append({
                        "type": "name_starts_with",
                        "label": f"Names starting with {partial_query}",
                        "values": [f"{name_prefix}", f"{name_prefix} (First Name)", f"{name_prefix} (Last Name)"]
                    })
                
                elif context == "theme":
                    # For theme search, suggest theme categories
                    theme_categories = []
                    if partial_query.lower().startswith("n"):
                        theme_categories.append("Neuroscience")
                    if partial_query.lower().startswith("p"):
                        theme_categories.append("Psychology")
                    if partial_query.lower().startswith("c"):
                        theme_categories.append("Computer Science")
                    if partial_query.lower().startswith("b"):
                        theme_categories.append("Biology")
                    if partial_query.lower().startswith("e"):
                        theme_categories.append("Engineering")
                    
                    # Only include filter if we have matching categories
                    if theme_categories:
                        refinements["filters"].append({
                            "type": "theme_category",
                            "label": "Research Areas",
                            "values": theme_categories
                        })
                
                elif context == "designation":
                    # Find designations that match our prefix
                    designations = []
                    common_roles = ["Professor", "Researcher", "Scientist", "Director", "Student", "Lecturer", "Fellow"]
                    
                    for role in common_roles:
                        if role.lower().startswith(partial_query.lower()):
                            designations.append(role)
                    
                    if designations:
                        refinements["filters"].append({
                            "type": "role_filter",
                            "label": "Job Roles",
                            "values": designations
                        })
            except Exception as filter_error:
                logger.error(f"Error generating context filters: {filter_error}")
        
        return refinements
    
    except Exception as e:
        logger.error(f"Error in dynamic refinements: {e}", exc_info=True)
        return {
            "filters": [],
            "related_queries": [],
            "expertise_areas": []
        }

def generate_name_fallback_suggestions(partial_query: str) -> Tuple[List[str], List[float]]:
    """Generate simple name suggestions when no good matches are found"""
    common_first_names = [
        "Abdul", "Abdullah", "Abdel", "Abdala"
    ]
    common_last_names = [
        "Rahman", "Ali", "Ahmed", "Muhammad"
    ]
    
    suggestions = []
    
    # If partial query is one word, try to match with names
    if len(partial_query.split()) == 1:
        # Prioritize combinations that include the partial query
        suggestions = [
            f"{first} {last}" 
            for first in common_first_names 
            for last in common_last_names
            if partial_query.lower() in (first + " " + last).lower()
        ]
        
        # If no matches, fall back to generic combinations
        if not suggestions:
            suggestions = [
                f"{first} {last}" 
                for first in common_first_names[:2] 
                for last in common_last_names[:2]
            ]
    else:
        # If already two words, just return as-is if it looks like a name
        suggestions = [partial_query] if is_person_name(partial_query, partial_query) else []
    
    # If still no suggestions, create some
    if not suggestions:
        suggestions = [f"{partial_query} {last}" for last in common_last_names[:2]]
    
    # Generate decreasing confidence scores
    scores = [0.9 - (i * 0.1) for i in range(len(suggestions))]
    
    return suggestions, scores

def generate_advanced_predictive_refinements(
    partial_query: str, 
    predictions: List[str], 
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate advanced predictive refinements with context-specific filtering.
    
    Args:
        partial_query: The partial query string
        predictions: List of predicted suggestions
        context: Optional context for filtering (name, theme, designation)
    
    Returns:
        Dict of refinement suggestions
    """
    logger.info(f"Generating advanced predictive refinements - Context: {context}")
    
    try:
        # Context-specific keywords for refinement
        context_keywords = {
            "name": {
                "filters": ["Personal Profiles", "Researcher Names", "Expert Identities"],
                "expertise_prefixes": ["Expert in", "Researcher with", "Professional"]
            },
            "theme": {
                "filters": ["Research Areas", "Study Domains", "Academic Fields"],
                "expertise_prefixes": ["Research in", "Specializing in", "Focus on"]
            },
            "designation": {
                "filters": ["Professional Roles", "Job Titles", "Career Levels"],
                "expertise_prefixes": ["Experts as", "Professionals with", "Specialists in"]
            }
        }
        
        # Prepare base refinements structure
        refinements = {
            "filters": [],
            "related_queries": predictions[:5],
            "expertise_areas": []
        }
        
        # Add context-specific filters if context is provided
        if context and context in context_keywords:
            refinements["filters"] = [{
                "type": f"{context}_filter",
                "label": context_keywords[context]["filters"][0],
                "values": context_keywords[context]["filters"]
            }]
            
            # Generate expertise areas
            expertise_areas = set()
            prefix = context_keywords[context]["expertise_prefixes"][0]
            
            for prediction in predictions:
                words = prediction.split()
                for word in words:
                    if len(word) > 3 and word.lower() != partial_query.lower():
                        expertise_areas.add(f"{prefix} {word.capitalize()}")
            
            # Ensure some expertise areas exist
            if not expertise_areas:
                expertise_areas.add(f"{prefix} {partial_query.capitalize()}")
            
            refinements["expertise_areas"] = list(expertise_areas)[:5]
        
        return refinements
    
    except Exception as e:
        logger.error(f"Error in advanced predictive refinements: {e}")
        return {
            "filters": [],
            "related_queries": predictions[:5],
            "expertise_areas": []
        }

