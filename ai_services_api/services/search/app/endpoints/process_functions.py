# process_functions.py
# Complete implementation with all imports

from fastapi import HTTPException
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
from datetime import datetime, timezone
import json
import uuid
from redis.asyncio import Redis

from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.ml.ml_predictor import MLPredictor, RedisConnectionManager
from ai_services_api.services.message.core.database import get_db_connection

# Import response models
from ai_services_api.services.search.api.models import (
    ExpertSearchResult,
    SearchResponse,
    PredictionResponse
)

# Configure logger
logger = logging.getLogger(__name__)

# Reuse existing session and analytics functions
async def get_or_create_session(conn, user_id: str) -> str:
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

async def record_search(conn, session_id: str, user_id: str, query: str, results: List[Dict], response_time: float):
    logger.info(f"Recording search analytics - Session: {session_id}, User: {user_id}")
    logger.debug(f"Recording search for query: {query} with {len(results)} results")
    
    cur = conn.cursor()
    try:
        # Record search analytics
        cur.execute("""
            INSERT INTO search_analytics
                (search_id, query, user_id, response_time,
                 result_count, search_type, timestamp)
            VALUES
                ((SELECT id FROM search_sessions WHERE session_id = %s),
                %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (
            session_id,
            query,
            user_id,
            response_time,
            len(results),
            'expert_search'
        ))
        
        search_id = cur.fetchone()[0]
        logger.debug(f"Created search analytics record with ID: {search_id}")

        # Record top 5 expert matches
        logger.debug(f"Recording top 5 matches from {len(results)} total results")
        for rank, result in enumerate(results[:5], 1):
            cur.execute("""
                INSERT INTO expert_search_matches
                    (search_id, expert_id, rank_position, similarity_score)
                VALUES (%s, %s, %s, %s)
            """, (
                search_id,
                result["id"],
                rank,
                result.get("score", 0.0)
            ))
            logger.debug(f"Recorded match - Expert ID: {result['id']}, Rank: {rank}")

        conn.commit()
        logger.info(f"Successfully recorded all search data for search ID: {search_id}")
        return search_id
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording search: {str(e)}", exc_info=True)
        logger.debug(f"Search recording failed: {str(e)}")
        raise
    finally:
        cur.close()

async def record_prediction(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    logger.info(f"Recording predictions for user {user_id}, session {session_id}")
    logger.debug(f"Recording predictions for partial query: {partial_query}")
    
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
            logger.debug(f"Recorded prediction: {pred} with confidence: {conf}")
        
        conn.commit()
        logger.info("Successfully recorded all predictions")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording prediction: {str(e)}", exc_info=True)
        logger.debug(f"Prediction recording failed: {str(e)}")
        raise
    finally:
        cur.close()

def generate_predictive_refinements(partial_query: str, predictions: List[str]) -> Dict[str, Any]:
    """
    Generate refinement suggestions based on partial query and predictions
    
    Args:
        partial_query (str): Initial partial query
        predictions (List[str]): Predicted full queries
    
    Returns:
        Dict: Refinement suggestions
    """
    try:
        # Initialize search manager
        from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
        search_manager = ExpertSearchIndexManager()
        
        # Generate related filters and expertise areas
        related_filters = []
        
        # Add a filter for query type if we have predictions
        if predictions:
            related_filters.append({
                "type": "query_type",
                "label": "Query Suggestions",
                "values": predictions[:3]
            })
        
        # Generate semantic variations to enhance related queries
        semantic_variations = []
        if partial_query:
            # Create variations like "expert in X", "research about X"
            variations = [
                f"expert in {partial_query}",
                f"{partial_query} research",
                f"{partial_query} specialist",
                f"studies on {partial_query}"
            ]
            semantic_variations = [v for v in variations if v not in predictions]
        
        # Combine predictions with variations
        combined_queries = list(predictions)
        for variation in semantic_variations:
            if len(combined_queries) < 8:  # Limit to 8 total suggestions
                combined_queries.append(variation)
        
        # Try to extract potential expertise areas from combined queries
        expertise_areas = set()
        for query in combined_queries:
            # Extract words that might be expertise areas
            words = query.lower().split()
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
            "related_queries": combined_queries[:5],
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

async def get_redis():
    """Initialize Redis connection with error handling."""
    logger.debug("Initializing Redis connection")
    try:
        # Use the RedisConnectionManager instead of creating new connections
        from ai_services_api.services.search.ml.ml_predictor import RedisConnectionManager
        
        redis_manager = RedisConnectionManager.get_instance()
        redis_client = redis_manager.get_connection(db=3)  # DB 3 for caching
        
        # Test connection
        ping_result = await redis_client.ping()
        if not ping_result:
            logger.warning("Redis connection established but ping failed")
            
        logger.info("Redis connection established successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}", exc_info=True)
        # Return None instead of raising - will handle gracefully in processing functions
        return None

async def process_query_prediction(partial_query: str, user_id: str, redis_client: Redis = None) -> PredictionResponse:
    """
    Enhanced query prediction with improved error handling and no fallbacks
    
    Args:
        partial_query (str): Partial search query
        user_id (str): User identifier
        redis_client (Redis, optional): Redis client for caching
    
    Returns:
        PredictionResponse: Prediction results with optional refinements
    """
    logger.info(f"Processing query prediction - Partial query: {partial_query}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID in case DB connection fails
    
    try:
        # Handle Redis client being None (connection failure)
        if redis_client is None:
            logger.warning("Redis client unavailable, proceeding without caching")
        
        # Try cache if Redis is available
        cache_hit = False
        if redis_client:
            try:
                cache_key = f"query_prediction:{user_id}:{partial_query}"
                cached_response = await redis_client.get(cache_key)
                if cached_response:
                    logger.info(f"Cache hit for prediction: {cache_key}")
                    cache_hit = True
                    return PredictionResponse(**json.loads(cached_response))
            except Exception as cache_error:
                logger.error(f"Cache retrieval error: {cache_error}", exc_info=True)
                # Continue processing instead of failing

        # Establish database connection
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Generate predictions - handle potentially empty results
        predictions = []
        try:
            predictor = MLPredictor()
            predictions = predictor.predict(partial_query, user_id=user_id)
            logger.debug(f"Generated {len(predictions)} predictions")
        except Exception as pred_error:
            logger.error(f"Prediction generation error: {pred_error}", exc_info=True)
        
        # Generate confidence scores (higher for first results)
        confidence_scores = []
        if predictions:
            confidence_scores = [max(0.1, 1.0 - (i * 0.1)) for i in range(len(predictions))]
        
        # Generate refinement suggestions
        refinements = {}
        try:
            refinements = generate_predictive_refinements(partial_query, predictions)
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements = {
                "filters": [],
                "related_queries": predictions[:5] if predictions else [],
                "expertise_areas": []
            }
        
        # Record prediction if we have a DB connection
        if conn and predictions:
            try:
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
        
        # Prepare response
        response = PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements
        )

        # Cache the response if Redis client is available
        if redis_client and not cache_hit:
            try:
                await redis_client.setex(
                    f"query_prediction:{user_id}:{partial_query}",
                    1800,  # 30 minutes
                    json.dumps(response.dict())
                )
                logger.debug(f"Cached predictions for: {partial_query}")
            except Exception as cache_error:
                logger.error(f"Cache storage error: {cache_error}", exc_info=True)
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in query prediction: {e}", exc_info=True)
        
        # Always return a valid response
        return PredictionResponse(
            predictions=[partial_query] if partial_query else [],
            confidence_scores=[0.5] if partial_query else [],
            user_id=user_id,
            refinements={
                "filters": [],
                "related_queries": [partial_query] if partial_query else [],
                "expertise_areas": []
            }
        )
    finally:
        # Always close connection if opened
        if conn:
            conn.close()
            logger.debug("Database connection closed")

async def process_expert_search(query: str, user_id: str, active_only: bool = True, redis_client: Redis = None) -> SearchResponse:
    """
    Process expert search with improved error handling
    
    Args:
        query (str): Search query
        user_id (str): User identifier
        active_only (bool): Filter for active experts only
        redis_client (Redis, optional): Redis client for caching
    
    Returns:
        SearchResponse: Search results including optional refinements
    """
    logger.info(f"Processing expert search - Query: {query}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID in case DB connection fails
    
    try:
        # Handle Redis client being None (connection failure)
        if redis_client is None:
            logger.warning("Redis client unavailable, proceeding without caching")
        
        # Try cache if Redis is available
        cache_hit = False
        if redis_client:
            try:
                cache_key = f"expert_search:{user_id}:{query}:{active_only}"
                cached_response = await redis_client.get(cache_key)
                if cached_response:
                    logger.info(f"Cache hit for search: {cache_key}")
                    cache_hit = True
                    return SearchResponse(**json.loads(cached_response))
            except Exception as cache_error:
                logger.error(f"Cache retrieval error: {cache_error}", exc_info=True)
                # Continue processing instead of failing

        # Establish database connection
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Execute search with timing
        start_time = datetime.utcnow()
        results = []
        try:
            search_manager = ExpertSearchIndexManager()
            results = search_manager.search_experts(query, k=5, active_only=active_only)
            logger.info(f"Search found {len(results)} results")
        except Exception as search_error:
            logger.error(f"Search execution error: {search_error}", exc_info=True)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Format results - handle empty results gracefully
        formatted_results = []
        try:
            formatted_results = [
                ExpertSearchResult(
                    id=str(result.get('id', 'unknown')),
                    first_name=result.get('first_name', ''),
                    last_name=result.get('last_name', ''),
                    designation=result.get('designation', ''),
                    theme=result.get('theme', ''),
                    unit=result.get('unit', ''),
                    contact=result.get('contact', ''),
                    is_active=result.get('is_active', True),
                    score=result.get('score'),
                    bio=result.get('bio'),
                    knowledge_expertise=result.get('knowledge_expertise', [])
                ) for result in results
            ]
        except Exception as format_error:
            logger.error(f"Result formatting error: {format_error}", exc_info=True)
        
        # Record search analytics if we have a connection
        if conn and results:
            try:
                await record_search(conn, session_id, user_id, query, results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions
        refinements = {}
        try:
            if results:
                refinements = search_manager.get_search_refinements(query, results)
            else:
                # Default empty refinements
                refinements = {
                    "filters": [],
                    "related_queries": [],
                    "expertise_areas": []
                }
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
        
        # Prepare response
        response = SearchResponse(
            total_results=len(formatted_results),
            experts=formatted_results,
            user_id=user_id,
            session_id=session_id,
            refinements=refinements
        )

        # Cache the response if Redis client is available and we have results
        if redis_client and not cache_hit and formatted_results:
            try:
                await redis_client.setex(
                    f"expert_search:{user_id}:{query}:{active_only}",
                    3600,  # Cache for 1 hour
                    json.dumps(response.dict())
                )
                logger.debug(f"Cached search results for: {query}")
            except Exception as cache_error:
                logger.error(f"Cache storage error: {cache_error}", exc_info=True)
        
        # Update ML predictor if search was successful
        if formatted_results:
            try:
                predictor = MLPredictor()
                predictor.update(query, user_id)
            except Exception as ml_error:
                logger.error(f"Error updating ML predictor: {ml_error}", exc_info=True)
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in expert search: {e}", exc_info=True)
        
        # Always return a valid response
        return SearchResponse(
            total_results=0,
            experts=[],
            user_id=user_id,
            session_id=session_id,
            refinements={
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        )
    finally:
        # Always close connection if opened
        if conn:
            conn.close()
            logger.debug("Database connection closed")