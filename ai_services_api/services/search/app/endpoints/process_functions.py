from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
from datetime import datetime
import json
import uuid
import asyncio
from redis import asyncio as aioredis

from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.search.core.models import PredictionResponse
from ai_services_api.services.search.core.personalization import personalize_suggestions



# Configure logger
logger = logging.getLogger(__name__)

# Global predictor instance (singleton)
_predictor = None

# Global Redis instance
_redis = None

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

async def record_prediction(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """Record prediction in database for analytics."""
    try:
        # Create a background task for recording to avoid blocking the response
        asyncio.create_task(
            _record_prediction_async(conn, session_id, user_id, partial_query, predictions, confidence_scores)
        )
    except Exception as e:
        logger.error(f"Failed to create background task for recording prediction: {e}")

async def _record_prediction_async(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """Asynchronous helper for recording predictions in the database."""
    logger.info(f"Recording predictions for user {user_id}, session {session_id}")
    
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

async def process_query_prediction(partial_query: str, user_id: str) -> PredictionResponse:
    """
    Process query prediction with personalization based on user history.
    
    Args:
        partial_query: The partial query to get suggestions for
        user_id: The ID of the user making the request
        
    Returns:
        PredictionResponse: Response containing predictions
    """
    logger.info(f"Processing query prediction - Query: '{partial_query}', User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID in case DB connection fails
    
    # Try to get Redis connection for caching
    redis = await get_redis()
    
    try:
        # Check cache first if Redis is available
        cache_hit = False
        if redis:
            try:
                cache_key = f"google_autocomplete:{partial_query}"
                cached_result = await redis.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache hit for query: {partial_query}")
                    cache_hit = True
                    
                    # Use cached suggestions
                    cached_data = json.loads(cached_result)
                    suggestions = cached_data["suggestions"]
                    confidence_scores = cached_data["confidence_scores"]
                    
                    # Add personalization to cached suggestions
                    try:
                        personalized_suggestions = await personalize_suggestions(
                            suggestions, user_id, partial_query
                        )
                        # Update scores after personalization
                        confidence_scores = [s.get("score", 0.5) for s in personalized_suggestions]
                        
                        # Extract just the texts for the response
                        predictions = [s.get("text", "") for s in personalized_suggestions]
                        
                        # Skip further processing
                        return PredictionResponse(
                            predictions=predictions,
                            confidence_scores=confidence_scores,
                            user_id=user_id,
                            refinements=generate_predictive_refinements(
                                partial_query, 
                                predictions
                            )
                        )
                    except Exception as personalize_error:
                        logger.error(f"Personalization error for cached results: {personalize_error}")
                        # Continue with non-personalized cached suggestions if personalization fails
                        predictions = [s["text"] for s in suggestions]
                        return PredictionResponse(
                            predictions=predictions,
                            confidence_scores=confidence_scores,
                            user_id=user_id,
                            refinements=generate_predictive_refinements(
                                partial_query, 
                                predictions
                            )
                        )
            except Exception as cache_error:
                logger.error(f"Cache retrieval error: {cache_error}", exc_info=True)
        
        # Establish database connection for analytics
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Get predictor instance (singleton)
        predictor = await get_predictor()
        
        # Generate predictions from Google Autocomplete API
        suggestion_objects = await predictor.predict(partial_query, limit=10)
        
        # Apply personalization to fresh suggestions
        try:
            personalized_suggestions = await personalize_suggestions(
                suggestion_objects, user_id, partial_query
            )
            suggestion_objects = personalized_suggestions
            logger.info(f"Applied personalization to suggestions")
        except Exception as personalize_error:
            logger.error(f"Personalization error for fresh results: {personalize_error}")
            # Continue with non-personalized suggestions if personalization fails
        
        # Extract suggestion texts
        predictions = [s["text"] for s in suggestion_objects]
        
        # Generate confidence scores (use the personalized scores)
        confidence_scores = [s.get("score", 0.5) for s in suggestion_objects]
        
        # Log the predictions
        logger.debug(f"Generated {len(predictions)} predictions: {predictions}")
        
        # Cache the results if Redis is available
        if redis and not cache_hit and predictions:
            try:
                cache_key = f"google_autocomplete:{partial_query}"
                cache_data = {
                    "suggestions": suggestion_objects,
                    "confidence_scores": confidence_scores
                }
                
                # Cache for 15 minutes (900 seconds)
                await redis.setex(
                    cache_key,
                    900,
                    json.dumps(cache_data)
                )
                logger.debug(f"Cached predictions for: {partial_query}")
            except Exception as cache_error:
                logger.error(f"Cache storage error: {cache_error}", exc_info=True)
        
        # Generate refinement suggestions
        refinements = generate_predictive_refinements(partial_query, predictions)
        
        # Record prediction if we have a DB connection
        if conn and predictions:
            await record_prediction(
                conn,
                session_id,
                user_id,
                partial_query,
                predictions,
                confidence_scores
            )
        
        # Prepare response
        response = PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements
        )
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in query prediction: {e}", exc_info=True)
        
        # Always return a valid response even on error
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