from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
from datetime import datetime
import json
import uuid

from ai_services_api.services.search.gemini.gemini_predictor import GeminiPredictor
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.search.core.models import PredictionResponse

# Configure logger
logger = logging.getLogger(__name__)

# Reuse existing session function
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

# Record prediction in database for analytics
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

# Generate refinements based on predictions
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

# The main process function that uses Gemini API directly
async def process_query_prediction(partial_query: str, user_id: str) -> PredictionResponse:
    """
    Enhanced query prediction using Gemini API with Google Search Retrieval
    
    Args:
        partial_query (str): Partial search query
        user_id (str): User identifier
    
    Returns:
        PredictionResponse: Prediction results with refinements
    """
    logger.info(f"Processing query prediction - Partial query: {partial_query}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID in case DB connection fails
    
    try:
        # Establish database connection for analytics
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Generate predictions directly from Gemini API
        predictor = GeminiPredictor()
        suggestion_objects = await predictor.predict(partial_query, limit=5)
        
        # Extract suggestion texts
        predictions = [s["text"] for s in suggestion_objects]
        
        # Generate confidence scores
        confidence_scores = predictor.generate_confidence_scores(suggestion_objects)
        
        # Log the predictions
        logger.debug(f"Generated {len(predictions)} predictions: {predictions}")
        
        # Generate refinement suggestions
        refinements = generate_predictive_refinements(partial_query, predictions)
        
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