from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
from datetime import datetime, timezone
import json
import pandas as pd
from redis.asyncio import Redis
import uuid

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

# Initialize ML Predictor
ml_predictor = MLPredictor()
logger.info("ML Predictor initialized successfully")

# Response Models [unchanged]
class ExpertSearchResult(BaseModel):
    id: str
    first_name: str
    last_name: str
    designation: str
    theme: str
    unit: str
    contact: str
    is_active: bool
    score: Optional[float] = None
    bio: Optional[str] = None  
    knowledge_expertise: List[str] = []

class SearchResponse(BaseModel):
    total_results: int
    experts: List[ExpertSearchResult]
    user_id: str
    session_id: str
    refinements: Optional[Dict[str, Any]] = None  # New optional field for refinements

class PredictionResponse(BaseModel):
    predictions: List[str]
    confidence_scores: List[float]
    user_id: str
    refinements: Optional[Dict[str, Any]] = None  # New optional field

async def get_redis():
    logger.debug("Initializing Redis connection")
    redis_client = Redis(host='redis', port=6379, db=3, decode_responses=True)
    logger.info("Redis connection established")
    return redis_client

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
    print(f"Test user ID being used: {TEST_USER_ID}")
    return TEST_USER_ID

async def get_or_create_session(conn, user_id: str) -> str:
    logger.info(f"Getting or creating session for user: {user_id}")
    cur = conn.cursor()
    try:
        session_id = int(str(int(datetime.utcnow().timestamp()))[-8:])
        print(f"Generated session ID: {session_id}")
        
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
        print(f"Session creation failed: {str(e)}")
        raise
    finally:
        cur.close()

async def record_search(conn, session_id: str, user_id: str, query: str, results: List[Dict], response_time: float):
    logger.info(f"Recording search analytics - Session: {session_id}, User: {user_id}")
    print(f"Recording search for query: {query} with {len(results)} results")
    
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
        print(f"Recording top 5 matches from {len(results)} total results")
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
        print(f"Search recording failed: {str(e)}")
        raise
    finally:
        cur.close()

async def record_prediction(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    logger.info(f"Recording predictions for user {user_id}, session {session_id}")
    print(f"Recording predictions for partial query: {partial_query}")
    
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
        print(f"Prediction recording failed: {str(e)}")
        raise
    finally:
        cur.close()

async def process_expert_search(query: str, user_id: str, active_only: bool = True, redis_client: Redis = None) -> SearchResponse:
    """
    Process expert search with added refinement suggestions.
    
    Args:
        query (str): Search query
        user_id (str): User identifier
        active_only (bool): Filter for active experts only
        redis_client (Redis, optional): Redis client for caching
    
    Returns:
        SearchResponse: Search results including optional refinements
    """
    logger.info(f"Processing expert search - Query: {query}, User: {user_id}")
    print(f"Starting expert search process for query: {query}")
    
    conn = None
    try:
        # Check Redis cache first
        if redis_client:
            cache_key = f"expert_search:{user_id}:{query}:{active_only}"
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for search: {cache_key}")
                print("Retrieved results from cache")
                return SearchResponse(**json.loads(cached_response))

        # Establish database connection
        conn = get_db_connection()
        session_id = await get_or_create_session(conn, user_id)
        logger.debug(f"Created session: {session_id}")
        
        # Execute search
        start_time = datetime.utcnow()
        search_manager = ExpertSearchIndexManager()
        results = search_manager.search_experts(query, k=5, active_only=active_only)
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Search completed in {response_time:.2f} seconds with {len(results)} results")
        print(f"Found {len(results)} experts in {response_time:.2f} seconds")

        # Format results
        formatted_results = [
            ExpertSearchResult(
                id=str(result['id']),
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
        
        # Record search analytics
        await record_search(conn, session_id, user_id, query, results, response_time)
        
        # Generate refinement suggestions
        refinements = search_manager.get_search_refinements(query, results)
        
        # Prepare response with refinements
        response = SearchResponse(
            total_results=len(formatted_results),
            experts=formatted_results,
            user_id=user_id,
            session_id=session_id,
            refinements=refinements  # Add refinement suggestions
        )

        # Cache the response if Redis client is available
        if redis_client:
            await redis_client.setex(
                cache_key,
                3600,  # Cache for 1 hour
                json.dumps(response.dict())
            )
            logger.debug(f"Cached search results with key: {cache_key}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching experts: {str(e)}", exc_info=True)
        print(f"Expert search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")

# Existing API endpoints remain the same
@router.get("/experts/search/{query}")
async def search_experts(
    query: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received expert search request - Query: {query}, User: {user_id}")
    return await process_expert_search(query, user_id, active_only, redis_client)

async def process_query_prediction(partial_query: str, user_id: str, redis_client: Redis = None) -> PredictionResponse:
    """
    Enhanced query prediction with refinement suggestions
    
    Args:
        partial_query (str): Partial search query
        user_id (str): User identifier
        redis_client (Redis, optional): Redis client for caching
    
    Returns:
        PredictionResponse: Prediction results with optional refinements
    """
    logger.info(f"Processing query prediction - Partial query: {partial_query}, User: {user_id}")
    print(f"Starting prediction process for partial query: {partial_query}")
    
    conn = None
    try:
        # Check Redis cache first
        if redis_client:
            cache_key = f"query_prediction:{user_id}:{partial_query}"
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for prediction: {cache_key}")
                print("Retrieved predictions from cache")
                return PredictionResponse(**json.loads(cached_response))

        # Establish database connection
        conn = get_db_connection()
        session_id = await get_or_create_session(conn, user_id)
        
        # Generate predictions
        predictions = ml_predictor.predict(partial_query, user_id=user_id)
        
        # Ensure we have at least one prediction even with empty results
        if not predictions and partial_query:
            predictions = [partial_query]
        
        # Generate confidence scores (higher for first results)
        confidence_scores = [max(0.1, 1.0 - (i * 0.1)) for i in range(len(predictions))]
        
        logger.debug(f"Generated {len(predictions)} predictions")
        print(f"Generated predictions: {predictions}")
        
        # Generate refinement suggestions - ENSURE THIS IS CALLED
        refinements = generate_predictive_refinements(partial_query, predictions)
        
        # Record prediction
        if conn and predictions:
            await record_prediction(
                conn,
                session_id,
                user_id,
                partial_query,
                predictions,
                confidence_scores
            )
        
        # Prepare response with optional refinements
        response = PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements  # This should now always have content
        )

        # Cache the response if Redis client is available
        if redis_client:
            await redis_client.setex(
                cache_key,
                1800,  # 30 minutes
                json.dumps(response.dict())
            )
            logger.debug(f"Cached predictions with key: {cache_key}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error predicting queries: {str(e)}", exc_info=True)
        print(f"Query prediction failed: {str(e)}")
        
        # Return a minimal valid response even on error
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
        if conn:
            conn.close()
            logger.debug("Database connection closed")
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
        # Import necessary modules
        from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
        
        # Initialize search manager
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
                f"studies on {partial_query}",
                f"{partial_query} analysis"
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

# Existing endpoint remains the same
@router.get("/experts/predict/{partial_query}")
async def predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received query prediction request - Partial query: {partial_query}, User: {user_id}")
    return await process_query_prediction(partial_query, user_id, redis_client)

# API Endpoints
@router.get("/experts/search/{query}")
async def search_experts(
    query: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received expert search request - Query: {query}, User: {user_id}")
    return await process_expert_search(query, user_id, active_only, redis_client)



@router.get("/test/experts/search/{query}")
async def test_search_experts(
    query: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_test_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received test expert search request - Query: {query}")
    return await process_expert_search(query, user_id, active_only, redis_client)

@router.get("/test/experts/predict/{partial_query}")
async def test_predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_test_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received test query prediction request - Partial query: {partial_query}")
    return await process_query_prediction(partial_query, user_id, redis_client)