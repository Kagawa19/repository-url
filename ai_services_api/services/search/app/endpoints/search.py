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
from ai_services_api.services.search.ml.ml_predictor import MLPredictor, RedisConnectionManager
from ai_services_api.services.message.core.database import get_db_connection

# Import the improved process functions
from ai_services_api.services.search.app.endpoints.process_functions import (
    process_expert_search,
    process_query_prediction,
    get_redis
)

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
    refinements: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    predictions: List[str]
    confidence_scores: List[float]
    user_id: str
    refinements: Optional[Dict[str, Any]] = None

# Helper functions moved to separate module
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

# Only define each endpoint once
@router.get("/experts/search/{query}")
async def search_experts(
    query: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Search for experts matching the query."""
    logger.info(f"Received expert search request - Query: {query}, User: {user_id}")
    return await process_expert_search(query, user_id, active_only, redis_client)

@router.get("/experts/predict/{partial_query}")
async def predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Predict query completion based on partial input."""
    logger.info(f"Received query prediction request - Partial query: {partial_query}, User: {user_id}")
    return await process_query_prediction(partial_query, user_id, redis_client)

@router.get("/test/experts/search/{query}")
async def test_search_experts(
    query: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_test_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Test endpoint for expert search."""
    logger.info(f"Received test expert search request - Query: {query}")
    return await process_expert_search(query, user_id, active_only, redis_client)

@router.get("/test/experts/predict/{partial_query}")
async def test_predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_test_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Test endpoint for query prediction."""
    logger.info(f"Received test query prediction request - Partial query: {partial_query}")
    return await process_query_prediction(partial_query, user_id, redis_client)

# Health check endpoint
@router.get("/health")
async def health_check():
    """API health check endpoint."""
    try:
        # Check Redis connectivity
        redis_manager = RedisConnectionManager.get_instance()
        redis_client = redis_manager.get_connection(0)
        redis_status = "healthy" if redis_client.ping() else "unhealthy"
        
        # Check if prediction is working
        prediction_status = "healthy"
        try:
            predictions = ml_predictor.predict("test", "health_check")
            if predictions is None:
                prediction_status = "unhealthy"
        except:
            prediction_status = "unhealthy"
            
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": redis_status,
                "prediction": prediction_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Cache management endpoint
@router.post("/cache/invalidate")
async def invalidate_cache(
    request: Request,
    user_id: Optional[str] = None,
    pattern: Optional[str] = None,
    user_id_from_header: str = Depends(get_user_id)
):
    """Invalidate specific cache entries."""
    try:
        # Use user_id from header if not specified in request
        target_user_id = user_id or user_id_from_header
        
        redis_manager = RedisConnectionManager.get_instance()
        cache_redis = redis_manager.get_connection(3)  # Cache DB
        
        keys_to_delete = []
        
        # Build pattern based on inputs
        if pattern and target_user_id:
            search_pattern = f"*{target_user_id}*{pattern}*"
        elif pattern:
            search_pattern = f"*{pattern}*"
        elif target_user_id:
            search_pattern = f"*{target_user_id}*"
        else:
            # Don't allow deleting all keys - require some pattern
            return {"status": "error", "message": "Pattern or user_id required"}
            
        # Find keys matching pattern
        keys_to_delete = cache_redis.keys(search_pattern)
        
        # Delete matched keys
        if keys_to_delete:
            cache_redis.delete(*keys_to_delete)
            
        return {
            "status": "success",
            "keys_deleted": len(keys_to_delete),
            "pattern": search_pattern
        }
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }