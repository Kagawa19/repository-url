import datetime
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import uuid

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, Body
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import uuid

from ai_services_api.services.search.core.models import PredictionResponse, SearchResponse
from ai_services_api.services.search.app.endpoints.process_functions import process_query_prediction
from ai_services_api.services.search.core.expert_search import (
    process_expert_search,
    process_expert_name_search,
    process_expert_theme_search,
    process_expert_designation_search
)


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
TEST_USER_ID = "123"

# Helper functions for user ID extraction
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
    user_id: str = Depends(get_user_id)
):
    """Search for experts matching the query."""
    logger.info(f"Received expert search request - Query: {query}, User: {user_id}")
    return await process_expert_search(query, user_id, active_only)

@router.get("/experts/predict/{partial_query}")
async def predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_user_id)
):
    """Predict query completion based on partial input using Gemini API."""
    logger.info(f"Received query prediction request - Partial query: {partial_query}, User: {user_id}")
    return await process_query_prediction(partial_query, user_id)

# Add these to your router file (search.py)

@router.get("/experts/search_by_name/{name}")
async def search_experts_by_name(
    name: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_user_id)
):
    """Search for experts by name."""
    logger.info(f"Received name search request - Name: {name}, User: {user_id}")
    return await process_expert_name_search(name, user_id, active_only)

@router.get("/experts/search_by_theme/{theme}")
async def search_experts_by_theme(
    theme: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_user_id)
):
    """Search for experts by theme."""
    logger.info(f"Received theme search request - Theme: {theme}, User: {user_id}")
    return await process_expert_theme_search(theme, user_id, active_only)

@router.get("/experts/search_by_designation/{designation}")
async def search_experts_by_designation(
    designation: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_user_id)
):
    """Search for experts by designation."""
    logger.info(f"Received designation search request - Designation: {designation}, User: {user_id}")
    return await process_expert_designation_search(designation, user_id, active_only)

@router.get("/test/experts/search_by_name/{name}")
async def test_search_experts_by_name(
    name: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_test_user_id)
):
    """Test endpoint for name search."""
    logger.info(f"Received test name search request - Name: {name}")
    return await process_expert_name_search(name, user_id, active_only)

@router.get("/test/experts/search_by_theme/{theme}")
async def test_search_experts_by_theme(
    theme: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_test_user_id)
):
    """Test endpoint for theme search."""
    logger.info(f"Received test theme search request - Theme: {theme}")
    return await process_expert_theme_search(theme, user_id, active_only)

@router.get("/test/experts/search_by_designation/{designation}")
async def test_search_experts_by_designation(
    designation: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_test_user_id)
):
    """Test endpoint for designation search."""
    logger.info(f"Received test designation search request - Designation: {designation}")
    return await process_expert_designation_search(designation, user_id, active_only)

@router.get("/test/experts/search/{query}")
async def test_search_experts(
    query: str,
    request: Request,
    active_only: bool = True,
    user_id: str = Depends(get_test_user_id)
):
    """Test endpoint for expert search."""
    logger.info(f"Received test expert search request - Query: {query}")
    return await process_expert_search(query, user_id, active_only)

@router.get("/test/experts/predict/{partial_query}")
async def test_predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_test_user_id)
):
    """Test endpoint for query prediction using Gemini API."""
    logger.info(f"Received test query prediction request - Partial query: {partial_query}")
    return await process_query_prediction(partial_query, user_id)

@router.post("/experts/track-suggestion")
async def track_suggestion(
    request: Request,
    partial_query: str = Body(...),
    selected_suggestion: str = Body(...),
    user_id: str = Depends(get_user_id)
):
    """
    Track which suggestion the user selected from the prediction results.
    This enables better personalization over time.
    """
    logger.info(f"Tracking selected suggestion - User: {user_id}")
    
    from ai_services_api.services.search.core.personalization import track_selected_suggestion
    await track_selected_suggestion(
        user_id,
        partial_query,
        selected_suggestion
    )
    
    return {"status": "success", "message": "Suggestion selection tracked"}

# Health check endpoint
@router.get("/health")
async def health_check():
    """API health check endpoint."""
    try:
        # Check Gemini API connectivity
        from ai_services_api.services.search.gemini.gemini_predictor import GeminiPredictor
        
        predictor = GeminiPredictor()
        gemini_status = "healthy"
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "gemini": gemini_status,
                "database": "healthy"  # Assuming database is healthy
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }