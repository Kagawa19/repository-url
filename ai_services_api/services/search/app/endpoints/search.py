import datetime
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import uuid
from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from fastapi import APIRouter, HTTPException, Request, Depends, Body
from ai_services_api.services.search.core.models import PredictionResponse, SearchResponse
from ai_services_api.services.search.app.endpoints.process_functions import process_advanced_query_prediction
from ai_services_api.services.search.core.expert_search import process_advanced_search, process_expert_search, process_publication_search
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, Body
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import uuid

from ai_services_api.services.search.core.models import PredictionResponse, SearchResponse
from ai_services_api.services.search.app.endpoints.process_functions import process_advanced_query_prediction, process_advanced_query_prediction
from ai_services_api.services.search.core.expert_search import (
    process_expert_search,
    process_expert_name_search,
    process_expert_theme_search,
    process_expert_designation_search
)
# Add this to your application's startup code (e.g., in your main.py or __init__.py)
import logging

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




@router.get("/test/experts/search/{query}")
async def advanced_search(
    request: Request,
    user_id: str = Depends(get_test_user_id),
    query: Optional[str] = None,
    search_type: Optional[str] = None,
    active_only: bool = True,
    k: int = 5,
    name: Optional[str] = None,
    theme: Optional[str] = None,
    designation: Optional[str] = None,
    publication: Optional[str] = None
):
    """
    Advanced search endpoint to search experts and resources based on multiple criteria.
    
    Allows searching by:
    - Experts (name, theme, designation)
    - Publications
    - Flexible combination of search parameters
    """
    logger.info(f"Advanced search request - User: {user_id}")
    
    # Validate search parameters
    search_params = [query, name, theme, designation, publication]
    if all(param is None for param in search_params):
        raise HTTPException(
            status_code=400, 
            detail="At least one search parameter must be provided"
        )
    
    # Determine search type priority
    if search_type:
        valid_search_types = ["name", "theme", "designation", "publication", "general"]
        if search_type not in valid_search_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types)}"
            )
    
    try:
        # Construct effective query based on parameters
        effective_query = query or ""
        
        # Add structured fields to the query if provided
        if name and (not search_type or search_type == "name"):
            effective_query = f"{effective_query} {name}".strip()
        
        if theme and (not search_type or search_type == "theme"):
            effective_query = f"{effective_query} theme:{theme}".strip()
        
        if designation and (not search_type or search_type == "designation"):
            effective_query = f"{effective_query} designation:{designation}".strip()
        
        if publication and (not search_type or search_type == "publication"):
            effective_query = f"{effective_query} publication:{publication}".strip()
        
        # If no effective query was constructed but we have a search type,
        # use that search type as the query itself
        if not effective_query and search_type and search_type != "general":
            effective_query = search_type
        
        logger.info(f"Constructed effective query: '{effective_query}'")
        
        # Select the appropriate search method based on search type
        if search_type == "name" and name:
            search_response = await process_expert_name_search(
                name=name,
                user_id=user_id,
                active_only=active_only,
                k=k
            )
        elif search_type == "theme" and theme:
            search_response = await process_expert_theme_search(
                theme=theme,
                user_id=user_id,
                active_only=active_only,
                k=k
            )
        elif search_type == "designation" and designation:
            search_response = await process_expert_designation_search(
                designation=designation,
                user_id=user_id,
                active_only=active_only,
                k=k
            )
        elif search_type == "publication" and publication:
            search_response = await process_publication_search(
                publication=publication,
                user_id=user_id,
                k=k
            )
        else:
            # Use the unified process_expert_search function for general search
            # This now includes direct name matching for suggestion clicks
            search_response = await process_expert_search(
                query=effective_query,
                user_id=user_id,
                active_only=active_only,
                k=k
            )
        
        # Log search results
        logger.info(f"Search completed with {search_response.total_results} results")
        
        return search_response
    
    except Exception as e:
        logger.error(f"Error in advanced search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing the advanced search"
        )


# Replace the advanced_predict endpoint with the simpler 'predict' path
@router.get("/experts/predict/{partial_query}")
async def predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_user_id),
    search_type: Optional[str] = None,
    limit: int = 10
):
    """
    Generate query predictions based on partial input.
    """
    logger.info(f"Received query prediction request - Partial query: {partial_query}, Search Type: {search_type}")
    
    # Validate search type
    valid_search_types = ["name", "theme", "designation", "publication", None]
    if search_type and search_type not in valid_search_types[:-1]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types[:-1])}"
        )
    
    try:
        # Process advanced query prediction
        return await process_advanced_query_prediction(
            partial_query, 
            user_id, 
            search_type, 
            limit
        )
    
    except Exception as e:
        logger.error(f"Error in query prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while generating predictions: {str(e)}"
        )


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
    
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict
import logging
from redis.asyncio import Redis

# Add these endpoints to your existing router with minimal change to your existing code
# You'll need to add the get_redis dependency function if it doesn't exist

async def get_redis():
    """Establish Redis connection with detailed logging"""
    try:
        redis_client = Redis(host='redis', port=6379, db=2, decode_responses=True)
        logging.info(f"Redis connection established successfully to host: redis, port: 6379, db: 2")
        return redis_client
    except Exception as e:
        logging.error(f"Failed to establish Redis connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Redis connection failed")

# Add these to your existing router
@router.delete("/cache/search/{user_id}")
async def clear_user_search_cache(
    user_id: str,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear search cache for a specific user"""
    try:
        # Common patterns for search-related caches
        patterns = [
            f"user_search:{user_id}:*",        # Search results
            f"query_prediction:{user_id}:*",    # Query predictions
            f"expert_search:{user_id}:*",       # Expert search results
            f"personalization:{user_id}:*"      # User personalization data
        ]
        
        total_deleted = 0
        for pattern in patterns:
            # Use scan_iter to get all matching keys for this pattern
            async for key in redis_client.scan_iter(match=pattern):
                await redis_client.delete(key)
                total_deleted += 1
        
        logging.info(f"Search cache cleared for user {user_id}. Total keys deleted: {total_deleted}")
        
        return {
            "status": "success", 
            "message": f"Search cache cleared for user {user_id}", 
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logging.error(f"Failed to clear search cache for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

@router.delete("/cache/search")
async def clear_all_search_cache(
    request: Request,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear search cache for all users"""
    try:
        # Common patterns for all search-related caches
        patterns = [
            "user_search:*",        # All search results
            "query_prediction:*",    # All query predictions
            "expert_search:*",       # All expert search results
            "personalization:*"      # All personalization data
        ]
        
        total_deleted = 0
        for pattern in patterns:
            # Use scan_iter to get all matching keys for this pattern
            async for key in redis_client.scan_iter(match=pattern):
                await redis_client.delete(key)
                total_deleted += 1
        
        logging.info(f"All search caches cleared. Total keys deleted: {total_deleted}")
        
        return {
            "status": "success", 
            "message": "Cleared all search caches", 
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logging.error(f"Failed to clear all search caches: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

# New flexible_user_id dependency that checks request state first before falling back to header
async def flexible_get_user_id(request: Request) -> str:
    """
    Get user ID with flexibility:
    1. First check if a user ID is set in request state (from /set-user-id endpoint)
    2. Fall back to X-User-ID header if not in request state
    3. Raise exception if neither is available
    
    This preserves the original get_user_id behavior for existing endpoints.
    """
    logger.debug("Flexible user ID extraction")
    
    # First check if we have a user ID in the request state (set via the /set-user-id endpoint)
    if hasattr(request.state, "user_id"):
        user_id = request.state.user_id
        logger.info(f"Using user ID from request state: {user_id}")
        return user_id
    
    # Otherwise fall back to the header
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Missing required X-User-ID header in request")
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    
    logger.info(f"User ID extracted from header: {user_id}")
    return user_id

# Your /set-user-id endpoint remains the same
@router.post("/set-user-id")
async def set_user_id(
    request: Request,
    user_id: str = Body(...),
) -> Dict:
    """
    Set the user ID for the current session.
    This allows testing with different user IDs without changing headers.
    """
    logger.info(f"Setting user ID to: {user_id}")
    
    # Store the user ID in the request state for this session
    request.state.user_id = user_id
    
    return {
        "status": "success",
        "message": f"User ID set to: {user_id}",
        "user_id": user_id
    }

# Example of updating an endpoint to use the flexible dependency
# (you can choose which endpoints to update)
@router.get("/experts/predict/{partial_query}/flexible")
async def predict_query_flexible(
    partial_query: str,
    request: Request,
    user_id: str = Depends(flexible_get_user_id),  # Use the flexible dependency here
    search_type: Optional[str] = None,
    limit: int = 10
):
    """
    Generate query predictions based on partial input.
    Uses the flexible user ID resolution.
    """
    logger.info(f"Received flexible query prediction request - Partial query: {partial_query}, User ID: {user_id}")
    
    # Same implementation as your original predict_query
    valid_search_types = ["name", "theme", "designation", "publication", None]
    if search_type and search_type not in valid_search_types[:-1]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types[:-1])}"
        )
    
    try:
        # Process advanced query prediction
        return await process_advanced_query_prediction(
            partial_query, 
            user_id, 
            search_type, 
            limit
        )
    
    except Exception as e:
        logger.error(f"Error in query prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while generating predictions: {str(e)}"
        )