import datetime
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import uuid
from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from fastapi import APIRouter, HTTPException, Request, Depends, Body
from ai_services_api.services.search.core.models import PredictionResponse, SearchResponse
from ai_services_api.services.search.app.endpoints.process_functions import process_query_prediction
from ai_services_api.services.search.core.expert_search import process_advanced_search, process_expert_search, process_publication_search
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, Body
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import uuid

from ai_services_api.services.search.core.models import PredictionResponse, SearchResponse
from ai_services_api.services.search.app.endpoints.process_functions import process_query_prediction, process_advanced_query_prediction
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

# Only define each endpoint once
@router.get("/advanced_searchh/{query}")
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
    return await process_advanced_query_prediction(partial_query, user_id)

# Add these to your router file (search.py)

@router.get("/advanced_search")
async def advanced_search(
    request: Request,
    user_id: str = Depends(get_user_id),
    query: Optional[str] = None,
    search_type: Optional[str] = None,
    active_only: bool = True,
    k: int = 5,
    name: Optional[str] = None,
    theme: Optional[str] = None,
    designation: Optional[str] = None,
    publication: Optional[str] = None
):
    # Function body

    """
    Advanced search endpoint to search experts and resources based on multiple criteria.
    
    Allows searching by:
    - Experts (name, theme, designation)
    - Publications
    - Flexible combination of search parameters
    """
    logger.info(f"Advanced search request - User: {user_id}")
    
    # Validate search parameters
    search_params = [search_type, name, theme, designation, publication]
    if all(param is None for param in search_params):
        raise HTTPException(
            status_code=400, 
            detail="At least one search parameter must be provided"
        )
    
    # Determine search type priority
    if search_type:
        valid_search_types = ["name", "theme", "designation", "publication"]
        if search_type not in valid_search_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types)}"
            )
    
    try:
        # Flexible routing based on available parameters
        if search_type == "name" or name:
            return await process_expert_name_search(
                query or name, 
                user_id, 
                active_only, 
                k
            )
        
        elif search_type == "theme" or theme:
            return await process_expert_theme_search(
                query or theme, 
                user_id, 
                active_only, 
                k
            )
        
        elif search_type == "designation" or designation:
            return await process_expert_designation_search(
                query or designation, 
                user_id, 
                active_only, 
                k
            )
        
        elif search_type == "publication" or publication:
            # Assuming you have a similar function for publication search
            return await process_publication_search(
                query or publication, 
                user_id, 
                k
            )
        
        # Fallback to a general search if no specific type is matched
        return await process_advanced_search(
            query, 
            user_id, 
            active_only, 
            k,
            name=name,
            theme=theme,
            designation=designation,
            publication=publication
        )
    
    except Exception as e:
        logger.error(f"Error in advanced search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing the advanced search"
        )

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
@router.get("/experts/advanced_predict/{partial_query}")
async def advanced_predict_query(
    partial_query: str,
    request: Request,
    user_id: str = Depends(get_user_id),
    search_type: Optional[str] = None,
    limit: int = 10
):
    """
    Advanced query prediction with context-specific suggestions.
    """
    logger.info(f"Received advanced query prediction request - Partial query: {partial_query}, Search Type: {search_type}")
    
    # Validate search type
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
        logger.error(f"Error in advanced query prediction: {e}", exc_info=True)
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

@router.get("/public/advanced_search")
async def public_advanced_search(
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
    Public advanced search endpoint to search experts and resources based on multiple criteria.
    
    Allows searching by:
    - Experts (name, theme, designation)
    - Publications
    - Flexible combination of search parameters
    
    Does not require a user ID
    """
    logger.info(f"Public advanced search request")
    
    # Validate search parameters
    search_params = [search_type, name, theme, designation, publication]
    if all(param is None for param in search_params):
        raise HTTPException(
            status_code=400, 
            detail="At least one search parameter must be provided"
        )
    
    # Determine search type priority
    if search_type:
        valid_search_types = ["name", "theme", "designation", "publication"]
        if search_type not in valid_search_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types)}"
            )
    
    try:
        # Use a generic user ID for public searches
        public_user_id = "public_user"
        
        # Flexible routing based on available parameters
        if search_type == "name" or name:
            return await process_expert_name_search(
                query or name, 
                public_user_id, 
                active_only, 
                k
            )
        
        elif search_type == "theme" or theme:
            return await process_expert_theme_search(
                query or theme, 
                public_user_id, 
                active_only, 
                k
            )
        
        elif search_type == "designation" or designation:
            return await process_expert_designation_search(
                query or designation, 
                public_user_id, 
                active_only, 
                k
            )
        
        elif search_type == "publication" or publication:
            # Assuming you have a similar function for publication search
            return await process_publication_search(
                query or publication, 
                public_user_id, 
                k
            )
        
        # Fallback to a general search if no specific type is matched
        return await process_advanced_search(
            query, 
            public_user_id, 
            active_only, 
            k,
            name=name,
            theme=theme,
            designation=designation,
            publication=publication
        )
    
    except Exception as e:
        logger.error(f"Error in public advanced search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing the advanced search"
        )

@router.get("/public/experts/advanced_predict/{partial_query}")
async def public_advanced_predict_query(
    partial_query: str,
    search_type: Optional[str] = None,
    limit: int = 10
):
    """
    Public advanced query prediction with context-specific suggestions.
    Does not require a user ID.
    """
    logger.info(f"Received public advanced query prediction request - Partial query: {partial_query}, Search Type: {search_type}")
    
    # Validate search type
    # Validate search type
    valid_search_types = ["name", "theme", "designation", "publication", None]
    if search_type and search_type not in valid_search_types[:-1]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types[:-1])}"
    )
    
    try:
        # Use a generic user ID for public predictions
        public_user_id = "public_user"
        
        # Process advanced query prediction
        return await process_advanced_query_prediction(
            partial_query, 
            public_user_id, 
            search_type, 
            limit
        )
    
    except Exception as e:
        logger.error(f"Error in public advanced query prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while generating predictions: {str(e)}"
        )

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