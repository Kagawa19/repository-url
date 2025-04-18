import datetime
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
import logging
import redis
from redis.asyncio import Redis
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
from ai_services_api.services.search.app.endpoints.process_functions import process_advanced_query_prediction, get_search_categorizer
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

# Initialize global categorizer instance (will be lazily initialized when needed)
_categorizer = None

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
        
        # Try to categorize the query if not already specified in search_type
        if effective_query and not search_type:
            try:
                # Get categorizer
                categorizer = await get_search_categorizer()
                
                if categorizer:
                    # Categorize the query
                    category_info = await categorizer.categorize_query(effective_query, user_id)
                    detected_category = category_info.get("category")
                    
                    # Use detected category as search type if it's one of our valid types
                    if detected_category in ["name", "theme", "designation", "publication"]:
                        search_type = detected_category
                        logger.info(f"Query '{effective_query}' categorized as '{search_type}'")
            except Exception as category_error:
                logger.warning(f"Error categorizing query: {category_error}")
        
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
        
        # Add category information to response if available
        if effective_query:
            try:
                categorizer = await get_search_categorizer()
                if categorizer:
                    category_info = await categorizer.categorize_query(effective_query, user_id)
                    
                    # Add category information to response metadata
                    if hasattr(search_response, "metadata"):
                        search_response.metadata = search_response.metadata or {}
                        search_response.metadata["category"] = category_info.get("category", "general")
                        search_response.metadata["category_confidence"] = category_info.get("confidence", 0.0)
            except Exception as category_error:
                logger.warning(f"Error adding category to response: {category_error}")
        
        # NEW: Record the search in user history
        if effective_query:  # Only record non-empty queries
            try:
                redis_client = await get_redis()
                await track_user_search(
                    redis_client, 
                    user_id, 
                    effective_query, 
                    search_type or "general"
                )
            except Exception as track_error:
                logger.warning(f"Error tracking search history: {track_error}")
        
        # Log search results
        logger.info(f"Search completed with {search_response.total_results} results")
        
        return search_response
    
    except Exception as e:
        logger.error(f"Error in advanced search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing the advanced search"
        )

async def track_user_search(redis_client: Redis, user_id: str, query: str, category: str = "general") -> None:
    """
    Track a user's search query in their history.
    
    Args:
        redis_client: Redis client instance
        user_id: User ID
        query: The search query
        category: Search category
    """
    try:
        # Key for this user's search history
        key = f"user_search_history:{user_id}"
        
        # Current timestamp for sorting
        timestamp = datetime.now().timestamp()
        
        # Create a search record with timestamp and category
        search_record = {
            "query": query,
            "timestamp": timestamp,
            "category": category
        }
        
        # Add to a list in Redis, limited to most recent 20 searches
        # Using Redis sorted set for automatic timestamp-based ordering
        await redis_client.zadd(
            key,
            {
                # Store as JSON string with timestamp as score for sorting
                # Timestamp in score ensures chronological order
                f"{timestamp}:{query}": timestamp
            }
        )
        
        # Store search details in a hash
        detail_key = f"user_search_details:{user_id}:{timestamp}:{query}"
        await redis_client.hset(
            detail_key,
            mapping={
                "query": query,
                "timestamp": str(timestamp),
                "category": category
            }
        )
        
        # Set expiration for the detail record (7 days)
        await redis_client.expire(detail_key, 60 * 60 * 24 * 7)
        
        # Trim to keep only the latest 20 searches
        await redis_client.zremrangebyrank(key, 0, -21)
        
        # Set expiration for the search history (30 days)
        await redis_client.expire(key, 60 * 60 * 24 * 30)
        
        logger.debug(f"Tracked search for user {user_id}: {query}")
    except Exception as e:
        logger.error(f"Error tracking search history: {e}")
        # Don't raise exception - this should not affect the main search functionality

@router.get("/experts/recent-searches")
async def get_recent_searches(
    request: Request,
    user_id: str = Depends(get_user_id),
    limit: int = 5
):
    """
    Get the user's most recent searches.
    
    Args:
        user_id: User ID to get searches for
        limit: Maximum number of searches to return (default: 5)
    
    Returns:
        List of recent searches with timestamps and categories
    """
    logger.info(f"Fetching recent searches for user: {user_id}")
    
    try:
        redis_client = await get_redis()
        
        # Get recent searches from Redis sorted set
        key = f"user_search_history:{user_id}"
        
        # Get the latest entries (highest scores) with their scores
        # ZREVRANGE returns newest first (highest score)
        recent_searches = await redis_client.zrevrange(
            key, 
            0, 
            limit - 1,  # Zero-indexed, so -1 for limit
            withscores=True
        )
        
        # Format results
        result = []
        for search_key, score in recent_searches:
            # Parse the search key (timestamp:query)
            parts = search_key.split(':', 1)
            if len(parts) != 2:
                continue
                
            timestamp, query = parts
            
            # Get detailed information
            detail_key = f"user_search_details:{user_id}:{timestamp}:{query}"
            details = await redis_client.hgetall(detail_key)
            
            if not details:
                # Fall back to just the basic info if details not found
                details = {
                    "query": query,
                    "timestamp": str(score),
                    "category": "general"
                }
            
            # Add to results
            result.append({
                "query": query,
                "timestamp": float(details.get("timestamp", score)),
                "category": details.get("category", "general"),
                "formatted_time": datetime.fromtimestamp(float(details.get("timestamp", score))).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "recent_searches": result,
            "total": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error fetching recent searches: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching recent searches: {str(e)}"
        )
    
@router.delete("/experts/recent-searches")
async def clear_recent_searches(
    request: Request,
    user_id: str = Depends(get_user_id)
):
    """
    Clear a user's recent search history.
    
    Args:
        user_id: User ID to clear searches for
    """
    logger.info(f"Clearing recent searches for user: {user_id}")
    
    try:
        redis_client = await get_redis()
        
        # Get recent searches from Redis sorted set to find detail keys
        key = f"user_search_history:{user_id}"
        recent_searches = await redis_client.zrange(key, 0, -1, withscores=True)
        
        # Delete detail records
        pipeline = redis_client.pipeline()
        for search_key, _ in recent_searches:
            # Parse the search key (timestamp:query)
            parts = search_key.split(':', 1)
            if len(parts) != 2:
                continue
                
            timestamp, query = parts
            
            # Delete the detail key
            detail_key = f"user_search_details:{user_id}:{timestamp}:{query}"
            pipeline.delete(detail_key)
        
        # Delete the history key itself
        pipeline.delete(key)
        
        # Execute all delete operations
        await pipeline.execute()
        
        return {
            "status": "success",
            "message": f"Search history cleared for user {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error clearing recent searches: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while clearing recent searches: {str(e)}"
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
    If partial_query is empty or just whitespace, returns recent searches instead.
    """
    # Clean the partial query
    cleaned_query = partial_query.strip()
    
    logger.info(f"Received query prediction request - Partial query: '{cleaned_query}', Search Type: {search_type}")
    
    # If query is empty or just whitespace, return recent searches
    if not cleaned_query:
        try:
            # Get recent searches
            redis_client = await get_redis()
            recent_searches = await get_user_recent_searches(redis_client, user_id, limit)
            
            # Format as prediction response
            predictions = []
            for search in recent_searches:
                predictions.append({
                    "text": search["query"],
                    "category": search["category"],
                    "source": "history",
                    "score": 1.0  # Give high score to prioritize history items
                })
            
            return PredictionResponse(
                query=partial_query,
                predictions=predictions,
                total_results=len(predictions),
                metadata={"source": "history"}
            )
        except Exception as e:
            logger.warning(f"Error getting recent searches: {e}")
            # If there's an error, continue with normal prediction
    
    # Validate search type
    valid_search_types = ["name", "theme", "designation", "publication", None]
    if search_type and search_type not in valid_search_types[:-1]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search type. Must be one of: {', '.join(valid_search_types[:-1])}"
        )
    
    try:
        # Process advanced query prediction (original logic)
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
    
async def get_user_recent_searches(redis_client: Redis, user_id: str, limit: int = 5) -> list:
    """
    Get a user's recent searches from Redis.
    
    Args:
        redis_client: Redis client instance
        user_id: User ID
        limit: Maximum number of searches to return
        
    Returns:
        List of recent search objects
    """
    # Key for this user's search history
    key = f"user_search_history:{user_id}"
    
    # Get the latest entries (highest scores) with their scores
    recent_searches = await redis_client.zrevrange(
        key, 
        0, 
        limit - 1,
        withscores=True
    )
    
    # Format results
    result = []
    for search_key, score in recent_searches:
        # Parse the search key (timestamp:query)
        parts = search_key.split(':', 1)
        if len(parts) != 2:
            continue
            
        timestamp, query = parts
        
        # Get detailed information
        detail_key = f"user_search_details:{user_id}:{timestamp}:{query}"
        details = await redis_client.hgetall(detail_key)
        
        if not details:
            # Fall back to just the basic info if details not found
            details = {
                "query": query,
                "timestamp": str(score),
                "category": "general"
            }
        
        # Add to results
        result.append({
            "query": query,
            "timestamp": float(details.get("timestamp", score)),
            "category": details.get("category", "general")
        })
    
    return result

@router.post("/experts/track-suggestion")
async def track_suggestion(
    request: Request,
    partial_query: str = Body(...),
    selected_suggestion: str = Body(...),
    user_id: str = Depends(get_user_id),
    category: Optional[str] = Body(None)
):
    """
    Track which suggestion the user selected from the prediction results.
    This enables better personalization over time.
    """
    logger.info(f"Tracking selected suggestion - User: {user_id}")
    
    # Try to determine category if not provided
    if not category:
        try:
            categorizer = await get_search_categorizer()
            if categorizer:
                category_info = await categorizer.categorize_query(selected_suggestion, user_id)
                category = category_info.get("category", "general")
                logger.info(f"Suggestion '{selected_suggestion}' categorized as '{category}'")
        except Exception as e:
            logger.warning(f"Error categorizing selected suggestion: {e}")
            category = "general"
    
    # Track the suggestion with enhanced category information
    try:
        # Import with enhanced category support
        from ai_services_api.services.search.core.personalization import track_selected_suggestion
        await track_selected_suggestion(
            user_id,
            partial_query,
            selected_suggestion,
            category
        )
        
        # Also track in predictor for immediate use
        try:
            from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
            predictor = GoogleAutocompletePredictor()
            await predictor.track_selection(
                partial_query,
                selected_suggestion,
                user_id,
                category
            )
        except Exception as pred_error:
            logger.warning(f"Error tracking in predictor: {pred_error}")
        
        return {
            "status": "success", 
            "message": "Suggestion selection tracked",
            "category": category
        }
    except Exception as track_error:
        logger.error(f"Error tracking suggestion: {track_error}")
        return {
            "status": "error",
            "message": "Failed to track suggestion",
            "error": str(track_error)
        }

# Health check endpoint with categorizer status
@router.get("/health")
async def health_check():
    """API health check endpoint."""
    try:
        components = {
            "database": "healthy",  # Assuming database is healthy
            "categorizer": "not_configured"
        }
        
        # Check Gemini API connectivity
        from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
        predictor = GoogleAutocompletePredictor()
        components["gemini"] = "healthy"
        
        # Check categorizer status
        try:
            categorizer = await get_search_categorizer()
            if categorizer:
                if categorizer.gemini_model:
                    components["categorizer"] = "healthy"
                else:
                    components["categorizer"] = "no_gemini_api"
        except Exception as cat_error:
            components["categorizer"] = f"error: {str(cat_error)}"
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": components
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
            f"personalization:{user_id}:*",     # User personalization data
            f"category:{user_id}:*"             # Category cache for user queries
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
            "personalization:*",     # All personalization data
            "category:*"             # All category cache
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

# New endpoint to get categories for multiple queries in batch
@router.post("/experts/categorize-queries")
async def categorize_queries(
    request: Request,
    queries: List[str] = Body(...),
    user_id: str = Depends(get_user_id)
):
    """
    Categorize multiple queries in batch.
    
    This is useful for analyzing search patterns or pre-categorizing a set of queries.
    """
    try:
        categorizer = await get_search_categorizer()
        if not categorizer:
            raise HTTPException(
                status_code=503,
                detail="Categorizer service is not available"
            )
        
        results = {}
        for query in queries:
            if not query or not query.strip():
                results[query] = {"category": "general", "confidence": 0.0}
                continue
                
            category_info = await categorizer.categorize_query(query, user_id)
            results[query] = category_info
        
        return {
            "status": "success",
            "results": results,
            "total": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error categorizing queries: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while categorizing queries: {str(e)}"
        )

# New endpoint to get category statistics
@router.get("/experts/category-stats")
async def get_category_stats(
    request: Request,
    user_id: Optional[str] = None,
    days: int = 30
):
    """
    Get statistics about query categories.
    
    Args:
        user_id: Optional user ID to filter stats for a specific user
        days: Number of days to include in stats (default: 30)
    """
    try:
        from ai_services_api.services.message.core.database import get_db_connection
        
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Check if search_categories table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'search_categories'
                    );
                """)
                
                if not cur.fetchone()[0]:
                    return {
                        "status": "error",
                        "message": "Category statistics not available",
                        "categories": {}
                    }
                
                # Build query based on filters
                query = """
                    SELECT 
                        category,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM search_categories
                    WHERE created_at >= NOW() - INTERVAL %s DAY
                """
                params = [days]
                
                # Add user filter if provided
                if user_id:
                    query += " AND user_id = %s"
                    params.append(user_id)
                
                # Group by category
                query += " GROUP BY category ORDER BY count DESC"
                
                # Execute query
                cur.execute(query, params)
                rows = cur.fetchall()
                
                # Process results
                categories = {}
                total = 0
                for category, count, avg_confidence in rows:
                    categories[category] = {
                        "count": count,
                        "avg_confidence": float(avg_confidence)
                    }
                    total += count
                
                # Add percentages
                for category in categories:
                    categories[category]["percentage"] = categories[category]["count"] / total if total > 0 else 0
                
                return {
                    "status": "success",
                    "total_queries": total,
                    "days": days,
                    "user_id": user_id,
                    "categories": categories
                }
                
        except Exception as db_error:
            logger.error(f"Database error in category stats: {db_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(db_error)}"
            )
        finally:
            if conn:
                conn.close()
    
    except Exception as e:
        logger.error(f"Error getting category stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while getting category stats: {str(e)}"
        )