from fastapi import APIRouter, HTTPException, Request, Depends
from typing import List, Dict, Any
from datetime import datetime
import logging
import json
import psycopg2
from redis.asyncio import Redis
from ai_services_api.services.recommendation.services.expert_matching import ExpertMatchingService
from ai_services_api.services.message.core.database import get_db_connection

router = APIRouter()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_router.log')
    ]
)
logger = logging.getLogger(__name__)

TEST_USER_ID = "test_user_123"

async def get_redis():
    """Establish Redis connection with detailed logging"""
    try:
        redis_client = Redis(host='redis', port=6379, db=3, decode_responses=True)
        logger.info(f"Redis connection established successfully to host: redis, port: 6379, db: 3")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to establish Redis connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Redis connection failed")
    
async def get_user_id(request: Request) -> str:
    logger.debug("Extracting user ID from request headers")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Missing required X-User-ID header in request")
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    logger.info(f"User ID extracted successfully: {user_id}")
    return user_id


async def get_test_user_id(request: Request) -> str:
    """Return test user ID with logging"""
    logger.info(f"Using test user ID: {TEST_USER_ID}")
    return TEST_USER_ID

async def process_recommendations(user_id: str, redis_client: Redis) -> Dict:
    """Process expert recommendations with comprehensive logging and error handling"""
    try:
        logger.info(f"Starting recommendation process for user: {user_id}")
        
        # Check cache first
        cache_key = f"user_recommendations:{user_id}"
        cached_response = await redis_client.get(cache_key)
        
        if cached_response:
            logger.info(f"Cache hit for user recommendations: {cache_key}")
            try:
                return json.loads(cached_response)
            except json.JSONDecodeError as json_err:
                logger.error(f"Error decoding cached response: {json_err}")
        
        start_time = datetime.utcnow()
        logger.debug(f"Recommendation generation started at: {start_time}")
        
        expert_matching = ExpertMatchingService()
        try:
            logger.info(f"Generating recommendations for user: {user_id}")
            
            recommendations = await expert_matching.get_recommendations_for_user(user_id)
            
            response_data = {
                "user_id": user_id,
                "recommendations": recommendations or [],
                "total_matches": len(recommendations) if recommendations else 0,
                "timestamp": datetime.utcnow().isoformat(),
                "response_time": (datetime.utcnow() - start_time).total_seconds()
            }
            
            logger.info(f"Recommendation generation completed for user {user_id}. "
                        f"Total matches: {response_data['total_matches']}, "
                        f"Response time: {response_data['response_time']} seconds")
            
            try:
                await redis_client.setex(
                    cache_key,
                    1800,  # 30 minutes expiry
                    json.dumps(response_data)
                )
                logger.info(f"Recommendations cached successfully for user {user_id}")
            except Exception as cache_err:
                logger.error(f"Failed to cache recommendations: {cache_err}")
            
            return response_data
        
        except Exception as matching_err:
            logger.error(f"Error in expert matching for user {user_id}: {str(matching_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Expert matching error: {str(matching_err)}")
        
        finally:
            expert_matching.close()
            logger.debug(f"Expert matching service closed for user {user_id}")
    
    except Exception as e:
        logger.error(f"Unhandled error in recommendation process for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Comprehensive recommendation error")

@router.get("/recommend/{user_id}")
async def get_expert_recommendations(
    user_id: str,
    request: Request,
    redis_client: Redis = Depends(get_redis)
):
    """Get expert recommendations for the user based on their behavior"""
    logger.info(f"Recommendation request received for user: {user_id}")
    return await process_recommendations(user_id, redis_client)

@router.get("/test/recommend")
async def test_get_expert_recommendations(
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Test endpoint for getting user recommendations"""
    logger.info(f"Test recommendation request for user: {user_id}")
    return await process_recommendations(user_id, redis_client)

# New cache clearing endpoints
@router.delete("/cache/recommendations/{user_id}")
async def clear_user_recommendations_cache(
    user_id: str,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear recommendations cache for a specific user"""
    try:
        cache_key = f"user_recommendations:{user_id}"
        deleted = await redis_client.delete(cache_key)
        
        logger.info(f"Cache clearing request for user: {user_id}, result: {deleted > 0}")
        
        if deleted:
            return {"status": "success", "message": f"Cache cleared for user {user_id}"}
        else:
            return {"status": "success", "message": f"No cache found for user {user_id}"}
    
    except Exception as e:
        logger.error(f"Failed to clear cache for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

@router.delete("/cache/recommendations")
async def clear_all_recommendations_cache(
    request: Request,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear recommendations cache for all users"""
    try:
        # Find all keys matching the pattern
        pattern = "user_recommendations:*"
        total_deleted = 0
        
        # Use scan_iter to get all matching keys and delete them
        async for key in redis_client.scan_iter(match=pattern):
            await redis_client.delete(key)
            total_deleted += 1
        
        logger.info(f"All recommendation caches cleared. Total deleted: {total_deleted}")
        
        return {
            "status": "success", 
            "message": f"Cleared all recommendation caches", 
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logger.error(f"Failed to clear all recommendation caches: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")