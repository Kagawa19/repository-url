from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional, Dict
from pydantic import BaseModel
import json
from datetime import datetime
import logging
import time
import asyncio
from slowapi import Limiter
from slowapi.util import get_remote_address
from redis.asyncio import Redis
from ai_services_api.services.chatbot.utils.llm_manager import GeminiLLMManager
from ai_services_api.services.chatbot.utils.message_handler import MessageHandler
from ai_services_api.services.chatbot.utils.db_utils import DatabaseConnector
from ai_services_api.services.chatbot.utils.redis_connection import redis_pool, get_redis
from ai_services_api.services.chatbot.utils.rate_limiter import DynamicRateLimiter
from ai_services_api.services.chatbot.utils.redis_connection import DynamicRedisPool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router and services
router = APIRouter()
llm_manager = GeminiLLMManager()
message_handler = MessageHandler(llm_manager)
rate_limiter = DynamicRateLimiter()
redis_pool = DynamicRedisPool()


class ChatResponse(BaseModel):
    response: str
    timestamp: datetime
    user_id: str
    response_time: Optional[float] = None

async def get_user_id(request: Request) -> str:
    """Fetch user ID from request headers."""
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    return user_id

@DatabaseConnector.retry_on_failure(max_retries=3)
async def process_chat_request(query: str, user_id: str, redis_client) -> ChatResponse:
    """
    Handle chat request with optimized processing and enhanced logging.
    
    Args:
        query (str): The user's chat query
        user_id (str): Unique identifier for the user
        redis_client: Redis client for caching
    
    Returns:
        ChatResponse: Processed chat response
    """
    # Capture overall start time
    overall_start_time = datetime.utcnow()
    logger.info(f"Starting chat request processing")
    logger.debug(f"Request details - User ID: {user_id}, Query length: {len(query)}")

    try:
        # Generate unique redis key for caching
        redis_key = f"chat:{user_id}:{query}"
        logger.debug(f"Generated Redis key: {redis_key}")

        # Concurrent cache check and session creation
        logger.info("Performing concurrent cache check and session initialization")
        try:
            cache_check, session_id = await asyncio.gather(
                redis_client.get(redis_key),
                message_handler.start_chat_session(user_id)
            )
        except Exception as concurrent_error:
            logger.error(f"Error in concurrent operations: {concurrent_error}", exc_info=True)
            raise

        # Check cache hit
        if cache_check:
            logger.info(f"Cache hit detected for query")
            logger.debug(f"Cached response size: {len(cache_check)} bytes")
            return ChatResponse(**json.loads(cache_check))

        # Initialize response collection
        response_parts = []
        error_response = None

        # Process message stream with timeout
        try:
            logger.info("Initiating message stream processing with 30-second timeout")
            async with asyncio.timeout(30):
                async for part in message_handler.send_message_async(
                    message=query,
                    user_id=user_id,
                    session_id=session_id
                ):
                    # Decode bytes if necessary
                    if isinstance(part, bytes):
                        part = part.decode('utf-8')
                    
                    # Log each response part
                    logger.debug(f"Received response part (length: {len(part)})")
                    response_parts.append(part)

        except asyncio.TimeoutError:
            logger.warning("Request processing timed out after 30 seconds")
            error_response = "The request took too long to process. Please try again."
        
        except Exception as processing_error:
            logger.error(f"Error in chat processing: {processing_error}", exc_info=True)
            
            # Specific error handling
            if "503" in str(processing_error) or "overloaded" in str(processing_error):
                logger.warning("Service overload detected")
                error_response = "I apologize, but the service is currently experiencing high load. Please try again in a few moments."
            else:
                logger.error("Unexpected error during chat processing")
                error_response = "An unexpected error occurred. Please try again."

        # Handle error scenario
        if error_response:
            logger.info("Generating error response")
            chat_data = {
                "response": error_response,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            return ChatResponse(**chat_data)

        # Process successful response
        response_time = (datetime.utcnow() - overall_start_time).total_seconds()
        complete_response = ''.join(response_parts)
        
        logger.info(f"Successfully processed chat request")
        logger.debug(f"Response details - Total length: {len(complete_response)}, "
                     f"Response time: {response_time:.2f} seconds")

        # Prepare chat data
        chat_data = {
            "response": complete_response,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "response_time": response_time
        }

        # Concurrent cache update and DB save
        logger.info("Performing concurrent cache update and database save")
        try:
            await asyncio.gather(
                redis_client.setex(redis_key, 3600, json.dumps(chat_data)),
                message_handler.save_chat_to_db(
                    user_id,
                    query,
                    complete_response,
                    response_time
                )
            )
        except Exception as save_error:
            logger.error(f"Error in cache/DB save: {save_error}", exc_info=True)
            # Note: We don't raise here to ensure the response is still returned

        # Final logging of total processing time
        total_processing_time = (datetime.utcnow() - overall_start_time).total_seconds()
        logger.info(f"Chat request processing completed. Total time: {total_processing_time:.2f} seconds")

        return ChatResponse(**chat_data)

    except Exception as critical_error:
        # Critical error handling
        logger.critical(f"Unhandled error in chat endpoint: {critical_error}", exc_info=True)
        
        # Log additional context
        logger.error(f"Error details - User ID: {user_id}, Query: {query}")
        
        # Raise HTTP exception for proper error handling
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(critical_error)}"
        )
@router.get("/chat/{query}")
async def chat_endpoint(
    query: str,
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
    ):
    """Chat endpoint with improved rate limiting and error handling."""
    # Log request information
    logger.info(f"Chat request received - User: {user_id}, Query length: {len(query)}")
    
    # Check rate limit with enhanced error handling
    if not await rate_limiter.check_rate_limit(user_id):
        remaining_time = await rate_limiter.get_window_remaining(user_id)
        retry_after = max(1, min(60, remaining_time))  # Cap retry between 1-60 seconds
        
        # Log rate limit event
        logger.warning(f"Rate limit exceeded for user {user_id}, retry after {retry_after}s")
        
        # Check if there's a global circuit breaker active
        redis_circuit_key = "global:api_circuit_breaker"
        circuit_active = await redis_client.get(redis_circuit_key)
        
        if circuit_active:
            # Service-wide circuit breaker is active
            circuit_ttl = await redis_client.ttl(redis_circuit_key)
            error_detail = {
                "error": "Service temporarily unavailable",
                "retry_after": circuit_ttl,
                "message": "Our service is experiencing high demand. Please try again shortly."
            }
            logger.warning(f"Circuit breaker active, returning 503 response with {circuit_ttl}s retry")
            return JSONResponse(
                status_code=503,
                content=error_detail,
                headers={"Retry-After": str(circuit_ttl)}
            )
        else:
            # Standard rate limit for this user
            error_detail = {
                "error": "Rate limit exceeded",
                "retry_after": retry_after,
                "limit": await rate_limiter.get_user_limit(user_id),
                "message": "Please reduce your request frequency"
            }
            return JSONResponse(
                status_code=429,
                content=error_detail,
                headers={"Retry-After": str(retry_after)}
            )
    
    try:
        # Process the request
        start_time = time.time()
        response = await process_chat_request(query, user_id, redis_client)
        processing_time = time.time() - start_time
        
        # Log successful processing
        logger.info(f"Chat request processed successfully - Time: {processing_time:.2f}s, User: {user_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        
        # Check for Google API rate limit errors
        if any(x in str(e).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
            logger.warning(f"Google API rate limit detected: {str(e)}")
            
            # Record the rate limit error to trigger global circuit breaker if needed
            await rate_limiter._record_api_rate_limit_error(redis_client)
            
            # Return appropriate error response
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "Our service is experiencing high demand. Please try again in a moment.",
                    "retry_after": 30
                },
                headers={"Retry-After": "30"}
            )
            
        # Check for timeout errors
        elif any(x in str(e).lower() for x in ["timeout", "deadline exceeded"]):
            logger.warning(f"Request timeout: {str(e)}")
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "message": "Your request took too long to process. Please try a shorter query."
                }
            )
            
        # Generic error handling
        else:
            logger.error(f"Unhandled error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred. Please try again."
                }
            )

@router.get("/chat/limit/status")
async def get_rate_limit_status(user_id: str = Depends(get_user_id)):
    """Get current rate limit status for user."""
    try:
        current_limit = await rate_limiter.get_user_limit(user_id)
        limit_key = await rate_limiter.get_limit_key(user_id)
        redis_client = await redis_pool.get_redis()
        current_count = int(await redis_client.get(limit_key) or 0)
        
        return {
            "current_limit": current_limit,
            "requests_remaining": max(0, current_limit - current_count),
            "reset_time": int(time.time() / rate_limiter.window_size + 1) * rate_limiter.window_size
        }
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        raise HTTPException(status_code=500, detail="Error checking rate limit status")

@router.get("/test/chat/{query}")
async def test_chat_endpoint(
        query: str,
        request: Request,
        user_id: str = "test_user_123",
        redis_client: Redis = Depends(get_redis)
    ):
        """
        Test endpoint with same rate limiting and processing.
        
        This endpoint mirrors the main chat endpoint but uses a test user ID.
        Useful for testing and development purposes.
        """
        # Log request information
        logger.info(f"Test endpoint called with query: {query}, User: {user_id}")
        
        # Check rate limit with enhanced error handling
        if not await rate_limiter.check_rate_limit(user_id):
            remaining_time = await rate_limiter.get_window_remaining(user_id)
            retry_after = max(1, min(60, remaining_time))  # Cap retry between 1-60 seconds
            
            # Log rate limit event
            logger.warning(f"Rate limit exceeded for test user: {user_id}, retry after {retry_after}s")
            
            # Check if there's a global circuit breaker active
            redis_circuit_key = "global:api_circuit_breaker"
            circuit_active = await redis_client.get(redis_circuit_key)
            
            if circuit_active:
                # Service-wide circuit breaker is active
                circuit_ttl = await redis_client.ttl(redis_circuit_key)
                error_detail = {
                    "error": "Service temporarily unavailable",
                    "retry_after": circuit_ttl,
                    "message": "Our service is experiencing high demand. Please try again shortly.",
                    "test_mode": True
                }
                logger.warning(f"Circuit breaker active, returning 503 response with {circuit_ttl}s retry")
                return JSONResponse(
                    status_code=503,
                    content=error_detail,
                    headers={"Retry-After": str(circuit_ttl)}
                )
            else:
                # Standard rate limit for this user
                error_detail = {
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "limit": await rate_limiter.get_user_limit(user_id),
                    "message": "Please wait before making another request",
                    "test_mode": True
                }
                return JSONResponse(
                    status_code=429,
                    content=error_detail,
                    headers={"Retry-After": str(retry_after)}
                )
        
        try:
            # Process the request
            start_time = time.time()
            
            # Process the chat request with database connection
            async with DatabaseConnector.get_connection() as conn:
                response = await process_chat_request(query, user_id, redis_client)
                
                # Add test-specific metadata
                response_dict = response.dict()
                response_dict["test_mode"] = True
                response_dict["test_user_id"] = user_id
                
                # Log successful processing
                processing_time = time.time() - start_time
                logger.info(f"Test chat request processed successfully - Time: {processing_time:.2f}s, User: {user_id}")
                
                return response_dict
            
        except Exception as e:
            logger.error(f"Error in test chat endpoint: {e}", exc_info=True)
            
            # Check for Google API rate limit errors
            if any(x in str(e).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
                logger.warning(f"Google API rate limit detected in test endpoint: {str(e)}")
                
                # Record the rate limit error to trigger global circuit breaker if needed
                await rate_limiter._record_api_rate_limit_error(redis_client)
                
                # Return appropriate error response
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Service temporarily unavailable",
                        "message": "Our service is experiencing high demand. Please try again in a moment.",
                        "retry_after": 30,
                        "test_mode": True
                    },
                    headers={"Retry-After": "30"}
                )
                
            # Check for timeout errors
            elif any(x in str(e).lower() for x in ["timeout", "deadline exceeded"]):
                logger.warning(f"Request timeout in test endpoint: {str(e)}")
                return JSONResponse(
                    status_code=504,
                    content={
                        "error": "Request timeout",
                        "message": "Your request took too long to process. Please try a shorter query.",
                        "test_mode": True
                    }
                )
                
            # Generic error handling
            else:
                logger.error(f"Unhandled error in test endpoint: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal server error in test endpoint",
                        "message": str(e),
                        "test_mode": True
                    }
                )
# Startup and shutdown events
async def startup_event():
    """Initialize database and Redis connections on startup."""
    await DatabaseConnector.initialize()
    await redis_pool.get_redis()

async def shutdown_event():
    """Cleanup connections on shutdown."""
    await DatabaseConnector.close()
    await redis_pool.close()