from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict
from pydantic import BaseModel, Field
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

# Request and response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's chat message", min_length=1)

class ErrorResponse(BaseModel):
    error: str
    message: str
    retry_after: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime
    user_id: str
    response_time: Optional[float] = None
    metadata: Optional[dict] = None  # Added metadata field to model

async def get_user_id(request: Request) -> str:
    """Fetch user ID from request headers."""
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    return user_id

@DatabaseConnector.retry_on_failure(max_retries=3)
# Enhanced logging for process_chat_request function
async def process_chat_request(query: str, user_id: str, redis_client) -> ChatResponse:
    """Handle chat request with enhanced logging for debugging."""
    # Capture overall start time
    overall_start_time = datetime.utcnow()
    logger.info(f"Starting process_chat_request")
    logger.info(f"Request details - User ID: {user_id}, Query: {query}")

    try:
        # Generate unique redis key for caching
        redis_key = f"chat:{user_id}:{query}"
        
        # Concurrent cache check and session creation
        logger.info("Performing concurrent cache check and session initialization")
        try:
            cache_check, session_id = await asyncio.gather(
                redis_client.get(redis_key),
                message_handler.start_chat_session(user_id)
            )
            logger.info(f"Created chat session: {session_id}")
        except Exception as concurrent_error:
            logger.error(f"Error in concurrent operations: {concurrent_error}", exc_info=True)
            raise

        # Check cache hit
        if cache_check:
            logger.info(f"Cache hit detected for query")
            return ChatResponse(**json.loads(cache_check))

        # Initialize response collection
        response_parts = []
        error_response = None

        # Process message stream with timeout
        try:
            logger.info("Initiating message stream processing with timeout")
            async with asyncio.timeout(200):
                part_counter = 0
                async for part in message_handler.send_message_async(
                    message=query,
                    user_id=user_id,
                    session_id=session_id
                ):
                    part_counter += 1
                    logger.info(f"Processing stream part #{part_counter}")
                    
                    # Log exact content type of this part
                    logger.info(f"Part type: {type(part).__name__}")
                    
                    # Handle different part types
                    if isinstance(part, dict):
                        logger.info(f"Received dictionary part: {str(part)[:100]}...")
                        # Skip metadata dictionaries in the response parts
                        if part.get('is_metadata'):
                            logger.info("Skipping metadata dictionary in response parts")
                            continue
                        # Convert other dictionaries to strings (this should rarely happen)
                        logger.info("Converting dictionary to string for response")
                        response_parts.append(json.dumps(part))
                    elif isinstance(part, bytes):
                        logger.info(f"Received bytes part, length: {len(part)}")
                        # Decode bytes to string
                        decoded = part.decode('utf-8')
                        logger.debug(f"Decoded bytes: {decoded[:50]}...")
                        response_parts.append(decoded)
                    elif isinstance(part, str):
                        logger.info(f"Received string part, length: {len(part)}")
                        logger.debug(f"String content: {part[:50]}...")
                        response_parts.append(part)
                    else:
                        # Handle unexpected types
                        logger.warning(f"Unexpected part type: {type(part).__name__}")
                        logger.debug(f"Converting to string: {str(part)[:100]}...")
                        response_parts.append(str(part))

                logger.info(f"Completed processing {part_counter} stream parts")

        except asyncio.TimeoutError:
            logger.warning("Request processing timed out")
            error_response = "The request took too long to process. Please try again."
        
        except Exception as processing_error:
            logger.error(f"Error in chat processing: {processing_error}", exc_info=True)
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
        
        logger.info(f"Joining {len(response_parts)} response parts")
        try:
            complete_response = ''.join(response_parts)
            logger.info(f"Successfully joined all response parts, total length: {len(complete_response)}")
        except Exception as join_error:
            logger.error(f"Error joining response parts: {join_error}", exc_info=True)
            logger.error(f"Response parts types: {[type(p).__name__ for p in response_parts]}")
            return ChatResponse(
                response="Error processing response. Please try again.",
                timestamp=datetime.utcnow().isoformat(),
                user_id=user_id,
                response_time=response_time
            )
        
        logger.info(f"Successfully processed chat request")
        
        # Prepare chat data
        chat_data = {
            "response": complete_response,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "response_time": response_time
        }

        # Save to cache and database
        logger.info("Saving response to cache and database")
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
            logger.info("Successfully saved to cache and database")
        except Exception as save_error:
            logger.error(f"Error in cache/DB save: {save_error}", exc_info=True)

        # Final logging
        total_processing_time = (datetime.utcnow() - overall_start_time).total_seconds()
        logger.info(f"Chat request processing completed. Total time: {total_processing_time:.2f} seconds")

        return ChatResponse(**chat_data)

    except Exception as critical_error:
        # Critical error handling
        logger.critical(f"Unhandled error in chat endpoint: {critical_error}", exc_info=True)
        logger.error(f"Error details - User ID: {user_id}, Query: {query}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(critical_error)}"
        )

# Updated POST endpoint for chat
@router.post("/chat", response_model=ChatResponse, responses={
    400: {"model": ErrorResponse, "description": "Missing user ID or invalid request"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
    503: {"model": ErrorResponse, "description": "Service temporarily unavailable"},
    504: {"model": ErrorResponse, "description": "Request timeout"}
})
async def chat_endpoint(
    request_data: ChatRequest,
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
    ):
    """
    Process chat messages sent by users.
    
    This endpoint accepts a chat message in the request body and returns the AI's response.
    Each request must include the X-User-ID header for user identification and rate limiting.
    
    Rate limits apply to prevent abuse.
    """
    # Extract message from request body
    query = request_data.message
    
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
        
@router.post("/chat/anonymous", response_model=ChatResponse, responses={
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
    503: {"model": ErrorResponse, "description": "Service temporarily unavailable"},
    504: {"model": ErrorResponse, "description": "Request timeout"}
})
async def anonymous_chat_endpoint(
    request_data: ChatRequest,
    request: Request,
    redis_client: Redis = Depends(get_redis)
    ):
    """
    Process chat messages without requiring a user ID.
    
    This endpoint generates a temporary anonymous user ID for each request.
    Useful for quick interactions without user registration.
    
    Rate limits still apply to prevent abuse.
    """
    # Generate a temporary anonymous user ID
    import uuid
    user_id = f"anon_{uuid.uuid4().hex}"
    
    # Extract message from request body
    query = request_data.message
    
    # Log request information
    logger.info(f"Anonymous chat request received - Temp User: {user_id}, Query length: {len(query)}")
    
    # Check rate limit with enhanced error handling
    if not await rate_limiter.check_rate_limit(user_id):
        remaining_time = await rate_limiter.get_window_remaining(user_id)
        retry_after = max(1, min(60, remaining_time))  # Cap retry between 1-60 seconds
        
        # Log rate limit event
        logger.warning(f"Rate limit exceeded for anonymous user, retry after {retry_after}s")
        
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
            # Standard rate limit for this temporary user
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
        logger.info(f"Anonymous chat request processed successfully - Time: {processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in anonymous chat endpoint: {e}", exc_info=True)
        
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

async def shutdown_event():
    """Cleanup connections on shutdown."""
    await DatabaseConnector.close()
    await redis_pool.close()

@router.get("/chat/limit/status", response_model=Dict)
async def get_rate_limit_status(user_id: str = Depends(get_user_id)):
    """
    Get current rate limit status for the user.
    
    Returns information about the user's current rate limit, 
    remaining requests, and reset time.
    
    Requires X-User-ID header for user identification.
    """
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

# Startup and shutdown events
async def startup_event():
    """Initialize database and Redis connections on startup."""
    await DatabaseConnector.initialize()
    await redis_pool.get_redis()



