from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.message.core.config import get_settings
from redis.asyncio import Redis
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
from datetime import datetime
import logging
import json
import re
from psycopg2.extras import RealDictCursor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

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
    logger.debug("Extracting test user ID from request headers")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.info("No X-User-ID provided, using default test ID: 123")
        user_id = "1"
    return user_id

def clean_message_content(text: str, receiver_name: str, context: str) -> str:
    """
    Clean the message content and replace empty placeholders.
    
    Args:
        text: The raw message text from the AI
        receiver_name: The name of the message recipient
        context: The original context provided by the user
        
    Returns:
        str: The cleaned message text
    """
    logger.debug("Cleaning message content")
    
    # Remove Subject line if present
    text = re.sub(r'Subject:.*?\n', '', text)
    
    # Remove salutation (Dear X,)
    text = re.sub(r'Dear [^,\n]*,?\s*', '', text)
    
    # Remove signature block (Sincerely, etc.)
    text = re.sub(r'\s*Sincerely,\s*(\n.*)*$', '', text, flags=re.DOTALL)
    
    # Fix empty placeholders - match patterns like "I am a [role] at [institution]"
    text = re.sub(r'I am a\s+(?:at\s+)?\.', 'I am reaching out to you', text)
    text = re.sub(r'I am\s+at\s+\.', 'I am reaching out to you', text)
    text = re.sub(r'expertise in\s+\.', f'expertise in {context}.', text)
    text = re.sub(r'research on\s+\.', f'research on {context}.', text)
    text = re.sub(r'Your work on\s+particularly', 'Your work particularly', text)
    text = re.sub(r'specifically\s+\.', 'specifically.', text)
    
    # Replace all remaining empty placeholders [Your X] or just empty brackets
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Replace all newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Trim whitespace
    text = text.strip()
    
    logger.debug(f"Message after cleaning: {text[:100]}...")
    return text

@retry(
    retry=retry_if_exception_type((
        google_exceptions.ResourceExhausted,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        ConnectionError,
        TimeoutError,
        Exception  # Catch-all for any unexpected errors
    )),
    wait=wait_exponential(multiplier=1.5, min=4, max=60),
    stop=stop_after_attempt(5)
)
def generate_with_retry(model, prompt):
    """Generate content with retry logic for rate limits."""
    try:
        return model.generate_content(prompt)
    except Exception as e:
        logger.warning(f"API error in generate_content, retrying: {e}")
        raise

async def process_message_draft(
    user_id: str,
    receiver_id: str, 
    content: str,
    redis_client: Redis = None
):
    logger.info(f"Starting message draft process for receiver {receiver_id}")
    logger.debug(f"Draft request parameters - user_id: {user_id}, content length: {len(content)}")
    
    # Check cache if Redis client is provided
    if redis_client:
        # Check for exact match first
        cache_key = f"message_draft:{user_id}:{receiver_id}:{content}"
        logger.debug(f"Checking cache with key: {cache_key}")
        
        cached_response = await redis_client.get(cache_key)
        if cached_response:
            logger.info("Cache hit for message draft")
            return json.loads(cached_response)
            
        # If no exact match, check for similar content to maximize cache usage
        similar_key_pattern = f"message_draft:{user_id}:{receiver_id}:*"
        try:
            keys = await redis_client.keys(similar_key_pattern)
            
            # Process only if there are a reasonable number of keys
            if keys and len(keys) < 100:  # Limit search to avoid performance issues
                logger.debug(f"Found {len(keys)} similar cache keys to check")
                for key in keys:
                    try:
                        # Extract the content part from the key
                        key_parts = key.split(':', 3)
                        if len(key_parts) < 4:
                            continue
                            
                        cached_content = key_parts[3]
                        # Skip very short content
                        if len(cached_content) < 10 or len(content) < 10:
                            continue
                            
                        # Calculate word overlap for similarity
                        content_words = set(content.lower().split())
                        cached_words = set(cached_content.lower().split())
                        
                        if len(content_words) == 0 or len(cached_words) == 0:
                            continue
                            
                        overlap = content_words & cached_words
                        overlap_ratio = len(overlap) / max(len(content_words), len(cached_words))
                        
                        if overlap_ratio > 0.7:  # 70% similar words
                            cached_response = await redis_client.get(key)
                            if cached_response:
                                logger.info(f"Found similar cached content (similarity: {overlap_ratio:.2f})")
                                return json.loads(cached_response)
                    except Exception as key_error:
                        logger.warning(f"Error processing cache key {key}: {str(key_error)}")
                        continue
        except Exception as cache_error:
            logger.warning(f"Error checking similar cache keys: {str(cache_error)}")
            # Continue execution even if cache similarity check fails
    
    conn = None
    cur = None
    start_time = datetime.utcnow()
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        logger.debug("Database connection established successfully")
            
        # Fetch receiver details
        cur.execute("""
            SELECT id, first_name, last_name, designation, theme, domains, fields 
            FROM experts_expert 
            WHERE id = %s AND is_active = true
        """, (receiver_id,))
        receiver = cur.fetchone()
        
        if not receiver:
            logger.error(f"Receiver not found or inactive: {receiver_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Receiver with ID {receiver_id} not found or is inactive"
            )

        logger.info(f"Receiver found: {receiver['first_name']} {receiver['last_name']}")

        # Configure Gemini
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.debug("Gemini model configured successfully")
        
        # Keep your original prompt exactly the same
        prompt = f"""
            Draft a professional message to {receiver['first_name']} {receiver['last_name']} ({receiver['designation'] or 'Expert'}).
            The message should be about: {content}

            Context about receiver:
            - Theme: {receiver['theme'] or 'Research'}
            - Domains: {', '.join(receiver['domains'] if receiver.get('domains') else ['Research'])}
            - Fields: {', '.join(receiver['fields'] if receiver.get('fields') else ['Expertise'])}

            Important: DO NOT include placeholders or brackets in your response. Replace any information you don't have with general professional language.
            DO NOT use phrases like "I am a [role] at [institution]" or any brackets. If you don't know certain information, write around it naturally.
            """
        
        logger.debug(f"Generated prompt for Gemini: {prompt}")
        
        try:
            # Use the retry-wrapped function
            response = generate_with_retry(model, prompt)
            draft_content = response.text
            logger.info(f"Generated draft content of length: {len(draft_content)}")
            
            # Apply cleaning function to the response with all required parameters
            receiver_name = f"{receiver['first_name']} {receiver['last_name']}"
            cleaned_content = clean_message_content(
                text=draft_content,
                receiver_name=receiver_name,
                context=content
            )
            logger.info(f"Cleaned content of length: {len(cleaned_content)}")
            
        except Exception as api_error:
            logger.error(f"Error generating content after retries: {str(api_error)}")
            # More descriptive fallback message
            cleaned_content = f"I'm reaching out regarding {content}. I would appreciate the opportunity to discuss this with you. [Note: This is a simplified message as our AI service is experiencing high demand at the moment.]"
            logger.info("Using fallback content due to API error")

        # Insert the draft message with cleaned content
        cur.execute("""
            INSERT INTO expert_messages 
                (sender_id, receiver_id, content, draft, created_at, updated_at) 
            VALUES 
                (%s, %s, %s, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, created_at
        """, (1, receiver_id, cleaned_content))
        
        new_message = cur.fetchone()
        logger.info(f"Created new draft message with ID: {new_message['id']}")

        conn.commit()
        logger.info(f"Successfully committed transaction for message {new_message['id']}")

        # Convert datetime to string for JSON serialization
        created_at = new_message['created_at'].isoformat() if new_message['created_at'] else None

        response_data = {
            "id": str(new_message['id']),
            "content": cleaned_content,  # Use cleaned content in response
            "sender_id": user_id,
            "receiver_id": str(receiver_id),
            "created_at": created_at,
            "draft": False,
            "receiver_name": f"{receiver['first_name']} {receiver['last_name']}",
            "sender_name": "Test User"
        }
        
        # Cache the response if Redis client is provided
        if redis_client:
            try:
                logger.debug(f"Caching response data with key: {cache_key}")
                await redis_client.setex(
                    cache_key,
                    3600,  # Cache for 1 hour
                    json.dumps(response_data)
                )
                logger.info("Response data cached successfully")
                
                # Also save a timestamp of the last successful API call
                await redis_client.setex(
                    "last_successful_gemini_call",
                    3600,  # Keep for 1 hour
                    datetime.utcnow().isoformat()
                )
            except Exception as cache_error:
                logger.error(f"Error caching response: {str(cache_error)}")
                # Continue even if caching fails

        logger.debug(f"Preparing response data: {response_data}")
        return response_data

    except Exception as e:
        if conn:
            conn.rollback()
            logger.warning("Transaction rolled back due to error")
        logger.error(f"Error in process_message_draft: {str(e)}", exc_info=True)
        
        # Check if this is likely a rate limit related error
        if any(x in str(e).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
            # Try to get the most recent successful draft as fallback
            if redis_client:
                try:
                    # Find any cached draft for this user-receiver pair
                    fallback_pattern = f"message_draft:{user_id}:{receiver_id}:*"
                    fallback_keys = await redis_client.keys(fallback_pattern)
                    
                    if fallback_keys:
                        # Get the most recently cached response
                        fallback_response = await redis_client.get(fallback_keys[0])
                        if fallback_response:
                            fallback_data = json.loads(fallback_response)
                            fallback_data["content"] += " [Note: This is a previously generated message as our service is currently experiencing high demand.]"
                            logger.info("Returning fallback message from cache due to rate limiting")
                            return fallback_data
                except Exception as fallback_error:
                    logger.error(f"Error getting fallback message: {str(fallback_error)}")
        
        # If we get here, we couldn't find a fallback or it's not a rate limit error
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.debug("Database connections closed")

@router.get("/test/draft/{receiver_id}/{content}")
async def test_create_message_draft(
    receiver_id: str,
    content: str,
    request: Request,
    user_id: str = Depends(get_test_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received test draft message request for receiver: {receiver_id}")
    return await process_message_draft(user_id, receiver_id, content, redis_client)

@router.get("/draft/{receiver_id}/{content}")
async def create_message_draft(
    receiver_id: str,
    content: str,
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    logger.info(f"Received draft message request for receiver: {receiver_id}")
    return await process_message_draft(user_id, receiver_id, content, redis_client)