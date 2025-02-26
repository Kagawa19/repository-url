from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.message.core.config import get_settings
from redis.asyncio import Redis
import google.generativeai as genai
from datetime import datetime
import logging
import json
from psycopg2.extras import RealDictCursor

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
        user_id = "123"
    return user_id

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
        cache_key = f"message_draft:{user_id}:{receiver_id}:{content}"
        logger.debug(f"Checking cache with key: {cache_key}")
        
        cached_response = await redis_client.get(cache_key)
        if cached_response:
            logger.info("Cache hit for message draft")
            return json.loads(cached_response)
    
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
        model = genai.GenerativeModel('gemini-pro')
        logger.debug("Gemini model configured successfully")
        
        prompt = f"""
        Draft a professional message to {receiver['first_name']} {receiver['last_name']} ({receiver['designation'] or 'Expert'}).
        
        Context about receiver:
        - Theme: {receiver['theme'] or 'Not specified'}
        - Domains: {', '.join(receiver['domains'] if receiver.get('domains') else ['Not specified'])}
        - Fields: {', '.join(receiver['fields'] if receiver.get('fields') else ['Not specified'])}
        
        Additional context: {content}
        """
        
        logger.debug(f"Generated prompt for Gemini: {prompt}")
        response = model.generate_content(prompt)
        draft_content = response.text
        logger.info(f"Generated draft content of length: {len(draft_content)}")

        # Insert the draft message
        cur.execute("""
            INSERT INTO expert_messages 
                (sender_id, receiver_id, content, draft, created_at, updated_at) 
            VALUES 
                (%s, %s, %s, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, created_at
        """, (1, receiver_id, draft_content))
        
        new_message = cur.fetchone()
        logger.info(f"Created new draft message with ID: {new_message['id']}")

        conn.commit()
        logger.info(f"Successfully committed transaction for message {new_message['id']}")

        # Convert datetime to string for JSON serialization
        created_at = new_message['created_at'].isoformat() if new_message['created_at'] else None

        response_data = {
            "id": str(new_message['id']),
            "content": draft_content,
            "sender_id": user_id,
            "receiver_id": str(receiver_id),
            "created_at": created_at,  # Now a string in ISO format
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
                    json.dumps(response_data)  # Now serializable with datetime as string
                )
                logger.info("Response data cached successfully")
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