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


def generate_with_retry(model, prompt):
    """Generate content with retry logic for rate limits."""
    try:
        return model.generate_content(prompt)
    except Exception as e:
        logger.warning(f"API error in generate_content, retrying: {e}")
        raise

def clean_message_content(text: str, receiver: dict, sender: dict, context: str) -> str:
    """
    Enhanced cleaning of message content with real sender information.
    
    Args:
        text: The raw message text from the AI
        receiver: Dictionary containing receiver details
        sender: Dictionary containing sender details
        context: The original context provided by the user
        
    Returns:
        str: The cleaned and enhanced message text
    """
    logger.debug("Cleaning message content with sender information")
    
    # Get name information
    receiver_name = f"{receiver.get('first_name', '')} {receiver.get('last_name', '').strip()}".strip()
    sender_name = f"{sender.get('first_name', '')} {sender.get('last_name', '').strip()}".strip()
    sender_position = sender.get('position', 'Researcher')
    sender_department = sender.get('department', 'APHRC')
    
    # Fix repetitive salutations
    text = re.sub(rf'Dear {receiver_name},\s*Dear {receiver_name},', f'Dear {receiver_name},', text)
    
    # Ensure introduction includes sender details
    intro_pattern = r'My name is.*?and I am a.*?at the African Population and Health Research Center \(APHRC\)'
    proper_intro = f"My name is {sender_name}, and I am a {sender_position} at the African Population and Health Research Center (APHRC)"
    
    if re.search(intro_pattern, text):
        text = re.sub(intro_pattern, proper_intro, text)
    else:
        # If no introduction found, add it after the salutation
        salutation_pattern = rf'Dear {receiver_name},'
        if re.search(salutation_pattern, text):
            text = re.sub(
                rf'(Dear {receiver_name},\s*)', 
                f'\\1\n\n{proper_intro}. ', 
                text
            )
    
    # Ensure proper signature
    signature_pattern = r'Best regards,\s*.*?\s*.*?African Population and Health Research Center \(APHRC\)'
    proper_signature = f"Best regards,\n{sender_name}\n{sender_department}\nAfrican Population and Health Research Center (APHRC)"
    
    if re.search(signature_pattern, text, re.DOTALL):
        text = re.sub(signature_pattern, proper_signature, text, flags=re.DOTALL)
    else:
        # If no signature found, add it at the end
        text = text.rstrip() + f"\n\n{proper_signature}"
    
    # Remove any remaining placeholders
    text = re.sub(r'\[.*?\]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def generate_expert_draft_prompt(receiver: dict, content: str, sender: dict) -> str:
    """
    Generate a sophisticated prompt with real sender information.
    
    Args:
        receiver: Dictionary containing receiver details
        content: The original message context
        sender: Dictionary containing sender details
        
    Returns:
        str: Comprehensive prompt for message generation
    """
    # Prepare receiver details
    first_name = receiver.get('first_name', 'Respected')
    last_name = receiver.get('last_name', 'Colleague')
    designation = receiver.get('designation', 'Research Expert')
    theme = receiver.get('theme', 'Population Health Research')
    domains = ', '.join(receiver.get('domains', ['Interdisciplinary Research']))
    fields = ', '.join(receiver.get('fields', ['Health Systems']))
    
    # Prepare sender details
    sender_first_name = sender.get('first_name', 'Research')
    sender_last_name = sender.get('last_name', 'Team')
    sender_position = sender.get('position', 'Researcher')
    sender_department = sender.get('department', 'APHRC')
    
    # Construct a nuanced, context-rich prompt
    prompt = f"""
    You are an AI assistant helping a researcher from the African Population and Health Research Center (APHRC) draft a professional communication.

    Recipient Details:
    - Name: {first_name} {last_name}
    - Designation: {designation}
    - Research Focus: {theme}
    - Primary Domains: {domains}
    - Specialized Fields: {fields}
    
    Sender Details:
    - Name: {sender_first_name} {sender_last_name}
    - Position: {sender_position}
    - Department: {sender_department}

    Communication Context: {content}

    IMPORTANT FORMATTING REQUIREMENTS:
    1. Begin with "Dear {first_name} {last_name},"
    2. Introduce the sender using their real name: "My name is {sender_first_name} {sender_last_name}, and I am a {sender_position} at the African Population and Health Research Center (APHRC)."
    3. End with "Best regards,\\n{sender_first_name} {sender_last_name}\\n{sender_department}\\nAfrican Population and Health Research Center (APHRC)"
    4. DO NOT use placeholder text anywhere in the message

    Drafting Guidelines:
    1. Craft a concise, professional message that reflects APHRC's research excellence
    2. Demonstrate genuine interest in potential collaboration or knowledge exchange
    3. Highlight the relevance of the proposed communication to the recipient's expertise
    4. Maintain a tone of academic respect and professional curiosity
    5. Ensure the message is culturally sensitive and aligned with APHRC's mission

    Draft the message accordingly, focusing on creating a meaningful professional connection.
    """
    
    return prompt

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
            
        # Fetch receiver details - MOVED THIS CODE UP 
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

        # Get sender details from experts_expert table instead of users
        cur.execute("""
            SELECT id, first_name, last_name, designation as position, unit as department 
            FROM experts_expert
            WHERE id = %s AND is_active = true
        """, (user_id,))
        sender = cur.fetchone()
        
        if not sender:
            logger.warning(f"Sender with ID {user_id} not found, using default sender info")
            # Create a default sender object if the user is not found
            sender = {
                "id": user_id,
                "first_name": "Research",
                "last_name": "Team",
                "position": "Researcher",
                "department": "APHRC"
            }

        logger.info(f"Using sender: {sender['first_name']} {sender['last_name']}")

        # Configure Gemini
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.debug("Gemini model configured successfully")
        
        # Use the new prompt generation function with sender information
        prompt = generate_expert_draft_prompt(receiver, content, sender)
        
        logger.debug(f"Generated sophisticated prompt for Gemini: {prompt[:500]}...")
        
        try:
            # Use the retry-wrapped function
            response = generate_with_retry(model, prompt)
            draft_content = response.text
            logger.info(f"Generated draft content of length: {len(draft_content)}")
            
            # Apply enhanced cleaning function to the response with sender info
            cleaned_content = clean_message_content(
                text=draft_content,
                receiver=receiver,
                sender=sender,
                context=content
            )
            logger.info(f"Cleaned content of length: {len(cleaned_content)}")
            
        except Exception as api_error:
            logger.error(f"Error generating content after retries: {str(api_error)}")
            
            # More descriptive APHRC-specific fallback message with sender details
            receiver_name = f"{receiver.get('first_name', '')} {receiver.get('last_name', '')}".strip()
            sender_name = f"{sender.get('first_name', '')} {sender.get('last_name', '')}".strip()
            sender_position = sender.get('position', 'Researcher')
            sender_department = sender.get('department', 'APHRC')
            
            cleaned_content = f"""Dear {receiver_name},

My name is {sender_name}, and I am a {sender_position} at the African Population and Health Research Center (APHRC). I am reaching out regarding {content}. 
Despite current service constraints, I would appreciate the opportunity to discuss this matter further.

Best regards,
{sender_name}
{sender_department}
African Population and Health Research Center (APHRC)"""
            logger.info("Using APHRC-specific fallback content due to API error")

        # Insert the draft message with cleaned content
        cur.execute("""
            INSERT INTO expert_messages 
                (sender_id, receiver_id, content, draft, created_at, updated_at) 
            VALUES 
                (%s, %s, %s, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, created_at
        """, (user_id, receiver_id, cleaned_content))
        
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
            "draft": True,
            "receiver_name": f"{receiver['first_name']} {receiver['last_name']}",
            "sender_name": f"{sender['first_name']} {sender['last_name']}"
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

from fastapi import Body
from typing import Dict

# Add this to your router - Set User ID endpoint for message service
@router.post("/message/set-user-id")
async def set_message_user_id(
    request: Request,
    user_id: str = Body(...),
):
    """
    Set the user ID for the message service.
    This allows testing with different user IDs without changing headers.
    
    Returns:
        Dict with status and the set user ID
    """
    logger.info(f"Setting message service user ID to: {user_id}")
    
    # Store the user ID in the request state for this session
    request.state.user_id = user_id
    
    return {
        "status": "success",
        "message": f"Message service user ID set to: {user_id}",
        "user_id": user_id
    }

# Create a flexible user ID dependency for the message service
async def flexible_message_user_id(request: Request) -> str:
    """
    Get user ID with flexibility for message endpoints:
    1. First check if a user ID is set in request state (from /message/set-user-id endpoint)
    2. Fall back to X-User-ID header if not in request state
    3. Raise exception if neither is available
    
    This preserves the original get_user_id behavior for existing endpoints.
    """
    logger.debug("Flexible message user ID extraction")
    
    # First check if we have a user ID in the request state
    if hasattr(request.state, "user_id"):
        user_id = request.state.user_id
        logger.info(f"Using message user ID from request state: {user_id}")
        return user_id
    
    # Otherwise fall back to the header
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Missing required X-User-ID header in request")
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    
    logger.info(f"Message user ID extracted from header: {user_id}")
    return user_id

# Add a flexible version of the draft message endpoint
@router.get("/draft/{receiver_id}/{content}/flexible")
async def flexible_create_message_draft(
    receiver_id: str,
    content: str,
    request: Request,
    user_id: str = Depends(flexible_message_user_id),  # Use the flexible dependency here
    redis_client: Redis = Depends(get_redis)
):
    """
    Create a message draft with flexible user ID handling.
    This endpoint uses the user ID from either the state or the header.
    """
    logger.info(f"Received flexible draft message request - User: {user_id}, Receiver: {receiver_id}")
    return await process_message_draft(user_id, receiver_id, content, redis_client)

# Also add a flexible version of the test endpoint
@router.get("/test/draft/{receiver_id}/{content}/flexible")
async def flexible_test_create_message_draft(
    receiver_id: str,
    content: str,
    request: Request,
    user_id: str = Depends(flexible_message_user_id),  # Use the flexible dependency
    redis_client: Redis = Depends(get_redis)
):
    """
    Test endpoint for creating a message draft with flexible user ID handling.
    This endpoint uses the user ID from either the state or the header.
    """
    logger.info(f"Received flexible test draft message request - User: {user_id}, Receiver: {receiver_id}")
    return await process_message_draft(user_id, receiver_id, content, redis_client)

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