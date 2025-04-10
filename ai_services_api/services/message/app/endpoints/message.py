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

def clean_message_content(text: str, receiver: dict, context: str) -> str:
    """
    Enhanced cleaning of message content with more aggressive placeholder removal.
    
    Args:
        text: The raw message text from the AI
        receiver: Dictionary containing receiver details
        context: The original context provided by the user
        
    Returns:
        str: The cleaned and enhanced message text
    """
    logger.debug("Cleaning message content with enhanced placeholder removal")
    
    # Remove Subject line if present
    text = re.sub(r'Subject:.*?\n', '', text)
    
    # Prepare receiver and context details
    receiver_name = f"{receiver.get('first_name', '')} {receiver.get('last_name', '').strip()}"
    receiver_name = receiver_name.strip()
    designation = receiver.get('designation', 'Researcher')
    
    # Remove duplicate salutations
    text = re.sub(rf'^Dear {re.escape(receiver_name)},\s*Dear {re.escape(receiver_name)},', f'Dear {receiver_name},', text, flags=re.MULTILINE)
    
    # Aggressive placeholder removal and replacement
    placeholders = [
        (r'My name is\s*and\s*I am a\s*at', f'I am a researcher'),
        (r'particularly your contributions to\s*\.', 'your significant research contributions.'),
        (r'current research on\s*\.', f'current research exploring population health.'),
        (r'exploring\s*\.', 'investigating emerging health challenges.'),
        (r'expertise in\s*could', 'expertise could'),
        (r'understanding of\s*\.', 'understanding of complex health dynamics.')
    ]
    
    for pattern, replacement in placeholders:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove specific acronyms or vague references if not properly contextualized
    text = re.sub(r'\bHAW\b', 'health and wellness', text, flags=re.IGNORECASE)
    
    # Ensure a single, clean salutation
    text = re.sub(r'^Dear [^,\n]*,', f'Dear {receiver_name},', text, flags=re.MULTILINE)
    
    # Ensure professional closing
    if not re.search(r'Best regards,\s*Research Team', text, re.IGNORECASE):
        text += "\n\nBest regards,\nResearch Team at APHRC"
    
    # Remove multiple newlines and excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    logger.debug(f"Message after cleaning: {text[:500]}...")
    return text

def generate_expert_draft_prompt(receiver: dict, content: str) -> str:
    """
    Generate a sophisticated prompt for drafting an expert-level message.
    
    Args:
        receiver: Dictionary containing receiver details
        content: The original message context
        
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
    
    # Construct a nuanced, context-rich prompt
    prompt = f"""
    You are an AI assistant helping a researcher from the African Population and Health Research Center (APHRC) draft a professional communication.

    Recipient Details:
    - Name: {first_name} {last_name}
    - Designation: {designation}
    - Research Focus: {theme}
    - Primary Domains: {domains}
    - Specialized Fields: {fields}

    Communication Context: {content}

    Drafting Guidelines:
    1. Craft a concise, professional message that reflects APHRC's research excellence
    2. Demonstrate genuine interest in potential collaboration or knowledge exchange
    3. Highlight the relevance of the proposed communication to the recipient's expertise
    4. Maintain a tone of academic respect and professional curiosity
    5. Ensure the message is culturally sensitive and aligned with APHRC's mission

    Additional Considerations:
    - Be specific about the purpose of reaching out
    - Show preliminary knowledge of the recipient's work
    - Propose a clear next step or desired outcome
    - Avoid generic language; make the message feel personalized

    Prohibited Elements:
    - Do not use placeholder brackets
    - Avoid overly formal or stiff academic language
    - Do not make unsupported claims
    - Refrain from using boilerplate text

    Preferred Communication Style:
    - Clear and direct
    - Intellectually engaging
    - Respectful of the recipient's time and expertise
    - Demonstrating APHRC's commitment to impactful research

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
        
        # Use the new prompt generation function
        prompt = generate_expert_draft_prompt(receiver, content)
        
        logger.debug(f"Generated sophisticated prompt for Gemini: {prompt[:500]}...")
        
        try:
            # Use the retry-wrapped function
            response = generate_with_retry(model, prompt)
            draft_content = response.text
            logger.info(f"Generated draft content of length: {len(draft_content)}")
            
            # Apply enhanced cleaning function to the response
            cleaned_content = clean_message_content(
                text=draft_content,
                receiver=receiver,
                context=content
            )
            logger.info(f"Cleaned content of length: {len(cleaned_content)}")
            
        except Exception as api_error:
            logger.error(f"Error generating content after retries: {str(api_error)}")
            # More descriptive APHRC-specific fallback message
            receiver_name = f"{receiver.get('first_name', '')} {receiver.get('last_name', '')}".strip()
            cleaned_content = f"""Dear {receiver_name},

I am reaching out from the African Population and Health Research Center (APHRC) regarding {content}. 
Despite current service constraints, I would appreciate the opportunity to discuss this matter further.

Best regards,
Research Team at APHRC"""
            logger.info("Using APHRC-specific fallback content due to API error")

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
            "sender_name": "APHRC Research Team"
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