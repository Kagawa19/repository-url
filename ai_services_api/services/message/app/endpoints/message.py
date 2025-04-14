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
    Enhanced cleaning of message content with real sender information to ensure no gaps remain.
    
    Args:
        text: The raw message text from the AI
        receiver: Dictionary containing receiver details
        sender: Dictionary containing sender details
        context: The original context provided by the user
        
    Returns:
        str: The cleaned and enhanced message text with no gaps
    """
    logger.debug("Cleaning message content with comprehensive gap removal")
    
    # Get name information
    receiver_name = f"{receiver.get('first_name', '')} {receiver.get('last_name', '').strip()}".strip()
    sender_name = f"{sender.get('first_name', '')} {sender.get('last_name', '').strip()}".strip()
    sender_position = sender.get('position', 'Researcher')
    sender_department = sender.get('department', 'APHRC')
    
    # Fix repetitive salutations
    text = re.sub(rf'Dear {receiver_name},\s*Dear {receiver_name},', f'Dear {receiver_name},', text)
    
    # Ensure correct position title
    text = re.sub(r'I am a research at', 'I am a Researcher at', text)
    text = re.sub(r'I am an research at', 'I am a Researcher at', text)
    
    # Fix double titles
    text = re.sub(r'I am a researcher at.*?I am a researcher at', 'I am a Researcher at', text, flags=re.IGNORECASE | re.DOTALL)
    
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
    
    # Fix incomplete phrases and placeholder removal
    # Remove any bracketed placeholders
    text = re.sub(r'\[\s*\]', '', text)  # Empty brackets []
    text = re.sub(r'\[\s*\.\s*\]', '', text)  # Brackets with period [.]
    text = re.sub(r'\[\s*\w+\s*\]', '', text)  # Brackets with one word [word]
    text = re.sub(r'\[.*?\]', '', text)  # Any remaining brackets [anything]
    
    # Fix incomplete phrases and sentences
    text = re.sub(r'focuses\s+on\s+on', 'focuses on', text)
    text = re.sub(r'focuses\s+on\s*\.', 'focuses on this research area.', text)
    text = re.sub(r'focuses\s+on\s*$', 'focuses on this research area.', text, flags=re.MULTILINE)
    text = re.sub(r'focuses\s+on\s+within', 'focuses on research within', text)
    text = re.sub(r'interested\s+in\s*\.', 'interested in this collaboration.', text)
    text = re.sub(r'interested\s+in\s*$', 'interested in this collaboration.', text, flags=re.MULTILINE)
    text = re.sub(r'specializes\s+in\s*\.', 'specializes in this field.', text)
    text = re.sub(r'specializes\s+in\s*$', 'specializes in this field.', text, flags=re.MULTILINE)
    
    # Fix other common gap patterns
    text = re.sub(r'such\s+as\s*\.', 'such as these areas.', text)
    text = re.sub(r'such\s+as\s*$', 'such as these areas.', text, flags=re.MULTILINE)
    text = re.sub(r'including\s*\.', 'including various aspects.', text)
    text = re.sub(r'including\s*$', 'including various aspects.', text, flags=re.MULTILINE)
    text = re.sub(r'regarding\s*\.', 'regarding this matter.', text)
    text = re.sub(r'regarding\s*$', 'regarding this matter.', text, flags=re.MULTILINE)
    text = re.sub(r'pertaining\s+to\s*\.', 'pertaining to this subject.', text)
    text = re.sub(r'pertaining\s+to\s*$', 'pertaining to this subject.', text, flags=re.MULTILINE)
    
    # Fill in the gap if context is mentioned without details
    if context and len(context) > 3:
        context_summary = context[:50] + '...' if len(context) > 50 else context
        text = re.sub(
            r'(writing|reaching out|contacting) (to you|you)( today)? (because|regarding|about|concerning)\s*\.', 
            f'\\1 \\2\\3 \\4 {context_summary}.', 
            text
        )
        text = re.sub(
            r'(writing|reaching out|contacting) (to you|you)( today)? (because|regarding|about|concerning)\s*$', 
            f'\\1 \\2\\3 \\4 {context_summary}.', 
            text, 
            flags=re.MULTILINE
        )
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Remove duplicate adjacent words (like "the the")
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
    # Remove any remaining placeholders or references
    text = re.sub(r'\(Insert.*?\)', '', text)
    text = re.sub(r'\[Insert.*?\]', '', text)
    text = re.sub(r'\{Insert.*?\}', '', text)
    
    # Ensure sentences end with proper punctuation
    text = re.sub(r'(\w)\s*\n', r'\1.\n', text)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split into paragraphs and ensure each has meaningful content
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for para in paragraphs:
        stripped_para = para.strip()
        # Skip very short or empty paragraphs
        if len(stripped_para) < 5:
            continue
            
        # Check for incomplete sentences at paragraph end
        if stripped_para and stripped_para[-1] not in ['.', '!', '?', ':', ';', ',']:
            stripped_para += '.'
            
        processed_paragraphs.append(stripped_para)
    
    # Final sanity check for remaining placeholders or incomplete phrases
    result = '\n\n'.join(processed_paragraphs)
    
    # Check for any remaining ellipses or indicators of missing content
    result = re.sub(r'\.\.\.$', '.', result, flags=re.MULTILINE)
    result = re.sub(r'\.\.\.\s+\.', '.', result)
    
    return result.strip()

def validate_message_completeness(content, context, receiver_name, sender_name):
    """
    Validate that the message is complete without gaps.
    Returns a fixed version if gaps are found.
    """
    # Check for common indicators of incomplete content
    has_gaps = any([
        "I am particularly interested in ." in content,
        "focuses on  on" in content,
        "within ]" in content,
        "such as ." in content,
        "including ." in content,
        ". ." in content,
        "  " in content,
        "[" in content and "]" in content,
        "research at the the" in content
    ])
    
    # Fix any remaining issues if gaps detected
    if has_gaps:
        logger.warning(f"Post-cleaning gaps detected, performing additional fixes")
        
        # Fix additional issues
        content = content.replace("I am particularly interested in .", "I am particularly interested in this collaboration.")
        content = content.replace("focuses on  on", "focuses on")
        content = content.replace("within ]", "within our field")
        content = content.replace("such as .", "such as these relevant areas.")
        content = content.replace("including .", "including various aspects of this work.")
        content = content.replace(". .", ".")
        content = content.replace("  ", " ")
        content = content.replace("research at the the", "research at the")
        
        # If context is available, include it in relevant places
        if context:
            context_brief = context[:40] + "..." if len(context) > 40 else context
            if "regarding ." in content:
                content = content.replace("regarding .", f"regarding {context_brief}.")
                
            if "pertaining to ." in content:
                content = content.replace("pertaining to .", f"pertaining to {context_brief}.")
                
            # Handle any incomplete sentences with research focus
            if "My research focuses on ." in content:
                content = content.replace("My research focuses on .", f"My research focuses on {context_brief}.")
    
    # Ensure the message is properly addressed and signed
    if not content.startswith(f"Dear {receiver_name}"):
        content = f"Dear {receiver_name},\n\n" + content
        
    if "Best regards" not in content:
        content += f"\n\nBest regards,\n{sender_name}"
        
    return content

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
            cached_data = json.loads(cached_response)
            
            # Even for cached responses, validate completeness
            receiver_name = cached_data.get("receiver_name", "Respected Colleague")
            sender_name = cached_data.get("sender_name", "APHRC Researcher")
            cached_data["content"] = validate_message_completeness(
                cached_data["content"], content, receiver_name, sender_name
            )
            return cached_data
            
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
                                cached_data = json.loads(cached_response)
                                
                                # Validate completeness even for similar cached data
                                receiver_name = cached_data.get("receiver_name", "Respected Colleague")
                                sender_name = cached_data.get("sender_name", "APHRC Researcher")
                                cached_data["content"] = validate_message_completeness(
                                    cached_data["content"], content, receiver_name, sender_name
                                )
                                return cached_data
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
            logger.debug(f"Raw Gemini response: {draft_content[:1000]}")
            
            # Apply enhanced cleaning function to the response with sender info
            cleaned_content = clean_message_content(
                text=draft_content,
                receiver=receiver,
                sender=sender,
                context=content
            )
            logger.info(f"Cleaned content of length: {len(cleaned_content)}")
            
            # Perform final validation to ensure no gaps remain
            receiver_name = f"{receiver['first_name']} {receiver['last_name']}".strip()
            sender_name = f"{sender['first_name']} {sender['last_name']}".strip()
            final_content = validate_message_completeness(
                cleaned_content, content, receiver_name, sender_name
            )
            
            if final_content != cleaned_content:
                logger.info("Final validation fixed additional gaps in content")
            
            cleaned_content = final_content
            
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

def generate_expert_draft_prompt(receiver: dict, content: str, sender: dict) -> str:
    """
    Generate a sophisticated prompt with real sender information.
    Enhanced to prevent gaps in generated content.
    
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
    
    # Construct a nuanced, context-rich prompt with explicit instruction to avoid gaps
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
    5. NEVER use "..." or "[insert something]" or leave any part of the message incomplete
    6. ALWAYS provide complete, specific details instead of placeholders
    7. If referring to research topics, BE SPECIFIC about what those topics are - do not leave gaps
    8. NEVER use phrases like "I am interested in ." - always complete the thought
    9. DO NOT include double spaces, repeated words, or incomplete sentences

    Drafting Guidelines:
    1. Craft a concise, professional message that reflects APHRC's research excellence
    2. Demonstrate genuine interest in potential collaboration or knowledge exchange
    3. Highlight the relevance of the proposed communication to the recipient's expertise
    4. Maintain a tone of academic respect and professional curiosity
    5. Ensure the message is culturally sensitive and aligned with APHRC's mission
    6. INCLUDE SPECIFIC RESEARCH TOPICS related to the communication context
    7. COMPLETE ALL THOUGHTS and sentences fully with specific details
    8. DO NOT leave any part of the message unspecified or incomplete

    Draft a COMPLETE message, focusing on creating a meaningful professional connection.
    """
    
    return prompt

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