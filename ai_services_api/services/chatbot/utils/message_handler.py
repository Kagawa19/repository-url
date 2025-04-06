import asyncio
import logging
import re
import time
from typing import AsyncIterable, Optional, Dict
from datetime import datetime
from .llm_manager import GeminiLLMManager
from ai_services_api.services.message.core.database import get_db_connection
import asyncio
from typing import Dict, Any, AsyncGenerator
from ai_services_api.services.chatbot.utils.db_utils import DatabaseConnector
import json

logger = logging.getLogger(__name__)

class MessageHandler:
    def __init__(self, llm_manager):
        self.metadata = None
        self.llm_manager = llm_manager
    @staticmethod
    def clean_response_text(text: str) -> str:
        """
        Clean and format the response text with DOI preservation.
        
        Args:
            text (str): The input text with markdown formatting
            
        Returns:
            str: Cleaned and reformatted text
        """
        # Explicit DOI URL protection pattern
        doi_url_pattern = r'(https?://doi\.org/10\.\d+/[^\s]+)'
        doi_pattern = r'(10\.\d+/[^\s]+)'
        
        # If the entire text is a DOI or DOI URL, return it exactly
        if re.match(f'^({doi_url_pattern}|{doi_pattern})$', text.strip()):
            return text.strip()
        
        # Preserve DOI lines exactly
        if re.match(r'^DOI:', text.strip()):
            return text.strip()
        
        # Skip cleaning for numbered publication list items
        if re.match(r'^\d+\.\s+Title:', text.strip()):
            cleaned = text.replace('\n**', ' ')
            cleaned = cleaned.replace('**', '')
            return cleaned.strip()
        
        # Temporarily protect DOI URLs
        doi_matches = re.findall(doi_url_pattern, text)
        doi_placeholders = {}
        
        # Replace DOI URLs with unique placeholders
        for i, doi in enumerate(doi_matches):
            placeholder = f"__DOI_PLACEHOLDER_{i}__"
            doi_placeholders[placeholder] = doi
            text = text.replace(doi, placeholder)
        
        # Regular cleaning for other text
        cleaned = text.replace('\n**', ' ')
        cleaned = cleaned.replace('**', '')
        
        # Preserve numbered lists
        if not re.search(r'\n\d+\.', cleaned):
            cleaned = re.sub(r'\n\d+\.', '', cleaned)
        
        # Handle bullet points without converting DOIs
        bullet_points = re.findall(r'\*\s*([^*\n]+)', cleaned)
        if bullet_points:
            # Only convert bullets to flowing text outside of structured lists
            if "Title:" not in cleaned and "Authors:" not in cleaned:
                cleaned = re.sub(r'\*\s*[^*\n]+\n*', '', cleaned)
                bullet_list = ', '.join(point.strip() for point in bullet_points)
                if 'Key Findings:' in cleaned:
                    cleaned = cleaned.replace('Key Findings:', f'Key findings include: {bullet_list}.')
                else:
                    cleaned += f" {bullet_list}."
        
        # Fix spacing and formatting
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,:])', r'\1', cleaned)
        cleaned = cleaned.strip()
        
        # Add proper spacing after periods
        cleaned = re.sub(r'\.(?! )', '. ', cleaned)
        
        # Clean up special characters
        cleaned = cleaned.replace('\\n', ' ')
        cleaned = cleaned.strip()
        
        # Restore DOI URLs
        for placeholder, doi in doi_placeholders.items():
            cleaned = cleaned.replace(placeholder, doi)
        
        return cleaned
    
    async def process_stream_response(self, response_stream):
        """
        Process the streaming response with enhanced formatting, structure preservation, and metadata handling.
        
        Args:
            response_stream: Async generator of response chunks
            
        Yields:
            str or dict: Cleaned and formatted response chunks or metadata
        """
        buffer = ""
        metadata = None
        in_publication_list = False
        list_count = 0
        
        try:
            async for chunk in response_stream:
                # Capture metadata with improved detection
                if isinstance(chunk, dict) and chunk.get('is_metadata'):
                    # Store metadata both in method instance and for return
                    metadata = chunk.get('metadata', chunk)
                    self.metadata = metadata
                    
                    # Log metadata capture
                    logger.debug(f"Captured metadata: {json.dumps(metadata, default=str) if metadata else 'None'}")
                    continue
                
                # Extract text from chunk with enhanced format handling
                if isinstance(chunk, dict):
                    if 'chunk' in chunk:
                        text = chunk['chunk']
                    elif 'content' in chunk:
                        text = chunk['content']
                    elif 'text' in chunk:
                        text = chunk['text']
                    else:
                        # Try to stringify the dict as fallback
                        text = str(chunk)
                elif isinstance(chunk, (str, bytes)):
                    text = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                elif hasattr(chunk, 'content'):
                    # Handle AIMessage or similar objects
                    text = chunk.content
                elif hasattr(chunk, 'text'):
                    # Handle objects with text attribute
                    text = chunk.text
                else:
                    # Skip unknown formats
                    logger.debug(f"Skipping unknown chunk format: {type(chunk)}")
                    continue
                
                # Skip empty chunks
                if not text:
                    continue
                    
                buffer += text
                
                # Detect publication lists (unique preservation logic)
                list_pattern = r'\d+\.\s+Title:'
                if re.search(list_pattern, buffer) and not in_publication_list:
                    in_publication_list = True
                    list_count += 1
                
                # Unique list handling logic
                if in_publication_list:
                    # Try to find complete list items
                    items = re.split(r'(?=\d+\.\s+Title:)', buffer)
                    if len(items) > 1:
                        # Keep the last (potentially incomplete) item in buffer
                        complete_items = items[:-1]
                        buffer = items[-1]
                        
                        for item in complete_items:
                            if item.strip():
                                # Use cleaning method, but preserve DOI exactly
                                formatted_item = self.clean_response_text(item)
                                yield formatted_item
                else:
                    # Normal sentence processing
                    while '.' in buffer:
                        split_idx = buffer.find('.') + 1
                        sentence = self.clean_response_text(buffer[:split_idx])
                        buffer = buffer[split_idx:].lstrip()
                        
                        if sentence.strip():
                            yield sentence
            
            # Handle remaining buffer
            if buffer.strip():
                yield self.clean_response_text(buffer)
            
            # Yield metadata if exists
            if metadata:
                # Wrap in proper format if needed
                if not isinstance(metadata, dict) or 'is_metadata' not in metadata:
                    metadata_wrapper = {'is_metadata': True, 'metadata': metadata}
                    yield metadata_wrapper
                else:
                    yield metadata
                    
        except Exception as e:
            logger.error(f"Error processing stream response: {e}", exc_info=True)
            # Yield any remaining buffer to avoid losing content
            if buffer.strip():
                yield self.clean_response_text(buffer)
            
            # If metadata was captured but not yet yielded, yield it now
            if metadata and not isinstance(metadata, dict):
                yield {'is_metadata': True, 'metadata': metadata}

    
    async def start_chat_session(self, user_id: str) -> str:
        """
        Start a new chat session.
        
        Args:
            user_id (str): Unique identifier for the user
        
        Returns:
            str: Generated session identifier
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Generate unique session identifier
                    session_id = f"session_{user_id}_{int(time.time())}"
                    
                    try:
                        cursor.execute("""
                            INSERT INTO chat_sessions 
                                (session_id, user_id, start_timestamp)
                            VALUES (%s, %s, CURRENT_TIMESTAMP)
                            RETURNING session_id
                        """, (session_id, user_id))
                        
                        conn.commit()
                        logger.info(f"Created chat session: {session_id}")
                        return session_id
                    
                    except Exception as insert_error:
                        conn.rollback()
                        logger.error(f"Error inserting chat session: {insert_error}")
                        raise
        
        except Exception as e:
            logger.error(f"Error in start_chat_session: {e}")
            raise

    async def send_message_async(self, message: str, user_id: str, session_id: str = None):
        """
        Process message and handle responses with enhanced rate limit handling.
        
        Args:
            message (str): The message to be processed
            user_id (str): Unique identifier for the user
            session_id (str, optional): Unique identifier for the conversation session
        """
        start_time = time.time()
        logger.info("Starting async message processing")
        logger.debug(f"Message details - User ID: {user_id}, Session ID: {session_id}, Message length: {len(message)}")
        
        # Rate limit tracking
        rate_limit_encountered = False
        # Response tracking
        response_chunks = []
        has_error_occurred = False
        
        try:
            self.metadata = None
            logger.debug("Metadata reset before processing")
            
            # Check if LLM manager is in rate-limited state
            if hasattr(self.llm_manager, '_rate_limited') and self.llm_manager._rate_limited:
                logger.warning(f"LLM is in rate-limited state for user {user_id}")
                rate_limit_encountered = True
            
            # Additional local throttling if needed
            if not rate_limit_encountered and hasattr(self, '_local_rate_limit_until'):
                if time.time() < self._local_rate_limit_until:
                    logger.warning(f"Local rate limit active for another {self._local_rate_limit_until - time.time():.1f} seconds")
                    rate_limit_encountered = True
                else:
                    # Reset expired local rate limit
                    delattr(self, '_local_rate_limit_until')
            
            # If we're rate limited, consider alternative response strategies
            if rate_limit_encountered:
                # Check cache for similar queries
                cached_response = await self._check_response_cache(message, user_id)
                if cached_response:
                    logger.info("Using cached response due to rate limiting")
                    yield "Note: Using cached response due to high traffic. "
                    yield cached_response
                    return
            
            # Get the response stream from LLM
            try:
                raw_response_stream = self.llm_manager.generate_async_response(message)
            except Exception as e:
                # Check specifically for rate limit errors
                if any(x in str(e).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
                    logger.warning(f"Rate limit detected at stream initialization: {e}")
                    rate_limit_encountered = True
                    
                    # Set local rate limit cooldown
                    self._local_rate_limit_until = time.time() + 60  # 1 minute cooldown
                    
                    # Provide fallback response
                    yield "I apologize, but our service is experiencing high demand right now. Please try again in a moment."
                    has_error_occurred = True
                    return
                else:
                    raise
            
            # Initialize response tracking
            captured_metadata = None
            
            # Process and format the response stream
            try:
                async for chunk in self.process_stream_response(raw_response_stream):
                    # Check if this is a metadata chunk
                    if isinstance(chunk, dict) and chunk.get('is_metadata'):
                        captured_metadata = chunk['metadata']
                        
                        # Check if the response was generated in rate-limited mode
                        if captured_metadata.get('metrics', {}).get('rate_limited_mode', False):
                            rate_limit_encountered = True
                            logger.warning("Response was generated in rate-limited mode")
                        
                        logger.debug(f"Captured metadata: {json.dumps(captured_metadata, default=str)}")
                        continue
                        
                    logger.debug(f"Yielding formatted response chunk (length: {len(chunk)})")
                    response_chunks.append(chunk)
                    yield chunk
                    
            except Exception as stream_error:
                # Detect rate limit errors in streaming
                if any(x in str(stream_error).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
                    logger.warning(f"Rate limit detected during streaming: {stream_error}")
                    rate_limit_encountered = True
                    
                    # Set local rate limit cooldown
                    self._local_rate_limit_until = time.time() + 60  # 1 minute cooldown
                    
                    # If we have partial response, yield it with a note
                    if response_chunks:
                        complete_response = ''.join(response_chunks)
                        if len(complete_response) > 30:  # If we have a substantial partial response
                            yield "\n\nNote: Response was truncated due to high demand. Please try again in a moment."
                    else:
                        # Otherwise provide a fallback response
                        yield "I apologize, but our service is experiencing high demand right now. Please try again in a moment."
                        has_error_occurred = True
                    return
                else:
                    raise
            
            # Prepare complete response
            complete_response = ''.join(response_chunks)
            response_time = time.time() - start_time
            
            logger.info("Message processing completed successfully")
            logger.debug(f"Total processing time: {response_time:.2f} seconds")
            
            # Cache the response for future rate-limited situations if it's good quality
            if complete_response and len(complete_response) > 50:
                await self._cache_response(message, user_id, complete_response)
            
            # Save chat to database
            await self.save_chat_to_db(user_id, message, complete_response, response_time)
            
            # Prepare response data with either captured metadata or fallback to sentiment
            if captured_metadata and 'metrics' in captured_metadata:
                # Use the quality metrics from the LLM's response
                response_data = {
                    'response': complete_response,
                    'timestamp': captured_metadata.get('timestamp', datetime.utcnow().isoformat()),
                    'metrics': captured_metadata['metrics']
                }
                logger.debug("Using quality metrics from LLM metadata")
            else:
                # Fallback to sentiment analysis if no metadata was captured
                logger.warning("No metadata captured, falling back to sentiment analysis")
                sentiment_data = await self.llm_manager.analyze_sentiment(message)
                response_data = {
                    'response': complete_response,
                    'metrics': {
                        'response_time': response_time,
                        'sentiment': sentiment_data
                    }
                }
            
            # Record interaction with metrics
            await self.record_interaction(session_id, user_id, message, response_data)
            
        except Exception as e:
            logger.error("Critical error in message stream processing", exc_info=True)
            logger.error(f"Error details - User ID: {user_id}, Session ID: {session_id}")
            
            if hasattr(e, '__traceback__'):
                tb = e.__traceback__
                logger.error(f"Error occurred in file: {tb.tb_frame.f_code.co_filename}, line: {tb.tb_lineno}")
            
            # Check if this is a rate limit error
            if any(x in str(e).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
                logger.warning(f"Rate limit detected in exception handler: {e}")
                rate_limit_encountered = True
                
                # Set local rate limit cooldown
                self._local_rate_limit_until = time.time() + 120  # 2 minute cooldown
                
                error_message = "I apologize, but our service is experiencing high demand right now. Please try again in a few minutes."
            else:
                error_message = "I encountered an error processing your message. Please try again."
            
            # Only yield error message if no successful response has been generated yet
            if not response_chunks:
                logger.warning(f"Yielding error message to user: {error_message}")
                yield error_message
                has_error_occurred = True
            else:
                # If we have some content already, don't add an error message that would confuse the user
                logger.info("Error occurred but partial response exists; not appending error message")
            
        finally:
            # Update session stats
            if session_id:
                await self.update_session_stats(session_id, not (rate_limit_encountered or has_error_occurred))
            
            logger.info("Async message processing concluded")
            total_time = time.time() - start_time
            logger.debug(f"Total method execution time: {total_time:.2f} seconds")
    async def _check_response_cache(self, message: str, user_id: str) -> str:
        """Check for cached similar responses during rate limiting."""
        try:
            # Get Redis client
            from ai_services_api.services.chatbot.utils.redis_connection import redis_pool
            redis_client = await redis_pool.get_redis()
            
            # Generate a simplified message for similarity matching
            simple_message = ' '.join(word.lower() for word in message.split() 
                                        if len(word) > 3 and word.lower() not in 
                                        ('what', 'when', 'where', 'which', 'who', 'why', 'how', 
                                        'the', 'and', 'for', 'that', 'this', 'are', 'with'))
            
            # Look for similar cached queries
            pattern = f"chat:{user_id}:*"
            keys = await redis_client.keys(pattern)
            
            for key in keys:
                # Get the original query from the key
                try:
                    parts = key.split(':', 2)
                    if len(parts) < 3:
                        continue
                        
                    cached_query = parts[2]
                    
                    # Create simplified cached query for comparison
                    simple_cached = ' '.join(word.lower() for word in cached_query.split() 
                                                if len(word) > 3 and word.lower() not in 
                                                ('what', 'when', 'where', 'which', 'who', 'why', 'how', 
                                                'the', 'and', 'for', 'that', 'this', 'are', 'with'))
                    
                    # Check for significant overlap in key terms
                    if simple_message and simple_cached:
                        msg_terms = set(simple_message.split())
                        cached_terms = set(simple_cached.split())
                        
                        if len(msg_terms) > 0 and len(cached_terms) > 0:
                            overlap = len(msg_terms.intersection(cached_terms))
                            overlap_ratio = overlap / max(len(msg_terms), len(cached_terms))
                            
                            if overlap_ratio > 0.6:  # At least 60% term overlap
                                # Retrieve cached response
                                cached_data = await redis_client.get(key)
                                if cached_data:
                                    data = json.loads(cached_data)
                                    return data.get('response', '')
                except Exception as parse_error:
                    logger.error(f"Error parsing cache key: {parse_error}")
                    continue
                    
            return None
        except Exception as e:
            logger.error(f"Error checking response cache: {e}")
            return None

    async def _cache_response(self, message: str, user_id: str, response: str):
        """Cache response for future use during rate limiting."""
        try:
            # Get Redis client
            from ai_services_api.services.chatbot.utils.redis_connection import redis_pool
            redis_client = await redis_pool.get_redis()
            
            # Generate cache key
            redis_key = f"chat:{user_id}:{message}"
            
            # Prepare data with timestamp
            chat_data = {
                "response": response,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "cached_for_rate_limit": True
            }
            
            # Cache for 24 hours
            await redis_client.setex(redis_key, 86400, json.dumps(chat_data))
            logger.debug(f"Cached response for potential rate limit situations: {redis_key}")
        except Exception as e:
            logger.error(f"Error caching response: {e}")
        async def start_chat_session(self, user_id: str) -> str:
            """
            Start a new chat session.
            
            Args:
                user_id (str): Unique identifier for the user
            
            Returns:
                str: Generated session identifier
            """
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        # Generate unique session identifier
                        session_id = f"session_{user_id}_{int(time.time())}"
                        
                        try:
                            cursor.execute("""
                                INSERT INTO chat_sessions 
                                    (session_id, user_id, start_timestamp)
                                VALUES (%s, %s, CURRENT_TIMESTAMP)
                                RETURNING session_id
                            """, (session_id, user_id))
                            
                            conn.commit()
                            logger.info(f"Created chat session: {session_id}")
                            return session_id
                        
                        except Exception as insert_error:
                            conn.rollback()
                            logger.error(f"Error inserting chat session: {insert_error}")
                            raise
            
            except Exception as e:
                logger.error(f"Error in start_chat_session: {e}")
                raise

                
    async def save_chat_to_db(self, user_id: str, query: str, response: str, response_time: float):
        """Save chat interaction to database."""
        try:
            async with DatabaseConnector.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO chatbot_logs 
                        (user_id, query, response, response_time, timestamp)
                    VALUES ($1, $2, $3, $4, NOW())
                """, user_id, query, response, response_time)
        except Exception as e:
            logger.error(f"Error saving chat to database: {e}")

    
    async def record_interaction(self, session_id: str, user_id: str, query: str, response_data: dict):
        """
        Record response quality metrics for a chat interaction.
        
        Args:
            session_id (str): Unique identifier for the conversation session
            user_id (str): Unique identifier for the user
            query (str): The user's message
            response_data (dict): Contains response and metrics data
        """
        try:
            # Log the entire response_data dictionary to see what's being received
            logger.debug(f"Full response_data received in record_interaction: {json.dumps(response_data, default=str)}")
            
            async with DatabaseConnector.get_connection() as conn:
                # Start a transaction
                async with conn.transaction():
                    metrics = response_data.get('metrics', {})
                    
                    # Log the entire metrics dictionary
                    logger.debug(f"Full metrics dictionary: {json.dumps(metrics, default=str)}")
                    
                    # Get the chat log id from the chatbot_logs table
                    chat_log_result = await conn.fetchrow("""
                        SELECT id FROM chatbot_logs 
                        WHERE user_id = $1 AND query = $2 
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, user_id, query)
                    
                    if not chat_log_result:
                        logger.warning("Corresponding chat log not found for response quality metrics")
                        logger.debug(f"Query parameters - user_id: {user_id}, query: {query}")
                        return
                    
                    log_id = chat_log_result['id']
                    
                    # Get response quality data from metrics
                    quality_data = metrics.get('quality', {})
                    
                    # Log the extracted quality_data
                    logger.debug(f"Extracted quality_data: {json.dumps(quality_data, default=str)}")
                    
                    # Check if quality data is in a different location as fallback
                    if not quality_data and 'sentiment' in metrics:
                        logger.warning("No quality data found, checking sentiment data as fallback")
                        quality_data = metrics.get('sentiment', {})
                        logger.debug(f"Using sentiment data instead: {json.dumps(quality_data, default=str)}")
                    
                    # Extract values with explicit logging of each
                    helpfulness_score = quality_data.get('helpfulness_score', 0.0)
                    logger.debug(f"Extracted helpfulness_score: {helpfulness_score}")
                    
                    hallucination_risk = quality_data.get('hallucination_risk', 0.0)
                    logger.debug(f"Extracted hallucination_risk: {hallucination_risk}")
                    
                    factual_grounding_score = quality_data.get('factual_grounding_score', 0.0)
                    logger.debug(f"Extracted factual_grounding_score: {factual_grounding_score}")
                    
                    # REMOVED: unclear_elements and potentially_fabricated_elements
                    
                    # Modified SQL - removed columns that don't exist in the database
                    await conn.execute("""
                        INSERT INTO response_quality_metrics
                            (interaction_id, helpfulness_score, hallucination_risk, 
                            factual_grounding_score)
                        VALUES ($1, $2, $3, $4)
                    """,
                        log_id,  # Using chatbot_logs.id instead of interaction_id
                        helpfulness_score,
                        hallucination_risk,
                        factual_grounding_score
                    )
                    
                    logger.info(f"Recorded response quality metrics for chat log ID: {log_id}")
                    logger.info(f"Inserted values - helpfulness: {helpfulness_score}, hallucination: {hallucination_risk}, factual: {factual_grounding_score}")
        
        except Exception as e:
            logger.error(f"Error recording interaction quality metrics: {e}", exc_info=True)
            # Printing the full exception for debugging
            import traceback
            logger.error(f"Detailed traceback: {traceback.format_exc()}")
            raise

    async def update_session_stats(self, session_id: str, successful: bool = True):
        """Update session statistics with async database."""
        try:
            async with DatabaseConnector.get_connection() as conn:
                await conn.execute("""
                    UPDATE chat_sessions 
                    SET total_messages = total_messages + 1,
                        successful = $1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = $2
                """, successful, session_id)
                
                logger.info(f"Updated session stats for {session_id}")
                
        except Exception as e:
            logger.error(f"Error updating session stats: {e}", exc_info=True)
            raise

    async def _create_error_metadata(self, start_time: float, error_type: str) -> Dict:
        """Create standardized error metadata."""
        return {
            'metrics': {
                'response_time': time.time() - start_time,
                'intent': {'type': 'error', 'confidence': 0.0},
                'sentiment': {
                    'sentiment_score': 0.0,
                    'emotion_labels': ['error'],
                    'aspects': {
                        'satisfaction': 0.0,
                        'urgency': 0.0,
                        'clarity': 0.0
                    }
                },
                'content_matches': [],
                'content_types': {
                    'navigation': 0,
                    'publication': 0
                },
                'error_type': error_type
            },
            'error_occurred': True
        }

    async def _cache_response(self, message: str, user_id: str, response: str):
        """Cache response for future use during rate limiting."""
        try:
            # Get Redis client
            from ai_services_api.services.chatbot.utils.redis_connection import redis_pool
            redis_client = await redis_pool.get_redis()
            
            # Generate cache key
            redis_key = f"chat:{user_id}:{message}"
            
            # Prepare data with timestamp
            chat_data = {
                "response": response,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "cached_for_rate_limit": True
            }
            
            # Cache for 24 hours
            await redis_client.setex(redis_key, 86400, json.dumps(chat_data))
            logger.debug(f"Cached response for potential rate limit situations: {redis_key}")
        except Exception as e:
            logger.error(f"Error caching response: {e}")

    async def update_content_click(self, interaction_id: int, content_id: str):
        """
        Update when a user clicks on a content match.
        
        Args:
            interaction_id (int): Interaction identifier
            content_id (str): Clicked content identifier
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    try:
                        cursor.execute("""
                            UPDATE chat_analytics 
                            SET clicked = true 
                            WHERE interaction_id = %s AND content_id = %s
                        """, (interaction_id, content_id))
                        conn.commit()
                    except Exception as update_error:
                        conn.rollback()
                        logger.error(f"Error updating content click: {update_error}")
                        raise
        except Exception as e:
            logger.error(f"Error in update_content_click: {e}")
            raise
            
    async def flush_conversation_cache(self, conversation_id: str):
        """Clears the conversation history stored in the memory."""
        try:
            memory = self.llm_manager.create_memory()
            memory.clear()
            logger.info(f"Successfully flushed conversation cache for ID: {conversation_id}")
        except Exception as e:
            logger.error(f"Error while flushing conversation cache for ID {conversation_id}: {e}")
            raise RuntimeError(f"Failed to clear conversation history: {str(e)}")
