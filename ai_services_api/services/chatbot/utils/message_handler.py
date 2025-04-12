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
        Clean and format the response text with improved structure preservation.
        
        Args:
            text (str): The input text with markdown formatting
            
        Returns:
            str: Cleaned and reformatted text that preserves structure for lists
        """
        # ⚠️ Changed: This method now calls our new helper method for text cleaning
        # This maintains backward compatibility while using enhanced cleaning
        return MessageHandler._clean_text_for_user(text)
    
    
    async def process_stream_response(self, response_stream):
        """
        Process the streaming response with enhanced formatting and structure preservation.
        
        Args:
            response_stream: Async generator of response chunks
            
        Yields:
            str or dict: Cleaned and formatted response chunks or metadata
        """
        buffer = ""
        metadata = None
        in_publication_list = False
        list_count = 0
        publication_buffer = ""
        
        try:
            async for chunk in response_stream:
                # Capture metadata with improved detection
                if isinstance(chunk, dict) and chunk.get('is_metadata'):
                    # Store metadata both in method instance and for internal use
                    metadata = chunk.get('metadata', chunk)
                    self.metadata = metadata
                    
                    # Log metadata capture but don't yield it to user
                    logger.debug(f"Captured metadata: {json.dumps(metadata, default=str) if metadata else 'None'}")
                    # ⚠️ Changed: Don't yield metadata to user, just store it internally
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
                
                # Detect publication lists
                if re.search(r'\d+\.\s+Title:', buffer) or re.search(r'Title:', buffer):
                    if not in_publication_list:
                        in_publication_list = True
                        publication_buffer = buffer
                        buffer = ""
                        continue
                    else:
                        publication_buffer += text
                        
                        # Check if we have a complete publication entry or multiple entries
                        if re.search(r'\n\d+\.', publication_buffer):
                            # Multiple entries detected, try to split
                            entries = re.split(r'(?=\d+\.\s+Title:)', publication_buffer)
                            if len(entries) > 1:
                                # Process all but the last entry
                                for entry in entries[:-1]:
                                    if entry.strip():
                                        # ⚠️ Changed: Apply enhanced text cleaning for publication entries
                                        cleaned_entry = self._clean_text_for_user(entry)
                                        yield cleaned_entry
                                
                                # Keep the last entry in the buffer
                                publication_buffer = entries[-1]
                            
                        # Check if entry appears complete (has multiple fields and ending line)
                        elif len(re.findall(r'(Title:|Authors:|Publication Year:|DOI:|Abstract:|Summary:)', publication_buffer)) >= 3 and '\n\n' in publication_buffer:
                            # This looks like a complete entry
                            # ⚠️ Changed: Apply enhanced text cleaning
                            cleaned_entry = self._clean_text_for_user(publication_buffer)
                            yield cleaned_entry
                            publication_buffer = ""
                            in_publication_list = False
                            
                        continue
                
                # If we were in a publication list but now we're not, yield the publication buffer
                if in_publication_list and not re.search(r'(Title:|Authors:|Publication Year:|DOI:|Abstract:|Summary:)', text):
                    if publication_buffer.strip():
                        # ⚠️ Changed: Apply enhanced text cleaning
                        cleaned_entry = self._clean_text_for_user(publication_buffer)
                        yield cleaned_entry
                    publication_buffer = ""
                    in_publication_list = False
                
                # Normal sentence processing for non-publication content
                if not in_publication_list:
                    # Check if buffer contains complete sentences
                    sentences = re.split(r'(?<=[.!?])\s+', buffer)
                    if len(sentences) > 1:
                        # Process all but the last sentence
                        for sentence in sentences[:-1]:
                            if sentence.strip():
                                # ⚠️ Changed: Apply enhanced text cleaning
                                cleaned_sentence = self._clean_text_for_user(sentence)
                                yield cleaned_sentence
                        
                        # Keep the last sentence in the buffer
                        buffer = sentences[-1]
            
            # Handle any remaining text in buffers
            if publication_buffer.strip():
                yield self._clean_text_for_user(publication_buffer)
            elif buffer.strip():
                yield self._clean_text_for_user(buffer)
            
            # ⚠️ Changed: Don't yield metadata to user, it's already been processed internally
            # Metadata is stored in self.metadata for internal use but not exposed to the user
                
        except Exception as e:
            logger.error(f"Error processing stream response: {e}", exc_info=True)
            # Yield any remaining buffer to avoid losing content
            if publication_buffer.strip():
                yield self._clean_text_for_user(publication_buffer)
            elif buffer.strip():
                yield self._clean_text_for_user(buffer)

    @staticmethod
    def _clean_text_for_user(text: str) -> str:
        """
        Enhanced text cleaning for user-facing responses.
        Removes technical artifacts and properly formats markdown.
        
        Args:
            text (str): The input text that may contain technical artifacts or raw markdown
            
        Returns:
            str: Clean, well-formatted text suitable for user display
        """
        if not text:
            return ""
        
        # Remove any JSON metadata that might have slipped through
        metadata_pattern = r'^\s*\{\"is_metadata\"\s*:\s*true.*?\}\s*'
        text = re.sub(metadata_pattern, '', text)
        
        # Check if text is part of a structured publication list
        is_publication_item = bool(re.search(r'(Title:|Authors:|Publication Year:|DOI:|Abstract:|Summary:|Research Domains:)', text))
        
        # Explicit DOI URL protection pattern
        doi_url_pattern = r'(https?://doi\.org/10\.\d+/[^\s]+)'
        doi_pattern = r'(10\.\d+/[^\s]+)'
        
        # If the entire text is a DOI or DOI URL, return it exactly
        if re.match(f'^({doi_url_pattern}|{doi_pattern})$', text.strip()):
            return text.strip()
        
        # Preserve DOI lines exactly
        if re.match(r'^DOI:', text.strip()):
            return text.strip()
        
        # Convert markdown formatting: replace ** with proper formatting or remove
        # In this implementation, we'll remove them for clean plain text
        # If UI supports formatting, this could be modified
        text = text.replace('***', '')
        text = text.replace('**', '')
        
        # Handle structured publication data differently
        if is_publication_item:
            # For publication data, preserve line structure but clean up extra whitespace
            cleaned = text
            
            # Fix line breaks to ensure each field is on its own line
            for field in ['Title:', 'Authors:', 'Publication Year:', 'DOI:', 'Abstract:', 'Summary:', 'Research Domains:', 'Citation:']:
                # Ensure field starts on a new line
                if field in cleaned and not re.search(f'(\n|^){field}', cleaned):
                    cleaned = cleaned.replace(field, f'\n{field}')
            
            # Clean up extra whitespace while preserving line structure
            lines = [line.strip() for line in cleaned.split('\n')]
            cleaned = '\n'.join(lines)
            
            return cleaned
        
        # Skip cleaning for numbered publication list items
        if re.match(r'^\d+\.\s+Title:', text.strip()):
            cleaned = text.replace('\n**', ' ')
            cleaned = cleaned.replace('**', '')
            
            # Ensure each field is on a new line
            for field in ['Title:', 'Authors:', 'Publication Year:', 'DOI:', 'Abstract:', 'Summary:', 'Research Domains:']:
                pattern = f'({field})'
                replacement = f'\n{field}'
                cleaned = re.sub(pattern, replacement, cleaned)
            
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

    def _extract_metadata(self, chunk):
        """
        Extract metadata from a response chunk for internal use.
        
        Args:
            chunk: A response chunk that may contain metadata
            
        Returns:
            dict or None: Extracted metadata if present, None otherwise
        """
        try:
            if isinstance(chunk, dict) and chunk.get('is_metadata'):
                # Extract the metadata portion
                metadata = chunk.get('metadata', {})
                # Store it in the instance for later use
                self.metadata = metadata
                logger.debug(f"Extracted metadata: {json.dumps(metadata, default=str) if metadata else 'None'}")
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return None
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

    def _format_publication_response(self, text):
        """Special formatter for publication list responses that creates clean, structured output."""
        # Remove markdown heading symbols
        text = re.sub(r'#{1,6}\s*', '', text)
        
        # Fix spaces in DOI links
        text = re.sub(r'(https?://doi\.org/\s*)([\d\.]+(/\s*)?[^\s\)]*)', lambda m: m.group(1).replace(' ', '') + m.group(2).replace(' ', ''), text)
        text = re.sub(r'(10\.\s*\d+\s*/\s*[^\s\)]+)', lambda m: m.group(1).replace(' ', ''), text)
        
        # Remove duplicate content at the end
        duplicate_patterns = [
            r'These publications address different aspects of.*?\.',
            r'Research Domains:.*?(?=\n|\Z)',
            r'Summary:.*?(?=\n|\Z)'
        ]
        
        for pattern in duplicate_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) > 1:
                # Keep only the first instance
                first_match_end = matches[0].end()
                for match in matches[1:]:
                    text = text[:match.start()] + text[match.end():]
                    
        # Extract the introduction
        intro_match = re.search(r'^(.*?)(?=\d+\.|$)', text, re.DOTALL)
        intro = intro_match.group(1).strip() if intro_match else ""
        
        # Split publications by numbered items
        items = re.split(r'(\d+\.)', text)
        if len(items) < 3:  # Not a properly structured list
            return text
        
        # Start with the introduction
        formatted_parts = [intro] if intro else []
        
        # Process each publication
        current_pub = ""
        for i, item in enumerate(items):
            if re.match(r'\d+\.', item):
                # Start of a new publication
                if current_pub:
                    formatted_parts.append(current_pub.strip())
                current_pub = item
            elif i > 0 and re.match(r'\d+\.', items[i-1]):
                # Content following a publication number
                # Extract and format key fields
                content = item.strip()
                
                # Format the title
                title_match = re.search(r'^(.+?)(?=\s*-\s*Authors:|\s*-\s*Publication|\s*-\s*DOI:|\s*-\s*Research|\s*-\s*Summary:|\Z)', content, re.DOTALL)
                title = title_match.group(1).strip() if title_match else content
                
                # Build structured output with fields aligned
                lines = [title]
                
                # Extract each field
                field_patterns = [
                    (r'-\s*Authors:\s*(.+?)(?=\s*-\s*|\Z)', "Authors: "),
                    (r'-\s*Publication Year:\s*(.+?)(?=\s*-\s*|\Z)', "Publication Year: "),
                    (r'-\s*DOI:\s*\[?(.+?)\]?(?=\s*-\s*|\Z)', "DOI: ")
                ]
                
                for pattern, field_name in field_patterns:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        field_content = match.group(1).strip()
                        # Clean up any markdown link formatting
                        field_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', field_content)
                        lines.append(f"{field_name}{field_content}")
                
                # Combine publication content
                current_pub += "\n" + "\n".join(lines)
        
        # Add the last publication
        if current_pub:
            formatted_parts.append(current_pub.strip())
        
        # Join with double newlines for clear separation
        formatted_text = "\n\n".join(formatted_parts)
        
        # Final cleanup - remove any remaining markdown or excess whitespace
        formatted_text = re.sub(r'\*\*|\*', '', formatted_text)  # Remove bold/italic markers
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)  # Normalize multiple newlines
        formatted_text = re.sub(r'\s+$', '', formatted_text, flags=re.MULTILINE)  # Remove trailing whitespace
        
        # Remove any summary and research domains sections if this is just meant to be a list
        if "list" in formatted_text.lower() or len(items) > 3:  # Likely a list request
            formatted_text = re.sub(r'\s*-\s*Summary:.*?(?=\n\n|\Z)', '', formatted_text)
            formatted_text = re.sub(r'\s*-\s*Research Domains:.*?(?=\n\n|\Z)', '', formatted_text)
        
        return formatted_text

    async def send_message_async(self, message: str, user_id: str, session_id: str) -> AsyncGenerator:
        """Stream messages with enhanced logging for debugging."""
        logger.info(f"Starting send_message_async - User: {user_id}, Session: {session_id}")
        logger.info(f"Message content: {message[:50]}... (truncated)")
        
        try:
            # Log intent detection start
            logger.info("Detecting message intent")
            intent_result = await self.llm_manager.detect_intent(message)
            intent_type = intent_result.get('intent', 'unknown')
            logger.info(f"Detected intent: {intent_type}, confidence: {intent_result.get('confidence', 0)}")
            
            # Special handling for publication and expert list requests
            if "list" in message.lower() and "publication" in message.lower():
                logger.info("Detected publication list request - will apply special formatting")
            elif "list" in message.lower() and ("expert" in message.lower() or "researcher" in message.lower()):
                logger.info("Detected expert list request - will apply special formatting")
                
            # Start the async response generator
            logger.info("Starting async response generation")
            response_generator = self.llm_manager.generate_async_response(message)
            
            # ⚠️ CRITICAL CHANGE: Process the response through process_stream_response
            # This ensures responses are cleaned and formatted properly before being sent to users
            logger.info("Processing response through cleaning pipeline")
            async for part in self.process_stream_response(response_generator):
                # The part is now cleaned by process_stream_response
                
                # Log the type and partial content of each processed part
                part_type = type(part).__name__
                
                if isinstance(part, dict):
                    # This should rarely happen now as metadata is filtered by process_stream_response
                    logger.info(f"Yielding processed dictionary part type: {part_type}")
                    logger.debug(f"Dictionary content: {str(part)[:100]}... (truncated)")
                elif isinstance(part, bytes):
                    logger.info(f"Yielding processed bytes part, length: {len(part)}")
                elif isinstance(part, str):
                    logger.info(f"Yielding processed string part, length: {len(part)}")
                    logger.debug(f"String content: {part[:50]}... (truncated)")
                else:
                    logger.warning(f"Unexpected processed part type: {part_type}")
                    logger.debug(f"Content representation: {str(part)[:100]}")
                    
                # Yield the cleaned part
                yield part
                
            logger.info("Completed send_message_async stream generation")
            
        except Exception as e:
            logger.error(f"Error in send_message_async: {e}", exc_info=True)
            yield f"Error processing message: {str(e)}"

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
                try:
                    # First, check if the table exists before attempting to insert
                    table_check_result = await conn.fetchrow("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'response_quality_metrics'
                        );
                    """)
                    
                    if not table_check_result or not table_check_result[0]:
                        logger.warning("Table 'response_quality_metrics' does not exist, skipping quality metrics recording")
                        return
                    
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
                        
                        # Insert quality metrics with proper exception handling
                        try:
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
                        except Exception as insert_error:
                            # Handle specific error cases if needed
                            logger.error(f"Database error: {str(insert_error)}")
                            
                            # If the error is about missing columns, we might try a different schema
                            if "column" in str(insert_error).lower() and "does not exist" in str(insert_error).lower():
                                logger.warning("Attempting alternative schema for quality metrics recording")
                                try:
                                    # Simplified schema with just the essential columns
                                    await conn.execute("""
                                        INSERT INTO response_quality_metrics
                                            (interaction_id, quality_score)
                                        VALUES ($1, $2)
                                    """,
                                        log_id,
                                        helpfulness_score
                                    )
                                    logger.info(f"Recorded simplified quality metrics for chat log ID: {log_id}")
                                except Exception as fallback_error:
                                    logger.error(f"Failed to record quality metrics using alternative schema: {fallback_error}")
                except Exception as db_error:
                    logger.error(f"Database error in record_interaction: {db_error}")
        
        except Exception as e:
            logger.error(f"Error recording interaction quality metrics: {e}", exc_info=True)
            # Do not re-raise the exception so the conversation flow is not disrupted

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
            # Check if LLM manager has a conversation history attribute
            if hasattr(self.llm_manager, 'context_window'):
                # Clear the context window
                self.llm_manager.context_window = []
                logger.info(f"Successfully flushed conversation context for ID: {conversation_id}")
            elif hasattr(self.llm_manager, 'conversation_history'):
                # Alternative attribute name
                self.llm_manager.conversation_history = []
                logger.info(f"Successfully flushed conversation history for ID: {conversation_id}")
            else:
                # If no direct memory attribute exists, try to find a method to clear it
                if hasattr(self.llm_manager, 'clear_history'):
                    await self.llm_manager.clear_history()
                    logger.info(f"Successfully cleared conversation history using clear_history() for ID: {conversation_id}")
                else:
                    logger.warning(f"No known method to clear conversation history for ID: {conversation_id}")
                    
            # Also clear any local cache in the MessageHandler if it exists
            if hasattr(self, 'cached_responses'):
                self.cached_responses = {}
            
            # Clear metadata from previous interactions
            self.metadata = None
            
            logger.info(f"Successfully flushed all conversation data for ID: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error while flushing conversation data for ID {conversation_id}: {e}")
            raise RuntimeError(f"Failed to clear conversation history: {str(e)}")