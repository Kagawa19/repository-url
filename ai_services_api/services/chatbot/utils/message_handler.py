import asyncio
import logging
import random
import re
import time
from typing import AsyncIterable, List, Optional, Dict, Tuple
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
        Process the streaming response with enhanced formatting, structure preservation,
        and improved natural language elements.
        
        Args:
            response_stream: Async generator of response chunks
            
        Yields:
            str or dict: Cleaned and formatted response chunks with natural language improvements
        """
        buffer = ""
        metadata = None
        in_publication_list = False
        list_count = 0
        publication_buffer = ""
        
        # Track intent information for content-aware enhancements
        detected_intent = None
        is_first_chunk = True
        transition_inserted = False
        
        try:
            async for chunk in response_stream:
                # Capture metadata with improved detection
                if isinstance(chunk, dict) and chunk.get('is_metadata'):
                    # Store metadata both in method instance and for internal use
                    metadata = chunk.get('metadata', chunk)
                    self.metadata = metadata
                    
                    # Extract intent information if available for content-aware styling
                    if metadata and 'intent' in metadata:
                        detected_intent = metadata.get('intent')
                        logger.debug(f"Detected intent for response styling: {detected_intent}")
                    
                    # Log metadata capture but don't yield it to user
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
                    text = chunk.decode('utf8') if isinstance(chunk, bytes) else chunk
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
                
                # Add natural language improvements for first chunk based on intent
                if is_first_chunk and detected_intent and not transition_inserted:
                    # Prepare intent-specific natural introductions
                    intro_text = ""
                    if detected_intent == "publication":
                        intro_prefix = self._get_random_intro("publication")
                        if intro_prefix and not text.startswith(intro_prefix):
                            intro_text = intro_prefix
                    elif detected_intent == "expert":
                        intro_prefix = self._get_random_intro("expert")
                        if intro_prefix and not text.startswith(intro_prefix):
                            intro_text = intro_prefix
                    elif detected_intent == "navigation":
                        intro_prefix = self._get_random_intro("navigation")
                        if intro_prefix and not text.startswith(intro_prefix):
                            intro_text = intro_prefix
                    
                    # Insert introduction if we have one
                    if intro_text:
                        yield intro_text
                        transition_inserted = True
                    
                    is_first_chunk = False
                    
                buffer += text
                
                # Detect publication lists
                if re.search(r'\d+\.\s+Title:', buffer) or re.search(r'Title:', buffer):
                    if not in_publication_list:
                        in_publication_list = True
                        
                        # Add natural transition for publication lists if not already done
                        if not transition_inserted and detected_intent == "publication":
                            transition = self._get_random_transition("publication_list")
                            if transition:
                                yield transition
                                transition_inserted = True
                        
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
                                        # Apply enhanced text cleaning for publication entries
                                        cleaned_entry = self._clean_text_for_user(entry)
                                        yield cleaned_entry
                                        
                                        # Add transition between publications if appropriate
                                        if detected_intent == "publication" and random.random() < 0.5:
                                            yield self._get_random_transition("between_publications")
                                
                                # Keep the last entry in the buffer
                                publication_buffer = entries[-1]
                            
                        # Check if entry appears complete (has multiple fields and ending line)
                        elif len(re.findall(r'(Title:|Authors:|Publication Year:|DOI:|Abstract:|Summary:)', publication_buffer)) >= 3 and '\n\n' in publication_buffer:
                            # This looks like a complete entry
                            # Apply enhanced text cleaning
                            cleaned_entry = self._clean_text_for_user(publication_buffer)
                            yield cleaned_entry
                            publication_buffer = ""
                            in_publication_list = False
                            
                            # Add transition after publication if appropriate
                            if detected_intent == "publication" and random.random() < 0.3:
                                yield self._get_random_transition("after_publication")
                            
                        continue
                
                # If we were in a publication list but now we're not, yield the publication buffer
                if in_publication_list and not re.search(r'(Title:|Authors:|Publication Year:|DOI:|Abstract:|Summary:)', text):
                    if publication_buffer.strip():
                        # Apply enhanced text cleaning
                        cleaned_entry = self._clean_text_for_user(publication_buffer)
                        yield cleaned_entry
                        
                        # Add appropriate closing transition
                        if detected_intent == "publication":
                            yield self._get_random_transition("after_publications")
                    
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
                                # Apply enhanced text cleaning
                                cleaned_sentence = self._clean_text_for_user(sentence)
                                yield cleaned_sentence
                        
                        # Keep the last sentence in the buffer
                        buffer = sentences[-1]
            
            # Handle any remaining text in buffers
            if publication_buffer.strip():
                cleaned_text = self._clean_text_for_user(publication_buffer)
                yield cleaned_text
                
                # Add appropriate closing for publication list
                if detected_intent == "publication":
                    yield self._get_random_transition("publication_conclusion")
                    
            elif buffer.strip():
                cleaned_text = self._clean_text_for_user(buffer)
                yield cleaned_text
                
                # Add appropriate closing based on intent
                if detected_intent:
                    yield self._get_random_transition(f"{detected_intent}_conclusion")
                    
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
        Removes technical artifacts, properly formats markdown, and 
        improves language quality and readability.
        
        Args:
            text (str): The input text that may contain technical artifacts or raw markdown
            
        Returns:
            str: Clean, well-formatted text suitable for user display with improved readability
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
            
            # Enhance publication formatting for better readability
            enhanced_lines = []
            for line in lines:
                # Improve Title formatting
                if line.startswith('Title:'):
                    title_content = line[6:].strip()
                    if title_content and not title_content.endswith(('.', '?', '!')):
                        # Add period if title doesn't end with punctuation
                        title_content += '.'
                    enhanced_lines.append(f'Title: {title_content}')
                
                # Improve Authors formatting with proper joining
                elif line.startswith('Authors:'):
                    authors_content = line[8:].strip()
                    if ',' in authors_content and ' and ' not in authors_content and authors_content.count(',') > 1:
                        # Fix author lists that have commas but no "and"
                        last_comma = authors_content.rindex(',')
                        authors_content = authors_content[:last_comma] + ' and' + authors_content[last_comma+1:]
                    enhanced_lines.append(f'Authors: {authors_content}')
                
                # Improve Abstract for readability
                elif line.startswith('Abstract:'):
                    abstract_content = line[9:].strip()
                    if abstract_content:
                        # Ensure abstract has proper sentence structure
                        if not abstract_content.endswith(('.', '?', '!')):
                            abstract_content += '.'
                        
                        # Split overly long abstracts into better formatted paragraphs
                        if len(abstract_content) > 200:
                            sentences = re.split(r'(?<=[.!?])\s+', abstract_content)
                            formatted_abstract = []
                            current_paragraph = ""
                            
                            for sentence in sentences:
                                if len(current_paragraph) + len(sentence) > 100:
                                    if current_paragraph:
                                        formatted_abstract.append(current_paragraph.strip())
                                    current_paragraph = sentence
                                else:
                                    if current_paragraph:
                                        current_paragraph += " " + sentence
                                    else:
                                        current_paragraph = sentence
                            
                            if current_paragraph:
                                formatted_abstract.append(current_paragraph.strip())
                            
                            abstract_content = "\n  ".join(formatted_abstract)
                            enhanced_lines.append(f'Abstract: \n  {abstract_content}')
                        else:
                            enhanced_lines.append(f'Abstract: {abstract_content}')
                    else:
                        enhanced_lines.append(line)
                else:
                    enhanced_lines.append(line)
            
            cleaned = '\n'.join(enhanced_lines)
            
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
                bullet_list = []
                
                # Improve bullet point formatting
                for point in bullet_points:
                    point = point.strip()
                    if point:
                        # Ensure proper sentence structure in bullet points
                        if not point.endswith(('.', '?', '!')):
                            point += '.'
                        bullet_list.append(point)
                
                # Combine bullet points with better transitions
                if bullet_list:
                    if 'Key Findings:' in cleaned:
                        if len(bullet_list) > 1:
                            bullet_text = f"Key findings include: {bullet_list[0]} Additionally, {' '.join(bullet_list[1:])}"
                        else:
                            bullet_text = f"Key findings include: {bullet_list[0]}"
                        cleaned = cleaned.replace('Key Findings:', bullet_text)
                    else:
                        if len(bullet_list) > 1:
                            bullet_text = f"{bullet_list[0]} Additionally, {' '.join(bullet_list[1:])}"
                        else:
                            bullet_text = bullet_list[0]
                        cleaned += f" {bullet_text}"
        
        # Fix spacing and formatting
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,:])', r'\1', cleaned)
        cleaned = cleaned.strip()
        
        # Add proper spacing after periods
        cleaned = re.sub(r'\.(?! )', '. ', cleaned)
        
        # Clean up special characters
        cleaned = cleaned.replace('\\n', ' ')
        
        # Improve sentence structures for better readability
        # Fix sentences that start with lowercase after periods
        cleaned = re.sub(r'\.\s+([a-z])', lambda m: f". {m.group(1).upper()}", cleaned)
        
        # Fix common awkward phrasings
        cleaned = cleaned.replace(' i.e. ', ', specifically, ')
        cleaned = cleaned.replace(' e.g. ', ', for example, ')
        cleaned = cleaned.replace(' etc.', ', and similar resources.')
        
        # Restore DOI URLs
        for placeholder, doi in doi_placeholders.items():
            cleaned = cleaned.replace(placeholder, doi)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        # Fix double spacing
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        return cleaned
    
    async def send_message_async(self, message: str, user_id: str, session_id: str) -> AsyncGenerator:
        """
        Stream messages with enhanced conversation awareness, contextual references,
        improved flow between multiple interactions, and user interest tracking.
        """
        logger.info(f"Starting send_message_async - User: {user_id}, Session: {session_id}")
        logger.info(f"Message content: {message[:50]}... (truncated)")
        
        try:
            # Retrieve conversation history for this session
            conversation_history = await self._get_conversation_history(user_id, session_id)
            logger.info(f"Retrieved conversation history with {len(conversation_history)} previous turns")
            
            # Analyze if this is a follow-up question
            is_followup, previous_topic = self._analyze_followup(message, conversation_history)
            if is_followup:
                logger.info(f"Detected follow-up question about: {previous_topic}")
            
            # Log intent detection start
            logger.info("Detecting message intent")
            intent_result = await self.llm_manager.detect_intent(message)
            intent_type = intent_result.get('intent', 'unknown')
            logger.info(f"Detected intent: {intent_type}, confidence: {intent_result.get('confidence', 0)}")
            
            # Determine interaction category for logging
            interaction_category = 'general'
            if hasattr(intent_type, 'value'):  # Handle enum type
                intent_value = intent_type.value
                if intent_value == 'publication':
                    interaction_category = 'publication'
                elif intent_value == 'expert':
                    interaction_category = 'expert'
            else:  # Handle string type
                if str(intent_type).lower() == 'publication':
                    interaction_category = 'publication'
                elif str(intent_type).lower() == 'expert':
                    interaction_category = 'expert'
            
            # Get the interest tracker
            interest_tracker = self._get_interest_tracker()
            
            # Use it for logging
            await interest_tracker.log_interaction(
                user_id=user_id,
                session_id=session_id,
                query=message,
                interaction_type=interaction_category
            )
            
            # Special handling for publication and expert list requests
            if "list" in message.lower() and "publication" in message.lower():
                logger.info("Detected publication list request - will apply special formatting")
            elif "list" in message.lower() and ("expert" in message.lower() or "researcher" in message.lower()):
                logger.info("Detected expert list request - will apply special formatting")
            
            # Enhance the message with conversation context if this is a follow-up
            enhanced_message = message
            if is_followup and previous_topic and conversation_history:
                enhanced_message = self._enhance_message_with_context(message, previous_topic, conversation_history)
                logger.info(f"Enhanced message with conversation context: {enhanced_message[:50]}... (truncated)")
            
            # Use the interest tracker to build context
            user_interests = await interest_tracker.build_user_interest_context(user_id)
            if user_interests:
                # Only add interests context for relevant intents
                if interaction_category != 'general':
                    enhanced_message = f"{user_interests}\n\n{enhanced_message}"
                    logger.info("Enhanced message with user interest context")
            
            # Create a tracker for content shown to the user
            content_tracker = {
                'publications': [],
                'experts': [],
                'topics': [],
                'domains': [],
                'expertise': []
            }
            
            # Start the async response generator
            logger.info("Starting async response generation")
            response_generator = self.llm_manager.generate_async_response(enhanced_message)
            
            # Add conversation-aware introduction for follow-ups if appropriate
            if is_followup and len(conversation_history) > 0:
                intro = self._get_followup_introduction(previous_topic)
                if intro:
                    yield intro
            
            # Process and yield the main response
            async for part in self.process_stream_response(response_generator):
                # Extract any content information from the response metadata
                if isinstance(part, dict) and not part.get('is_metadata', False):
                    # Check for publication data
                    if 'publications' in part:
                        content_tracker['publications'].extend(part['publications'])
                    # Check for expert data
                    if 'experts' in part:
                        content_tracker['experts'].extend(part['experts'])
                
                # Check metadata in self.metadata if set by process_stream_response
                if hasattr(self, 'metadata') and self.metadata:
                    # This could contain information about shown content
                    meta = self.metadata
                    if isinstance(meta, dict):
                        if 'publications' in meta:
                            content_tracker['publications'].extend(meta['publications'])
                        if 'experts' in meta:
                            content_tracker['experts'].extend(meta['experts'])
                
                # Log the type and partial content of each processed part
                part_type = type(part).__name__
                if isinstance(part, dict):
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
            
            # Process any tracked content after response is complete
            if content_tracker['publications']:
                # Extract topics and domains from publications
                topics, domains = await interest_tracker.extract_topics_from_publications(
                    content_tracker['publications']
                )
                
                # Track these as user interests
                if topics:
                    await interest_tracker.track_topic_interests(
                        user_id, topics, 'publication_topic'
                    )
                    content_tracker['topics'] = topics
                    
                if domains:
                    await interest_tracker.track_topic_interests(
                        user_id, domains, 'publication_domain'
                    )
                    content_tracker['domains'] = domains
            
            if content_tracker['experts']:
                # Extract expertise areas from experts
                expertise = await interest_tracker.extract_expertise_from_experts(
                    content_tracker['experts']
                )
                
                # Track these as user interests
                if expertise:
                    await interest_tracker.track_topic_interests(
                        user_id, expertise, 'expert_expertise'
                    )
                    content_tracker['expertise'] = expertise
            
            # Add a coherent closing based on conversation context if appropriate
            if random.random() < 0.3:  # Only add this sometimes to avoid being repetitive
                yield self._get_conversation_closing(intent_type, len(conversation_history))
                
            # Save this interaction to conversation history
            await self._save_to_conversation_history(user_id, session_id, message, intent_type)
                
            logger.info("Completed send_message_async stream generation")
            
        except Exception as e:
            logger.error(f"Error in send_message_async: {e}", exc_info=True)
            yield "I apologize, but I encountered an issue processing your request. Could you please try again or rephrase your question?"
    
    def _get_interest_tracker(self):
        """Lazy initialization of the interest tracker."""
        if not hasattr(self, 'interest_tracker'):
            from ai_services_api.services.chatbot.utils.db_utils import DatabaseConnector
            from ai_services_api.services.chatbot.utils.interest_tracker import UserInterestTracker
            self.interest_tracker = UserInterestTracker(DatabaseConnector)
            logger.info("Initialized user interest tracker")
        return self.interest_tracker

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

    async def _save_to_conversation_history(self, user_id: str, session_id: str, message: str, intent_type):
        """
        Save the current interaction to conversation history.
        
        Args:
            user_id (str): User identifier
            session_id (str): Session identifier
            message (str): The user's query
            intent_type: Detected intent type (will be converted to string)
        """
        try:
            # Initialize conversation cache if it doesn't exist
            if not hasattr(self, 'conversation_cache'):
                self.conversation_cache = {}
            
            # Initialize session history if it doesn't exist
            if session_id not in self.conversation_cache:
                self.conversation_cache[session_id] = []
            
            # Convert intent_type to string if it's an enum
            if hasattr(intent_type, 'value'):
                intent_type_str = intent_type.value
            else:
                intent_type_str = str(intent_type)
            
            # Add current message to history (in-memory cache)
            self.conversation_cache[session_id].append({
                'query': message,
                'intent': intent_type_str,
                'timestamp': datetime.now()
            })
            
            # Limit history size in memory
            if len(self.conversation_cache[session_id]) > 10:
                self.conversation_cache[session_id] = self.conversation_cache[session_id][-10:]
            
            # Also update database history if available
            try:
                async with DatabaseConnector.get_connection() as conn:
                    await conn.execute("""
                        UPDATE chatbot_logs 
                        SET intent_type = $1, session_id = $2
                        WHERE user_id = $3 AND query = $4
                        AND timestamp = (
                            SELECT MAX(timestamp) 
                            FROM chatbot_logs 
                            WHERE user_id = $3 AND query = $4
                        )
                    """, intent_type_str, session_id, user_id, message)
            except Exception as db_error:
                logger.warning(f"Error updating conversation history in database: {db_error}")
                # Continue even if database update fails - we still have memory cache
        
        except Exception as e:
            logger.error(f"Error saving to conversation history: {e}")
            # Non-critical error, can continue without history

    def _get_conversation_closing(self, intent_type: str, history_length: int) -> str:
        """
        Get a contextually appropriate conversation closing based on intent and history.
        
        Args:
            intent_type (str): The type of intent for the current response
            history_length (int): Number of previous conversation turns
            
        Returns:
            str: A natural language closing appropriate for the context
        """
        # General closings for any topic
        general_closings = [
            "Is there anything else you'd like to know about APHRC?",
            "Can I help you with anything else today?",
            "Would you like more information on this or another topic?",
            "Do you have any other questions about APHRC's work?"
        ]
        
        # Intent-specific closings
        publication_closings = [
            "Would you like me to recommend more publications on this topic?",
            "Are you interested in other research areas from APHRC?",
            "Can I help you find more specific research on this subject?",
            "Would you like details about the researchers behind these publications?"
        ]
        
        expert_closings = [
            "Would you like more information about any of these experts?",
            "Are you interested in the specific publications of any of these researchers?",
            "Would you like to know about other experts in related fields?",
            "Can I help you connect with any of these APHRC experts?"
        ]
        
        navigation_closings = [
            "Can I help you navigate to any other sections of our website?",
            "Is there specific content you're looking for on the APHRC site?",
            "Would you like information about other APHRC resources?",
            "Is there anything else I can help you find on our website?"
        ]
        
        # For conversations with history, offer more personalized closings
        extended_closings = [
            "Based on our conversation, you might also be interested in APHRC's work on related topics. Would you like me to suggest some?",
            "Is there a specific aspect of what we've discussed that you'd like to explore further?",
            "I hope our conversation has been helpful. Is there anything else you'd like to clarify about APHRC?",
            "Would you like me to summarize the key points we've covered in our discussion?"
        ]
        
        # Select appropriate closings based on intent and conversation history
        if history_length >= 2:
            closings = extended_closings
        elif intent_type == "publication":
            closings = publication_closings
        elif intent_type == "expert":
            closings = expert_closings
        elif intent_type == "navigation":
            closings = navigation_closings
        else:
            closings = general_closings
        
        return random.choice(closings)

    def _get_followup_introduction(self, previous_topic: str) -> str:
        """
        Get a natural introduction for a follow-up response.
        
        Args:
            previous_topic (str): The topic from previous conversation
            
        Returns:
            str: A natural language introduction for a follow-up
        """
        followup_intros = [
            f"Regarding {previous_topic}, ",
            f"Continuing our discussion about {previous_topic}, ",
            f"To follow up on {previous_topic}, ",
            f"Building on what we discussed about {previous_topic}, ",
            f"To address your follow-up question about {previous_topic}, "
        ]
        
        # Only return an intro sometimes to avoid being repetitive
        if random.random() < 0.7:  # 70% chance
            return random.choice(followup_intros)
        else:
            return ""

    def _enhance_message_with_context(self, message: str, previous_topic: str, conversation_history: List[Dict]) -> str:
        """
        Enhance a follow-up message with relevant context from previous conversation.
        
        Args:
            message (str): The current user message
            previous_topic (str): The identified previous topic
            conversation_history (List[Dict]): Previous interactions
            
        Returns:
            str: Enhanced message with relevant context
        """
        if not conversation_history:
            return message
        
        # Get the last 1-2 turns for relevant context
        recent_context = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        
        # Extract the most relevant previous turn
        last_turn = recent_context[-1]
        last_query = last_turn.get('query', '')
        
        # Extract key information from previous response
        last_response = last_turn.get('response', '')
        
        # Create a condensed context summary
        context_summary = f"This question refers to our previous discussion about {previous_topic}. "
        context_summary += f"Previous question: '{last_query}'. "
        
        # Add condensed previous response if not too long
        # (Just include the beginning to help establish context without overwhelming)
        if last_response:
            response_summary = last_response[:200] + "..." if len(last_response) > 200 else last_response
            context_summary += f"My previous response was about {previous_topic}."
        
        # Format the enhanced message
        enhanced_message = f"{context_summary}\n\nCurrent question: {message}"
        
        return enhanced_message
    
    async def _get_conversation_history(self, user_id: str, session_id: str) -> List[Dict]:
        """
        Retrieve conversation history for the current session.
        
        Args:
            user_id (str): User identifier
            session_id (str): Session identifier
            
        Returns:
            List[Dict]: Previous interactions in this conversation
        """
        try:
            # Try to get history from database
            async with DatabaseConnector.get_connection() as conn:
                # Get the last 5 interactions for this session, ordered by timestamp
                history = await conn.fetch("""
                    SELECT query, response, intent_type, timestamp
                    FROM chatbot_logs
                    WHERE user_id = $1 AND session_id = $2
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, user_id, session_id)
                
                # Format history as a list of dictionaries
                conversation = []
                for item in history:
                    conversation.append({
                        'query': item.get('query', ''),
                        'response': item.get('response', ''),
                        'intent': item.get('intent_type', 'unknown'),
                        'timestamp': item.get('timestamp', datetime.now())
                    })
                
                # Return reversed to get chronological order
                return list(reversed(conversation))
        
        except Exception as e:
            logger.warning(f"Error retrieving conversation history: {e}")
            # Fall back to any conversation history stored in instance
            if hasattr(self, 'conversation_cache'):
                return self.conversation_cache.get(session_id, [])
            return []

    def _analyze_followup(self, message: str, conversation_history: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Determine if the current message is a follow-up to previous conversation.
        
        Args:
            message (str): Current user message
            conversation_history (List[Dict]): Previous interactions
            
        Returns:
            Tuple[bool, Optional[str]]: Whether this is a follow-up and the previous topic
        """
        if not conversation_history:
            return False, None
        
        # Convert message to lowercase for analysis
        message_lower = message.lower().strip()
        
        # Check for explicit follow-up indicators
        followup_indicators = [
            'more about', 'tell me more', 'elaborate', 'expand', 'additional',
            'another', 'also', 'what about', 'and', 'further', 'furthermore',
            'additionally', 'besides', 'next', 'then', 'other', 'related',
            'similarly', 'likewise', 'too', 'as well', 'them', 'they', 'those',
            'these', 'that', 'this', 'it', 'he', 'she', 'their', 'this one'
        ]
        
        has_indicator = any(indicator in message_lower for indicator in followup_indicators)
        
        # Check for very short queries that likely rely on context
        is_short_query = len(message.split()) < 5
        
        # Check for questions without clear subjects
        has_dangling_reference = re.search(r'\b(it|they|them|those|these|this|that)\b', message_lower) is not None
        
        # Get the most recent conversation turn
        last_turn = conversation_history[-1] if conversation_history else None
        if not last_turn:
            return False, None
        
        
        previous_query = last_turn.get('query', '')
        previous_response = last_turn.get('response', '')
        previous_intent = last_turn.get('intent', '')
        
        # Extract key terms from previous query using simple NLP
        previous_terms = set(re.findall(r'\b\w{4,}\b', previous_query.lower()))
        current_terms = set(re.findall(r'\b\w{4,}\b', message_lower))
        
        # Check term overlap for topical continuity
        common_terms = previous_terms.intersection(current_terms)
        has_term_overlap = len(common_terms) > 0
        
        # Determine if this is likely a follow-up
        is_followup = (has_indicator or is_short_query or has_dangling_reference) and (has_term_overlap or len(common_terms) > 0)
        
        # Extract the probable topic of conversation
        if is_followup:
            # Try to extract topic from previous query
            key_terms = [term for term in previous_terms if len(term) > 5]
            if key_terms:
                topic = " and ".join(key_terms[:2])  # Use top 2 key terms
            else:
                # Fall back to intent type
                intent_topics = {
                    'publication': 'publications',
                    'expert': 'experts',
                    'navigation': 'website navigation',
                    'general': 'general information'
                }
                topic = intent_topics.get(previous_intent, 'your previous question')
            
            return True, topic
        
        return False, None
    
    
    
    def _get_random_transition(self, transition_type: str) -> str:
        """
        Get a random, natural-sounding transition phrase based on the transition type.
        
        Args:
            transition_type (str): The type of transition needed
            
        Returns:
            str: A natural language transition appropriate for the context
        """
        publication_list_transitions = [
            "Here are the publications I found: ",
            "Let me share the relevant publications with you: ",
            "I've compiled these publications that address your query: ",
            "Based on your interest, these publications stand out: ",
            "The following research publications might be helpful: "
        ]
        
        between_publications_transitions = [
            " Moving on to another relevant publication, ",
            " Another study you might find valuable is ",
            " Related to this, researchers also published ",
            " In a similar vein, ",
            " Additionally, "
        ]
        
        after_publication_transitions = [
            " This research provides valuable insights into this topic. ",
            " These findings have important implications for this field. ",
            " This work represents a significant contribution to the area. ",
            " The methodology used in this study was particularly rigorous. ",
            " The authors drew several important conclusions from this work. "
        ]
        
        after_publications_transitions = [
            "These publications collectively provide a comprehensive view of the topic. ",
            "Together, these studies highlight the important work APHRC is doing in this area. ",
            "This body of research demonstrates APHRC's commitment to evidence-based approaches. ",
            "These publications showcase the depth of APHRC's expertise in this field. ",
            "The findings from these studies have informed policy and practice in the region. "
        ]
        
        publication_conclusion_transitions = [
            "Would you like more specific information about any of these publications? ",
            "I can provide more details about specific aspects of this research if you're interested. ",
            "Is there a particular aspect of these publications you'd like to explore further? ",
            "Would you like to know about other related research from APHRC? ",
            "I hope these publications are helpful. Let me know if you need more specific information. "
        ]
        
        expert_conclusion_transitions = [
            "Would you like more information about any of these experts or their work? ",
            "I can provide more details about specific experts if you're interested. ",
            "Is there a particular expert whose work you'd like to explore further? ",
            "Would you like contact information for any of these researchers? ",
            "I hope this information about our experts is helpful. Let me know if you need anything else. "
        ]
        
        navigation_conclusion_transitions = [
            "I hope this helps you find what you're looking for on our website. ",
            "Is there anything specific within these sections you're trying to locate? ",
            "Would you like more detailed navigation instructions for any of these areas? ",
            "Let me know if you need help finding anything else on our website. ",
            "If you have any trouble accessing these resources, please let me know. "
        ]
        
        general_conclusion_transitions = [
            "Is there anything else you'd like to know about this topic? ",
            "I hope this information is helpful. Let me know if you have any other questions. ",
            "Would you like me to elaborate on any part of this response? ",
            "Is there a specific aspect of this topic you'd like to explore further? ",
            "Please let me know if you need any clarification or have follow-up questions. "
        ]
        
        # Select appropriate transitions based on type
        if transition_type == "publication_list":
            transitions = publication_list_transitions
        elif transition_type == "between_publications":
            transitions = between_publications_transitions
        elif transition_type == "after_publication":
            transitions = after_publication_transitions
        elif transition_type == "after_publications":
            transitions = after_publications_transitions
        elif transition_type == "publication_conclusion":
            transitions = publication_conclusion_transitions
        elif transition_type == "expert_conclusion":
            transitions = expert_conclusion_transitions
        elif transition_type == "navigation_conclusion":
            transitions = navigation_conclusion_transitions
        else:
            transitions = general_conclusion_transitions
        
        # Return a random transition
        return random.choice(transitions)

    def _get_random_intro(self, intent_type: str) -> str:
        """
        Get a random, natural-sounding introduction based on intent type.
        
        Args:
            intent_type (str): The type of intent detected
            
        Returns:
            str: A natural language introduction appropriate for the intent
        """
        publication_intros = [
            "I've found some relevant publications that might interest you. ",
            "Here are some APHRC publications related to your query. ",
            "Based on your question, these publications seem most relevant: ",
            "APHRC researchers have published several studies on this topic. ",
            "Let me share some publications that address your question. "
        ]
        
        expert_intros = [
            "I've identified APHRC experts who specialize in this area. ",
            "Several researchers at APHRC work on topics related to your question. ",
            "These APHRC experts might be able to provide insights on your question: ",
            "Let me introduce you to some APHRC experts in this field. ",
            "The following researchers at APHRC have expertise in this area: "
        ]
        
        navigation_intros = [
            "Let me help you find what you're looking for on the APHRC website. ",
            "I can guide you to the relevant sections of our website. ",
            "Here's how you can navigate to the information you need: ",
            "You can find this information in the following sections: ",
            "Let me point you to the right resources on our website. "
        ]
        
        general_intros = [
            "I'd be happy to help with that. ",
            "Let me address your question. ",
            "That's an interesting question about APHRC's work. ",
            "I can provide some information on that. ",
            "Thanks for your question about APHRC. "
        ]
        
        # Select appropriate introductions based on intent
        if intent_type == "publication":
            intros = publication_intros
        elif intent_type == "expert":
            intros = expert_intros
        elif intent_type == "navigation":
            intros = navigation_intros
        else:
            intros = general_intros
        
        # Return a random introduction
        return random.choice(intros)

    

    async def get_user_interests_for_response(self, user_id: str) -> str:
        """
        Retrieve user interests formatted for inclusion in response generation.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            str: Formatted user interests for inclusion in LLM context
        """
        try:
            # Skip if interest tracker is not available
            if not hasattr(self, 'interest_tracker'):
                return ""
                
            # Get user interests from tracker
            interests = await self.interest_tracker.get_user_interests(user_id)
            
            if not any(interests.values()):
                return ""
                
            # Format interests for the response context
            context_parts = []
            
            # Add publication topics if available
            if interests.get('publication_topic', []):
                topics_str = ", ".join(interests['publication_topic'][:3])
                context_parts.append(f"Research topics: {topics_str}")
                
            # Add publication domains if available
            if interests.get('publication_domain', []):
                domains_str = ", ".join(interests['publication_domain'][:3])
                context_parts.append(f"Research domains: {domains_str}")
                
            # Add expert expertise if available
            if interests.get('expert_expertise', []):
                expertise_str = ", ".join(interests['expert_expertise'][:3])
                context_parts.append(f"Expert areas: {expertise_str}")
                
            if not context_parts:
                return ""
                
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving user interests for response: {e}")
            return ""
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