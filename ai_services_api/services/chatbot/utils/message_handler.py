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

    def _format_publication_field(self, entry_text: str, field_name: str, display_name: str = None) -> str:
        """
        Helper method to extract and format a specific field from a publication entry.
        
        Args:
            entry_text (str): The full text of the publication entry
            field_name (str): The name of the field to extract
            display_name (str, optional): Display name for the field. Defaults to field_name.
            
        Returns:
            str: Formatted field string with proper indentation and structure
        """
        if not display_name:
            display_name = field_name
            
        # Create various patterns to match different field formatting
        patterns = [
            rf'[-–•]?\s*\*\*{field_name}:\*\*\s*(.*?)(?=\s*[-–•]?\s*\*\*|\Z)',  # Bold marker format
            rf'[-–•]?\s*{field_name}:\s*(.*?)(?=\s*[-–•]?\s*|\Z)',              # Regular format
            rf'\b{field_name}\b:\s*(.*?)(?=\s*\b\w+\b:\s*|\Z)'                  # Simple format
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, entry_text, re.IGNORECASE | re.DOTALL)
            if match:
                field_content = match.group(1).strip()
                
                # Special handling for DOI links
                if field_name.lower() == 'doi':
                    # Clean the DOI string and create a proper link
                    doi_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', field_content)
                    doi_text = doi_text.replace(' ', '')
                    
                    # Check if it's a complete URL or just a DOI
                    if doi_text.startswith('http'):
                        return f"    - **{display_name}:** {doi_text}\n\n"
                    else:
                        return f"    - **{display_name}:** {doi_text}\n\n"
                
                # Truncate abstract if too long
                if field_name.lower() == 'abstract' and len(field_content) > 300:
                    field_content = field_content[:297] + "..."
                    
                # Return properly formatted field
                return f"    - **{display_name}:** {field_content}\n\n"
                
        # Return empty string if field not found
        return ""

    def _clean_structured_content(self, text: str, content_type: str) -> str:
        """
        Specialized cleaning for structured content like publication lists or expert profiles.
        Applies format-specific cleaning rules based on content type.
        
        Args:
            text (str): The structured content text
            content_type (str): Type of content ("list", "expert", "publication", etc.)
            
        Returns:
            str: Cleaned and properly formatted structured content
        """
        if not text or not content_type:
            return self._clean_text_for_user(text)  # Fallback to general cleaning
            
        # Remove JSON metadata that might have slipped through
        metadata_pattern = r'^\s*\{\"is_metadata\"\s*:\s*true.*?\}\s*'
        text = re.sub(metadata_pattern, '', text, flags=re.MULTILINE)
        
        # CRITICAL FIX: Remove any stray curly braces at the beginning or end of the text
        text = re.sub(r'^\s*\}\s*', '', text)  # Remove leading }
        text = re.sub(r'\s*\{\s*$', '', text)  # Remove trailing {
        
        # Fix missing line breaks before numbered list items (e.g., "including:1. " → "including:\n\n1. ")
        text = re.sub(r'([:.\n])\s*(\d+\.\s+)', r'\1\n\n\2', text)
        
        if content_type == "list":
            # Check if this is a publication list specifically
            if re.search(r'(publication|paper|article|study|research)', text.lower()):
                # ENHANCED: Publication list formatting with improved structure
                
                # Extract introduction paragraph if present
                intro_match = re.search(r'^(.*?)(?=\d+\.|$)', text, re.DOTALL)
                intro = intro_match.group(1).strip() if intro_match else ""
                
                # Remove duplicate headers/intros that might be included in each item
                for pattern in [
                    r'Research Domains:.*?(?=\n|\Z)',
                    r'Summary:.*?(?=\n|\Z)',
                    r'Regarding.*?publications:.*?(?=\n|\Z)'
                ]:
                    matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
                    if len(matches) > 1:
                        # Keep only the first instance
                        first_match_end = matches[0].end()
                        for match in matches[1:]:
                            text = text[:match.start()] + text[match.end():]
                
                # NEW: Extract publication entries with improved pattern matching
                entries = []
                entry_pattern = r'(\d+\.\s+(?:\*\*)?[^*\n]+(?:\*\*)?(?:(?!\d+\.\s+(?:\*\*)?).)*)'
                raw_entries = re.findall(entry_pattern, text, re.DOTALL)
                
                for entry in raw_entries:
                    if not entry.strip():
                        continue
                    
                    # Extract the number and title
                    number_title_match = re.match(r'(\d+)\.\s+(?:\*\*)?([^*\n]+)(?:\*\*)?', entry)
                    if not number_title_match:
                        entries.append(entry)  # Just add as-is if pattern doesn't match
                        continue
                        
                    number = number_title_match.group(1)
                    title = number_title_match.group(2).strip()
                    
                    # Format the entry with consistent structure
                    formatted_entry = f"{number}. **{title}**\n"
                    
                    # Extract and format fields with improved patterns
                    # Note: Using helper functions for consistent field formatting
                    formatted_entry += self._format_publication_field(entry, "Publication Year", "Publication Year")
                    formatted_entry += self._format_publication_field(entry, "Authors", "Authors")
                    formatted_entry += self._format_publication_field(entry, "DOI", "DOI")
                    formatted_entry += self._format_publication_field(entry, "Abstract", "Abstract")
                    
                    entries.append(formatted_entry)
                
                # Find any closing message
                closing_match = re.search(r'Would you like more detailed.*$', text, re.DOTALL)
                closing = closing_match.group(0) if closing_match else ""
                
                # Reconstruct the text with proper formatting
                if intro:
                    formatted_text = f"{intro}\n\n"
                else:
                    formatted_text = ""
                    
                formatted_text += "\n\n".join(entries)
                
                if closing:
                    formatted_text += f"\n\n{closing}"
                    
                # Clean DOI links specifically for publications
                formatted_text = re.sub(
                    r'(DOI:?\s*)(10\.\s*\d+\s*/\s*[^\s\)]+)',
                    lambda m: f"{m.group(1)}{m.group(2).replace(' ', '')}",
                    formatted_text
                )
                
                # Ensure consistent newlines between sections
                formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
                
                # Fix missing line breaks before numbered list items (e.g., "including:1. " → "including:\n\n1. ")
                formatted_text = re.sub(r'([:.\n])\s*(\d+\.\s+)', r'\1\n\n\2', formatted_text)
                
                return formatted_text
                
            # Check if this is an expert list specifically (existing code)
            elif re.search(r'(expert|researcher|scientist|professor|specialist)', text.lower()):
                # CRITICAL FIX: Parse and reformat expert list with proper structure
                
                # Extract the header/intro
                intro_match = re.search(r'^(.*?)(?=\d+\.)', text, re.DOTALL)
                intro = intro_match.group(1).strip() if intro_match else ""
                
                # Extract the expert entries
                expert_entries = []
                
                # Split by numbered list items
                entry_pattern = r'(\d+\.\s+(?:\*\*)?[^*\n]+(?:\*\*)?(?:(?!\d+\.\s+(?:\*\*)?).)*)'
                entries = re.findall(entry_pattern, text, re.DOTALL)
                
                # Process each expert
                formatted_experts = []
                
                for entry in entries:
                    if not entry.strip():
                        continue
                    
                    # Extract numbered part and name
                    # Updated regex to handle cases with or without asterisks
                    number_name_match = re.match(r'(\d+)\.\s+(?:\*\*)?([^*\n]+?)(?:\*\*)?$', entry, re.MULTILINE)
                    if not number_name_match:
                        # Try alternative pattern
                        number_name_match = re.match(r'(\d+)\.\s+([^\n]+?)(?=\s*Email:|\s*$)', entry)
                        if not number_name_match:
                            # If still can't match, add as-is
                            formatted_experts.append(entry)
                            continue
                        
                    number = number_name_match.group(1)
                    name = number_name_match.group(2).strip()
                    
                    # Extract email if present
                    email_match = re.search(r'Email:\s*([^\s]+@[^\s]+)', entry)
                    email = email_match.group(1) if email_match else ""
                    
                    # Format the expert entry properly with name and email together
                    formatted_entry = f"{number}. **{name}**"
                    if email:
                        formatted_entry += f"\n    Email: {email}"
                    
                    formatted_experts.append(formatted_entry)
                
                # Find the closing message
                closing_match = re.search(r'Would you like more detailed.*$', text, re.DOTALL)
                closing = closing_match.group(0) if closing_match else ""
                
                # Reconstruct the text
                if intro:
                    text = f"{intro}\n\n"
                else:
                    text = ""
                    
                text += "\n\n".join(formatted_experts)
                
                if closing:
                    text += f"\n\n{closing}"
                
                # Perform general cleaning for any remaining issues
                cleaned_text = self._clean_text_for_user(text)
                
                # Additional cleaning specific to expert list formatting
                if re.search(r'(expert|researcher|scientist|professor|specialist)', text.lower()):
                    # CRITICAL FIX: Ensure email addresses are properly attached to expert names
                    # This prevents the "1.- Expert Name* Email: email@domain.com" formatting issue
                    cleaned_text = re.sub(r'(\d+\.\s+\*\*[^*]+\*\*)\s*\n\s*Email:', r'\1\nEmail:', cleaned_text)
                    
                    # Remove any stray asterisks that might appear
                    cleaned_text = re.sub(r'([A-Za-z0-9])\*\s+', r'\1 ', cleaned_text)
                    
                    # Fix formatting for email lines
                    cleaned_text = re.sub(r'Email:\s*([^\s]+@[^\s]+)', r'Email: \1', cleaned_text)
                    
                    # Ensure proper line breaks between experts
                    cleaned_text = re.sub(r'([^\s]+@[^\s]+)(\s+\d+\.)', r'\1\n\n\2', cleaned_text)
                    
                    # Fix any instances where expert names have missing bold formatting
                    cleaned_text = re.sub(r'(\d+)\.\s+([^*\n]+?)(?=\n)', r'\1. **\2**', cleaned_text)
                
                # Fix missing line breaks before numbered list items (e.g., "including:1. " → "including:\n\n1. ")
                cleaned_text = re.sub(r'([:.\n])\s*(\d+\.\s+)', r'\1\n\n\2', cleaned_text)
                
                return cleaned_text
            
            # For non-list content, just apply general cleaning
            return self._clean_text_for_user(text)
        
        # For non-list content, just apply general cleaning
        cleaned_text = self._clean_text_for_user(text)
        
        # Fix missing line breaks before numbered list items (e.g., "including:1. " → "including:\n\n1. ")
        cleaned_text = re.sub(r'([:.\n])\s*(\d+\.\s+)', r'\1\n\n\2', cleaned_text)
        
        return cleaned_text
    
    
  

    @staticmethod
    def _clean_text_for_user(text: str) -> str:
        """
        Enhanced text cleaning for user-facing responses in Markdown format.
        Removes technical artifacts, properly formats Markdown, and improves readability.
        Args:
            text (str): The input text that may contain technical artifacts or raw Markdown
        Returns:
            str: Clean, well-formatted text suitable for user display with improved readability
        """
        if not text:
            return ""

        # Remove JSON metadata that might have slipped through
        metadata_pattern = r'^\s*\{\"is_metadata\"\s*:\s*true.*?\}\s*'
        text = re.sub(metadata_pattern, '', text, flags=re.MULTILINE)
        
        # Fix potential duplicate heading markers
        text = re.sub(r'(#+)\s*(#+)\s*', r'\1 ', text)
        
        # Normalize heading formatting
        text = re.sub(r'(#+)(\w)', r'\1 \2', text)  # Ensure space after heading markers
        
        # Preserve markdown bold formatting (ensure spaces are correct)
        text = re.sub(r'\*\*\s*(.+?)\s*\*\*', r'**\1**', text)
        
        # Normalize bullet points (could be * or - in markdown)
        text = re.sub(r'^\s*[-*]\s*', '- ', text, flags=re.MULTILINE)  # Standardize to dash style
        
        # Fix spaces in DOI links - special case for academic content
        # First, handle standard DOI URLs
        text = re.sub(
            r'(https?://doi\.org/\s*)([\d\.]+(/\s*)?[^\s\)]*)',
            lambda m: m.group(1).replace(' ', '') + m.group(2).replace(' ', ''),
            text
        )
        
        # Then handle bare DOI references
        text = re.sub(
            r'(DOI:?\s*)(10\.\s*\d+\s*/\s*[^\s\)]+)',
            lambda m: m.group(1) + m.group(2).replace(' ', ''),
            text
        )
        
        # Handle numbered lists consistently
        text = re.sub(r'(\d+)\.\s+([A-Z])', r'\1. \2', text)  # Ensure proper spacing after numbers
        
        # Fix missing line breaks before numbered list items (e.g., "including:1. " → "including:\n\n1. ")
        text = re.sub(r'([:.\n])\s*(\d+\.\s+)', r'\1\n\n\2', text)
        
        # Clean up excess whitespace while preserving meaningful structure
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        
        # Ensure proper Markdown line breaks
        # Single newlines become Markdown line breaks (with two spaces)
        text = re.sub(r'(?<!\n)\n(?!\n)', '  \n', text)
        
        # But multiple newlines are preserved for paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize multiple newlines to exactly two
        
        # Remove trailing whitespace
        text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)
        
        # Ensure consistent formatting for publication information
        text = re.sub(r'(Title|Authors|Publication Year|DOI|Abstract|Summary):\s*', r'**\1**: ', text)
        
        # CRITICAL FIX: Ensure each bold section starts on a new line
        text = re.sub(r'([^\n])\s*\*\*([^*:]+):\*\*', r'\1\n\n**\2:**', text)
        
        # Fix line breaks for expert entries and emails
        text = re.sub(r'(\d+\.\s+\*\*[^*]+\*\*)\s+Email:', r'\1\nEmail:', text)
        
        # Final trim of any leading/trailing whitespace
        return text.strip()
        
    
    async def process_stream_response(self, response_stream):
        """
        Process the streaming response with enhanced formatting, structure preservation,
        and improved natural language elements. Now with better publication handling.
        Args:
            response_stream: Async generator of response chunks
        Yields:
            str or dict: Cleaned and formatted response chunks with natural language improvements
        """
        buffer = ""
        metadata = None
        detected_intent = None
        is_first_chunk = True
        transition_inserted = False
        
        # Track different content segments for better processing
        in_structured_content = False
        structured_buffer = ""
        content_type = None  # Will be "publication_list" or "expert_list" for more specific handling
        
        try:
            async for chunk in response_stream:
                # Capture metadata with improved detection
                if isinstance(chunk, dict) and chunk.get('is_metadata'):
                    metadata = chunk.get('metadata', chunk)
                    self.metadata = metadata
                    if metadata and 'intent' in metadata:
                        detected_intent = metadata.get('intent')
                        logger.debug(f"Detected intent for response styling: {detected_intent}")
                    logger.debug(f"Captured metadata: {json.dumps(metadata, default=str) if metadata else 'None'}")
                    yield chunk  # Pass metadata through for downstream handling
                    continue

                # Extract text from chunk
                if isinstance(chunk, dict):
                    text = chunk.get('chunk', chunk.get('content', chunk.get('text', '')))
                elif isinstance(chunk, (str, bytes)):
                    text = chunk.decode('utf8') if isinstance(chunk, bytes) else chunk
                elif hasattr(chunk, 'content'):
                    text = chunk.content
                elif hasattr(chunk, 'text'):
                    text = chunk.text
                else:
                    logger.debug(f"Skipping unknown chunk format: {type(chunk)}")
                    continue

                if not text.strip():
                    continue

                # Add natural language improvements for first chunk based on intent
                if is_first_chunk and detected_intent and not transition_inserted:
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
                    
                    if intro_text:
                        yield intro_text
                        transition_inserted = True
                    is_first_chunk = False

                # IMPROVED: Checking for multiple types of structured content with better detection
                # Check if we're entering a publication list
                if not in_structured_content and (
                    re.search(r'\d+\.\s+(Title:|Publication|Author|DOI:)', text, re.IGNORECASE) or
                    re.search(r'# Publications|# APHRC Publications', text) or
                    re.search(r'publications matching your query', text)
                ):
                    in_structured_content = True
                    content_type = "publication_list"  # More specific content type
                    structured_buffer = buffer + text
                    buffer = ""
                    
                    # Insert appropriate transition if needed
                    if not transition_inserted and detected_intent:
                        if "publication" in str(detected_intent).lower():
                            yield self._get_random_transition("publication_list")
                        transition_inserted = True
                    continue
                
                # Check if we're entering an expert list
                elif not in_structured_content and (
                    re.search(r'\d+\.\s+\*\*[^*]+\*\*', text) or
                    re.search(r'# Experts|experts (in|at) APHRC|Expert Profile', text)
                ):
                    in_structured_content = True
                    content_type = "expert_list"  # More specific content type
                    structured_buffer = buffer + text
                    buffer = ""
                    
                    # Insert appropriate transition if needed
                    if not transition_inserted and detected_intent:
                        if "expert" in str(detected_intent).lower():
                            yield self._get_random_transition("expert_list")
                        transition_inserted = True
                    continue
                
                # If we're in a structured content section, keep collecting it
                if in_structured_content:
                    structured_buffer += text
                    
                    # Check if we have a complete structured content item to yield
                    # Different handling based on content type
                    if content_type == "publication_list" and re.search(r'\n\d+\.', structured_buffer):
                        # We have at least one complete numbered item
                        items = re.split(r'(?=\n\d+\.)', structured_buffer)
                        
                        # Keep the last (potentially incomplete) item
                        if len(items) > 1:
                            for item in items[:-1]:
                                if item.strip():
                                    # Apply specialized cleaning for publication content
                                    cleaned_item = self._clean_structured_content(item, "list")
                                    yield cleaned_item
                                    
                                    # Maybe add transition between items
                                    if random.random() < 0.3:
                                        yield self._get_random_transition("between_publications")
                            
                            # Keep the potentially incomplete last item in the buffer
                            structured_buffer = items[-1]
                    
                    # Similar processing for expert lists, with pattern adjusted for expert content
                    elif content_type == "expert_list" and re.search(r'\n\d+\.\s+\*\*', structured_buffer):
                        items = re.split(r'(?=\n\d+\.\s+\*\*)', structured_buffer)
                        
                        if len(items) > 1:
                            for item in items[:-1]:
                                if item.strip():
                                    cleaned_item = self._clean_structured_content(item, "list")
                                    yield cleaned_item
                            
                            structured_buffer = items[-1]
                    
                    # Check if we're exiting structured content - different patterns for different types
                    exit_conditions = False
                    
                    if content_type == "publication_list":
                        # Check for publication list exit conditions
                        exit_conditions = (
                            len(structured_buffer) > 100 and  # Reasonable size to have complete content
                            not re.search(r'(Title:|Authors:|Publication Year:|DOI:|Abstract:|Summary:)', 
                                        text, re.IGNORECASE) and
                            re.search(r'(Would you like more|In conclusion|To summarize|Is there anything else)', 
                                    structured_buffer, re.IGNORECASE)
                        )
                    elif content_type == "expert_list":
                        # Check for expert list exit conditions
                        exit_conditions = (
                            len(structured_buffer) > 100 and
                            not re.search(r'(Expert|Position:|Email:|Notable publication)', 
                                        text, re.IGNORECASE) and
                            re.search(r'(Would you like more|In conclusion|To summarize|Is there anything else)',
                                    structured_buffer, re.IGNORECASE)
                        )
                    
                    if exit_conditions:
                        # We're likely leaving the structured content section
                        if structured_buffer.strip():
                            cleaned_content = self._clean_structured_content(structured_buffer, "list")
                            yield cleaned_content
                            
                            # Add appropriate conclusion transition
                            if detected_intent:
                                if "publication" in str(detected_intent).lower():
                                    yield self._get_random_transition("after_publications")
                                elif "expert" in str(detected_intent).lower():
                                    yield self._get_random_transition("after_experts")
                        
                        # Reset for normal text processing
                        in_structured_content = False
                        structured_buffer = ""
                        content_type = None
                        
                    continue
                
                # Normal text processing (not in structured content)
                buffer += text
                
                # Process complete sentences from the buffer
                sentences = re.split(r'(?<=[.!?])\s+', buffer)
                if len(sentences) > 1:
                    # Process and yield complete sentences
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            # Apply general text cleaning
                            cleaned_sentence = self._clean_text_for_user(sentence)
                            yield cleaned_sentence
                    
                    # Keep the last incomplete sentence in the buffer
                    buffer = sentences[-1]

            # Process any remaining content
            if in_structured_content and structured_buffer.strip():
                cleaned_content = self._clean_structured_content(structured_buffer, "list")
                yield cleaned_content
                
                # Add appropriate closing transition
                if detected_intent:
                    if "publication" in str(detected_intent).lower():
                        yield self._get_random_transition("publication_conclusion")
                    elif "expert" in str(detected_intent).lower():
                        yield self._get_random_transition("expert_conclusion")
                        
            elif buffer.strip():
                # Clean any remaining content in the buffer
                cleaned_text = self._clean_text_for_user(buffer)
                yield cleaned_text
                
                # Add general conclusion if appropriate
                if detected_intent and not in_structured_content:
                    yield self._get_random_transition(f"{str(detected_intent).lower()}_conclusion")

        except Exception as e:
            logger.error(f"Error processing stream response: {e}", exc_info=True)
            # Return any remaining buffered content if there's an error
            if in_structured_content and structured_buffer.strip():
                yield self._clean_structured_content(structured_buffer, "list")
            elif buffer.strip():
                yield self._clean_text_for_user(buffer)
        
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