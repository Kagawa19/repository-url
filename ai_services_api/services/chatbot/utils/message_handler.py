```python
import asyncio
import logging
import random
import re
import time
from typing import AsyncIterable, List, Optional, Dict, Tuple, AsyncGenerator
from datetime import datetime
from .llm_manager import GeminiLLMManager
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.chatbot.utils.db_utils import DatabaseConnector
import json

logger = logging.getLogger(__name__)

class MessageHandler:
    def __init__(self, llm_manager):
        self.metadata = None
        self.llm_manager = llm_manager

    def _format_publication_field(self, entry_text: str, field_name: str, display_name: str = None) -> str:
        # Unchanged, kept for context
        if not display_name:
            display_name = field_name
        patterns = [
            rf'[-–•]?\s*\*\*{field_name}:\*\*\s*(.*?)(?=\s*[-–•]?\s*\*\*|\Z)',
            rf'[-–•]?\s*{field_name}:\s*(.*?)(?=\s*[-–•]?\s*|\Z)',
            rf'\b{field_name}\b:\s*(.*?)(?=\s*\b\w+\b:\s*|\Z)',
            rf'\s+[-–•]\s*\*\*{field_name}:\*\*\s*(.*?)(?=\s+[-–•]|\Z)'
        ]
        for pattern in patterns:
            match = re.search(pattern, entry_text, re.IGNORECASE | re.DOTALL)
            if match:
                field_content = match.group(1).strip()
                if field_name.lower() == 'doi':
                    doi_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', field_content)
                    doi_text = re.sub(r'\s+', '', doi_text)
                    if doi_text.startswith('http'):
                        return f"    - **{display_name}:** {doi_text}\n\n"
                    elif doi_text.startswith('10.'):
                        return f"    - **{display_name}:** {doi_text}\n\n"
                    else:
                        return f"    - **{display_name}:** {doi_text}\n\n"
                if field_name.lower() == 'abstract' and len(field_content) > 300:
                    field_content = field_content[:297] + "..."
                if field_name.lower() in ['author', 'authors']:
                    if '[' in field_content and ']' in field_content:
                        try:
                            author_list = json.loads(field_content.replace("'", '"'))
                            if isinstance(author_list, list) and len(author_list) > 0:
                                if len(author_list) > 3:
                                    authors_display = f"{', '.join(author_list[:3])} et al."
                                else:
                                    authors_display = ', '.join(author_list)
                                field_content = authors_display
                        except:
                            pass
                if field_name.lower() in ['year', 'publication year', 'published']:
                    year_match = re.search(r'\b(19|20)\d{2}\b', field_content)
                    if year_match:
                        field_content = year_match.group(0)
                return f"    - **{display_name}:** {field_content}\n\n"
        return ""

    def _extract_experts_from_text(self, text: str) -> List[Dict[str, Any]]:
        # Unchanged, kept for context
        experts = []
        expert_blocks = re.findall(r'(\d+\.\s+\*\*[^*\n]+\*\*(?:(?!\d+\.\s+\*\*).)*)', text, re.DOTALL)
        for block in expert_blocks:
            expert = {}
            name_match = re.search(r'\d+\.\s+\*\*([^*\n]+)\*\*', block)
            if name_match:
                full_name = name_match.group(1).strip()
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    expert['first_name'] = ' '.join(name_parts[:-1])
                    expert['last_name'] = name_parts[-1]
                else:
                    expert['first_name'] = full_name
                    expert['last_name'] = ''
            else:
                continue
            designation_match = re.search(r'(?:Designation|Position):\s*([^\n]+)', block, re.IGNORECASE)
            if designation_match:
                expert['designation'] = designation_match.group(1).strip()
            theme_match = re.search(r'Theme:\s*([^\n]+)', block, re.IGNORECASE)
            if theme_match:
                theme_text = theme_match.group(1).strip()
                theme_abbr_match = re.search(r'([A-Z]+)\s*\(', theme_text)
                if theme_abbr_match:
                    expert['theme'] = theme_abbr_match.group(1).strip()
                else:
                    expert['theme'] = theme_text
            unit_match = re.search(r'Unit:\s*([^\n]+)', block, re.IGNORECASE)
            if unit_match:
                unit_text = unit_match.group(1).strip()
                unit_abbr_match = re.search(r'([A-Z]+)\s*\(', unit_text)
                if unit_abbr_match:
                    expert['unit'] = unit_abbr_match.group(1).strip()
                else:
                    expert['unit'] = unit_text
            expertise_match = re.search(r'(?:Expertise|Knowledge).*?:\s*([^\n]+)', block, re.IGNORECASE)
            if expertise_match:
                expert['knowledge_expertise'] = expertise_match.group(1).strip()
            experts.append(expert)
        return experts

    def _extract_publications_from_text(self, text: str) -> List[Dict[str, Any]]:
        # Unchanged, kept for context
        publications = []
        pub_blocks = re.findall(r'(\d+\.\s+\*\*[^*\n]+\*\*(?:(?!\d+\.\s+\*\*).)*)', text, re.DOTALL)
        for block in pub_blocks:
            pub = {}
            title_match = re.search(r'\d+\.\s+\*\*([^*\n]+)\*\*', block)
            if title_match:
                pub['title'] = title_match.group(1).strip()
            else:
                continue
            authors_match = re.search(r'Authors?:\s*([^\n]+)', block, re.IGNORECASE)
            if authors_match:
                pub['authors'] = authors_match.group(1).strip()
            source_match = re.search(r'(?:Published in|Journal|Source):\s*([^\n]+)', block, re.IGNORECASE)
            if source_match:
                pub['source'] = source_match.group(1).strip()
            year_match = re.search(r'(?:Year|Published|Publication date):\s*([^\n]+)', block, re.IGNORECASE)
            if year_match:
                pub['publication_year'] = year_match.group(1).strip()
            theme_match = re.search(r'Theme:\s*([^\n]+)', block, re.IGNORECASE)
            if theme_match:
                pub['theme'] = theme_match.group(1).strip()
            summary_match = re.search(r'(?:Summary|Abstract):\s*([^\n]+)', block, re.IGNORECASE)
            if summary_match:
                pub['summary'] = summary_match.group(1).strip()
            doi_match = re.search(r'DOI:\s*([^\n]+)', block, re.IGNORECASE)
            if doi_match:
                pub['doi'] = doi_match.group(1).strip()
            publications.append(pub)
        return publications

    def _extract_navigation_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured navigation section data from text response.
        
        Args:
            text: Text containing navigation information
            
        Returns:
            List of navigation section dictionaries with structured fields
        """
        sections = []
        section_blocks = re.findall(r'(\d+\.\s+\*\*[^*\n]+\*\*(?:(?!\d+\.\s+\*\*).)*)', text, re.DOTALL)
        
        for block in section_blocks:
            section = {}
            title_match = re.search(r'\d+\.\s+\*\*([^*\n]+)\*\*', block)
            if title_match:
                section['title'] = title_match.group(1).strip()
            else:
                continue
            
            description_match = re.search(r'Description:\s*([^\n]+)', block, re.IGNORECASE)
            if description_match:
                section['description'] = description_match.group(1).strip()
            
            url_match = re.search(r'Link:\s*\[([^\]]+)\]\(([^\)]+)\)', block, re.IGNORECASE)
            if url_match:
                section['url'] = url_match.group(2).strip()
            
            keywords_match = re.search(r'Keywords:\s*([^\n]+)', block, re.IGNORECASE)
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                section['keywords'] = [k.strip() for k in keywords_text.split(',')]
            
            sections.append(section)
        
        return sections

    def clean_response(self, response: str) -> str:
        # Unchanged, kept for context
        pattern = r'^\s*\{.*?"is_metadata"\s*:\s*true.*?\}\s*'
        match = re.match(pattern, response, re.DOTALL)
        if match:
            return response[match.end():].lstrip()
        return response
    

    def _extract_navigation_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured navigation data from text response.
        
        Args:
            text: Text containing navigation information
            
        Returns:
            List of navigation dictionaries with structured fields
        """
        sections = []
        
        # Look for numbered navigation entries
        section_blocks = re.findall(r'(\d+\.\s+\*\*[^*\n]+\*\*(?:(?!\d+\.\s+\*\*).)*)', text, re.DOTALL)
        
        for block in section_blocks:
            section = {}
            
            # Extract title
            title_match = re.search(r'\d+\.\s+\*\*([^*\n]+)\*\*', block)
            if title_match:
                section['title'] = title_match.group(1).strip()
            else:
                continue  # Skip if no title found
            
            # Extract description
            description_match = re.search(r'Description:\s*([^\n]+)', block, re.IGNORECASE)
            if description_match:
                section['description'] = description_match.group(1).strip()
            
            # Extract URL/Link
            link_match = re.search(r'Link:\s*\[([^\]]+)\]\(([^)]+)\)', block, re.IGNORECASE)
            if link_match:
                section['url'] = link_match.group(2).strip()
            else:
                # Try alternate format for URL
                url_match = re.search(r'(?:URL|Link|Location):\s*([^\n]+)', block, re.IGNORECASE)
                if url_match:
                    section['url'] = url_match.group(1).strip()
            
            # Extract keywords
            keywords_match = re.search(r'Keywords:\s*([^\n]+)', block, re.IGNORECASE)
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                # Split by commas and clean each keyword
                section['keywords'] = [k.strip() for k in keywords_text.split(',') if k.strip()]
            
            # Extract section/category
            section_match = re.search(r'Section:\s*([^\n]+)', block, re.IGNORECASE)
            if section_match:
                section['parent_section'] = section_match.group(1).strip()
            
            # Extract related sections
            related_match = re.search(r'Related sections:\s*([^\n]+)', block, re.IGNORECASE)
            if related_match:
                related_text = related_match.group(1).strip()
                # Split by commas and clean each section
                section['related_sections'] = [r.strip() for r in related_text.split(',') if r.strip()]
            
            sections.append(section)
        
        return sections

    async def process_stream_response(self, response_stream):
        """
        Process the streaming response with enhanced formatting for expert, publication, and navigation lists.
        Detects and applies special formatting for structured content.
        """
        buffer = ""
        metadata = None
        detected_intent = None
        is_first_chunk = True
        transition_inserted = False

        in_structured_content = False
        structured_buffer = ""
        content_type = None

        is_expert_list = False
        is_publication_list = False
        is_navigation_list = False  # Added navigation list detection

        try:
            async for chunk in response_stream:
                print(f"[DEBUG] Received chunk: {chunk}")

                if isinstance(chunk, dict) and chunk.get('is_metadata'):
                    metadata = chunk.get('metadata', chunk)
                    self.metadata = metadata
                    if metadata and 'intent' in metadata:
                        detected_intent = metadata.get('intent')
                        is_expert_list = detected_intent == 'expert' and metadata.get('is_list_request', False)
                        is_publication_list = detected_intent == 'publication' and metadata.get('is_list_request', False)
                        # Add navigation list detection
                        is_navigation_list = detected_intent == 'navigation' and metadata.get('is_list_request', False)

                    print(f"[INFO] Detected intent: {detected_intent}")
                    print(f"[INFO] Metadata: {json.dumps(metadata, default=str)}")
                    yield chunk
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
                    print(f"[DEBUG] Skipping unknown chunk format: {type(chunk)}")
                    continue

                # Clean the text to remove any leading metadata JSON
                text = self.clean_response(text)

                print(f"[DEBUG] Cleaned text chunk: {text}")

                # Safeguard against invalid cleaned output
                if not text or text.strip() == '}':
                    print(f"[DEBUG] Skipping empty or invalid cleaned text: {text}")
                    continue

                if is_first_chunk:
                    text = re.sub(r'^[}\]]*', '', text)
                    is_first_chunk = False

                text = re.sub(r'^(\s*[}\]]+\s*)+', '', text)

                # Detect Expert List
                if not in_structured_content and (
                    re.search(r'\d+\.\s+\*\*[^*]+\*\*', text) or
                    re.search(r'\d+\.\s+[^*\n]+', text) or
                    re.search(r'#\s*Experts|experts\s+(?:in|at)\s+APHRC|Expert\s+Profile', text, re.IGNORECASE) or
                    re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)[^A-Z]*?specializes in', text) or
                    re.search(r'\d+\.\s+\*\*[^*]+\*\*\s*\n\s*\*\*Designation:', text) or
                    re.search(r'\d+\.\s+\*\*[^*]+\*\*\s*\n\s*\*\*Theme:', text) or
                    re.search(r'\d+\.\s*\*[^*]+\*', text)
                ):
                    if is_expert_list or re.search(r'(expert|researcher|scientist)', text.lower()) or detected_intent == 'expert':
                        in_structured_content = True
                        content_type = "expert_list"
                        structured_buffer = buffer + text
                        buffer = ""
                        print(f"[INFO] Entered expert list formatting mode")
                        continue

                # Detect Publication List
                if not in_structured_content and (
                    re.search(r'\d+\.\s+\*\*[^*]+\*\*', text) or
                    re.search(r'\d+\.\s+[^*\n]+', text) or
                    re.search(r'#\s*Publications|publications\s+(?:on|about)\s+APHRC|Publication\s+List', text, re.IGNORECASE) or
                    re.search(r'publications\s+(?:related\s+to|on\s+the\s+topic\s+of)', text, re.IGNORECASE) or
                    re.search(r'(paper|article|publication|research)\s+(?:titled|published\s+in|by)', text, re.IGNORECASE) or
                    re.search(r'\d+\.\s*\*[^*]+\*', text)
                ):
                    if is_publication_list or re.search(r'publication|paper|article|research|study|doi', text.lower()) or detected_intent == 'publication':
                        in_structured_content = True
                        content_type = "publication_list"
                        structured_buffer = buffer + text
                        buffer = ""
                        print(f"[INFO] Entered publication list formatting mode")
                        continue

                # ADDED: Detect Navigation List
                if not in_structured_content and (
                    re.search(r'\d+\.\s+\*\*[^*]+\*\*', text) or
                    re.search(r'\d+\.\s+[^*\n]+', text) or
                    re.search(r'#\s*Website Navigation|Navigation\s+(?:on|about)\s+APHRC|Navigation\s+Section', text, re.IGNORECASE) or
                    re.search(r'sections\s+(?:related\s+to|on\s+the\s+topic\s+of)', text, re.IGNORECASE) or
                    re.search(r'(section|page|resource|navigation)\s+(?:titled|found\s+at|location)', text, re.IGNORECASE) or
                    re.search(r'\d+\.\s*\*[^*]+\*', text)
                ):
                    if is_navigation_list or re.search(r'navigation|webpage|website|section|page|url|link', text.lower()) or detected_intent == 'navigation':
                        in_structured_content = True
                        content_type = "navigation_list"
                        structured_buffer = buffer + text
                        buffer = ""
                        print(f"[INFO] Entered navigation list formatting mode")
                        continue

                # Expert List Processing
                if in_structured_content and content_type == "expert_list":
                    structured_buffer += text
                    print(f"[DEBUG] Appending to expert buffer: {text[:60]}...")
                    if re.search(r'Would\s+you\s+like\s+more\s+detailed|more\s+information\s+about|anything\s+else', structured_buffer, re.IGNORECASE):
                        structured_buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', structured_buffer)
                        print(f"[INFO] Yielding expert list")
                        yield structured_buffer
                        in_structured_content = False
                        structured_buffer = ""
                        content_type = None
                    continue

                # Publication List Processing
                if in_structured_content and content_type == "publication_list":
                    structured_buffer += text
                    print(f"[DEBUG] Appending to publication buffer: {text[:60]}...")
                    if re.search(r'Would\s+you\s+like\s+more\s+detailed|Is\s+there\s+anything\s+else', structured_buffer, re.IGNORECASE):
                        structured_buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', structured_buffer)
                        print(f"[INFO] Yielding publication list")
                        yield structured_buffer
                        in_structured_content = False
                        structured_buffer = ""
                        content_type = None
                    continue
                    
                # ADDED: Navigation List Processing
                if in_structured_content and content_type == "navigation_list":
                    structured_buffer += text
                    print(f"[DEBUG] Appending to navigation buffer: {text[:60]}...")
                    if re.search(r'Would\s+you\s+like\s+more\s+detailed|more\s+information\s+about|navigating\s+to\s+a\s+specific', structured_buffer, re.IGNORECASE):
                        structured_buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', structured_buffer)
                        print(f"[INFO] Yielding navigation list")
                        yield structured_buffer
                        in_structured_content = False
                        structured_buffer = ""
                        content_type = None
                    continue

                # Regular content processing
                buffer += text
                sentences = re.split(r'(?<=[.!?])\s+', buffer)
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            print(f"[DEBUG] Yielding sentence: {sentence.strip()}")
                            yield sentence.strip()
                    buffer = sentences[-1]

            # Final cleanup
            if in_structured_content and structured_buffer.strip():
                structured_buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', structured_buffer)
                print(f"[INFO] Final yield: structured content")
                yield structured_buffer
            elif buffer.strip():
                buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', buffer)
                print(f"[DEBUG] Final yield: cleaned buffer: {buffer.strip()}")
                yield buffer.strip()

        except Exception as e:
            print(f"[ERROR] Exception during stream processing: {e}")
            import traceback
            traceback.print_exc()

            if in_structured_content and structured_buffer.strip():
                structured_buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', structured_buffer)
                try:
                    print(f"[ERROR-RECOVERY] Yielding structured content after error")
                    yield structured_buffer
                except Exception as fallback_e:
                    print(f"[ERROR-RECOVERY] Failed structured recovery: {fallback_e}")
                    yield structured_buffer
            elif buffer.strip():
                buffer = re.sub(r'^(\s*[}\]]+\s*)+', '', buffer)
                print(f"[ERROR-RECOVERY] Yielding final cleaned buffer after error")
                yield buffer.strip()

    async def send_message_async(self, message: str, user_id: str, session_id: str) -> AsyncGenerator:
        """
        Stream messages with enhanced conversation awareness, contextual references,
        improved flow between multiple interactions, and user interest tracking.
        Now supports navigation context.
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
                # ADDED: Handle navigation intent
                elif intent_value == 'navigation':
                    interaction_category = 'navigation'
            else:  # Handle string type
                if str(intent_type).lower() == 'publication':
                    interaction_category = 'publication'
                elif str(intent_type).lower() == 'expert':
                    interaction_category = 'expert'
                # ADDED: Handle navigation intent
                elif str(intent_type).lower() == 'navigation':
                    interaction_category = 'navigation'
            
            # Get the interest tracker
            interest_tracker = self._get_interest_tracker()
            
            # Use it for logging
            await interest_tracker.log_interaction(
                user_id=user_id,
                session_id=session_id,
                query=message,
                interaction_type=interaction_category
            )
            
            # Special handling for publication, expert, and navigation list requests
            if "list" in message.lower() and "publication" in message.lower():
                logger.info("Detected publication list request - will apply special formatting")
            elif "list" in message.lower() and ("expert" in message.lower() or "researcher" in message.lower()):
                logger.info("Detected expert list request - will apply special formatting")
            # ADDED: Handle navigation list requests
            elif "list" in message.lower() and ("section" in message.lower() or "page" in message.lower() or "navigation" in message.lower()):
                logger.info("Detected navigation list request - will apply special formatting")
            
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
                'expertise': [],
                'navigation': []  # ADDED: Track navigation sections
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
                    # ADDED: Check for navigation data
                    if 'navigation' in part:
                        content_tracker['navigation'].extend(part['navigation'])
                
                # Check metadata in self.metadata if set by process_stream_response
                if hasattr(self, 'metadata') and self.metadata:
                    # This could contain information about shown content
                    meta = self.metadata
                    if isinstance(meta, dict):
                        if 'publications' in meta:
                            content_tracker['publications'].extend(meta['publications'])
                        if 'experts' in meta:
                            content_tracker['experts'].extend(meta['experts'])
                        # ADDED: Check for navigation data in metadata
                        if 'navigation' in meta:
                            content_tracker['navigation'].extend(meta['navigation'])
                
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
            
            # ADDED: Process navigation content
            if content_tracker['navigation']:
                # Extract navigation sections and keywords
                nav_sections = []
                nav_keywords = []
                
                for section in content_tracker['navigation']:
                    if 'title' in section and section['title']:
                        nav_sections.append(section['title'])
                    
                    if 'keywords' in section and section['keywords']:
                        if isinstance(section['keywords'], list):
                            nav_keywords.extend(section['keywords'])
                        elif isinstance(section['keywords'], str):
                            keywords = [k.strip() for k in section['keywords'].split(',')]
                            nav_keywords.extend(keywords)
                
                # Track navigation interests
                if nav_sections:
                    await interest_tracker.track_topic_interests(
                        user_id, nav_sections, 'navigation_section'
                    )
                
                if nav_keywords:
                    await interest_tracker.track_topic_interests(
                        user_id, nav_keywords, 'navigation_topic'
                    )
            
            # Add a coherent closing based on conversation context if appropriate
            if random.random() < 0.3:  # Only add this sometimes to avoid being repetitive
                yield self._get_conversation_closing(intent_type, len(conversation_history))
                
            # Save this interaction to conversation history
            await self._save_to_conversation_history(user_id, session_id, message, intent_type)
                
            logger.info("Completed send_message_async stream generation")
            
        except Exception as e:
            logger.error(f"Error in send_message_async: {e}", exc_info=True)
            yield "I apologize, but I encountered an issue processing your request. Could you please try again or rephrase your question?"

    def _get_conversation_closing(self, intent_type: str, history_length: int) -> str:
        """
        Get a contextually appropriate conversation closing based on intent and history.
        Enhanced with navigation-specific closings.
        
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
        
        # ENHANCED: More variety in navigation closings
        navigation_closings = [
            "Can I help you navigate to any other sections of our website?",
            "Is there specific content you're looking for on the APHRC site?",
            "Would you like information about other APHRC resources?",
            "Is there anything else I can help you find on our website?",
            "Would you like me to explain more about what you can find in these sections?",
            "Can I help you understand how these sections relate to your interests?",
            "Are you looking for specific resources within these sections?",
            "Would you like me to guide you through the structure of our website?"
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
        elif str(intent_type).lower() == "publication" or (hasattr(intent_type, 'value') and intent_type.value == 'publication'):
            closings = publication_closings
        elif str(intent_type).lower() == "expert" or (hasattr(intent_type, 'value') and intent_type.value == 'expert'):
            closings = expert_closings
        elif str(intent_type).lower() == "navigation" or (hasattr(intent_type, 'value') and intent_type.value == 'navigation'):
            closings = navigation_closings
        else:
            closings = general_closings
        
        return random.choice(closings)
    def _get_random_intro(self, intent_type: str) -> str:
        """
        Get a random, natural-sounding introduction based on intent type.
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
            "Let me help you navigate the APHRC website to find what you need. ",
            "I can guide you to the relevant sections of our website. ",
            "Here’s how you can find the information you're looking for: ",
            "You can explore these sections on the APHRC website: ",
            "Let me point you to the right resources on our site. "
        ]
        general_intros = [
            "I'd be happy to help with that. ",
            "Let me address your question. ",
            "That's an interesting question about APHRC's work. ",
            "I can provide some information on that. ",
            "Thanks for your question about APHRC. "
        ]
        
        if str(intent_type).lower() == "publication":
            intros = publication_intros
        elif str(intent_type).lower() == "expert":
            intros = expert_intros
        elif str(intent_type).lower() == "navigation":
            intros = navigation_intros
        else:
            intros = general_intros
        
        return random.choice(intros)

    def _get_random_transition(self, transition_type: str) -> str:
        """
        Get a random, natural-sounding transition phrase based on the transition type.
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
        navigation_list_transitions = [
            "Here are the website sections I found: ",
            "Let me guide you to these relevant pages: ",
            "I've listed these sections for you: ",
            "Based on your query, these website areas might help: ",
            "The following pages are available: "
        ]
        between_navigation_transitions = [
            " Moving to another relevant section, ",
            " Another page you might find useful is ",
            " Related to this, you can also explore ",
            " Next, let me show you ",
            " Additionally, here’s "
        ]
        after_navigation_transitions = [
            " This section provides key information about that area. ",
            " You can find more details by visiting this page. ",
            " This page is a great resource for that topic. ",
            " That section includes helpful content for your needs. ",
            " You might find what you’re looking for here. "
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
        elif transition_type == "navigation_list":
            transitions = navigation_list_transitions
        elif transition_type == "between_navigation":
            transitions = between_navigation_transitions
        elif transition_type == "after_navigation":
            transitions = after_navigation_transitions
        elif transition_type == "navigation_conclusion":
            transitions = navigation_conclusion_transitions
        else:
            transitions = general_conclusion_transitions
        
        return random.choice(transitions)

    def _get_interest_tracker(self):
        # Unchanged, kept for context
        if not hasattr(self, 'interest_tracker'):
            from ai_services_api.services.chatbot.utils.interest_tracker import UserInterestTracker
            self.interest_tracker = UserInterestTracker(DatabaseConnector)
            logger.info("Initialized user interest tracker")
        return self.interest_tracker

    def _extract_metadata(self, chunk):
        # Unchanged, kept for context
        try:
            if isinstance(chunk, dict) and chunk.get('is_metadata'):
                metadata = chunk.get('metadata', {})
                self.metadata = metadata
                logger.debug(f"Extracted metadata: {json.dumps(metadata, default=str) if metadata else 'None'}")
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return None
