import os
import asyncio
import json
import logging
import random
import re
import time
import numpy as np
from typing import Dict, Tuple, Any, List, AsyncGenerator, Optional
from datetime import datetime
from enum import Enum
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager



logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enum for different types of query intents."""
    NAVIGATION = "navigation"
    PUBLICATION = "publication"
    GENERAL = "general"

class CustomAsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """Custom callback handler for streaming responses."""
    
    async def on_llm_start(self, *args, **kwargs):
        """Handle LLM start."""
        pass

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        """Handle new token."""
        if token:
            self.queue.put_nowait(token)

    async def on_llm_end(self, *args, **kwargs):
        """Handle LLM end."""
        self.queue.put_nowait(None)

    async def on_llm_error(self, error: Exception, *args, **kwargs):
        """Handle LLM error."""
        self.queue.put_nowait(f"Error: {str(error)}")

class GeminiLLMManager:
    def __init__(self):
        """Initialize the LLM manager with required components."""
        try:
            # Load API key
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            # Initialize callback handler
            self.callback = CustomAsyncCallbackHandler()
            
            # Initialize Redis manager for accessing real data
            try:
                self.redis_manager = ExpertRedisIndexManager()
                logger.info("Redis manager initialized successfully")
            except Exception as redis_error:
                logger.error(f"Error initializing Redis manager: {redis_error}")
                self.redis_manager = None
                logger.warning("Continuing without Redis integration - will use generative responses")
            
            # Initialize context management
            self.context_window = []
            self.max_context_items = 5
            self.context_expiry = 1800  # 30 minutes
            self.confidence_threshold = 0.6
            
            # Initialize intent patterns
            self.intent_patterns = {
                QueryIntent.NAVIGATION: {
                    'patterns': [
                        (r'website', 1.0),
                        (r'page', 0.9),
                        (r'find', 0.8),
                        (r'where', 0.8),
                        (r'how to', 0.7),
                        (r'navigate', 0.9),
                        (r'section', 0.8),
                        (r'content', 0.7),
                        (r'information about', 0.7)
                    ],
                    'threshold': 0.6
                },
                QueryIntent.PUBLICATION: {
                    'patterns': [
                        (r'research', 1.0),
                        (r'paper', 1.0),
                        (r'publication', 1.0),
                        (r'study', 0.9),
                        (r'article', 0.9),
                        (r'journal', 0.8),
                        (r'doi', 0.9),
                        (r'published', 0.8),
                        (r'authors', 0.8),
                        (r'findings', 0.7)
                    ],
                    'threshold': 0.6
                }
            }
            
            logger.info("GeminiLLMManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GeminiLLMManager: {e}", exc_info=True)
            raise

    def get_gemini_model(self):
        """Initialize and return the Gemini model with built-in retry logic."""
        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            stream=True,
            model="gemini-2.0-flash-thinking-exp-01-21",
            convert_system_message_to_human=True,
            callbacks=[self.callback],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_retries=5,  # Use built-in retry mechanism
            timeout=30  # Set a reasonable timeout
        )
            
    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model is not available."""
        logger.info("Creating fallback embedding")
        # Create a deterministic embedding based on character values
        embedding = np.zeros(384)  # Standard dimension for simple embeddings
        for i, char in enumerate(text):
            embedding[i % len(embedding)] += ord(char) / 1000
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def format_publication_context(self, publications: List[Dict[str, Any]]) -> str:
        """Format publication data into a readable context for the LLM."""
        if not publications:
            return "No specific publications found."
            
        context_parts = ["Here are relevant APHRC publications:"]
        
        # Count publications with DOIs for debugging
        doi_count = 0
        
        for pub in publications:
            pub_context = []
            # Title
            pub_context.append(f"Title: {pub.get('title', 'Untitled')}")
            
            # Publication year
            if pub.get('publication_year'):
                pub_context.append(f"Year: {pub.get('publication_year')}")
                
            # DOI (crucial for preventing fictional DOIs)
            if pub.get('doi') and pub.get('doi').strip() and pub.get('doi') != 'None':
                doi_value = pub.get('doi').strip()
                # Check if DOI is already a URL
                if doi_value.startswith('https://doi.org/'):
                    # It's already a full URL, use as is
                    pub_context.append(f"DOI: [{doi_value}]({doi_value})")
                else:
                    # It's just the DOI identifier, format as URL
                    pub_context.append(f"DOI: [{doi_value}](https://doi.org/{doi_value})")
                doi_count += 1
            
            # Authors
            authors = pub.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    if authors:
                        authors_text = ", ".join(str(a) for a in authors if a)
                        pub_context.append(f"Authors: {authors_text}")
                elif isinstance(authors, str):
                    pub_context.append(f"Authors: {authors}")
                elif isinstance(authors, dict):
                    authors_text = ", ".join(str(v) for v in authors.values() if v)
                    pub_context.append(f"Authors: {authors_text}")
            
            # Field and subfield
            if pub.get('field'):
                field_text = f"Field: {pub.get('field')}"
                if pub.get('subfield'):
                    field_text += f", Subfield: {pub.get('subfield')}"
                pub_context.append(field_text)
                
            # Summary
            if pub.get('summary_snippet'):
                pub_context.append(f"Summary: {pub.get('summary_snippet')}")
                
            context_parts.append("\n".join(pub_context))
        
        logger.info(f"Formatted {len(publications)} publications, {doi_count} with DOIs")
        
        return "\n\n".join(context_parts)

    async def get_relevant_publications(self, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Enhanced publication search with suggestions"""
        try:
            if not self.redis_manager:
                logger.warning("Redis manager not available, cannot retrieve publications")
                return [], "Our publication database is currently unavailable. Please try again later."
            
            # Check Redis for any keys with DOIs
            try:
                all_keys = []
                cursor = 0
                while True:
                    cursor, keys = self.redis_manager.redis_text.scan(cursor, match="meta:resource:*", count=100)
                    all_keys.extend(keys)
                    if cursor == 0:
                        break
                
                doi_count = 0
                for key in all_keys[:20]:
                    metadata = self.redis_manager.redis_text.hgetall(key)
                    if metadata.get('doi') and metadata.get('doi').strip() and metadata.get('doi') != 'None':
                        doi_count += 1
                        logger.info(f"Found publication with DOI: {metadata.get('doi')} - Title: {metadata.get('title')}")
                logger.info(f"Found {doi_count} publications with DOIs in sample of {min(20, len(all_keys))}")
            except Exception as e:
                logger.error(f"Error checking for DOIs: {e}")
            
            publication_keys = []
            try:
                cursor = 0
                pattern = "meta:resource:*"
                while True:
                    cursor, keys = self.redis_manager.redis_text.scan(cursor, match=pattern, count=100)
                    publication_keys.extend(keys)
                    if cursor == 0:
                        break
                        
                logger.info(f"Found {len(publication_keys)} total publications in Redis")
                
                if not publication_keys:
                    logger.warning("No publications found in Redis database")
                    return [], "No publications found in the database."
            except Exception as e:
                logger.error(f"Error scanning Redis keys: {e}")
                return [], "Error scanning publication records."
            
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            results = []
            
            doi_pattern = r'\b(10\.\d+\/[a-zA-Z0-9./-]+)\b'
            url_doi_pattern = r'https?://doi\.org/(10\.\d+\/[a-zA-Z0-9./-]+)\b'
            doi_matches = re.findall(doi_pattern, query) + [m for m in re.findall(url_doi_pattern, query)]
            
            if doi_matches:
                logger.info(f"Found DOI in query: {doi_matches[0]}")
                for key in publication_keys:
                    try:
                        metadata = self.redis_manager.redis_text.hgetall(key)
                        doi = metadata.get('doi', '')
                        if doi:
                            if doi.startswith('https://doi.org/'):
                                extracted_doi = doi.replace('https://doi.org/', '')
                            else:
                                extracted_doi = doi
                            if any(d in doi or d == extracted_doi for d in doi_matches):
                                metadata['match_score'] = 100
                                if metadata.get('authors'):
                                    try:
                                        metadata['authors'] = json.loads(metadata.get('authors', '[]'))
                                    except:
                                        metadata['authors'] = []
                                if metadata.get('domains'):
                                    try:
                                        metadata['domains'] = json.loads(metadata.get('domains', '[]'))
                                    except:
                                        metadata['domains'] = []
                                results.append(metadata)
                                logger.info(f"Found exact DOI match: {metadata.get('doi')}")
                    except Exception as e:
                        logger.error(f"Error processing publication {key} for DOI match: {e}")
                        continue
            
            if results:
                logger.info(f"Returning {len(results)} exact DOI matches")
                return results[:limit], None
            
            for key in publication_keys:
                try:
                    resource_id = key.split(':')[-1]
                    metadata = self.redis_manager.redis_text.hgetall(key)
                    if not metadata:
                        continue
                    
                    score = 0
                    title = metadata.get('title', '').lower()
                    summary = metadata.get('summary_snippet', '').lower()
                    field = metadata.get('field', '').lower()
                    
                    authors = []
                    domains = []
                    if metadata.get('authors'):
                        try:
                            authors_data = json.loads(metadata.get('authors', '[]'))
                            if isinstance(authors_data, list):
                                authors = [str(a).lower() for a in authors_data if a]
                            elif isinstance(authors_data, dict):
                                authors = [str(v).lower() for v in authors_data.values() if v]
                        except json.JSONDecodeError:
                            authors = []
                    if metadata.get('domains'):
                        try:
                            domains_data = json.loads(metadata.get('domains', '[]'))
                            domains = [str(d).lower() for d in domains_data if d]
                        except json.JSONDecodeError:
                            domains = []
                    
                    for keyword in query_keywords:
                        if keyword in title:
                            score += 3
                        if keyword in summary:
                            score += 2
                        if keyword in field:
                            score += 2
                        if any(keyword in author for author in authors):
                            score += 1
                        if any(keyword in domain for domain in domains):
                            score += 1
                    
                    year_matches = re.findall(r'\b(19|20)\d{2}\b', query)
                    pub_year = metadata.get('publication_year', '')
                    if year_matches and pub_year in year_matches:
                        score += 3
                    
                    if metadata.get('doi') and metadata.get('doi').strip() and metadata.get('doi') != 'None':
                        score += 5
                    
                    if score > 0:
                        metadata['match_score'] = score
                        if metadata.get('authors'):
                            try:
                                metadata['authors'] = json.loads(metadata.get('authors', '[]'))
                            except:
                                metadata['authors'] = []
                        if metadata.get('domains'):
                            try:
                                metadata['domains'] = json.loads(metadata.get('domains', '[]'))
                            except:
                                metadata['domains'] = []
                        results.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing publication {key}: {e}")
                    continue
            
            sorted_results = sorted(results, key=lambda x: x.get('match_score', 0), reverse=True)
            top_publications = sorted_results[:limit]
            
            with_doi = [p for p in top_publications if p.get('doi') and p.get('doi').strip() and p.get('doi') != 'None']
            logger.info(f"Found {len(top_publications)} relevant publications for query")
            logger.info(f"Of these, {len(with_doi)} have DOIs")

            if not top_publications:
                # Generate suggestions for empty results
                suggestions = self._generate_search_suggestions(query)
                return [], suggestions

            return top_publications, None
        
        except Exception as e:
            logger.error(f"Error retrieving publications: {e}")
            return [], "We encountered an error searching publications. Try simplifying your query."

        
    def _generate_search_suggestions(self, query: str) -> str:
        """Generate search suggestions for empty results"""
        suggestions = [
            "Try different keywords or more general terms",
            "Search by publication year (e.g., 'studies from 2020-2022')",
            "Include author names if known",
            "Browse our publications by topic at [APHRC Publications](https://aphrc.org/publications)"
        ]
        
        # Simple keyword analysis
        if len(query.split()) > 4:
            suggestions.insert(0, "Try a shorter, more focused query")
        if any(word in query.lower() for word in ["recent", "new", "latest"]):
            suggestions.append("For recent work, try 'publications from the last 3 years'")
        
        return "No exact matches found. Suggestions:\n- " + "\n- ".join(suggestions)

    def _create_system_message(self, intent: QueryIntent) -> str:
        """
        Create appropriate system message based on intent with natural, flowing responses.
        """
        base_prompts = {
            "common": (
                "You are APHRC's AI assistant. Provide concise, clear responses. "
                "Limit responses to 2-3 paragraphs. Include only essential details. "
                "Focus on accuracy and brevity."
            ),
            QueryIntent.NAVIGATION: (
                "Guide website navigation concisely. Include direct URLs. "
                "Format: [Section Name](URL) - Brief description. "
                "Prioritize most relevant sections first."
            ),
            QueryIntent.PUBLICATION: (
                "Summarize research publications briefly. "
                "IMPORTANT: When DOIs are available, use them as clickable links. "
                "If the DOI is already a full URL like 'https://doi.org/10.xxxx/yyyy', "
                "format it as [https://doi.org/10.xxxx/yyyy](https://doi.org/10.xxxx/yyyy). "
                "Always format citations as: Author et al. (Year) - Key finding. Include the DOI link at the end of each citation. "
                "Focus on main conclusions and practical implications."
            ),
            QueryIntent.GENERAL: (
                "Provide focused overview of APHRC's work. "
                "Balance between navigation help and research insights. "
                "Direct users to specific resources when applicable."
            )
        }

        response_guidelines = (
            "Structure responses in a natural conversational style. "
            "Start with a direct answer, then provide supporting details, and end with relevant links if applicable. "
            "Use natural, flowing language without numbered points. "
            "Avoid repetition and technical jargon."
        )

        # Add specific instruction for publication queries to prevent making up information
        if intent == QueryIntent.PUBLICATION:
            publication_instruction = (
                "VERY IMPORTANT: Only reference the specific publications mentioned in the context. "
                "Do NOT create hypothetical or fictional DOIs. "
                "Make sure to include DOI links when they are present in the context. "
                "If no specific publications are provided in the context, "
                "recommend the user visit the APHRC website for publication information "
                "instead of making up publication details. "
                "Never invent research findings or publication information."
            )
            return f"{base_prompts['common']} {base_prompts[intent]} {response_guidelines} {publication_instruction}"
        
        return f"{base_prompts['common']} {base_prompts[intent]} {response_guidelines}"

    

    async def analyze_quality(self, message: str, response: str = "") -> Dict:
        """
        Analyze the quality of a response in terms of helpfulness, factual accuracy, and potential hallucination.
        Includes improved rate limit handling.
        
        Args:
            message (str): The user's original query
            response (str): The chatbot's response to analyze (if available)
        
        Returns:
            Dict: Quality metrics including helpfulness, hallucination risk, and factual grounding
        """
        try:
            # Limit analysis during high load periods or if a rate limit was recently hit
            if hasattr(self, '_rate_limited') and self._rate_limited:
                logger.warning("Skipping quality analysis due to recent rate limit")
                return self._get_default_quality()
            
            # If no response provided, we can only analyze the query
            if not response:
                prompt = f"""Analyze this query for an APHRC chatbot and return a JSON object with quality expectations.
                The chatbot helps users find publications and navigate APHRC resources.
                Return ONLY the JSON object with no markdown formatting, no code blocks, and no additional text.
                
                Required format:
                {{
                    "helpfulness_score": <float between 0 and 1, representing expected helpfulness>,
                    "hallucination_risk": <float between 0 and 1, representing risk based on query complexity>,
                    "factual_grounding_score": <float between 0 and 1, representing how much factual knowledge is needed>,
                    "unclear_elements": [<array of strings representing potential unclear aspects of the query>],
                    "potentially_fabricated_elements": []
                }}
                
                Query to analyze: {message}
                """
            else:
                # If we have both query and response, analyze the response quality
                prompt = f"""Analyze the quality of this chatbot response for the given query and return a JSON object.
                The APHRC chatbot helps users find publications and navigate APHRC resources.
                Evaluate helpfulness, factual accuracy, and potential hallucination.
                Return ONLY the JSON object with no markdown formatting, no code blocks, and no additional text.
                
                Required format:
                {{
                    "helpfulness_score": <float between 0 and 1>,
                    "hallucination_risk": <float between 0 and 1>,
                    "factual_grounding_score": <float between 0 and 1>,
                    "unclear_elements": [<array of strings representing unclear aspects of the response>],
                    "potentially_fabricated_elements": [<array of strings representing statements that may be hallucinated>]
                }}
                
                User query: {message}
                
                Chatbot response: {response}
                """
            
            # Use model with retry logic already built in from get_gemini_model()
            try:
                model = self.get_gemini_model()
                response = await model.invoke(prompt)
                
                # Reset any rate limit flag if successful
                if hasattr(self, '_rate_limited'):
                    self._rate_limited = False
                
                cleaned_response = response.content.strip()
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
                
                try:
                    quality_data = json.loads(cleaned_response)
                    logger.info(f"Response quality analysis result: {quality_data}")
                    return quality_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse quality analysis response: {cleaned_response}")
                    logger.error(f"JSON parse error: {e}")
                    return self._get_default_quality()
                    
            except Exception as api_error:
                # Check if this was a rate limit error
                if any(x in str(api_error).lower() for x in ["429", "quota", "rate limit", "resource exhausted"]):
                    logger.warning("Rate limit detected in quality analysis, marking for cooldown")
                    self._rate_limited = True
                    # Set expiry for rate limit status
                    asyncio.create_task(self._reset_rate_limited_after(300))  # 5 minutes cooldown
                
                logger.error(f"API error in quality analysis: {api_error}")
                return self._get_default_quality()
                    
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return self._get_default_quality()

    async def _reset_rate_limited_after(self, seconds: int):
        """Reset rate limited flag after specified seconds."""
        await asyncio.sleep(seconds)
        self._rate_limited = False
        logger.info(f"Rate limit cooldown expired after {seconds} seconds")

    def _get_default_quality(self) -> Dict:
        """Return default quality metric values."""
        return {
            'helpfulness_score': 0.5,
            'hallucination_risk': 0.5,
            'factual_grounding_score': 0.5,
            'unclear_elements': [],
            'potentially_fabricated_elements': []
        }

    # This maintains backwards compatibility with code still calling analyze_sentiment
    async def analyze_sentiment(self, message: str) -> Dict:
        """
        Legacy method maintained for backwards compatibility.
        Now redirects to analyze_quality.
        """
        logger.warning("analyze_sentiment is deprecated, using analyze_quality instead")
        quality_data = await self.analyze_quality(message)
        
        # Transform quality data to match the expected sentiment structure
        # This ensures old code expecting sentiment data continues to work
        return {
            'sentiment_score': quality_data.get('helpfulness_score', 0.5) * 2 - 1,  # Map 0-1 to -1-1
            'emotion_labels': [],
            'confidence': 1.0 - quality_data.get('hallucination_risk', 0.5),
            'aspects': {
                'satisfaction': quality_data.get('helpfulness_score', 0.5),
                'urgency': 0.5,
                'clarity': quality_data.get('factual_grounding_score', 0.5)
            }
        }
    
    async def detect_intent(self, message: str, is_followup: bool = False) -> Tuple[QueryIntent, float, Optional[str]]:
        """Enhanced intent detection with clarification support."""
        try:
            message = message.lower()
            intent_scores = {intent: 0.0 for intent in QueryIntent}
            matched_keywords = {intent: [] for intent in QueryIntent}
            
            # Score patterns and collect matched keywords
            for intent, config in self.intent_patterns.items():
                score = 0.0
                matches = 0
                
                for pattern, weight in config['patterns']:
                    if re.search(pattern, message):
                        score += weight
                        matches += 1
                        matched_keywords[intent].append(pattern)
                
                if matches > 0:
                    intent_scores[intent] = score / matches
            
            max_intent, max_score = max(intent_scores.items(), key=lambda x: x[1])
            clarification = None
            
            # Generate clarification only for initial queries (not followups)
            if not is_followup:
                # Borderline confidence case
                if 0.4 <= max_score < 0.6:
                    top_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                    if len(top_intents) > 1 and top_intents[0][1] - top_intents[1][1] < 0.2:
                        clarification = (
                            f"I found both '{top_intents[0][0].value}' and '{top_intents[1][0].value}' relevant. "
                            f"Are you looking for {self._get_intent_examples(top_intents[0][0])} or "
                            f"{self._get_intent_examples(top_intents[1][0])}?"
                        )
                
                # Low confidence case
                elif max_score < 0.4:
                    clarification = (
                        "Could you clarify what you're looking for? For example:\n"
                        "- Publications: 'Find papers about maternal health'\n"
                        "- Navigation: 'Where can I find research datasets?'"
                    )

            # Return only if score meets threshold
            if max_score >= self.intent_patterns.get(max_intent, {}).get('threshold', 0.6):
                return max_intent, max_score, clarification

            return QueryIntent.GENERAL, max_score, clarification

        except Exception as e:
            # Safe fallback and error logging
            logger.error(f"Intent detection failed: {e}", exc_info=True)
            return QueryIntent.GENERAL, 0.0, "Sorry, I couldn't understand your request. Can you rephrase it?"

        finally:
            # Optional cleanup or logging
            logger.debug(f"Intent detection executed for message: '{message}' (follow-up: {is_followup})")


    def create_context(self, relevant_data: List[Dict]) -> str:
        """
        Create a flowing context narrative from relevant content.
        """
        if not relevant_data:
            return ""
        
        navigation_content = []
        publication_content = []
        
        for item in relevant_data:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            content_type = metadata.get('type', 'unknown')
            
            if content_type == 'navigation':
                navigation_content.append(
                    f"The {metadata.get('title', 'section')} of our website ({metadata.get('url', '')}) "
                    f"provides information about {text[:300].strip()}..."
                )
            
            elif content_type == 'publication':
                authors = metadata.get('authors', 'our researchers')
                date = metadata.get('date', '')
                date_text = f" in {date}" if date else ""
                
                publication_content.append(
                    f"In a study published{date_text}, {authors} explored {metadata.get('title', 'research')}. "
                    f"Their work revealed that {text[:300].strip()}..."
                )
        
        context_parts = []
        
        if navigation_content:
            context_parts.append(
                "Regarding our online resources: " + 
                " ".join(navigation_content)
            )
        
        if publication_content:
            context_parts.append(
                "Our research has produced several relevant findings: " + 
                " ".join(publication_content)
            )
        
        return "\n\n".join(context_parts)

    def manage_context_window(self, new_context: Dict):
        """Manage sliding window of conversation context."""
        current_time = datetime.now().timestamp()
        
        # Remove expired contexts
        self.context_window = [
            ctx for ctx in self.context_window 
            if current_time - ctx.get('timestamp', 0) < self.context_expiry
        ]
        
        # Add new context
        new_context['timestamp'] = current_time
        self.context_window.append(new_context)
        
        # Maintain maximum window size
        if len(self.context_window) > self.max_context_items:
            self.context_window.pop(0)

  
    async def generate_async_response(self, message: str, conversation_history: Optional[List[Dict]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Enhanced response generation with conversation awareness"""
        start_time = time.time()
        logger.info(f"Starting async response generation for message: {message}")
        
        try:
            # Detect intent with conversation context
            is_followup = conversation_history is not None and len(conversation_history) > 0
            logger.info(f"Follow-up detected: {is_followup}")
            
            # Detect intent with conversation context, handling follow-ups
            try:
                intent_result, clarification = await self.detect_intent(message, is_followup)
                intent, confidence = intent_result
                
                # Handle clarification questions first
                if clarification:
                    logger.info(f"Clarification needed: {clarification}")
                    yield {'chunk': clarification, 'is_metadata': False}
                    return
                
            except Exception as task_error:
                logger.error(f"Error in task processing: {task_error}", exc_info=True)
                intent = QueryIntent.GENERAL
                confidence = 0.0
                clarification = None
            
            # Log intent results
            logger.info(f"Intent detected: {intent} (Confidence: {confidence})")
            
            # Initialize default context
            context = "I'll help you find information about APHRC's work."
            
            # If publication intent, try to retrieve real publications
            if intent == QueryIntent.PUBLICATION and self.redis_manager:
                logger.info("Publication intent detected, retrieving real publication data")
                try:
                    # Get relevant publications from Redis
                    publications, suggestion = await self.get_relevant_publications(message)
                    
                    # Handle no publications found, but offer suggestion if available
                    if not publications and suggestion:
                        logger.info("No publications found, but suggestion available")
                        yield {'chunk': suggestion, 'is_metadata': False}
                        return
                    
                    if publications:
                        # Format publications into readable context
                        context = self.format_publication_context(publications)
                        logger.info(f"Retrieved and formatted {len(publications)} publications for context")
                    else:
                        context = "I don't have specific publications on this topic. I can provide general information about APHRC's research."
                        logger.info("No relevant publications found in Redis")
                except Exception as pub_error:
                    logger.error(f"Error retrieving publications: {pub_error}")
                    context = "I can help you find information about APHRC's publications, though I'm having trouble accessing specific details right now."
            
            # Manage context window
            logger.debug("Managing context window")
            self.manage_context_window({'text': context, 'query': message})
            
            # Prepare messages for model
            logger.debug("Preparing system and human messages")
            system_message = self._create_system_message(intent)
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Context: {context}\n\nQuery: {message}")
            ]
            
            # Initialize response tracking
            response_chunks = []
            buffer = ""
            
            try:
                logger.info("Initializing model streaming")
                model = self.get_gemini_model()
                
                try:
                    # Get response with proper error handling
                    response_data = await model.agenerate([messages])
                    
                    if not response_data.generations or not response_data.generations[0]:
                        logger.warning("Empty response received from model")
                        yield {'chunk': "I'm sorry, but I couldn't generate a response at this time.", 'is_metadata': False}
                        return
                    
                    # Process the main response content
                    content = response_data.generations[0][0].text
                    
                    if not content:
                        logger.warning("Empty content received from model")
                        yield {'chunk': "I'm sorry, but I couldn't generate a response at this time.", 'is_metadata': False}
                        return
                    
                    # Process the content in chunks for streaming-like behavior
                    remaining_content = content
                    while remaining_content:
                        # Find a good breaking point
                        end_pos = min(100, len(remaining_content))
                        if end_pos < len(remaining_content):
                            # Try to break at a sentence or paragraph
                            for break_char in ['. ', '! ', '? ', '\n']:
                                pos = remaining_content[:end_pos].rfind(break_char)
                                if pos > 0:
                                    end_pos = pos + len(break_char)
                                    break
                        
                        # Extract current chunk
                        current_chunk = remaining_content[:end_pos]
                        remaining_content = remaining_content[end_pos:]
                        
                        # Save and yield chunk
                        response_chunks.append(current_chunk)
                        logger.debug(f"Yielding chunk (length: {len(current_chunk)})")
                        yield {'chunk': current_chunk, 'is_metadata': False}
                    
                    # Prepare complete response
                    complete_response = ''.join(response_chunks)
                    logger.info(f"Complete response generated. Total length: {len(complete_response)}")
                    
                    # Yield metadata
                    logger.debug("Preparing and yielding metadata")
                    yield {
                        'is_metadata': True,
                        'metadata': {
                            'response': complete_response,
                            'timestamp': datetime.now().isoformat(),
                            'metrics': {
                                'response_time': time.time() - start_time,
                                'intent': {'type': intent.value, 'confidence': confidence},
                                'quality': self._get_default_quality()  # Quality metrics
                            },
                            'error_occurred': False
                        }
                    }
                except Exception as stream_error:
                    logger.error(f"Error in response processing: {stream_error}", exc_info=True)
                    error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
                    yield {'chunk': error_message, 'is_metadata': False}
            
            except Exception as e:
                logger.error(f"Critical error generating response: {e}", exc_info=True)
                error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
                
                # Yield error chunk
                yield {'chunk': error_message, 'is_metadata': False}
                
                # Yield error metadata
                yield {
                    'is_metadata': True,
                    'metadata': {
                        'response': error_message,
                        'timestamp': datetime.now().isoformat(),
                        'metrics': {
                            'response_time': time.time() - start_time,
                            'intent': {'type': 'error', 'confidence': 0.0},
                            'quality': self._get_default_quality()  # Use default quality metrics
                        },
                        'error_occurred': True
                    }
                }
        
        except Exception as e:
            logger.error(f"Critical error in generate_async_response: {e}", exc_info=True)
            error_message = "I apologize for the inconvenience. Could you please rephrase your question?"
            
            # Yield error chunk
            yield {'chunk': error_message, 'is_metadata': False}
            
            # Yield error metadata
            yield {
                'is_metadata': True,
                'metadata': {
                    'response': error_message,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'response_time': time.time() - start_time,
                        'intent': {'type': 'error', 'confidence': 0.0},
                        'quality': self._get_default_quality()
                    },
                    'error_occurred': True
                }
            }
        
        finally:
            logger.info(f"Async response generation completed. Total time: {time.time() - start_time:.2f} seconds")

    async def _throttle_request(self):
        """Apply throttling to incoming requests to help prevent rate limits."""
        await self.new_method()

    async def new_method(self):
        """Apply throttling to incoming requests to help prevent rate limits."""
        # Get global throttling status from class variable
        if not hasattr(self.__class__, '_last_request_time'):
            self.__class__._last_request_time = 0
            self.__class__._request_count = 0
            self.__class__._throttle_lock = asyncio.Lock()
            
        async with self.__class__._throttle_lock:
            current_time = time.time()
            time_since_last = current_time - self.__class__._last_request_time
                
            # Reset counter if more than 1 minute has passed
            if time_since_last > 60:
                self.__class__._request_count = 0
                    
            # Increment request counter
            self.__class__._request_count += 1
                
            # Calculate delay based on request count within the minute
            # As we approach Gemini's limits, add increasingly longer delays
            if self.__class__._request_count > 50:  # Getting close to limit
                delay = 2.0  # 2 seconds
            elif self.__class__._request_count > 30:
                delay = 1.0  # 1 second
            elif self.__class__._request_count > 20:
                delay = 0.5  # 0.5 seconds
            elif self.__class__._request_count > 10:
                delay = 0.2  # 0.2 seconds
            else:
                delay = 0
                    
            # Add randomization to prevent request bunching
            if delay > 0:
                jitter = delay * 0.2 * (random.random() * 2 - 1)  # ±20% jitter
                delay += jitter
                logger.debug(f"Adding throttling delay of {delay:.2f}s (request {self.__class__._request_count})")
                await asyncio.sleep(delay)
                    
            # Update last request time
            self.__class__._last_request_time = time.time()

class ConversationHelper:
    """New helper class for conversation enhancements"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.summary_cache = {}
        
    async def generate_summary(self, conversation_id: str, turns: List[Dict]) -> str:
        """Generate conversation summary"""
        if conversation_id in self.summary_cache:
            return self.summary_cache[conversation_id]
            
        key_points = []
        for turn in turns[-5:]:  # Last 5 turns
            if turn.get('intent') == QueryIntent.PUBLICATION:
                key_points.append(f"Discussed publications about {turn.get('topics', 'various topics')}")
            elif turn.get('intent') == QueryIntent.NAVIGATION:
                key_points.append(f"Asked about {turn.get('section', 'website sections')}")
        
        summary = "Our conversation so far:\n- " + "\n- ".join(key_points[-3:])  # Last 3 points
        self.summary_cache[conversation_id] = summary
        return summary
        
    def get_suggested_questions(self, last_intent: QueryIntent) -> List[str]:
        """Get context-aware follow-up questions"""
        suggestions = {
            QueryIntent.PUBLICATION: [
                "Find more publications by the same authors",
                "See related publications from recent years",
                "Get citation formats for these papers"
            ],
            QueryIntent.NAVIGATION: [
                "Show more sections in this category",
                "Find contact information for this department",
                "Open this page in my browser"
            ],
            QueryIntent.GENERAL: [
                "What other programs does APHRC offer?",
                "Tell me about upcoming events",
                "Who are the key researchers in this field?"
            ]
        }
        return suggestions.get(last_intent, [])