import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio

logger = logging.getLogger(__name__)

class UserInterestTracker:
    """
    Tracks user interests based on interactions with publications and experts.
    This class is designed to be plugged into the existing system without requiring
    changes elsewhere in the codebase.
    """
    
    def __init__(self, db_connector=None):
        """Initialize with optional database connector."""
        self.db_connector = db_connector
        # Cache to reduce database writes
        self._pending_logs = []
        self._flush_lock = asyncio.Lock()
    
    async def log_interaction(self, user_id: str, session_id: str, query: str, 
                            interaction_type: str, content_id: Optional[str] = None,
                            response_quality: float = 0.0) -> bool:
        """
        Log a user interaction with a publication or expert.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Current session identifier
            query: The user's original query
            interaction_type: Type of interaction ('publication', 'expert', 'general')
            content_id: ID of the content if applicable
            response_quality: Quality score of the response (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.db_connector:
                logger.warning("No database connector provided for UserInterestTracker")
                return False
                
            # Add to database
            async with self.db_connector.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO user_interest_logs 
                        (user_id, session_id, query, interaction_type, content_id, response_quality, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                """, user_id, session_id, query, interaction_type, content_id, response_quality)
                
            logger.debug(f"Logged {interaction_type} interaction for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging user interaction: {e}")
            return False
    
    async def track_topic_interests(self, user_id: str, topics: Union[List, Dict], 
                                topic_type: str) -> List[str]:
        """
        Track specific topics of interest for a user.
        
        Args:
            user_id: Unique identifier for the user
            topics: List or Dict of topics extracted from content
            topic_type: Type of topics ('publication_topic', 'publication_domain', 'expert_expertise')
            
        Returns:
            List[str]: List of tracked topic keys
        """
        if not topics:
            return []
            
        # Normalize topics to list
        topic_list = []
        if isinstance(topics, dict):
            # Handle JSONB format from database
            for key, value in topics.items():
                if isinstance(value, list):
                    topic_list.extend(value)
                else:
                    topic_list.append(str(value))
        elif isinstance(topics, list):
            topic_list = [str(t) for t in topics]
        else:
            topic_list = [str(topics)]
        
        # Filter empty or None values
        topic_list = [t for t in topic_list if t and t.strip()]
        
        if not topic_list:
            return []
            
        try:
            if not self.db_connector:
                logger.warning("No database connector provided for UserInterestTracker")
                return []
                
            tracked_topics = []
            async with self.db_connector.get_connection() as conn:
                for topic in topic_list:
                    # Clean the topic key
                    topic_key = topic.strip().lower()
                    if not topic_key:
                        continue
                        
                    # Check if topic exists and update counter, or insert new
                    await conn.execute("""
                        INSERT INTO user_topic_interests 
                            (user_id, topic_key, topic_type, interaction_count, last_interaction)
                        VALUES ($1, $2, $3, 1, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id, topic_key, topic_type) 
                        DO UPDATE SET 
                            interaction_count = user_topic_interests.interaction_count + 1,
                            last_interaction = CURRENT_TIMESTAMP,
                            engagement_score = user_topic_interests.engagement_score * 0.9 + 1.0
                    """, user_id, topic_key, topic_type)
                    
                    tracked_topics.append(topic_key)
            
            logger.debug(f"Tracked {len(tracked_topics)} {topic_type} interests for user {user_id}")
            return tracked_topics
            
        except Exception as e:
            logger.error(f"Error tracking topic interests: {e}")
            return []
    
    async def get_user_interests(self, user_id: str, limit: int = 5) -> Dict[str, List[str]]:
        """
        Get a user's top interests across all categories.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of interests per category
            
        Returns:
            Dict[str, List[str]]: Dictionary of interests by type
        """
        try:
            if not self.db_connector:
                logger.warning("No database connector provided for UserInterestTracker")
                return {}
                
            interests = {
                'publication_topic': [],
                'publication_domain': [], 
                'expert_expertise': []
            }
                
            async with self.db_connector.get_connection() as conn:
                # Query top interests for each type
                for topic_type in interests.keys():
                    rows = await conn.fetch("""
                        SELECT topic_key, interaction_count, engagement_score
                        FROM user_topic_interests
                        WHERE user_id = $1 AND topic_type = $2
                        ORDER BY engagement_score DESC, interaction_count DESC, last_interaction DESC
                        LIMIT $3
                    """, user_id, topic_type, limit)
                    
                    interests[topic_type] = [row['topic_key'] for row in rows]
            
            return interests
            
        except Exception as e:
            logger.error(f"Error retrieving user interests: {e}")
            return {}
    
    async def extract_topics_from_publications(self, publications: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extract topics and domains from publication data.
        
        Args:
            publications: List of publication dictionaries
            
        Returns:
            Tuple[List[str], List[str]]: (topics, domains)
        """
        topics = []
        domains = []
        
        for pub in publications:
            # Extract topics
            try:
                pub_topics = pub.get('topics', {})
                if isinstance(pub_topics, str):
                    try:
                        pub_topics = json.loads(pub_topics)
                    except:
                        pub_topics = {'other': [pub_topics]}
                
                if isinstance(pub_topics, dict):
                    for category, values in pub_topics.items():
                        if isinstance(values, list):
                            topics.extend(values)
                        else:
                            topics.append(str(values))
            except Exception as e:
                logger.debug(f"Error extracting publication topics: {e}")
            
            # Extract domains
            try:
                pub_domains = pub.get('domains', [])
                if isinstance(pub_domains, str):
                    try:
                        pub_domains = json.loads(pub_domains)
                    except:
                        pub_domains = [pub_domains]
                
                if isinstance(pub_domains, list):
                    domains.extend([str(d) for d in pub_domains])
            except Exception as e:
                logger.debug(f"Error extracting publication domains: {e}")
        
        # Clean and deduplicate
        topics = list(set([t.strip().lower() for t in topics if t and t.strip()]))
        domains = list(set([d.strip().lower() for d in domains if d and d.strip()]))
        
        return topics, domains
    
    async def build_user_interest_context(self, user_id: str) -> str:
        """
        Build a context string based on user interests for enhancing responses.
        Now includes navigation interests.
        
        Args:
            user_id: User identifier
            
        Returns:
            str: A formatted context string with user interests
        """
        interests = await self.get_user_interests(user_id)
        if not interests or not any(interests.values()):
            return ""
        
        context_parts = []
        context_parts.append("Based on your interests, I'll highlight relevant information about:")
        
        # Format publication topics
        if interests.get('publication_topic', []):
            topics = interests['publication_topic'][:3]  # Top 3 topics
            context_parts.append(f"- Research topics: {', '.join(topics)}")
        
        # Format publication domains
        if interests.get('publication_domain', []):
            domains = interests['publication_domain'][:3]  # Top 3 domains
            context_parts.append(f"- Research domains: {', '.join(domains)}")
        
        # Format expert expertise
        if interests.get('expert_expertise', []):
            expertise = interests['expert_expertise'][:3]  # Top 3 expertise areas
            context_parts.append(f"- Expertise areas: {', '.join(expertise)}")
        
        # ADDED: Format navigation interests
        if interests.get('navigation_section', []):
            sections = interests['navigation_section'][:3]  # Top 3 sections
            context_parts.append(f"- Website sections: {', '.join(sections)}")
        
        if interests.get('navigation_topic', []):
            nav_topics = interests['navigation_topic'][:3]  # Top 3 navigation topics
            context_parts.append(f"- Website resources: {', '.join(nav_topics)}")
        
        return "\n".join(context_parts)
    
    async def extract_navigation_from_sections(self, sections: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extract navigation sections and related keywords from navigation data.
        
        Args:
            sections: List of navigation section dictionaries
            
        Returns:
            Tuple[List[str], List[str]]: Extracted section names and keywords
        """
        section_names = []
        keywords = []
        
        for section in sections:
            # Extract section title
            if 'title' in section and section['title']:
                section_names.append(section['title'])
            
            # Extract section keywords
            if 'keywords' in section:
                if isinstance(section['keywords'], list):
                    keywords.extend([k for k in section['keywords'] if k])
                elif isinstance(section['keywords'], str):
                    try:
                        # Try to parse JSON
                        parsed_keywords = json.loads(section['keywords'])
                        if isinstance(parsed_keywords, list):
                            keywords.extend([k for k in parsed_keywords if k])
                    except json.JSONDecodeError:
                        # Handle as comma-separated string
                        keywords.extend([k.strip() for k in section['keywords'].split(',') if k.strip()])
        
        # Clean and deduplicate
        clean_sections = list(set([s.strip() for s in section_names if s.strip()]))
        clean_keywords = list(set([k.strip() for k in keywords if k.strip()]))
        
        return clean_sections, clean_keywords
    
    async def extract_expertise_from_experts(self, experts: List[Dict]) -> List[str]:
        """
        Extract expertise areas from expert data.
        
        Args:
            experts: List of expert dictionaries
            
        Returns:
            List[str]: List of expertise areas
        """
        expertise_areas = []
        
        for expert in experts:
            # Try different fields where expertise might be stored
            for field in ['knowledge_expertise', 'expertise', 'research_interests', 'keywords']:
                try:
                    exp_data = expert.get(field, {})
                    if isinstance(exp_data, str):
                        try:
                            exp_data = json.loads(exp_data)
                        except:
                            exp_data = [exp_data]
                    
                    if isinstance(exp_data, dict):
                        for category, values in exp_data.items():
                            if isinstance(values, list):
                                expertise_areas.extend(values)
                            else:
                                expertise_areas.append(str(values))
                    elif isinstance(exp_data, list):
                        expertise_areas.extend([str(e) for e in exp_data])
                except Exception as e:
                    logger.debug(f"Error extracting expert expertise: {e}")
        
        # Clean and deduplicate
        expertise_areas = list(set([e.strip().lower() for e in expertise_areas if e and e.strip()]))
        
        return expertise_areas
    
    async def build_user_interest_context(self, user_id: str) -> str:
        """
        Build a context string based on user interests for enhancing responses.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            str: Context string for the LLM
        """
        interests = await self.get_user_interests(user_id)
        
        if not any(interests.values()):
            return ""
            
        context_parts = ["Based on your previous interactions, you've shown interest in:"]
        
        # Add publication topics if available
        if interests['publication_topic']:
            topics_str = ", ".join(interests['publication_topic'][:3])
            context_parts.append(f"- Research topics: {topics_str}")
            
        # Add publication domains if available
        if interests['publication_domain']:
            domains_str = ", ".join(interests['publication_domain'][:3])
            context_parts.append(f"- Research domains: {domains_str}")
            
        # Add expert expertise if available
        if interests['expert_expertise']:
            expertise_str = ", ".join(interests['expert_expertise'][:3])
            context_parts.append(f"- Expert areas: {expertise_str}")
            
        return "\n".join(context_parts)