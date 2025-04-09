import asyncio
import logging
from typing import List, Dict, Any, Optional
import psycopg2
import re

class DatabaseSuggestionGenerator:
    """
    A flexible suggestion generator that queries the database 
    based on partial input across multiple fields.
    """
    
    def __init__(self, connection_params):
        """
        Initialize the suggestion generator with database connection parameters.
        
        Args:
            connection_params (dict): Database connection parameters
        """
        self.connection_params = connection_params
        self.logger = logging.getLogger(__name__)
    
    def _get_db_connection(self):
        """
        Create a database connection.
        
        Returns:
            psycopg2 connection object
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
    
    def _normalize_input(self, input_str: str) -> str:
        """
        Normalize input for consistent matching.
        
        Args:
            input_str (str): Input string to normalize
        
        Returns:
            Normalized string
        """
        return input_str.lower().strip()
    
    async def generate_suggestions(
        self, 
        partial_query: str, 
        limit: int = 10, 
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on partial query.
        
        Args:
            partial_query (str): Partial input to generate suggestions for
            limit (int): Maximum number of suggestions to return
            context (str, optional): Context of suggestion (name, theme, etc.)
        
        Returns:
            List of suggestion dictionaries
        """
        # Validate input
        if not partial_query or len(partial_query) < 2:
            return []
        
        # Normalize input
        normalized_query = self._normalize_input(partial_query)
        
        # Establish database connection
        conn = self._get_db_connection()
        
        try:
            # Create cursor
            cur = conn.cursor()
            
            # Prepare suggestions lists
            suggestions = []
            
            # 1. Expert Name Suggestions
            name_suggestions = self._get_expert_name_suggestions(
                cur, normalized_query, limit, context
            )
            suggestions.extend(name_suggestions)
            
            # 2. Theme/Designation Suggestions
            theme_suggestions = self._get_expert_theme_suggestions(
                cur, normalized_query, limit, context
            )
            suggestions.extend(theme_suggestions)
            
            # 3. Resource Title Suggestions
            resource_suggestions = self._get_resource_suggestions(
                cur, normalized_query, limit, context
            )
            suggestions.extend(resource_suggestions)
            
            # Close cursor and connection
            cur.close()
            conn.close()
            
            # Sort and deduplicate suggestions
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            
            # Sort by relevance and limit
            sorted_suggestions = sorted(
                unique_suggestions, 
                key=lambda x: x.get('score', 0), 
                reverse=True
            )[:limit]
            
            return sorted_suggestions
        
        except Exception as e:
            self.logger.error(f"Suggestion generation error: {e}")
            if conn:
                conn.close()
            return []
    
    def _get_expert_name_suggestions(
        self, 
        cur, 
        normalized_query: str, 
        limit: int, 
        context: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate name-based suggestions from experts table.
        
        Args:
            cur: Database cursor
            normalized_query: Normalized search input
            limit: Maximum suggestions to return
            context: Optional context filter
        
        Returns:
            List of name suggestions
        """
        # Query to match first or last names
        query = """
        SELECT 
            first_name, 
            last_name, 
            designation, 
            theme,
            (
                similarity(lower(first_name), %s) * 0.5 + 
                similarity(lower(last_name), %s) * 0.5
            ) as name_score
        FROM experts_expert
        WHERE 
            (lower(first_name) LIKE %s OR lower(last_name) LIKE %s)
            AND is_active = true
        ORDER BY name_score DESC
        LIMIT %s
        """
        
        # Prepare pattern for LIKE
        like_pattern = f"{normalized_query}%"
        
        # Execute query
        cur.execute(query, (
            normalized_query, 
            normalized_query, 
            like_pattern, 
            like_pattern, 
            limit
        ))
        
        # Process results
        suggestions = []
        for row in cur.fetchall():
            first_name, last_name, designation, theme, score = row
            full_name = f"{first_name} {last_name}".strip()
            
            suggestions.append({
                "text": full_name,
                "type": "name",
                "score": score,
                "metadata": {
                    "designation": designation,
                    "theme": theme
                }
            })
        
        return suggestions
    
    def _get_expert_theme_suggestions(
        self, 
        cur, 
        normalized_query: str, 
        limit: int, 
        context: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate theme/designation suggestions from experts table.
        
        Args:
            cur: Database cursor
            normalized_query: Normalized search input
            limit: Maximum suggestions to return
            context: Optional context filter
        
        Returns:
            List of theme suggestions
        """
        # Query to match themes and designations
        query = """
        SELECT DISTINCT
            theme, 
            designation,
            similarity(lower(theme), %s) as theme_score,
            similarity(lower(designation), %s) as designation_score
        FROM experts_expert
        WHERE 
            (lower(theme) LIKE %s OR lower(designation) LIKE %s)
            AND is_active = true
        ORDER BY 
            theme_score DESC, 
            designation_score DESC
        LIMIT %s
        """
        
        # Prepare pattern for LIKE
        like_pattern = f"{normalized_query}%"
        
        # Execute query
        cur.execute(query, (
            normalized_query, 
            normalized_query, 
            like_pattern, 
            like_pattern, 
            limit
        ))
        
        # Process results
        suggestions = []
        for row in cur.fetchall():
            theme, designation, theme_score, designation_score = row
            
            # Determine which field matched and use its score
            score = max(theme_score, designation_score)
            matched_text = theme if theme_score > designation_score else designation
            
            suggestions.append({
                "text": matched_text,
                "type": "theme" if theme_score > designation_score else "designation",
                "score": score,
                "metadata": {
                    "theme": theme,
                    "designation": designation
                }
            })
        
        return suggestions
    
    def _get_resource_suggestions(
        self, 
        cur, 
        normalized_query: str, 
        limit: int, 
        context: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate resource title suggestions.
        
        Args:
            cur: Database cursor
            normalized_query: Normalized search input
            limit: Maximum suggestions to return
            context: Optional context filter
        
        Returns:
            List of resource suggestions
        """
        # Query to match resource titles
        query = """
        SELECT 
            title, 
            type,
            domains,
            similarity(lower(title), %s) as title_score
        FROM resources_resource
        WHERE 
            lower(title) LIKE %s
        ORDER BY title_score DESC
        LIMIT %s
        """
        
        # Prepare pattern for LIKE
        like_pattern = f"{normalized_query}%"
        
        # Execute query
        cur.execute(query, (
            normalized_query, 
            like_pattern, 
            limit
        ))
        
        # Process results
        suggestions = []
        for row in cur.fetchall():
            title, resource_type, domains, score = row
            
            suggestions.append({
                "text": title,
                "type": "resource",
                "score": score,
                "metadata": {
                    "resource_type": resource_type,
                    "domains": domains
                }
            })
        
        return suggestions
    
    def _deduplicate_suggestions(
        self, 
        suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate suggestions while preserving highest score.
        
        Args:
            suggestions: List of suggestion dictionaries
        
        Returns:
            Deduplicated list of suggestions
        """
        # Use a dictionary to track unique suggestions
        unique_suggestions = {}
        
        for suggestion in suggestions:
            text = suggestion.get('text', '').lower()
            
            # If suggestion not seen or has higher score, update
            if (text not in unique_suggestions or 
                suggestion.get('score', 0) > unique_suggestions[text].get('score', 0)):
                unique_suggestions[text] = suggestion
        
        return list(unique_suggestions.values())