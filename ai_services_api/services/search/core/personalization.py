import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from ai_services_api.services.message.core.database import get_db_connection

# Configure logger
logger = logging.getLogger(__name__)

async def get_user_search_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent search history for a user from the database.
    
    Args:
        user_id: User identifier
        limit: Maximum number of history items to return
    
    Returns:
        List of recent searches with their timestamps
    """
    logger.info(f"Retrieving search history for user {user_id}")
    conn = None
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Query to get recent searches with timestamp
        cur.execute("""
            SELECT sa.query, sa.timestamp 
            FROM search_analytics sa
            JOIN search_sessions ss ON sa.search_id = ss.id
            WHERE ss.user_id = %s
            ORDER BY sa.timestamp DESC
            LIMIT %s
        """, (user_id, limit))
        
        history = []
        rows = cur.fetchall()
        for row in rows:
            history.append({
                "query": row[0],
                "timestamp": row[1].isoformat() if row[1] else None
            })
        
        logger.info(f"Retrieved {len(history)} search history items for user {user_id}")
        return history
    
    except Exception as e:
        logger.error(f"Error fetching search history: {e}")
        return []
    finally:
        if conn:
            conn.close()

async def personalize_suggestions(
    suggestions: List[Dict[str, Any]], 
    user_id: str, 
    partial_query: str
) -> List[Dict[str, Any]]:
    """
    Re-rank suggestions based on user's search history.
    
    Args:
        suggestions: Original search suggestions from predictor
        user_id: User identifier
        partial_query: The partial query being typed
        
    Returns:
        Personalized list of suggestions
    """
    logger.info(f"Personalizing suggestions for user {user_id}")
    
    # Get user history
    history = await get_user_search_history(user_id)
    
    # If no history, return original suggestions
    if not history:
        return suggestions
    
    # Extract search terms from history and create a frequency map
    term_weights = {}
    for item in history:
        query = item.get("query", "").lower()
        
        # Split into words
        words = query.split()
        for word in words:
            if len(word) >= 3:  # Only consider significant words
                term_weights[word] = term_weights.get(word, 0) + 1
    
    # Also get previously selected suggestions
    selected_suggestions = await get_selected_suggestions(user_id)
    for suggestion in selected_suggestions:
        suggestion_text = suggestion.lower()
        words = suggestion_text.split()
        for word in words:
            if len(word) >= 3:
                # Give higher weight to words in selected suggestions
                term_weights[word] = term_weights.get(word, 0) + 2
    
    # Boost scores for suggestions that contain frequently searched terms
    personalized_suggestions = []
    for suggestion in suggestions:
        text = suggestion.get("text", "").lower()
        base_score = suggestion.get("score", 0.5)
        
        # Calculate boost based on term frequency in user history
        boost = 0.0
        for term, weight in term_weights.items():
            if term in text:
                # More frequently searched terms give higher boost
                boost += min(0.3, weight * 0.05)  # Cap at 0.3 boost
        
        # Check for exact matches with selected suggestions (higher boost)
        for suggestion_text in selected_suggestions:
            if suggestion_text.lower() == text:
                boost += 0.2  # Significant boost for previously selected suggestions
                break
        
        # Final personalized score
        personalized_score = min(1.0, base_score + boost)
        
        personalized_suggestions.append({
            "text": suggestion.get("text", ""),
            "source": suggestion.get("source", "personalized"),
            "score": personalized_score
        })
    
    # Sort by personalized score
    personalized_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return personalized_suggestions

async def get_selected_suggestions(user_id: str, limit: int = 10) -> List[str]:
    """
    Get recently selected suggestions for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of items to return
        
    Returns:
        List of selected suggestion texts
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Query to get recent selections
        cur.execute("""
            SELECT selected_suggestion
            FROM suggestion_selections
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (user_id, limit))
        
        return [row[0] for row in cur.fetchall()]
        
    except Exception as e:
        logger.error(f"Error fetching selected suggestions: {e}")
        return []
    finally:
        if conn:
            conn.close()

async def track_selected_suggestion(user_id: str, partial_query: str, selected_suggestion: str):
    """
    Track which suggestion the user selected to improve future personalization.
    
    Args:
        user_id: User identifier
        partial_query: The partial query that was typed
        selected_suggestion: The suggestion that was selected
    """
    logger.info(f"Tracking selected suggestion for user {user_id}")
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Record the selection in the database
        cur.execute("""
            INSERT INTO suggestion_selections
                (user_id, partial_query, selected_suggestion, timestamp)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """, (user_id, partial_query, selected_suggestion))
        
        conn.commit()
        logger.info(f"Successfully recorded suggestion selection")
        
    except Exception as e:
        logger.error(f"Error tracking suggestion selection: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()