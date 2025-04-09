import asyncio
import logging
import math
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from ai_services_api.services.message.core.db_pool import get_pooled_connection, return_connection, DatabaseConnection, safe_background_task
import redis

from ai_services_api.services.message.core.database import get_db_connection


# Configure logger
logger = logging.getLogger(__name__)

# Modified version of update_trending_suggestions to work with safe_background_task
async def _update_trending_suggestions_task(conn, partial_query: str, selected_suggestion: str):
    """
    Background task implementation that accepts a connection.
    This version is designed to be used with safe_background_task.
    
    Args:
        conn: Database connection (provided by safe_background_task)
        partial_query: The partial query that was typed
        selected_suggestion: The suggestion that was selected
    """
    try:
        # Connect to Redis for trending data
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)), 
            db=int(os.getenv('REDIS_TRENDING_DB', 2)),
            decode_responses=True
        )
        
        current_time = int(time.time())
        
        # Normalize the query and suggestion
        normalized_query = partial_query.lower().strip()
        normalized_suggestion = selected_suggestion.lower().strip()
        
        # Update trending data with different time windows
        # 1-hour trending (short-term)
        hour_trending_key = f"trending:hour:{int(current_time / 3600)}"
        redis_client.zincrby(hour_trending_key, 1, normalized_suggestion)
        # Set expiration for 2 hours (to ensure overlap)
        redis_client.expire(hour_trending_key, 7200)
        
        # 24-hour trending (medium-term)
        day_trending_key = f"trending:day:{int(current_time / 86400)}"
        redis_client.zincrby(day_trending_key, 1, normalized_suggestion)
        # Set expiration for 48 hours (to ensure overlap)
        redis_client.expire(day_trending_key, 172800)
        
        # Update query-specific trending
        query_trending_key = f"trending:query:{normalized_query}"
        redis_client.zincrby(query_trending_key, 1, normalized_suggestion)
        redis_client.expire(query_trending_key, 604800)  # 1 week
        
        # Update popularity weights for hybrid ranking
        popularity_key = f"query_popularity:{normalized_query}"
        
        # Get existing popularity data or create new
        popularity_data = redis_client.get(popularity_key)
        popularity_weights = json.loads(popularity_data) if popularity_data else {}
        
        # Update weight for this suggestion
        current_weight = popularity_weights.get(normalized_suggestion, 0)
        popularity_weights[normalized_suggestion] = min(1.0, current_weight + 0.05)
        
        # Apply decay to other suggestions
        for suggestion in popularity_weights:
            if suggestion != normalized_suggestion:
                popularity_weights[suggestion] *= 0.99  # Slight decay
        
        # Store updated popularity weights
        redis_client.setex(
            popularity_key,
            604800,  # 1 week
            json.dumps(popularity_weights)
        )
        
        # Record this update in the database for analytics
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO suggestion_selections_analytics
                        (query, suggestion, timestamp)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                """, (normalized_query, normalized_suggestion))
                conn.commit()
                cur.close()
            except Exception as db_error:
                logger.error(f"Error recording trending data in database: {db_error}")
                if hasattr(conn, 'rollback'):
                    conn.rollback()
        
        logger.info(f"Updated trending data for suggestion: {normalized_suggestion}")
        
    except Exception as e:
        logger.error(f"Error updating trending suggestions: {e}")

# Wrapper function to launch the background task with proper connection handling
async def update_trending_suggestions(partial_query: str, selected_suggestion: str):
    """
    Update trending suggestions data - wrapper function that uses safe_background_task
    for proper connection management.
    
    Args:
        partial_query: The partial query that was typed
        selected_suggestion: The suggestion that was selected
    """
    try:
        await safe_background_task(
            _update_trending_suggestions_task, 
            partial_query, 
            selected_suggestion
        )
    except Exception as e:
        logger.error(f"Failed to start trending suggestions update task: {e}")

# Modified version of _record_search_history to work with safe_background_task
async def _record_search_history_task(conn, user_id: str, query: str, result_count: int):
    """
    Record search in user history - implementation for safe_background_task.
    
    Args:
        conn: Database connection (provided by safe_background_task)
        user_id: User identifier
        query: The search query
        result_count: Number of results found
    """
    try:
        # Connect to Redis for search history
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_HISTORY_DB', 3)),
            decode_responses=True
        )
        
        # Create history entry
        history_entry = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "result_count": result_count
        }
        
        # Add to user-specific history list with limit
        history_key = f"search_history:{user_id}"
        redis_client.lpush(history_key, json.dumps(history_entry))
        redis_client.ltrim(history_key, 0, 99)  # Keep most recent 100 searches
        redis_client.expire(history_key, 86400 * 30)  # 30-day expiry
        
        # Also add to global search history for trending analytics
        popular_key = "popular_searches"
        redis_client.zincrby(popular_key, 1, query.lower())
        redis_client.expire(popular_key, 86400 * 7)  # 7-day expiry for popular searches
        
        # Also record in the database if connection is provided
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO user_search_history
                        (user_id, query, result_count, timestamp)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                """, (user_id, query, result_count))
                conn.commit()
                cur.close()
            except Exception as db_error:
                logger.error(f"Error recording search history in database: {db_error}")
                if hasattr(conn, 'rollback'):
                    conn.rollback()
        
    except Exception as e:
        logger.error(f"Error recording search history: {e}")

# Wrapper function for _record_search_history with safe_background_task
async def record_search_history(user_id: str, query: str, result_count: int):
    """
    Record search in user history - wrapper that uses safe_background_task.
    
    Args:
        user_id: User identifier
        query: The search query
        result_count: Number of results found
    """
    try:
        await safe_background_task(_record_search_history_task, user_id, query, result_count)
    except Exception as e:
        logger.error(f"Failed to start search history recording task: {e}")


async def personalize_suggestions(
    suggestions: List[Dict[str, Any]], 
    user_id: str, 
    partial_query: str
) -> List[Dict[str, Any]]:
    """
    Re-rank suggestions based on user's search history with temporal relevance
    and collaborative filtering. Enhanced for dynamic filtering.
    
    Args:
        suggestions: Original search suggestions from predictor
        user_id: User identifier
        partial_query: The partial query being typed
        
    Returns:
        Personalized list of suggestions
    """
    logger.info(f"Personalizing suggestions for user {user_id}")
    
    # If partial_query is empty, just return suggestions as is
    if not partial_query:
        return suggestions
    
    # Get user history with timestamps
    history = await get_user_search_history(user_id)
    
    # Also get collaborative suggestions
    collaborative_suggestions = await get_collaborative_suggestions(
        user_id, partial_query, limit=3
    )
    
    # Combine with original suggestions
    combined_suggestions = suggestions.copy()
    seen_texts = set(s.get("text", "").lower() for s in suggestions)
    
    # Add collaborative suggestions if not already present and they match the prefix
    for collab_suggestion in collaborative_suggestions:
        suggestion_text = collab_suggestion.get("text", "").lower()
        if (suggestion_text not in seen_texts and 
            suggestion_text.startswith(partial_query.lower())):
            combined_suggestions.append(collab_suggestion)
            seen_texts.add(suggestion_text)
    
    # If no history and no collaborative data, return combined suggestions
    if not history and not collaborative_suggestions:
        return combined_suggestions
    
    # Extract search terms from history and create a weighted frequency map
    term_weights = {}
    
    # Get current time for temporal weighting
    current_time = datetime.now()
    
    # Process history items with temporal decay
    for item in history:
        query = item.get("query", "").lower()
        timestamp_str = item.get("timestamp")
        
        # Calculate recency weight based on timestamp
        recency_weight = 1.0  # Default weight
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                # Calculate days difference
                days_diff = (current_time - timestamp).days
                # Apply exponential decay: weight = exp(-days/30)
                # This gives ~0.97 weight for 1 day old, ~0.72 for 10 days old
                recency_weight = math.exp(-days_diff/30)
            except (ValueError, TypeError):
                pass
        
        # Split into words and apply weighted counting
        words = query.split()
        for word in words:
            if len(word) >= 3:  # Only consider significant words
                current_weight = term_weights.get(word, 0)
                # Add recency-weighted value
                term_weights[word] = current_weight + recency_weight
    
    # Also get previously selected suggestions with temporal weights
    selected_suggestions = await get_selected_suggestions_with_times(user_id)
    for suggestion, timestamp_str in selected_suggestions:
        # Calculate recency weight for selections (with higher base)
        recency_weight = 2.0  # Higher base weight for selections
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                days_diff = (current_time - timestamp).days
                # Selections decay slower than searches
                recency_weight = 2.0 * math.exp(-days_diff/45)
            except (ValueError, TypeError):
                pass
                
        suggestion_text = suggestion.lower()
        words = suggestion_text.split()
        for word in words:
            if len(word) >= 3:
                term_weights[word] = term_weights.get(word, 0) + recency_weight
    
    # Also consider domain-specific expertise relevance if available
    user_expertise = await get_user_expertise_areas(user_id)
    
    # Boost scores for suggestions that match user's expertise areas
    personalized_suggestions = []
    for suggestion in combined_suggestions:
        text = suggestion.get("text", "").lower()
        
        # Only include suggestions that start with our partial query
        if not text.startswith(partial_query.lower()):
            continue
            
        base_score = suggestion.get("score", 0.5)
        source = suggestion.get("source", "")
        
        # Calculate boost based on term frequency in user history
        history_boost = 0.0
        expertise_boost = 0.0
        exact_match_boost = 0.0
        collaborative_boost = 0.0
        
        # Apply term frequency boosts with diminishing returns
        for term, weight in term_weights.items():
            if term in text:
                # Apply diminishing returns: sqrt(weight) * 0.05
                # This prevents over-boosting highly frequent terms
                history_boost += min(0.25, math.sqrt(weight) * 0.05)
        
        # Check for exact matches with previously selected suggestions
        for suggestion_text, _ in selected_suggestions:
            if suggestion_text.lower() == text:
                exact_match_boost = 0.3  # Higher boost for exact matches
                break
                
        # Check expertise relevance
        for expertise in user_expertise:
            if expertise.lower() in text:
                expertise_boost = 0.15
                break
                
        # Apply collaborative filtering boost
        if source == "collaborative":
            collaborative_boost = 0.2
        
        # Combine all factors for final personalized score
        # History: 35%, Exact matches: 25%, Expertise: 15%, Collaborative: 10%, Base score: 15%
        history_component = history_boost * 0.35
        exact_match_component = exact_match_boost * 0.25
        expertise_component = expertise_boost * 0.15
        collaborative_component = collaborative_boost * 0.1
        base_component = base_score * 0.15
        
        personalized_score = min(1.0, 
            history_component + exact_match_component + expertise_component + 
            collaborative_component + base_component)
        
        personalized_suggestions.append({
            "text": suggestion.get("text", ""),
            "source": source,
            "score": personalized_score,
            "original_score": base_score
        })
    
    # Sort by personalized score
    personalized_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return personalized_suggestions


# Optimized Method #1: get_user_search_history
async def get_user_search_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent search history for a user from the database.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        user_id: User identifier
        limit: Maximum number of history items to return
    
    Returns:
        List of recent searches with their timestamps
    """
    logger.info(f"Retrieving search history for user {user_id}")
    
    try:
        with DatabaseConnection() as conn:
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
            
            cur.close()
            logger.info(f"Retrieved {len(history)} search history items for user {user_id}")
            return history
    
    except Exception as e:
        logger.error(f"Error fetching search history: {e}")
        return []

# Optimized Method #2: get_selected_suggestions_with_times
async def get_selected_suggestions_with_times(user_id: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    Get recently selected suggestions for a user with timestamps.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        user_id: User identifier
        limit: Maximum number of items to return
        
    Returns:
        List of tuples containing (selected_suggestion, timestamp)
    """
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Query to get recent selections with timestamps
            cur.execute("""
                SELECT selected_suggestion, timestamp
                FROM suggestion_selections
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            
            result = [(row[0], row[1].isoformat() if row[1] else None) for row in cur.fetchall()]
            cur.close()
            return result
        
    except Exception as e:
        logger.error(f"Error fetching selected suggestions with timestamps: {e}")
        return []

# Optimized Method #3: get_user_expertise_areas
async def get_user_expertise_areas(user_id: str) -> List[str]:
    """
    Get expertise areas for a user based on profile or past behavior.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of expertise areas
    """
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Try to get explicit expertise areas from user profile
            cur.execute("""
                SELECT expertise_areas 
                FROM user_profiles
                WHERE user_id = %s
            """, (user_id,))
            
            row = cur.fetchone()
            if row and row[0]:
                try:
                    # Try to parse as JSON array
                    return json.loads(row[0])
                except:
                    # If not valid JSON, treat as comma-separated list
                    return [area.strip() for area in row[0].split(',') if area.strip()]
            
            # If no explicit areas, derive from most common search areas
            cur.execute("""
                SELECT sa.query
                FROM search_analytics sa
                JOIN search_sessions ss ON sa.search_id = ss.id
                WHERE ss.user_id = %s
                ORDER BY sa.timestamp DESC
                LIMIT 50
            """, (user_id,))
            
            # Extract potential expertise areas from search queries
            potential_areas = {}
            for row in cur.fetchall():
                query = row[0].lower()
                words = query.split()
                
                # Count word frequency
                for word in words:
                    if len(word) > 3:  # Only consider significant words
                        potential_areas[word] = potential_areas.get(word, 0) + 1
            
            # Return top 5 most frequent terms as likely expertise areas
            sorted_areas = sorted(potential_areas.items(), key=lambda x: x[1], reverse=True)
            cur.close()
            return [area[0] for area in sorted_areas[:5]]
        
    except Exception as e:
        logger.error(f"Error fetching user expertise areas: {e}")
        return []

# Optimized Method #4: get_collaborative_suggestions
async def get_collaborative_suggestions(
    user_id: str, 
    partial_query: str, 
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get suggestions based on collaborative filtering of user behavior.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        user_id: User identifier
        partial_query: Partial query to find similar queries for
        limit: Maximum number of suggestions to return
        
    Returns:
        List of collaboratively filtered suggestions
    """
    logger.info(f"Getting collaborative suggestions for user {user_id} and query '{partial_query}'")
    
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Step 1: Find similar users who searched for similar queries
            # First, get this user's recent searches
            cur.execute("""
                SELECT sa.query 
                FROM search_analytics sa
                JOIN search_sessions ss ON sa.search_id = ss.id
                WHERE ss.user_id = %s
                ORDER BY sa.timestamp DESC
                LIMIT 10
            """, (user_id,))
            
            user_searches = [row[0].lower() for row in cur.fetchall()]
            
            # If user has no search history, return empty list
            if not user_searches:
                return []
                
            # Find similar users who searched for the same things
            placeholders = ','.join(['%s'] * len(user_searches))
            
            # This query finds users who searched for similar things
            cur.execute(f"""
                WITH similar_users AS (
                    SELECT DISTINCT ss.user_id, COUNT(*) as overlap_count
                    FROM search_analytics sa
                    JOIN search_sessions ss ON sa.search_id = ss.id
                    WHERE LOWER(sa.query) IN ({placeholders})
                    AND ss.user_id != %s
                    GROUP BY ss.user_id
                    ORDER BY overlap_count DESC
                    LIMIT 50
                )
                SELECT sa.query, COUNT(*) as frequency
                FROM search_analytics sa
                JOIN search_sessions ss ON sa.search_id = ss.id
                JOIN similar_users su ON ss.user_id = su.user_id
                WHERE LOWER(sa.query) LIKE %s
                AND LOWER(sa.query) NOT IN ({placeholders})
                GROUP BY sa.query
                ORDER BY frequency DESC
                LIMIT %s
            """, user_searches + [user_id, f'%{partial_query.lower()}%'] + user_searches + [limit])
            
            collaborative_suggestions = []
            for row in cur.fetchall():
                suggestion_text = row[0]
                frequency = row[1]
                
                # Normalize score between 0.6 and 0.9
                normalized_score = 0.6 + min(0.3, frequency / 10.0)
                
                collaborative_suggestions.append({
                    "text": suggestion_text,
                    "source": "collaborative",
                    "score": normalized_score
                })
            
            cur.close()
            return collaborative_suggestions
        
    except Exception as e:
        logger.error(f"Error getting collaborative suggestions: {e}")
        return []

# Optimized Method #5: get_selected_suggestions
async def get_selected_suggestions(user_id: str, limit: int = 10) -> List[str]:
    """
    Get recently selected suggestions for a user.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        user_id: User identifier
        limit: Maximum number of items to return
        
    Returns:
        List of selected suggestion texts
    """
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Query to get recent selections
            cur.execute("""
                SELECT selected_suggestion
                FROM suggestion_selections
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            
            result = [row[0] for row in cur.fetchall()]
            cur.close()
            return result
        
    except Exception as e:
        logger.error(f"Error fetching selected suggestions: {e}")
        return []

# Optimized Method #6: track_selected_suggestion
async def track_selected_suggestion(user_id: str, partial_query: str, selected_suggestion: str):
    """
    Track which suggestion the user selected to improve future personalization
    and update trending suggestions.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        user_id: User identifier
        partial_query: The partial query that was typed
        selected_suggestion: The suggestion that was selected
    """
    logger.info(f"Tracking selected suggestion for user {user_id}")
    
    try:
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Record the selection in the database
            cur.execute("""
                INSERT INTO suggestion_selections
                    (user_id, partial_query, selected_suggestion, timestamp)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (user_id, partial_query, selected_suggestion))
            
            selection_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Successfully recorded suggestion selection with ID {selection_id}")
            
            # Update trending suggestions in a separate background task
            # Background task now uses safe_background_task for proper connection management
            asyncio.create_task(
                update_trending_suggestions(partial_query, selected_suggestion)
            )
            
    except Exception as e:
        logger.error(f"Error tracking suggestion selection: {e}")

# Optimized Method #7: get_context_popular_queries
async def get_context_popular_queries(context: str, limit: int = 5) -> List[str]:
    """
    Get popular queries specific to a context type.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        context: Context type (name, theme, designation, publication)
        limit: Maximum number of items to return
        
    Returns:
        List of popular queries for this context
    """
    try:
        popular_queries = []
        
        with DatabaseConnection() as conn:
            cur = conn.cursor()
            
            # Get popular searches filtered by context
            if context == "name":
                # For name searches, look for patterns like "John Smith" or queries with "Name:"
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        (query LIKE '% %' AND length(query) < 30) OR
                        query LIKE 'name:%'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            elif context == "theme":
                # For theme searches, look for queries that were used in theme search
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        query LIKE 'theme:%' OR
                        search_type = 'theme'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            elif context == "designation":
                # For designation searches
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        query LIKE 'designation:%' OR
                        search_type = 'designation'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            elif context == "publication":
                # For publication searches
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE 
                        query LIKE 'publication:%' OR
                        search_type = 'publication'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            else:
                # For general searches
                cur.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics
                    WHERE query NOT LIKE '%:%'  -- Exclude prefixed queries
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT %s
                """, (limit,))
                
            # Process results
            for row in cur.fetchall():
                query = row[0]
                
                # Clean up query - remove context prefix if present
                if context and query.lower().startswith(f"{context}:"):
                    query = query[len(context)+1:].strip()
                
                if query and query.strip():
                    popular_queries.append(query.strip())
                    
            cur.close()
            return popular_queries
                
    except Exception as db_error:
        logger.error(f"Database error in context popular queries: {db_error}")
        
        # Fall back to Redis if available
        try:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_TRENDING_DB', 2)),
                decode_responses=True
            )
            
            # Try to get context-specific trending from Redis
            context_key = f"trending_context:{context or 'general'}"
            trending = redis_client.zrevrange(context_key, 0, limit-1)
            
            if trending:
                return trending
        except Exception as redis_error:
            logger.error(f"Redis error in context popular queries: {redis_error}")
        
        # If all else fails, return hardcoded defaults based on context
        if context == "name":
            return ["John Smith", "Jane Doe", "Michael Johnson", "David Williams", "Sara Chen"]
        elif context == "theme":
            return ["Neuroscience", "Data Science", "Artificial Intelligence", "Psychology", "Biochemistry"]
        elif context == "designation":
            return ["Professor", "Researcher", "Director", "Scientist", "Student"]
        elif context == "publication":
            return ["Journal of Neuroscience", "Nature", "Science", "IEEE Transactions", "PLOS ONE"]
        else:
            return ["Research", "Science", "Technology", "Medicine", "Engineering"]
        

async def get_trending_suggestions(partial_query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get trending suggestions relevant to the partial query.
    
    Args:
        partial_query: The partial query to get trending suggestions for
        limit: Maximum number of suggestions to return
        
    Returns:
        List of trending suggestion dictionaries
    """
    try:
        # Connect to Redis for trending data
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)), 
            db=int(os.getenv('REDIS_TRENDING_DB', 2)),
            decode_responses=True
        )
        
        current_time = int(time.time())
        normalized_query = partial_query.lower().strip()
        
        # First check for query-specific trending
        query_trending_key = f"trending:query:{normalized_query}"
        query_trending = redis_client.zrevrange(
            query_trending_key, 0, limit-1, withscores=True
        )
        
        # Then check the current hour's trending
        hour_trending_key = f"trending:hour:{int(current_time / 3600)}"
        hour_trending = redis_client.zrevrange(
            hour_trending_key, 0, limit*2-1, withscores=True
        )
        
        # Also check the current day's trending
        day_trending_key = f"trending:day:{int(current_time / 86400)}"
        day_trending = redis_client.zrevrange(
            day_trending_key, 0, limit*2-1, withscores=True
        )
        
        # Combine and deduplicate trending data
        trending_scores = {}
        
        # Add query-specific trending with highest weight
        for suggestion, score in query_trending:
            trending_scores[suggestion] = score * 2.0  # Higher weight for query-specific
        
        # Add hourly trending
        for suggestion, score in hour_trending:
            if suggestion not in trending_scores:
                trending_scores[suggestion] = score * 1.5  # Medium-high weight
            else:
                trending_scores[suggestion] += score * 0.5  # Add partial weight
        
        # Add daily trending with lower weight
        for suggestion, score in day_trending:
            if suggestion not in trending_scores:
                trending_scores[suggestion] = score * 1.0  # Base weight
            else:
                trending_scores[suggestion] += score * 0.3  # Add partial weight
        
        # Filter suggestions that match the partial query
        matching_suggestions = []
        for suggestion, score in trending_scores.items():
            # Only include suggestions relevant to the query
            if (normalized_query in suggestion or 
                any(term in suggestion for term in normalized_query.split())):
                matching_suggestions.append({
                    "text": suggestion,
                    "source": "trending",
                    "score": min(1.0, score / 10.0)  # Normalize score between 0-1
                })
        
        # Sort by score and return top results
        matching_suggestions.sort(key=lambda x: x.get("score", 0), reverse=True)
        return matching_suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Error getting trending suggestions: {e}")
        return []

async def update_trending_suggestions(partial_query: str, selected_suggestion: str):
    """
    Update trending suggestions data.
    
    This method updates both short-term (1 hour) and long-term (24 hours) trending data.
    
    Args:
        partial_query: The partial query that was typed
        selected_suggestion: The suggestion that was selected
    """
    try:
        # Connect to Redis for trending data
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)), 
            db=int(os.getenv('REDIS_TRENDING_DB', 2)),
            decode_responses=True
        )
        
        current_time = int(time.time())
        
        # Normalize the query and suggestion
        normalized_query = partial_query.lower().strip()
        normalized_suggestion = selected_suggestion.lower().strip()
        
        # Update trending data with different time windows
        # 1-hour trending (short-term)
        hour_trending_key = f"trending:hour:{int(current_time / 3600)}"
        redis_client.zincrby(hour_trending_key, 1, normalized_suggestion)
        # Set expiration for 2 hours (to ensure overlap)
        redis_client.expire(hour_trending_key, 7200)
        
        # 24-hour trending (medium-term)
        day_trending_key = f"trending:day:{int(current_time / 86400)}"
        redis_client.zincrby(day_trending_key, 1, normalized_suggestion)
        # Set expiration for 48 hours (to ensure overlap)
        redis_client.expire(day_trending_key, 172800)
        
        # Update query-specific trending
        query_trending_key = f"trending:query:{normalized_query}"
        redis_client.zincrby(query_trending_key, 1, normalized_suggestion)
        redis_client.expire(query_trending_key, 604800)  # 1 week
        
        # Update popularity weights for hybrid ranking
        popularity_key = f"query_popularity:{normalized_query}"
        
        # Get existing popularity data or create new
        popularity_data = redis_client.get(popularity_key)
        popularity_weights = json.loads(popularity_data) if popularity_data else {}
        
        # Update weight for this suggestion
        current_weight = popularity_weights.get(normalized_suggestion, 0)
        popularity_weights[normalized_suggestion] = min(1.0, current_weight + 0.05)
        
        # Apply decay to other suggestions
        for suggestion in popularity_weights:
            if suggestion != normalized_suggestion:
                popularity_weights[suggestion] *= 0.99  # Slight decay
        
        # Store updated popularity weights
        redis_client.setex(
            popularity_key,
            604800,  # 1 week
            json.dumps(popularity_weights)
        )
        
        logger.info(f"Updated trending data for suggestion: {normalized_suggestion}")
        
    except Exception as e:
        logger.error(f"Error updating trending suggestions: {e}")


