import asyncio
import logging
import math
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import redis
import datetime
import psycopg2
from urllib.parse import urlparse

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# File: ai_services_api/services/search/core/personalization.py
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict
import asyncpg

logger = logging.getLogger(__name__)

# Database connection pool (same as previous implementation)
_pool: asyncpg.pool.Pool = None

async def create_pool():
    """Initialize the database connection pool"""
    global _pool
    _pool = await asyncpg.create_pool(
        host='postgres',
        user='your_user',
        password='your_password',
        database='aphrc',
        min_size=2,
        max_size=10
    )

async def get_connection() -> asyncpg.Connection:
    """Get connection from pool"""
    if not _pool:
        await create_pool()
    return await _pool.acquire()

async def release_connection(conn: asyncpg.Connection):
    """Release connection back to pool"""
    await _pool.release(conn)

async def calculate_time_decay(user_id: str, current_time: datetime) -> float:
    """Calculate time decay factor based on user's last activity"""
    conn = await get_connection()
    try:
        result = await conn.fetchrow("""
            SELECT MAX(timestamp) as last_active 
            FROM user_activity 
            WHERE user_id = $1
        """, user_id)
        
        if result and result['last_active']:
            time_diff = current_time - result['last_active']
            decay = max(0.5, 1 - (time_diff.days / 30))  # Linear decay over 30 days
            return round(decay, 2)
        return 1.0  # Default if no activity found
        
    except Exception as e:
        logger.error(f"Time decay error: {str(e)}")
        return 1.0
    finally:
        await release_connection(conn)

async def personalize_suggestions(suggestions: List[Dict], user_id: str, query: str) -> List[Dict]:
    """Personalize suggestions with full context handling"""
    conn = None
    try:
        conn = await get_connection()
        
        # Get user preferences
        user_prefs = await conn.fetchrow(
            "SELECT preferences FROM user_settings WHERE user_id = $1", 
            user_id
        )
        
        # Get current time correctly
        now = datetime.now()
        time_decay = await calculate_time_decay(user_id, now)
        
        # Process suggestions
        personalized = []
        for item in suggestions:
            try:
                # Convert Decimal safely
                base_score = float(item['score']) if isinstance(item['score'], Decimal) else item['score']
                
                # Apply personalization factors
                boosted_score = base_score * user_prefs.get('boost', 1.0) * time_decay
                
                personalized.append({
                    **item,
                    "score": round(boosted_score, 4),
                    "timestamp": now.isoformat()
                })
                
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid suggestion: {str(e)}")
                continue
                
        return sorted(personalized, key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        logger.error(f"Personalization failed: {str(e)}", exc_info=True)
        return suggestions
    finally:
        if conn:
            await release_connection(conn)

def get_connection_params():
    """Get database connection parameters from environment variables."""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        parsed_url = urlparse(database_url)
        return {
            'host': parsed_url.hostname,
            'port': parsed_url.port,
            'dbname': parsed_url.path[1:],  # Remove leading '/'
            'user': parsed_url.username,
            'password': parsed_url.password,
            'connect_timeout': 10  # Timeout for connection attempt
        }
    else:
        in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres'),
            'connect_timeout': 10  # Timeout for connection attempt
        }

def get_db_connection(dbname=None):
    """Get a direct database connection (no pooling)."""
    params = get_connection_params()
    if dbname:
        params['dbname'] = dbname 
    
    try:
        conn = psycopg2.connect(**params)
        with conn.cursor() as cur:
            # Explicitly set the schema
            cur.execute('SET search_path TO public')
        logger.info(f"Successfully connected to database: {params['dbname']} at {params['host']}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to the database: {e}")
        logger.error(f"Connection params: {params}")
        raise

def close_connection(conn):
    """Close the database connection."""
    if conn:
        conn.close()
        logger.info("Connection closed.")

# Async context manager for database connections
class AsyncDatabaseConnection:
    async def __aenter__(self):
        """Async enter method to get a database connection."""
        try:
            self.conn = get_db_connection()
            return self.conn
        except Exception as e:
            logger.error(f"Error in async database connection: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit method to close the database connection."""
        if hasattr(self, 'conn'):
            close_connection(self.conn)
        return False  # Propagate any exceptions

# Update methods to use the new connection approach

async def get_user_search_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent search history for a user from the database.
    """
    logger.info(f"Retrieving search history for user {user_id}")
    
    try:
        async with AsyncDatabaseConnection() as conn:
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

async def get_selected_suggestions_with_times(user_id: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    Get recently selected suggestions for a user with timestamps.
    """
    try:
        async with AsyncDatabaseConnection() as conn:
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

async def get_user_expertise_areas(user_id: str) -> List[str]:
    """
    Get expertise areas for a user based on profile or past behavior.
    """
    try:
        async with AsyncDatabaseConnection() as conn:
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

async def get_collaborative_suggestions(
    user_id: str, 
    partial_query: str, 
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get suggestions based on collaborative filtering of user behavior.
    """
    logger.info(f"Getting collaborative suggestions for user {user_id} and query '{partial_query}'")
    
    try:
        async with AsyncDatabaseConnection() as conn:
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

async def get_selected_suggestions(user_id: str, limit: int = 10) -> List[str]:
    """
    Get recently selected suggestions for a user.
    """
    try:
        async with AsyncDatabaseConnection() as conn:
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
    
# Method: Personalization Module
# Location: ai_services_api/services/search/core/personalization.py
# Changes: Fixed datetime usage and session handling

# Method: Analytics Recording
# Location: ai_services_api/services/search/app/endpoints/process_functions.py
#
async def track_selected_suggestion(user_id: str, partial_query: str, selected_suggestion: str):
    """
    Track which suggestion the user selected to improve future personalization
    and update trending suggestions.
    """
    logger.info(f"Tracking selected suggestion for user {user_id}")
    
    try:
        async with AsyncDatabaseConnection() as conn:
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
            asyncio.create_task(
                update_trending_suggestions(partial_query, selected_suggestion)
            )
            
    except Exception as e:
        logger.error(f"Error tracking suggestion selection: {e}")

async def get_context_popular_queries(context: str, limit: int = 5) -> List[str]:
    """
    Get popular queries specific to a context type.
    """
    try:
        popular_queries = []
        
        async with AsyncDatabaseConnection() as conn:
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
        # Fall back to Redis if available
        try:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
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

# Rest of the previous file's functions remain the same

# Add these two imports at the top of the file if not already present
from functools import wraps

def safe_background_task(func):
    """
    Decorator to safely run background tasks with proper connection management.
    This ensures that database connections are properly handled in async tasks.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            async with AsyncDatabaseConnection() as conn:
                # Modify the task to accept the connection as the first argument
                return await func(conn, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in background task {func.__name__}: {e}")
    return wrapper

@safe_background_task
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
            host=os.getenv('REDIS_HOST', 'redis'),
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
        await asyncio.create_task(_update_trending_suggestions_task(partial_query, selected_suggestion))
    except Exception as e:
        logger.error(f"Failed to start trending suggestions update task: {e}")

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
            host=os.getenv('REDIS_HOST', 'redis'),
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

# Personalize suggestions function remains the same as in the previous artifact

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
            host=os.getenv('REDIS_HOST', 'redis'),
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


