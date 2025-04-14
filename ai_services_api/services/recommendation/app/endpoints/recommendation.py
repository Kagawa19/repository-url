import os
import re
import time
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import List, Dict, Any
from datetime import datetime
import logging
import json
import psycopg2
from redis.asyncio import Redis
from ai_services_api.services.recommendation.services.expert_matching import ExpertMatchingService
from ai_services_api.services.message.core.database import get_db_connection
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
import json
import os
import psycopg2
from urllib.parse import urlparse
import logging





router = APIRouter()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_router.log')
    ]
)
logger = logging.getLogger(__name__)

TEST_USER_ID = "test_user_123"

async def get_redis():
    """Establish Redis connection with detailed logging"""
    try:
        redis_client = Redis(host='redis', port=6379, db=3, decode_responses=True)
        logger.info(f"Redis connection established successfully to host: redis, port: 6379, db: 3")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to establish Redis connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Redis connection failed")
    
async def get_user_id(request: Request) -> str:
    logger.debug("Extracting user ID from request headers")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Missing required X-User-ID header in request")
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    logger.info(f"User ID extracted successfully: {user_id}")
    return user_id


async def get_test_user_id(request: Request) -> str:
    """Return test user ID with logging"""
    logger.info(f"Using test user ID: {TEST_USER_ID}")
    return TEST_USER_ID

async def process_recommendations(user_id: str, redis_client: Redis) -> Dict:
    """Process expert recommendations with comprehensive logging and error handling"""
    try:
        logger.info(f"Starting recommendation process for user: {user_id}")
        
        # Check if we need to check for recent database changes
        should_check_changes = True
        
        # Check cache first
        cache_key = f"user_recommendations:{user_id}"
        cached_response = await redis_client.get(cache_key)
        
        if cached_response:
            try:
                cached_data = json.loads(cached_response)
                
                # NEW: Check if we should use the cached data or if there might be DB changes
                # Only check for DB changes if the cache is older than 5 minutes
                cache_age = await redis_client.ttl(cache_key)
                if cache_age > 1500:  # If more than 5 minutes fresh (1800-300=1500)
                    should_check_changes = False
                    logger.info(f"Using fresh cache for user recommendations: {cache_key}")
                    return cached_data
                else:
                    # Cache exists but may be stale, check for DB changes before using
                    logger.info(f"Cache exists but may need validation: {cache_key}")
            except json.JSONDecodeError as json_err:
                logger.error(f"Error decoding cached response: {json_err}")
        
        # NEW: Check if there have been relevant database changes
        if should_check_changes:
            has_changes = await _check_for_database_changes(user_id, redis_client)
            
            # If there are no changes and we have cached data, use it
            if not has_changes and cached_response:
                try:
                    logger.info(f"No relevant database changes, using cache: {cache_key}")
                    return json.loads(cached_response)
                except json.JSONDecodeError:
                    pass  # Fall through to regenerating recommendations
        
        start_time = datetime.utcnow()
        logger.debug(f"Recommendation generation started at: {start_time}")
        
        # ---- Get search analytics to enhance recommendations ----
        search_analytics = await _get_user_search_analytics(user_id, redis_client)
        logger.info(f"Retrieved search analytics for user {user_id}: {len(search_analytics)} records")
        
        expert_matching = ExpertMatchingService()
        try:
            logger.info(f"Generating recommendations for user: {user_id}")
            
            # ---- Get recommendations ----
            recommendations = await expert_matching.get_recommendations_for_user(user_id)
            
            # ---- Apply search analytics boosting ----
            if search_analytics and recommendations:
                # Get search query patterns
                search_queries = [sa.get('query', '').lower() for sa in search_analytics if sa.get('query')]
                search_terms = _extract_search_terms(search_queries)
                
                # Apply boosting to recommendations based on search history
                if search_terms:
                    # Apply a boost if expert's name or expertise matches search terms
                    for rec in recommendations:
                        boost_score = 0
                        
                        # Check name matches
                        if rec.get('name'):
                            for term in search_terms:
                                if term in rec['name'].lower():
                                    boost_score += 0.1
                        
                        # Check expertise matches in match_details
                        if rec.get('match_details'):
                            for concept_list in rec['match_details'].values():
                                for concept in concept_list:
                                    for term in search_terms:
                                        if term in concept.lower():
                                            boost_score += 0.05
                        
                        # Apply the boost to similarity_score
                        if boost_score > 0:
                            rec['similarity_score'] = rec.get('similarity_score', 0) + boost_score
                            rec['search_boosted'] = True
                    
                    # Re-sort recommendations by updated similarity score
                    recommendations.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                    logger.info(f"Applied search-based boosting to recommendations")
            
            # ---- Format response data ----
            response_data = {
                "user_id": user_id,
                "recommendations": recommendations or [],
                "total_matches": len(recommendations) if recommendations else 0,
                "timestamp": datetime.utcnow().isoformat(),
                "response_time": (datetime.utcnow() - start_time).total_seconds(),
                # NEW: Add a data version marker for change detection
                "data_version": await _get_current_data_version(redis_client)
            }
            
            logger.info(f"Recommendation generation completed for user {user_id}. "
                        f"Total matches: {response_data['total_matches']}, "
                        f"Response time: {response_data['response_time']} seconds")
            
            try:
                await redis_client.setex(
                    cache_key,
                    1800,  # 30 minutes expiry
                    json.dumps(response_data)
                )
                logger.info(f"Recommendations cached successfully for user {user_id}")
            except Exception as cache_err:
                logger.error(f"Failed to cache recommendations: {cache_err}")
            
            # ---- Log this recommendation event ----
            await _log_recommendation_event(user_id, response_data, redis_client)
            
            return response_data
        
        except Exception as matching_err:
            logger.error(f"Error in expert matching for user {user_id}: {str(matching_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Expert matching error: {str(matching_err)}")
        
        finally:
            expert_matching.close()
            logger.debug(f"Expert matching service closed for user {user_id}")
    
    except Exception as e:
        logger.error(f"Unhandled error in recommendation process for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Comprehensive recommendation error")

async def _check_for_database_changes(user_id: str, redis_client: Redis) -> bool:
    """
    Check if there have been database changes that would affect recommendations for this user.
    
    Args:
        user_id: The user's identifier
        redis_client: Redis client for version checking
        
    Returns:
        Boolean indicating if changes have been detected
    """
    try:
        # Get the cached data version if it exists
        cache_key = f"user_recommendations:{user_id}"
        cached_data = await redis_client.get(cache_key)
        
        if not cached_data:
            # No cache exists, so consider this as needing fresh data

async def _check_for_database_changes(user_id: str, redis_client: Redis) -> bool:
    """
    Check if there have been database changes that would affect recommendations for this user.
    
    Args:
        user_id: The user's identifier
        redis_client: Redis client for version checking
        
    Returns:
        Boolean indicating if changes have been detected
    """
    try:
        # Get the cached data version if it exists
        cache_key = f"user_recommendations:{user_id}"
        cached_data = await redis_client.get(cache_key)
        
        if not cached_data:
            # No cache exists, so consider this as needing fresh data
            return True
            
        try:
            # Extract the data version from cached data
            cached_version = json.loads(cached_data).get('data_version', 0)
        except (json.JSONDecodeError, AttributeError):
            # If we can't parse the cache or get the version, assume we need fresh data
            return True
            
        # Get the current data version
        current_version = await _get_current_data_version(redis_client)
        
        # Check if the user has specific changes
        user_changes_key = f"user_data_changed:{user_id}"
        user_has_changes = await redis_client.exists(user_changes_key)
        
        if user_has_changes:
            # Clear the change flag since we're processing it now
            await redis_client.delete(user_changes_key)
            logger.info(f"Detected specific changes for user {user_id}")
            return True
            
        # Compare versions to see if there have been general data changes
        if current_version > cached_version:
            logger.info(f"Data version change detected: {cached_version} -> {current_version}")
            return True
            
        # No relevant changes detected
        return False
        
    except Exception as e:
        logger.error(f"Error checking for database changes: {e}")
        # On error, assume we should refresh data to be safe
        return True

async def _get_current_data_version(redis_client: Redis) -> int:
    """
    Get the current data version number that tracks database changes.
    
    Args:
        redis_client: Redis client for version tracking
        
    Returns:
        Current data version as integer
    """
    try:
        # Key for storing the global data version
        version_key = "recommendation_data_version"
        
        # Get current version
        version = await redis_client.get(version_key)
        
        if version is None:
            # Initialize if doesn't exist
            await redis_client.set(version_key, "1")
            return 1
            
        return int(version)
        
    except Exception as e:
        logger.error(f"Error getting data version: {e}")
        # Return a default version on error
        return 0

async def increment_data_version(redis_client: Redis) -> int:
    """
    Increment the data version to indicate database changes.
    This should be called whenever significant changes are made to the database
    that would affect recommendations.
    
    Args:
        redis_client: Redis client for version tracking
        
    Returns:
        New data version as integer
    """
    try:
        # Key for storing the global data version
        version_key = "recommendation_data_version"
        
        # Increment and return the new version
        new_version = await redis_client.incr(version_key)
        logger.info(f"Incremented data version to {new_version}")
        
        return new_version
        
    except Exception as e:
        logger.error(f"Error incrementing data version: {e}")
        return 0

async def mark_user_data_changed(user_id: str, redis_client: Redis) -> bool:
    """
    Mark a specific user as having data changes that would affect their recommendations.
    This is more targeted than incrementing the global version and should be used
    when changes only affect specific users.
    
    Args:
        user_id: The user's identifier
        redis_client: Redis client for tracking
        
    Returns:
        Boolean indicating success
    """
    try:
        # Key for tracking user-specific changes
        user_changes_key = f"user_data_changed:{user_id}"
        
        # Set a flag that will be checked and cleared when processing recommendations
        # Use a short expiry to prevent permanent flags
        await redis_client.setex(user_changes_key, 3600, "1")  # 1 hour expiry
        
        logger.info(f"Marked data changes for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error marking user data changes: {e}")
        return False

@router.delete("/cache/recommendations/{user_id}")
async def clear_user_recommendations_cache(
    user_id: str,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear recommendations cache for a specific user"""
    try:
        cache_key = f"user_recommendations:{user_id}"
        deleted = await redis_client.delete(cache_key)
        
        # NEW: Also mark this user as having data changes
        await mark_user_data_changed(user_id, redis_client)
        
        logger.info(f"Cache clearing request for user: {user_id}, result: {deleted > 0}")
        
        if deleted:
            return {"status": "success", "message": f"Cache cleared for user {user_id}"}
        else:
            return {"status": "success", "message": f"No cache found for user {user_id}"}
    
    except Exception as e:
        logger.error(f"Failed to clear cache for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

@router.delete("/cache/recommendations")
async def clear_all_recommendations_cache(
    request: Request,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear recommendations cache for all users"""
    try:
        # Find all keys matching the pattern
        pattern = "user_recommendations:*"
        total_deleted = 0
        
        # Use scan_iter to get all matching keys and delete them
        async for key in redis_client.scan_iter(match=pattern):
            await redis_client.delete(key)
            total_deleted += 1
        
        # NEW: Also increment the data version to ensure fresh recommendations
        await increment_data_version(redis_client)
        
        logger.info(f"All recommendation caches cleared. Total deleted: {total_deleted}")
        
        return {
            "status": "success", 
            "message": f"Cleared all recommendation caches", 
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logger.error(f"Failed to clear all recommendation caches: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

@router.post("/notify/database-change")
async def notify_database_change(
    request: Request,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """
    Endpoint for other services to notify of database changes that
    might affect recommendations.
    """
    try:
        # Get the request data
        data = await request.json()
        
        # Check if this change affects specific users
        affected_users = data.get("affected_users", [])
        
        if affected_users:
            # Mark specific users as having changes
            for user_id in affected_users:
                await mark_user_data_changed(user_id, redis_client)
                
            logger.info(f"Marked changes for {len(affected_users)} specific users")
            return {
                "status": "success",
                "message": f"Marked changes for {len(affected_users)} users",
                "affected_users": len(affected_users)
            }
        else:
            # Increment the global version for system-wide changes
            new_version = await increment_data_version(redis_client)
            
            logger.info(f"Incremented global data version to {new_version}")
            return {
                "status": "success",
                "message": "Updated global data version",
                "new_version": new_version
            }
    
    except Exception as e:
        logger.error(f"Failed to process database change notification: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Notification processing error: {str(e)}")

async def _get_user_search_analytics(user_id: str, redis_client: Redis) -> List[Dict]:
    """
    Retrieve recent search analytics for a user.
    
    Args:
        user_id: The user's identifier
        redis_client: Redis client for caching
        
    Returns:
        List of search analytics records
    """
    try:
        # Try to get from cache first for performance
        cache_key = f"user_search_analytics:{user_id}"
        cached_data = await redis_client.get(cache_key)
        
        if cached_data:
            logger.debug(f"Cache hit for user search analytics: {cache_key}")
            try:
                return json.loads(cached_data)
            except Exception:
                pass
        
        # Define connection parameters
        conn_params = {}
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            parsed_url = urlparse(database_url)
            conn_params = {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'dbname': parsed_url.path[1:],
                'user': parsed_url.username,
                'password': parsed_url.password
            }
        else:
            conn_params = {
                'host': os.getenv('POSTGRES_HOST', 'postgres'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
            }
        
        # Connect to database and fetch search analytics
        conn = None
        try:
            conn = psycopg2.connect(**conn_params)
            with conn.cursor() as cur:
                # Get recent searches, ordered by timestamp
                cur.execute("""
                    SELECT 
                        search_id, 
                        query, 
                        response_time, 
                        result_count,
                        search_type,
                        timestamp
                    FROM search_analytics
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT 20
                """, (user_id,))
                
                rows = cur.fetchall()
                
                # Convert to list of dictionaries
                analytics = []
                for row in rows:
                    analytics.append({
                        'search_id': row[0],
                        'query': row[1],
                        'response_time': row[2],
                        'result_count': row[3],
                        'search_type': row[4],
                        'timestamp': row[5].isoformat() if row[5] else None
                    })
                
                # Cache the results for 15 minutes
                if analytics:
                    await redis_client.setex(
                        cache_key,
                        900,  # 15 minutes
                        json.dumps(analytics)
                    )
                
                return analytics
                
        except Exception as db_error:
            logger.error(f"Database error retrieving search analytics: {db_error}")
            return []
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error retrieving search analytics: {e}")
        return []

def _extract_search_terms(search_queries: List[str]) -> List[str]:
    """
    Extract key terms from search queries for recommendation boosting.
    
    Args:
        search_queries: List of search queries
        
    Returns:
        List of key search terms
    """
    if not search_queries:
        return []
        
    # Combine all queries
    all_text = " ".join(search_queries).lower()
    
    # Split into words
    words = re.findall(r'\b\w+\b', all_text)
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 'an', 'is', 'are', 'from'}
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Keep terms appearing more than once or longer terms
    search_terms = [word for word, count in word_counts.items() 
                  if count > 1 or len(word) > 5]
    
    # Limit to 10 most frequent terms
    search_terms.sort(key=lambda w: word_counts[w], reverse=True)
    return search_terms[:10]

async def _log_recommendation_event(user_id: str, response_data: Dict, redis_client: Redis) -> None:
    """
    Log a recommendation event for analytics and tracking.
    
    Args:
        user_id: The user's identifier
        response_data: The recommendation response data
        redis_client: Redis client for potential caching
    """
    try:
        # Define connection parameters
        conn_params = {}
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            parsed_url = urlparse(database_url)
            conn_params = {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'dbname': parsed_url.path[1:],
                'user': parsed_url.username,
                'password': parsed_url.password
            }
        else:
            conn_params = {
                'host': os.getenv('POSTGRES_HOST', 'postgres'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
            }
        
        # Connect to database and log the event
        conn = None
        try:
            conn = psycopg2.connect(**conn_params)
            with conn.cursor() as cur:
                # Generate a recommendation event ID
                recommendation_id = f"rec_{int(time.time())}_{user_id}"
                
                # Insert an analytics record for the recommendation event
                cur.execute("""
                    INSERT INTO search_analytics 
                        (search_id, query, user_id, response_time, result_count, search_type, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    recommendation_id,
                    "recommendation_request",
                    user_id,
                    response_data.get('response_time', 0),
                    response_data.get('total_matches', 0),
                    "recommendation",
                ))
                
                # For each recommended expert, log an interest record
                for i, rec in enumerate(response_data.get('recommendations', [])):
                    expert_id = rec.get('id')
                    if not expert_id:
                        continue
                        
                    # Log this recommendation as an expert interaction
                    cur.execute("""
                        INSERT INTO user_interest_logs 
                            (user_id, session_id, query, interaction_type, content_id, response_quality, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        user_id,
                        recommendation_id,
                        f"position_{i+1}",  # Record position in results
                        'expert',
                        expert_id,
                        0.5  # Neutral initial quality, will be updated when clicked
                    ))
                
                conn.commit()
                logger.info(f"Logged recommendation event: {recommendation_id} with {len(response_data.get('recommendations', []))} experts")
                
        except Exception as db_error:
            logger.error(f"Database error logging recommendation event: {db_error}")
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"Error logging recommendation event: {e}")

@router.get("/recommend/{user_id}")
async def get_expert_recommendations(
    user_id: str,
    request: Request,
    redis_client: Redis = Depends(get_redis)
):
    """Get expert recommendations for the user based on their behavior"""
    logger.info(f"Recommendation request received for user: {user_id}")
    
    # Process recommendations
    recommendations = await process_recommendations(user_id, redis_client)
    
    # Record that these recommendations were shown to the user
    try:
        # Generate a view ID for tracking
        view_id = f"view_{int(time.time())}"
        
        # Store in Redis which experts were recommended in this view
        # This will be used to track clicks later
        experts_shown = [exp.get('id') for exp in recommendations.get('recommendations', [])]
        if experts_shown:
            # Store for 24 hours
            await redis_client.setex(
                f"rec_view:{view_id}",
                86400,  # 24 hours
                json.dumps({
                    'user_id': user_id,
                    'experts': experts_shown,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            
            # Add view_id to response for frontend tracking
            recommendations['view_id'] = view_id
            
            logger.info(f"Recorded recommendation view {view_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Error recording recommendation view: {e}")
    
    return recommendations


class RecommendationClick(BaseModel):
    view_id: str
    expert_id: str
@router.post("/recommend/track")
async def track_recommendation_click(
    request: Request,
    payload: RecommendationClick,
    user_id: str = Depends(get_user_id),
    redis_client = Depends(get_redis)
):
    """
    Track when a user clicks on a recommended expert.
    This creates a feedback loop for the recommendation system.
    """
    view_id = payload.view_id
    expert_id = payload.expert_id

    logger.info(f"Tracking recommendation click - User: {user_id}, Expert: {expert_id}, View: {view_id}")

    try:
        # Verify this was a valid recommendation
        view_key = f"rec_view:{view_id}"
        view_data = await redis_client.get(view_key)

        if not view_data:
            logger.warning(f"Invalid view ID: {view_id}")
            return {"status": "error", "message": "Invalid view ID"}

        try:
            view_info = json.loads(view_data)

            # Verify expert was in this recommendation set
            if expert_id not in view_info.get('experts', []):
                logger.warning(f"Expert {expert_id} not in recommendation view {view_id}")
                return {"status": "error", "message": "Expert not in recommendation set"}

            # Calculate position in recommendations (1-based)
            position = view_info.get('experts', []).index(expert_id) + 1

            # Log this click in user_interest_logs with high quality score
            # Connect to database
            conn_params = {}
            database_url = os.getenv('DATABASE_URL')

            if database_url:
                parsed_url = urlparse(database_url)
                conn_params = {
                    'host': parsed_url.hostname,
                    'port': parsed_url.port,
                    'dbname': parsed_url.path[1:],
                    'user': parsed_url.username,
                    'password': parsed_url.password
                }
            else:
                conn_params = {
                    'host': os.getenv('POSTGRES_HOST', 'postgres'),
                    'port': os.getenv('POSTGRES_PORT', '5432'),
                    'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                    'user': os.getenv('POSTGRES_USER', 'postgres'),
                    'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
                }

            conn = None
            try:
                conn = psycopg2.connect(**conn_params)
                with conn.cursor() as cur:
                    # Insert a high-quality interaction for the clicked expert
                    cur.execute("""
                        INSERT INTO user_interest_logs 
                            (user_id, session_id, query, interaction_type, content_id, response_quality, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        user_id,
                        f"click_{view_id}",
                        f"recommendation_click_position_{position}",
                        'expert',
                        expert_id,
                        0.9  # High quality score for explicit clicks
                    ))

                    # Also update the topic interests based on this expert's expertise
                    cur.execute("""
                        SELECT knowledge_expertise, fields, domains
                        FROM experts_expert
                        WHERE id = %s
                    """, (expert_id,))

                    expert_data = cur.fetchone()
                    if expert_data:
                        knowledge_expertise, fields, domains = expert_data

                        # Track expertise as interests
                        if knowledge_expertise:
                            for expertise in knowledge_expertise:
                                if not expertise:
                                    continue

                                cur.execute("""
                                    INSERT INTO user_topic_interests 
                                        (user_id, topic_key, topic_type, interaction_count, last_interaction)
                                    VALUES (%s, %s, %s, 1, CURRENT_TIMESTAMP)
                                    ON CONFLICT (user_id, topic_key, topic_type) 
                                    DO UPDATE SET 
                                        interaction_count = user_topic_interests.interaction_count + 1,
                                        last_interaction = CURRENT_TIMESTAMP,
                                        engagement_score = user_topic_interests.engagement_score * 0.9 + 1.0
                                """, (user_id, expertise.lower(), 'expert_expertise'))

                        # Track fields as interests
                        if fields:
                            for field in fields:
                                if not field:
                                    continue

                                cur.execute("""
                                    INSERT INTO user_topic_interests 
                                        (user_id, topic_key, topic_type, interaction_count, last_interaction)
                                    VALUES (%s, %s, %s, 1, CURRENT_TIMESTAMP)
                                    ON CONFLICT (user_id, topic_key, topic_type) 
                                    DO UPDATE SET 
                                        interaction_count = user_topic_interests.interaction_count + 1,
                                        last_interaction = CURRENT_TIMESTAMP,
                                        engagement_score = user_topic_interests.engagement_score * 0.9 + 1.0
                                """, (user_id, field.lower(), 'expert_expertise'))

                        # Track domains as interests
                        if domains:
                            for domain in domains:
                                if not domain:
                                    continue

                                cur.execute("""
                                    INSERT INTO user_topic_interests 
                                        (user_id, topic_key, topic_type, interaction_count, last_interaction)
                                    VALUES (%s, %s, %s, 1, CURRENT_TIMESTAMP)
                                    ON CONFLICT (user_id, topic_key, topic_type) 
                                    DO UPDATE SET 
                                        interaction_count = user_topic_interests.interaction_count + 1,
                                        last_interaction = CURRENT_TIMESTAMP,
                                        engagement_score = user_topic_interests.engagement_score * 0.9 + 1.0
                                """, (user_id, domain.lower(), 'publication_domain'))

                conn.commit()
                logger.info(f"Recorded recommendation click for user {user_id} on expert {expert_id}")

            except Exception as db_error:
                logger.error(f"Database error recording recommendation click: {db_error}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    conn.close()

            return {
                "status": "success",
                "message": "Click tracked successfully",
                "data": {
                    "user_id": user_id,
                    "expert_id": expert_id,
                    "view_id": view_id,
                    "position": position
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding view data: {e}")
            return {"status": "error", "message": "Invalid view data format"}

    except Exception as e:
        logger.error(f"Error tracking recommendation click: {e}")
        return {"status": "error", "message": f"Internal error: {str(e)}"}

 

@router.get("/test/recommend")
async def test_get_expert_recommendations(
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
):
    """Test endpoint for getting user recommendations"""
    logger.info(f"Test recommendation request for user: {user_id}")
    return await process_recommendations(user_id, redis_client)

# New cache clearing endpoints
@router.delete("/cache/recommendations/{user_id}")
async def clear_user_recommendations_cache(
    user_id: str,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear recommendations cache for a specific user"""
    try:
        cache_key = f"user_recommendations:{user_id}"
        deleted = await redis_client.delete(cache_key)
        
        logger.info(f"Cache clearing request for user: {user_id}, result: {deleted > 0}")
        
        if deleted:
            return {"status": "success", "message": f"Cache cleared for user {user_id}"}
        else:
            return {"status": "success", "message": f"No cache found for user {user_id}"}
    
    except Exception as e:
        logger.error(f"Failed to clear cache for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

@router.delete("/cache/recommendations")
async def clear_all_recommendations_cache(
    request: Request,
    redis_client: Redis = Depends(get_redis)
) -> Dict:
    """Clear recommendations cache for all users"""
    try:
        # Find all keys matching the pattern
        pattern = "user_recommendations:*"
        total_deleted = 0
        
        # Use scan_iter to get all matching keys and delete them
        async for key in redis_client.scan_iter(match=pattern):
            await redis_client.delete(key)
            total_deleted += 1
        
        logger.info(f"All recommendation caches cleared. Total deleted: {total_deleted}")
        
        return {
            "status": "success", 
            "message": f"Cleared all recommendation caches", 
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logger.error(f"Failed to clear all recommendation caches: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")