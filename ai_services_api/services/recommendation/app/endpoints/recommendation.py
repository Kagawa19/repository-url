import os
import re
from psycopg2.extras import RealDictCursor
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
from ai_services_api.services.recommendation.services.recommendation_monitor import RecommendationMonitor

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
recommendation_monitor = RecommendationMonitor()
logger.info("RecommendationMonitor initialized successfully")

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
            
            # ---- NEW: Record recommendation snapshot for monitoring ----
            try:
                # Get data source information
                data_sources = ["regular"]
                features_used = ["similarity", "organization"]
                
                if search_analytics and search_terms:
                    data_sources.append("search_history")
                    features_used.append("search_terms")
                
                # Gather weight information if available from the expertise service 
                weights = {
                    "concept_weight": getattr(expert_matching, "_concept_weight", 0.3),
                    "domain_weight": getattr(expert_matching, "_domain_weight", 0.2),
                    "field_weight": getattr(expert_matching, "_field_weight", 0.2),
                    "organizational_weight": getattr(expert_matching, "_org_weight", 0.1),
                    "search_weight": 0.1 if search_terms else 0.0
                }
                
                # Record the snapshot
                recommendation_monitor.record_recommendation_snapshot(
                    user_id=user_id,
                    recommendations=recommendations or [],
                    weights=weights,
                    features_used=features_used,
                    data_sources_used=data_sources
                )
                
                # Record timing metric
                recommendation_monitor.record_metric(
                    metric_type="performance",
                    metric_name="response_time",
                    metric_value=response_data['response_time'],
                    user_id=user_id
                )
                
                # Record quality metrics
                if recommendations:
                    recommendation_monitor.record_metric(
                        metric_type="quality",
                        metric_name="recommendation_count",
                        metric_value=len(recommendations),
                        user_id=user_id
                    )
                    
                    # Average similarity score
                    avg_score = sum(rec.get('similarity_score', 0) for rec in recommendations) / len(recommendations)
                    recommendation_monitor.record_metric(
                        metric_type="quality",
                        metric_name="average_similarity_score",
                        metric_value=avg_score,
                        user_id=user_id
                    )
                
                logger.info("Recommendation monitoring data recorded successfully")
            except Exception as monitoring_err:
                logger.error(f"Error recording recommendation monitoring data: {str(monitoring_err)}")
                # Continue anyway - monitoring shouldn't break the main functionality
            
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
    
@router.get("/recommendations/monitoring/report")
async def get_recommendation_monitoring_report(
    request: Request,
    user_id: str = Depends(get_user_id),
    days: int = 30,
    format: str = "json"
):
    """Generate a report on recommendation system adaptiveness"""
    try:
        from datetime import datetime, timedelta
        
        # Generate the report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        report = recommendation_monitor.generate_adaptation_report(
            start_date=start_date,
            end_date=end_date,
            output_format=format
        )
        
        return report
    except Exception as e:
        logger.error(f"Error generating monitoring report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

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

@router.get("/recommendations/based-on-messaging")
async def get_messaging_based_recommendations(
    request: Request,
    user_id: str = Depends(get_user_id),
    redis_client: Redis = Depends(get_redis)
) -> Dict[str, Any]:
    """
    Get expert recommendations for a user specifically based on their messaging history.
    This endpoint prioritizes domains and fields from message interactions.
    
    Args:
        request: Request object
        user_id: User ID to get recommendations for
        redis_client: Redis client for caching
        
    Returns:
        Dict containing recommended experts and metadata
    """
    logger.info(f"Getting messaging-based recommendations for user {user_id}")
    
    try:
        start_time = datetime.utcnow()
        
        # Check cache first
        cache_key = f"user_messaging_recommendations:{user_id}"
        cached_response = await redis_client.get(cache_key)
        
        if cached_response:
            logger.info(f"Cache hit for messaging-based recommendations: {cache_key}")
            response_data = json.loads(cached_response)
            
            # Even for cached responses, record a monitoring entry to track usage
            try:
                recommendation_monitor.record_metric(
                    metric_type="usage",
                    metric_name="messaging_recommendations_cache_hit",
                    metric_value=1.0,
                    user_id=user_id
                )
            except Exception as monitor_err:
                logger.error(f"Error recording cache hit metric: {monitor_err}")
                
            return response_data
        
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                
                # Query to get top domains and fields from messaging interactions
                cur.execute("""
                    SELECT 
                        unnest(domains) as domain,
                        COUNT(*) as domain_count
                    FROM message_expert_interactions
                    WHERE user_id = %s
                    AND created_at > NOW() - INTERVAL '90 days'
                    GROUP BY domain
                    ORDER BY domain_count DESC
                    LIMIT 5
                """, (user_id,))
                
                top_domains = [row['domain'] for row in cur.fetchall() if row['domain']]
                
                cur.execute("""
                    SELECT 
                        unnest(fields) as field,
                        COUNT(*) as field_count
                    FROM message_expert_interactions
                    WHERE user_id = %s
                    AND created_at > NOW() - INTERVAL '90 days'
                    GROUP BY field
                    ORDER BY field_count DESC
                    LIMIT 5
                """, (user_id,))
                
                top_fields = [row['field'] for row in cur.fetchall() if row['field']]
                
                logger.info(f"Found {len(top_domains)} top domains and {len(top_fields)} top fields from messaging history")
                
                # Record domain/field discovery metric
                try:
                    recommendation_monitor.record_metric(
                        metric_type="data_quality",
                        metric_name="messaging_domain_field_count",
                        metric_value=len(top_domains) + len(top_fields),
                        user_id=user_id,
                        details={
                            "domains_count": len(top_domains),
                            "fields_count": len(top_fields),
                            "top_domains": top_domains[:3],  # Just include top 3 for brevity
                            "top_fields": top_fields[:3]
                        }
                    )
                except Exception as monitor_err:
                    logger.error(f"Error recording domain/field metric: {monitor_err}")
                
                # If no messaging history, fall back to regular recommendations
                if not top_domains and not top_fields:
                    logger.info(f"No messaging history found for user {user_id}, using regular recommendations")
                    
                    # Record fallback metric
                    try:
                        recommendation_monitor.record_metric(
                            metric_type="adaptation",
                            metric_name="messaging_recommendations_fallback",
                            metric_value=1.0,
                            user_id=user_id,
                            details={"reason": "no_domains_or_fields"}
                        )
                    except Exception as monitor_err:
                        logger.error(f"Error recording fallback metric: {monitor_err}")
                    
                    recommendations = await process_recommendations(user_id, redis_client)
                    return recommendations
                
                # Fetch matching experts from Neo4j
                expert_matching = ExpertMatchingService()
                try:
                    # Use the Neo4j driver directly to run a custom query
                    with expert_matching._neo4j_driver.session() as session:
                        query = """
                        MATCH (e:Expert)
                        WHERE e.is_active = true
                        AND e.id <> $user_id
                        
                        // Match with messaging domains
                        OPTIONAL MATCH (e)-[:WORKS_IN_DOMAIN]->(d:Domain)
                        WHERE d.name IN $domains
                        
                        // Match with messaging fields
                        OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(f:Field)
                        WHERE f.name IN $fields
                        
                        // Count matches
                        WITH e, 
                            COLLECT(DISTINCT d.name) as matched_domains,
                            COLLECT(DISTINCT f.name) as matched_fields,
                            SIZE(COLLECT(DISTINCT d)) as domain_matches,
                            SIZE(COLLECT(DISTINCT f)) as field_matches
                        
                        // Calculate score
                        WITH e, 
                            matched_domains,
                            matched_fields,
                            domain_matches * 0.6 + field_matches * 0.4 as messaging_score
                        
                        // Only include experts with some match
                        WHERE messaging_score > 0
                        
                        // Also match on concepts, organization for the full profile
                        OPTIONAL MATCH (e)-[:HAS_CONCEPT]->(c:Concept)
                        OPTIONAL MATCH (e)-[:RESEARCHES_IN]->(ra:ResearchArea)
                        
                        // Return formatted results
                        RETURN {
                            id: e.id,
                            name: e.name,
                            designation: e.designation,
                            theme: e.theme,
                            unit: e.unit,
                            match_details: {
                                matched_domains: matched_domains,
                                matched_fields: matched_fields,
                                shared_concepts: COLLECT(DISTINCT c.name),
                                shared_research_areas: COLLECT(DISTINCT ra.name)
                            },
                            match_reason: CASE
                                WHEN SIZE(matched_domains) > 0 AND SIZE(matched_fields) > 0 
                                    THEN 'Shares your interests in ' + matched_domains[0] + ' and ' + matched_fields[0]
                                WHEN SIZE(matched_domains) > 0 
                                    THEN 'Works in your area of interest: ' + matched_domains[0]
                                WHEN SIZE(matched_fields) > 0 
                                    THEN 'Specializes in your field of interest: ' + matched_fields[0]
                                ELSE 'Matches your messaging interests'
                            END,
                            similarity_score: messaging_score
                        } as result
                        ORDER BY messaging_score DESC
                        LIMIT 5
                        """
                        
                        result = session.run(query, {
                            "user_id": str(user_id),
                            "domains": top_domains,
                            "fields": top_fields
                        })
                        
                        messaging_recommendations = [record["result"] for record in result]
                        
                        # Record match success metric
                        try:
                            recommendation_monitor.record_metric(
                                metric_type="quality",
                                metric_name="messaging_match_success",
                                metric_value=len(messaging_recommendations) / 5.0,  # Percentage of filled slots
                                user_id=user_id,
                                details={
                                    "recommendations_found": len(messaging_recommendations),
                                    "target_count": 5
                                }
                            )
                        except Exception as monitor_err:
                            logger.error(f"Error recording match success metric: {monitor_err}")
                        
                        # If we don't have enough recommendations, get some regular ones to supplement
                        if len(messaging_recommendations) < 5:
                            logger.info(f"Only found {len(messaging_recommendations)} messaging-based recommendations, adding regular ones")
                            
                            # Record supplementation metric
                            try:
                                recommendation_monitor.record_metric(
                                    metric_type="adaptation",
                                    metric_name="messaging_recommendations_supplemented",
                                    metric_value=(5 - len(messaging_recommendations)) / 5.0,  # Percentage that needed supplementing
                                    user_id=user_id,
                                    details={"messaging_count": len(messaging_recommendations)}
                                )
                            except Exception as monitor_err:
                                logger.error(f"Error recording supplementation metric: {monitor_err}")
                            
                            # Get IDs of experts we already recommended
                            existing_ids = {rec['id'] for rec in messaging_recommendations}
                            
                            # Get extra_needed regular recommendations
                            extra_needed = 5 - len(messaging_recommendations)
                            regular_recs = await expert_matching.get_recommendations_for_user(user_id, extra_needed + 5)
                            
                            # Filter out any that we already recommended and add up to extra_needed
                            filtered_regular = [rec for rec in regular_recs if rec['id'] not in existing_ids]
                            messaging_recommendations.extend(filtered_regular[:extra_needed])
                
                except Exception as matching_err:
                    logger.error(f"Neo4j query error: {str(matching_err)}", exc_info=True)
                    
                    # Record error metric
                    try:
                        recommendation_monitor.record_metric(
                            metric_type="errors",
                            metric_name="messaging_recommendations_query_error",
                            metric_value=1.0,
                            user_id=user_id,
                            details={"error": str(matching_err)[:200]}  # Truncate to avoid massive error messages
                        )
                    except Exception as monitor_err:
                        logger.error(f"Error recording error metric: {monitor_err}")
                    
                    # Fall back to regular recommendations
                    recommendations = await process_recommendations(user_id, redis_client)
                    return recommendations
                finally:
                    expert_matching.close()
                
                end_time = datetime.utcnow()
                process_time = (end_time - start_time).total_seconds()
                
                # Format the response
                response_data = {
                    "user_id": user_id,
                    "recommendations": messaging_recommendations,
                    "total_matches": len(messaging_recommendations),
                    "timestamp": datetime.utcnow().isoformat(),
                    "response_time": process_time,
                    "messaging_based": True,
                    "top_domains": top_domains,
                    "top_fields": top_fields,
                    "data_version": await _get_current_data_version(redis_client) if '_get_current_data_version' in globals() else 1
                }
                
                # Cache the response
                try:
                    await redis_client.setex(
                        cache_key,
                        1800,  # 30 minutes
                        json.dumps(response_data)
                    )
                    logger.info(f"Cached messaging-based recommendations for user {user_id}")
                except Exception as cache_err:
                    logger.error(f"Failed to cache recommendations: {cache_err}")
                
                # Log this recommendation event
                try:
                    view_id = f"msg_rec_{int(time.time())}"
                    experts_shown = [exp.get('id') for exp in messaging_recommendations]
                    
                    # Store for 24 hours
                    await redis_client.setex(
                        f"rec_view:{view_id}",
                        86400,  # 24 hours
                        json.dumps({
                            'user_id': user_id,
                            'experts': experts_shown,
                            'timestamp': datetime.utcnow().isoformat(),
                            'source': 'messaging'
                        })
                    )
                    
                    # Add view_id to response for frontend tracking
                    response_data['view_id'] = view_id
                    
                    logger.info(f"Recorded messaging recommendation view {view_id} for user {user_id}")
                    
                    # Also log to user_interest_logs if that function exists
                    if '_log_recommendation_event' in globals():
                        await _log_recommendation_event(user_id, response_data, redis_client)
                except Exception as log_err:
                    logger.error(f"Error logging recommendation event: {log_err}")
                
                # NEW: Record recommendation monitoring data
                try:
                    # Record this recommendation snapshot with domain/field information
                    weights = {
                        "domain_weight": 0.6,
                        "field_weight": 0.4,
                        "concept_weight": 0.0,  # Not used in this particular query
                        "organizational_weight": 0.0  # Not used in this particular query
                    }
                    
                    # Record the snapshot
                    recommendation_monitor.record_recommendation_snapshot(
                        user_id=user_id,
                        recommendations=messaging_recommendations,
                        weights=weights,
                        features_used=["messaging_history", "domain_matching", "field_matching"],
                        data_sources_used=["message_domains", "message_fields"]
                    )
                    
                    # Record adaptiveness metric - this recommendation was based on messaging data
                    recommendation_monitor.record_metric(
                        metric_type="adaptation",
                        metric_name="messaging_based_recommendations",
                        metric_value=1.0,  # Using this feature is 100% adoption
                        user_id=user_id,
                        details={
                            "domain_count": len(top_domains),
                            "field_count": len(top_fields),
                            "recommendation_count": len(messaging_recommendations),
                            "response_time": process_time
                        }
                    )
                    
                    # Monitor changes from previous recommendations
                    change_metrics = recommendation_monitor.monitor_recommendation_changes(
                        user_id=user_id,
                        new_recommendations=messaging_recommendations,
                        weights=weights,
                        features_used=["messaging_history", "domain_matching", "field_matching"],
                        data_sources_used=["message_domains", "message_fields"]
                    )
                    
                    # Record the change metrics if this isn't the first recommendation
                    if change_metrics and not change_metrics.get('is_first_recommendation', False):
                        recommendation_monitor.record_metric(
                            metric_type="adaptation",
                            metric_name="recommendation_turnover_rate",
                            metric_value=change_metrics.get('turnover_rate', 0.0),
                            user_id=user_id,
                            details={
                                "added_experts_count": len(change_metrics.get('added_experts', [])),
                                "removed_experts_count": len(change_metrics.get('removed_experts', [])),
                                "uses_messaging_data": True
                            }
                        )
                    
                    logger.info("Recommendation monitoring data recorded successfully")
                except Exception as monitoring_err:
                    logger.error(f"Error recording recommendation monitoring data: {str(monitoring_err)}")
                    # Continue anyway - monitoring shouldn't break the main functionality
                
                logger.info(f"Successfully returned {len(messaging_recommendations)} messaging-based recommendations for user {user_id}")
                return response_data
                
        except Exception as db_error:
            logger.error(f"Database error getting messaging history: {str(db_error)}")
            
            # Record error metric
            try:
                recommendation_monitor.record_metric(
                    metric_type="errors",
                    metric_name="messaging_recommendations_db_error",
                    metric_value=1.0,
                    user_id=user_id,
                    details={"error": str(db_error)[:200]}  # Truncate to avoid massive error messages
                )
            except Exception:
                pass  # If we can't record the metric, just continue
                
            # Fall back to regular recommendations
            recommendations = await process_recommendations(user_id, redis_client)
            return recommendations
        finally:
            if conn:
                conn.close()
        
    except Exception as e:
        logger.error(f"Error getting messaging-based recommendations for user {user_id}: {str(e)}", exc_info=True)
        
        # Record critical error metric
        try:
            recommendation_monitor.record_metric(
                metric_type="errors",
                metric_name="messaging_recommendations_critical_error",
                metric_value=1.0,
                user_id=user_id,
                details={"error": str(e)[:200]}  # Truncate to avoid massive error messages
            )
        except Exception:
            pass  # If we can't record the metric, just continue
            
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

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


import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import matplotlib.pyplot as plt

from io import BytesIO
import base64
import csv
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recommendation_monitoring.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class RecommendationMonitor:
    """
    Monitors the adaptiveness and performance of the recommendation system
    by tracking key metrics over time.
    """
    
    def __init__(self):
        """Initialize the RecommendationMonitor with database connection info"""
        self.logger = logging.getLogger(__name__)
        
        # Define connection parameters
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            parsed_url = urlparse(database_url)
            self.conn_params = {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'dbname': parsed_url.path[1:],
                'user': parsed_url.username,
                'password': parsed_url.password
            }
        else:
            self.conn_params = {
                'host': os.getenv('POSTGRES_HOST', 'postgres'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
            }
        
        self.logger.info("RecommendationMonitor initialized successfully")
        
        # Create the recommendation_metrics table if it doesn't exist
        self._ensure_metrics_table_exists()
        
    def _ensure_metrics_table_exists(self):
        """Create the necessary tables for monitoring if they don't exist."""
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                # Create recommendation_metrics table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metric_type VARCHAR(100) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    user_id VARCHAR(255) NULL,
                    expert_id VARCHAR(255) NULL,
                    details JSONB NULL
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_rec_metrics_timestamp ON recommendation_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rec_metrics_type ON recommendation_metrics(metric_type);
                CREATE INDEX IF NOT EXISTS idx_rec_metrics_user ON recommendation_metrics(user_id);
                """)
                
                # Create recommendation_feedback table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_feedback (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(255) NOT NULL,
                    expert_id VARCHAR(255) NOT NULL,
                    recommendation_method VARCHAR(100) NOT NULL,
                    interaction_type VARCHAR(100) NOT NULL,
                    feedback_score FLOAT NULL,
                    details JSONB NULL
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_rec_feedback_timestamp ON recommendation_feedback(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rec_feedback_user ON recommendation_feedback(user_id);
                CREATE INDEX IF NOT EXISTS idx_rec_feedback_expert ON recommendation_feedback(expert_id);
                """)
                
                # Create recommendation_snapshot table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_snapshot (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(255) NOT NULL,
                    recommendations JSONB NOT NULL,
                    weights JSONB NULL,
                    features_used JSONB NULL,
                    data_sources_used JSONB NULL
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_rec_snapshot_timestamp ON recommendation_snapshot(timestamp);
                CREATE INDEX IF NOT EXISTS idx_rec_snapshot_user ON recommendation_snapshot(user_id);
                """)
                
                conn.commit()
                self.logger.info("Recommendation monitoring tables created or already exist")
        except Exception as e:
            self.logger.error(f"Error creating monitoring tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def record_metric(self, metric_type: str, metric_name: str, metric_value: float, 
                     user_id: Optional[str] = None, expert_id: Optional[str] = None, 
                     details: Optional[Dict] = None):
        """
        Record a metric for monitoring recommendation system performance.
        
        Args:
            metric_type: Category of metric (e.g., 'adaptation', 'performance', 'diversity')
            metric_name: Specific metric name (e.g., 'click_through_rate', 'domain_coverage')
            metric_value: Numerical value of the metric
            user_id: Optional user ID if metric is user-specific
            expert_id: Optional expert ID if metric is expert-specific
            details: Optional dictionary with additional details
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO recommendation_metrics
                    (timestamp, metric_type, metric_name, metric_value, user_id, expert_id, details)
                VALUES
                    (NOW(), %s, %s, %s, %s, %s, %s)
                """, (
                    metric_type, 
                    metric_name, 
                    metric_value, 
                    user_id, 
                    expert_id, 
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                self.logger.info(f"Recorded metric: {metric_type}.{metric_name} = {metric_value}")
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def record_feedback(self, user_id: str, expert_id: str, recommendation_method: str,
                       interaction_type: str, feedback_score: Optional[float] = None,
                       details: Optional[Dict] = None):
        """
        Record user feedback on recommendations for adaptation analysis.
        
        Args:
            user_id: The user who received the recommendation
            expert_id: The expert who was recommended
            recommendation_method: Method used to generate recommendation (e.g., 'messaging', 'semantic')
            interaction_type: Type of interaction (e.g., 'click', 'message', 'ignore')
            feedback_score: Optional numerical score (-1 to 1)
            details: Optional dictionary with additional details
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO recommendation_feedback
                    (timestamp, user_id, expert_id, recommendation_method, interaction_type, feedback_score, details)
                VALUES
                    (NOW(), %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    expert_id,
                    recommendation_method,
                    interaction_type,
                    feedback_score,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                self.logger.info(f"Recorded feedback from user {user_id} for expert {expert_id}")
        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def record_recommendation_snapshot(self, user_id: str, recommendations: List[Dict],
                                     weights: Optional[Dict] = None,
                                     features_used: Optional[List[str]] = None,
                                     data_sources_used: Optional[List[str]] = None):
        """
        Record a snapshot of recommendations for later analysis of system adaptation.
        
        Args:
            user_id: The user who received the recommendations
            recommendations: List of recommendation objects
            weights: Optional dictionary of weights used in the recommendation algorithm
            features_used: Optional list of features used in making recommendations
            data_sources_used: Optional list of data sources used
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO recommendation_snapshot
                    (timestamp, user_id, recommendations, weights, features_used, data_sources_used)
                VALUES
                    (NOW(), %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    json.dumps(recommendations),
                    json.dumps(weights) if weights else None,
                    json.dumps(features_used) if features_used else None,
                    json.dumps(data_sources_used) if data_sources_used else None
                ))
                
                conn.commit()
                self.logger.info(f"Recorded recommendation snapshot for user {user_id} with {len(recommendations)} recommendations")
        except Exception as e:
            self.logger.error(f"Error recording recommendation snapshot: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def measure_domain_field_adoption(self, start_date: Optional[datetime] = None, 
                                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Measure how domain and field data from messages is being used in recommendations.
        
        Args:
            start_date: Optional start date for analysis window
            end_date: Optional end date for analysis window
            
        Returns:
            Dictionary with metrics about domain/field adoption
        """
        conn = None
        try:
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
                
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count messaging interactions
                cur.execute("""
                SELECT 
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(ARRAY_LENGTH(domains, 1)) as avg_domains_per_message,
                    AVG(ARRAY_LENGTH(fields, 1)) as avg_fields_per_message,
                    COUNT(*) FILTER (WHERE ARRAY_LENGTH(domains, 1) > 0) as messages_with_domains,
                    COUNT(*) FILTER (WHERE ARRAY_LENGTH(fields, 1) > 0) as messages_with_fields
                FROM message_expert_interactions
                WHERE created_at BETWEEN %s AND %s
                """, (start_date, end_date))
                
                message_stats = cur.fetchone()
                
                # Compare with recommendation snapshots
                cur.execute("""
                SELECT 
                    COUNT(*) as total_snapshots,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(*) FILTER (
                        WHERE data_sources_used @> '"messaging"' OR 
                              data_sources_used @> '"message_domains"' OR
                              data_sources_used @> '"message_fields"'
                    ) as snapshots_using_messaging
                FROM recommendation_snapshot
                WHERE timestamp BETWEEN %s AND %s
                """, (start_date, end_date))
                
                recommendation_stats = cur.fetchone()
                
                # Get top domains and fields mentioned
                cur.execute("""
                SELECT 
                    UNNEST(domains) as domain,
                    COUNT(*) as mention_count
                FROM message_expert_interactions
                WHERE created_at BETWEEN %s AND %s
                GROUP BY domain
                ORDER BY mention_count DESC
                LIMIT 10
                """, (start_date, end_date))
                
                top_domains = [dict(row) for row in cur.fetchall()]
                
                cur.execute("""
                SELECT 
                    UNNEST(fields) as field,
                    COUNT(*) as mention_count
                FROM message_expert_interactions
                WHERE created_at BETWEEN %s AND %s
                GROUP BY field
                ORDER BY mention_count DESC
                LIMIT 10
                """, (start_date, end_date))
                
                top_fields = [dict(row) for row in cur.fetchall()]
                
                # Calculate messaging influence metrics
                messaging_adoption = 0
                if recommendation_stats['total_snapshots'] > 0:
                    messaging_adoption = recommendation_stats['snapshots_using_messaging'] / recommendation_stats['total_snapshots']
                
                # Compile metrics
                metrics = {
                    "message_stats": message_stats,
                    "recommendation_stats": recommendation_stats,
                    "top_domains": top_domains,
                    "top_fields": top_fields,
                    "messaging_adoption": messaging_adoption,
                    "analysis_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": (end_date - start_date).days
                    }
                }
                
                # Record these metrics
                self.record_metric(
                    metric_type="adaptation",
                    metric_name="messaging_domain_field_adoption",
                    metric_value=messaging_adoption,
                    details={
                        "message_count": message_stats['total_interactions'],
                        "recommendation_count": recommendation_stats['total_snapshots'],
                        "period_days": (end_date - start_date).days
                    }
                )
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error measuring domain/field adoption: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def analyze_recommendation_evolution(self, user_id: str, 
                                        lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze how recommendations for a specific user have evolved over time.
        
        Args:
            user_id: User ID to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with metrics about recommendation evolution
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get snapshots for this user
                start_date = datetime.now() - timedelta(days=lookback_days)
                
                cur.execute("""
                SELECT 
                    id,
                    timestamp,
                    recommendations,
                    weights,
                    features_used,
                    data_sources_used
                FROM recommendation_snapshot
                WHERE user_id = %s AND timestamp > %s
                ORDER BY timestamp
                """, (user_id, start_date))
                
                snapshots = [dict(row) for row in cur.fetchall()]
                
                if not snapshots:
                    return {
                        "user_id": user_id,
                        "analysis": "No recommendation snapshots found for this user in the specified time period."
                    }
                
                # Extract expert IDs from each snapshot
                evolution_data = []
                for snapshot in snapshots:
                    expert_ids = []
                    try:
                        recommendations = json.loads(snapshot['recommendations'])
                        expert_ids = [rec.get('id') for rec in recommendations if rec.get('id')]
                        
                        # Extract data sources and features
                        data_sources = []
                        if snapshot['data_sources_used']:
                            data_sources = json.loads(snapshot['data_sources_used'])
                        
                        features = []
                        if snapshot['features_used']:
                            features = json.loads(snapshot['features_used'])
                        
                        evolution_data.append({
                            "timestamp": snapshot['timestamp'].isoformat(),
                            "expert_ids": expert_ids,
                            "expert_count": len(expert_ids),
                            "data_sources": data_sources,
                            "features": features,
                            "uses_messaging": any(
                                source in ['messaging', 'message_domains', 'message_fields'] 
                                for source in data_sources
                            )
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing snapshot {snapshot['id']}: {e}")
                
                # Calculate change metrics
                if len(evolution_data) >= 2:
                    first_experts = set(evolution_data[0]['expert_ids'])
                    last_experts = set(evolution_data[-1]['expert_ids'])
                    
                    # Calculate Jaccard similarity
                    intersection = len(first_experts.intersection(last_experts))
                    union = len(first_experts.union(last_experts))
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    # Calculate turnover rate
                    turnover_rate = 1.0 - jaccard_similarity
                    
                    # Check for increasing use of messaging data
                    messaging_adoption_trend = []
                    for data_point in evolution_data:
                        messaging_adoption_trend.append(1 if data_point['uses_messaging'] else 0)
                    
                    # Calculate if messaging usage is increasing
                    messaging_trend = 0
                    if len(messaging_adoption_trend) >= 2:
                        early_usage = sum(messaging_adoption_trend[:len(messaging_adoption_trend)//2])
                        late_usage = sum(messaging_adoption_trend[len(messaging_adoption_trend)//2:])
                        messaging_trend = late_usage - early_usage
                    
                    evolution_metrics = {
                        "jaccard_similarity": jaccard_similarity,
                        "turnover_rate": turnover_rate,
                        "snapshot_count": len(evolution_data),
                        "messaging_adoption_trend": messaging_trend,
                        "first_timestamp": evolution_data[0]['timestamp'],
                        "last_timestamp": evolution_data[-1]['timestamp']
                    }
                    
                    # Record the turnover metric
                    self.record_metric(
                        metric_type="adaptation",
                        metric_name="recommendation_turnover_rate",
                        metric_value=turnover_rate,
                        user_id=user_id,
                        details={
                            "jaccard_similarity": jaccard_similarity,
                            "snapshot_count": len(evolution_data),
                            "period_days": lookback_days
                        }
                    )
                else:
                    evolution_metrics = {
                        "message": "Not enough snapshots for evolution analysis"
                    }
                
                return {
                    "user_id": user_id,
                    "evolution_data": evolution_data,
                    "evolution_metrics": evolution_metrics,
                    "analysis_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": datetime.now().isoformat(),
                        "days": lookback_days
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing recommendation evolution: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def generate_adaptation_report(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  output_format: str = 'json') -> Any:
        """
        Generate a comprehensive report on recommendation system adaptation.
        
        Args:
            start_date: Optional start date for analysis window
            end_date: Optional end date for analysis window
            output_format: Format of the report ('json', 'csv', 'html', 'plot')
            
        Returns:
            Report in the specified format
        """
        conn = None
        try:
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
                
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Gather metrics from recommendation_metrics table
                cur.execute("""
                SELECT 
                    metric_type,
                    metric_name,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value,
                    COUNT(*) as measurement_count,
                    MIN(timestamp) as first_measurement,
                    MAX(timestamp) as last_measurement
                FROM recommendation_metrics
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY metric_type, metric_name
                ORDER BY metric_type, metric_name
                """, (start_date, end_date))
                
                metrics_summary = [dict(row) for row in cur.fetchall()]
                
                # Gather domain/field usage stats
                domain_field_stats = self.measure_domain_field_adoption(start_date, end_date)
                
                # Gather user feedback stats
                cur.execute("""
                SELECT 
                    recommendation_method,
                    interaction_type,
                    COUNT(*) as interaction_count,
                    AVG(feedback_score) as avg_feedback_score
                FROM recommendation_feedback
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY recommendation_method, interaction_type
                ORDER BY recommendation_method, interaction_type
                """, (start_date, end_date))
                
                feedback_stats = [dict(row) for row in cur.fetchall()]
                
                # Gather system adaptation metrics over time
                cur.execute("""
                SELECT 
                    DATE_TRUNC('day', timestamp) as day,
                    COUNT(*) as snapshot_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(*) FILTER (
                        WHERE data_sources_used @> '"message_domains"' OR
                              data_sources_used @> '"message_fields"'
                    ) as snapshots_using_messaging
                FROM recommendation_snapshot
                WHERE timestamp BETWEEN %s AND %s
                GROUP BY DATE_TRUNC('day', timestamp)
                ORDER BY DATE_TRUNC('day', timestamp)
                """, (start_date, end_date))
                
                daily_adaptation = [dict(row) for row in cur.fetchall()]
                
                # Compile the report
                report = {
                    "report_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": (end_date - start_date).days
                    },
                    "metrics_summary": metrics_summary,
                    "domain_field_stats": domain_field_stats,
                    "feedback_stats": feedback_stats,
                    "daily_adaptation": daily_adaptation,
                    "generated_at": datetime.now().isoformat()
                }
                
                # Format the report as requested
                if output_format == 'json':
                    return report
                elif output_format == 'csv':
                    return self._report_to_csv(report)
                elif output_format == 'html':
                    return self._report_to_html(report)
                elif output_format == 'plot':
                    return self._generate_adaptation_plots(report)
                else:
                    return report
                
        except Exception as e:
            self.logger.error(f"Error generating adaptation report: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
    
    def _report_to_csv(self, report: Dict) -> str:
        """Convert report data to CSV format"""
        try:
            # Create a temporary file to store CSV data
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
                filename = temp.name
                
                # Write metrics summary to CSV
                writer = csv.writer(temp)
                writer.writerow(['Metric Type', 'Metric Name', 'Average Value', 'Min Value', 'Max Value', 'Count'])
                
                for metric in report['metrics_summary']:
                    writer.writerow([
                        metric['metric_type'],
                        metric['metric_name'],
                        metric['avg_value'],
                        metric['min_value'],
                        metric['max_value'],
                        metric['measurement_count']
                    ])
                
                # Add separator
                writer.writerow([])
                writer.writerow(['Domain/Field Adoption Metrics'])
                writer.writerow([])
                
                # Write domain/field stats
                if 'message_stats' in report['domain_field_stats']:
                    stats = report['domain_field_stats']['message_stats']
                    writer.writerow(['Total Interactions', 'Unique Users', 'Avg Domains/Message', 'Avg Fields/Message'])
                    writer.writerow([
                        stats['total_interactions'],
                        stats['unique_users'],
                        stats['avg_domains_per_message'],
                        stats['avg_fields_per_message']
                    ])
                
                # Add separator
                writer.writerow([])
                writer.writerow(['Top Domains'])
                writer.writerow(['Domain', 'Mention Count'])
                
                # Write top domains
                if 'top_domains' in report['domain_field_stats']:
                    for domain in report['domain_field_stats']['top_domains']:
                        writer.writerow([domain['domain'], domain['mention_count']])
                
                # Add separator
                writer.writerow([])
                writer.writerow(['Daily Adaptation'])
                writer.writerow(['Day', 'Snapshot Count', 'Unique Users', 'Using Messaging'])
                
                # Write daily adaptation
                for day in report['daily_adaptation']:
                    writer.writerow([
                        day['day'].strftime('%Y-%m-%d') if isinstance(day['day'], datetime) else day['day'],
                        day['snapshot_count'],
                        day['unique_users'],
                        day['snapshots_using_messaging']
                    ])
            
            # Read the temporary file
            with open(filename, 'r') as f:
                csv_content = f.read()
            
            # Clean up the temporary file
            os.unlink(filename)
            
            return csv_content
            
        except Exception as e:
            self.logger.error(f"Error converting report to CSV: {e}")
            return f"Error: {str(e)}"
    
    def _report_to_html(self, report: Dict) -> str:
        """Convert report data to HTML format"""
        try:
            html = f"""
            <html>
            <head>
                <title>Recommendation System Adaptation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-good {{ color: green; }}
                    .metric-bad {{ color: red; }}
                    .metric-neutral {{ color: orange; }}
                </style>
            </head>
            <body>
                <h1>Recommendation System Adaptation Report</h1>
                <p>Period: {report['report_period']['start_date']} to {report['report_period']['end_date']} ({report['report_period']['days']} days)</p>
                
                <h2>Metrics Summary</h2>
                <table>
                    <tr>
                        <th>Metric Type</th>
                        <th>Metric Name</th>
                        <th>Average Value</th>
                        <th>Min Value</th>
                        <th>Max Value</th>
                        <th>Measurements</th>
                    </tr>
            """
            
            # Add metrics rows
            for metric in report['metrics_summary']:
                html += f"""
                    <tr>
                        <td>{metric['metric_type']}</td>
                        <td>{metric['metric_name']}</td>
                        <td>{metric['avg_value']:.4f}</td>
                        <td>{metric['min_value']:.4f}</td>
                        <td>{metric['max_value']:.4f}</td>
                        <td>{metric['measurement_count']}</td>
                    </tr>
                """
            
            html += """
                </table>
                
                <h2>Domain/Field Adoption</h2>
            """
            
            # Add domain/field stats
            if 'message_stats' in report['domain_field_stats']:
                stats = report['domain_field_stats']['message_stats']
                html += f"""
                <table>
                    <tr>
                        <th>Total Interactions</th>
                        <th>Unique Users</th>
                        <th>Avg Domains/Message</th>
                        <th>Avg Fields/Message</th>
                    </tr>
                    <tr>
                        <td>{stats['total_interactions']}</td>
                        <td>{stats['unique_users']}</td>
                        <td>{stats['avg_domains_per_message']:.2f}</td>
                        <td>{stats['avg_fields_per_message']:.2f}</td>
                    </tr>
                </table>
                """
            
            # Add messaging adoption metric
            if 'messaging_adoption' in report['domain_field_stats']:
                adoption = report['domain_field_stats']['messaging_adoption']
                adoption_class = 'metric-good' if adoption > 0.7 else ('metric-neutral' if adoption > 0.3 else 'metric-bad')
                html += f"""
                <p>Messaging Adoption Rate: <span class="{adoption_class}">{adoption:.2%}</span></p>
                """
            
            # Add top domains
            if 'top_domains' in report['domain_field_stats']:
                html += """
                <h3>Top Domains</h3>
                <table>
                    <tr>
                        <th>Domain</th>
                        <th>Mention Count</th>
                    </tr>
                """
                
                for domain in report['domain_field_stats']['top_domains']:
                    html += f"""
                    <tr>
                        <td>{domain['domain']}</td>
                        <td>{domain['mention_count']}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Add top fields
            if 'top_fields' in report['domain_field_stats']:
                html += """
                <h3>Top Fields</h3>
                <table>
                    <tr>
                        <th>Field</th>
                        <th>Mention Count</th>
                    </tr>
                """
                
                for field in report['domain_field_stats']['top_fields']:
                    html += f"""
                    <tr>
                        <td>{field['field']}</td>
                        <td>{field['mention_count']}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Add feedback stats
            if report['feedback_stats']:
                html += """
                <h2>User Feedback</h2>
                <table>
                    <tr>
                        <th>Recommendation Method</th>
                        <th>Interaction Type</th>
                        <th>Count</th>
                        <th>Avg Feedback Score</th>
                    </tr>
                """
                
                for feedback in report['feedback_stats']:
                    score_class = 'metric-neutral'
                    if feedback['avg_feedback_score'] is not None:
                        score_class = 'metric-good' if feedback['avg_feedback_score'] > 0.7 else ('metric-neutral' if feedback['avg_feedback_score'] > 0.3 else 'metric-bad')
                    
                    html += f"""
                    <tr>
                        <td>{feedback['recommendation_method']}</td>
                        <td>{feedback['interaction_type']}</td>
                        <td>{feedback['interaction_count']}</td>
                        <td class="{score_class}">{feedback['avg_feedback_score']:.2f if feedback['avg_feedback_score'] is not None else 'N/A'}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Close the HTML document
            html += f"""
                <p><em>Report generated at: {report['generated_at']}</em></p>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error converting report to HTML: {e}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def _generate_adaptation_plots(self, report: Dict) -> Dict[str, str]:
        """Generate plots visualizing system adaptation"""
        plots = {}
        
        try:
            # Set up the style
            plt.style.use('ggplot')
            sns.set_palette("viridis")
            
            # Plot 1: Messaging Adoption Over Time
            if report['daily_adaptation']:
                plt.figure(figsize=(10, 6))
                
                # Convert data to pandas DataFrame
                df = pd.DataFrame(report['daily_adaptation'])
                
                # Convert day strings to datetime if needed
                if not isinstance(df['day'].iloc[0], datetime):
                    df['day'] = pd.to_datetime(df['day'])
                
                # Calculate adoption percentage
                df['messaging_pct'] = df['snapshots_using_messaging'] / df['snapshot_count'] * 100
                
                # Plot
                ax = sns.lineplot(x='day', y='messaging_pct', data=df, marker='o', linewidth=2)
                ax.set_title('Messaging-Based Recommendations Adoption Over Time', fontsize=14)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Percentage of Recommendations (%)', fontsize=12)
                ax.set_ylim(0, 100)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Save to BytesIO and convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                
                # Convert to base64
                encoded = base64.b64encode(image_png).decode('utf-8')
                plots['messaging_adoption'] = f"data:image/png;base64,{encoded}"
                
                plt.close()
            
            # Plot 2: Top Domains and Fields
            if 'top_domains' in report['domain_field_stats'] or 'top_fields' in report['domain_field_stats']:
                plt.figure(figsize=(12, 10))
                
                # Create subplot for domains
                if 'top_domains' in report['domain_field_stats'] and report['domain_field_stats']['top_domains']:
                    ax1 = plt.subplot(2, 1, 1)
                    domains_df = pd.DataFrame(report['domain_field_stats']['top_domains'])
                    
                    # Sort by count
                    domains_df = domains_df.sort_values('mention_count', ascending=False)
                    
                    # Plot horizontal bar chart
                    sns.barplot(x='mention_count', y='domain', data=domains_df.head(10), ax=ax1)
                    ax1.set_title('Top Domains Mentioned in Messages', fontsize=14)
                    ax1.set_xlabel('Mention Count', fontsize=12)
                    ax1.set_ylabel('Domain', fontsize=12)
                
                # Create subplot for fields
                if 'top_fields' in report['domain_field_stats'] and report['domain_field_stats']['top_fields']:
                    ax2 = plt.subplot(2, 1, 2)
                    fields_df = pd.DataFrame(report['domain_field_stats']['top_fields'])
                    
                    # Sort by count
                    fields_df = fields_df.sort_values('mention_count', ascending=False)
                    
                    # Plot horizontal bar chart
                    sns.barplot(x='mention_count', y='field', data=fields_df.head(10), ax=ax2)
                    ax2.set_title('Top Fields Mentioned in Messages', fontsize=14)
                    ax2.set_xlabel('Mention Count', fontsize=12)
                    ax2.set_ylabel('Field', fontsize=12)
                
                plt.tight_layout()
                
                # Save to BytesIO and convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                
                # Convert to base64
                encoded = base64.b64encode(image_png).decode('utf-8')
                plots['top_domains_fields'] = f"data:image/png;base64,{encoded}"
                
                plt.close()
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error generating adaptation plots: {e}")
            return {"error": str(e)}
    
    def monitor_recommendation_changes(self, user_id: str, 
                                       new_recommendations: List[Dict], 
                                       weights: Optional[Dict] = None,
                                       features_used: Optional[List[str]] = None,
                                       data_sources_used: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Monitor and record changes in recommendations for a user.
        
        Args:
            user_id: The user who received the recommendations
            new_recommendations: List of new recommendation objects
            weights: Optional dictionary of weights used in the recommendation algorithm
            features_used: Optional list of features used in making recommendations
            data_sources_used: Optional list of data sources used
            
        Returns:
            Dictionary with change metrics
        """
        # First, record the snapshot
        self.record_recommendation_snapshot(
            user_id=user_id,
            recommendations=new_recommendations,
            weights=weights,
            features_used=features_used,
            data_sources_used=data_sources_used
        )
        
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get the most recent previous snapshot for this user
                cur.execute("""
                SELECT 
                    id,
                    timestamp,
                    recommendations
                FROM recommendation_snapshot
                WHERE user_id = %s AND id < (
                    SELECT MAX(id) FROM recommendation_snapshot WHERE user_id = %s
                )
                ORDER BY id DESC
                LIMIT 1
                """, (user_id, user_id))
                
                prev_snapshot = cur.fetchone()
                
                if not prev_snapshot:
                    return {
                        "user_id": user_id,
                        "message": "No previous snapshots available for comparison",
                        "is_first_recommendation": True
                    }
                
                # Extract expert IDs from snapshots
                try:
                    prev_recommendations = json.loads(prev_snapshot['recommendations'])
                    prev_expert_ids = [rec.get('id') for rec in prev_recommendations if rec.get('id')]
                    new_expert_ids = [rec.get('id') for rec in new_recommendations if rec.get('id')]
                    
                    # Calculate metrics
                    prev_set = set(prev_expert_ids)
                    new_set = set(new_expert_ids)
                    
                    # Calculate Jaccard similarity
                    intersection = len(prev_set.intersection(new_set))
                    union = len(prev_set.union(new_set))
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    # Calculate turnover rate
                    turnover_rate = 1.0 - jaccard_similarity
                    
                    # Get added and removed experts
                    added_experts = list(new_set - prev_set)
                    removed_experts = list(prev_set - new_set)
                    
                    # Get whether messaging factored into the changes
                    uses_messaging = any(
                        source in ['messaging', 'message_domains', 'message_fields'] 
                        for source in (data_sources_used or [])
                    )
                    
                    # Compile change metrics
                    change_metrics = {
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "jaccard_similarity": jaccard_similarity,
                        "turnover_rate": turnover_rate,
                        "added_experts": added_experts,
                        "removed_experts": removed_experts,
                        "prev_snapshot_id": prev_snapshot['id'],
                        "prev_snapshot_timestamp": prev_snapshot['timestamp'].isoformat(),
                        "time_since_last_snapshot": (datetime.now() - prev_snapshot['timestamp']).total_seconds() / 3600,  # in hours
                        "uses_messaging_data": uses_messaging
                    }
                    
                    # Record the turnover metric
                    self.record_metric(
                        metric_type="adaptation",
                        metric_name="recommendation_change_rate",
                        metric_value=turnover_rate,
                        user_id=user_id,
                        details={
                            "jaccard_similarity": jaccard_similarity,
                            "added_experts": len(added_experts),
                            "removed_experts": len(removed_experts),
                            "uses_messaging": uses_messaging
                        }
                    )
                    
                    return change_metrics
                    
                except Exception as e:
                    self.logger.error(f"Error comparing recommendation snapshots: {e}")
                    return {"error": str(e)}
                
        except Exception as e:
            self.logger.error(f"Error monitoring recommendation changes: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()
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