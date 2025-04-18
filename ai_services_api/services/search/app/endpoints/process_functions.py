import os
import redis
from fastapi import HTTPException
from typing import Any, List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json
import uuid
import asyncio
from redis import asyncio as aioredis
import traceback

from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.search.core.models import PredictionResponse
from ai_services_api.services.search.core.personalization import get_selected_suggestions, get_trending_suggestions, get_user_search_history, personalize_suggestions
import json
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict
import random

from fastapi import HTTPException
from redis.asyncio import Redis
# In your other Python script
from ai_services_api.services.message.core.database import get_db_connection

# Configure logger
logger = logging.getLogger(__name__)

# Global predictor instance (singleton)
_predictor = None

# Global Redis instance
_redis = None

# In ai_services_api/services/search/app/endpoints/process_functions.py

# Initialize global categorizer instance
_categorizer = None

async def get_search_categorizer():
    """
    Get or create a singleton instance of the SearchCategorizer.
    
    Returns:
        SearchCategorizer instance or None if not available
    """
    global _categorizer
    
    if _categorizer is None:
        try:
            # Get Redis client for caching
            redis_client = await get_redis()
            
            # Import the SearchCategorizer
            from ai_services_api.services.search.gemini.search_categorizer import SearchCategorizer
            
            # Create categorizer instance
            _categorizer = SearchCategorizer(redis_client=redis_client)
            
            # Start background worker
            await _categorizer.start_background_worker()
            
            logger.info("Search categorizer initialized successfully")
        except ImportError:
            logger.warning("SearchCategorizer module not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing search categorizer: {e}")
            return None
    
    return _categorizer







# Helper function to extract refinements list
def extract_refinements_list(refinements_dict):
    """Extract a flat list of refinements from the refinements dictionary."""
    refinements_list = []
    if isinstance(refinements_dict, dict):
        # Extract related queries if they exist
        if "related_queries" in refinements_dict and isinstance(refinements_dict["related_queries"], list):
            refinements_list.extend(refinements_dict["related_queries"])
        
        # Extract expertise areas if they exist
        if "expertise_areas" in refinements_dict and isinstance(refinements_dict["expertise_areas"], list):
            refinements_list.extend(refinements_dict["expertise_areas"])
        
        # Extract values from filters if they exist
        if "filters" in refinements_dict and isinstance(refinements_dict["filters"], list):
            for filter_item in refinements_dict["filters"]:
                if isinstance(filter_item, dict) and "values" in filter_item:
                    if isinstance(filter_item["values"], list):
                        refinements_list.extend(filter_item["values"])
                    else:
                        refinements_list.append(str(filter_item["values"]))
    
    return refinements_list

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

async def get_redis():
    """Get or create Redis connection."""
    global _redis
    if _redis is None:
        try:
            # Create Redis connection
            _redis = await aioredis.Redis.from_url("redis://redis:6379/0", decode_responses=True)
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}", exc_info=True)
            _redis = None
    return _redis

async def get_predictor():
    """Get or create predictor instance (singleton pattern)."""
    global _predictor
    if _predictor is None:
        _predictor = GoogleAutocompletePredictor()
    return _predictor


async def _record_prediction_async(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """Asynchronous helper for recording predictions in the database."""
    logger.info(f"Recording predictions for user {user_id}, session {session_id}")
    
    # Create a new cursor for this async task to avoid connection issues
    try:
        cur = conn.cursor()
        try:
            for pred, conf in zip(predictions, confidence_scores):
                cur.execute("""
                    INSERT INTO query_predictions
                        (partial_query, predicted_query, confidence_score, 
                        user_id, timestamp)
                    VALUES 
                        (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (partial_query, pred, conf, user_id))
            
            conn.commit()
            logger.info("Successfully recorded all predictions")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error recording prediction: {str(e)}", exc_info=True)
        finally:
            cur.close()
    except Exception as conn_error:
        logger.error(f"Connection error in record_prediction: {conn_error}")
    # Note: Don't close the connection here as it might be used elsewhere



async def record_prediction(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """
    Record prediction in database for analytics.
    Uses an existing connection instead of creating a new one.
    """
    if not conn:
        logger.error("Cannot record prediction: No database connection provided")
        return
        
    try:
        # Create a cursor for this operation using the existing connection
        cur = conn.cursor()
        try:
            for pred, conf in zip(predictions, confidence_scores):
                cur.execute("""
                    INSERT INTO query_predictions
                        (partial_query, predicted_query, confidence_score, 
                        user_id, timestamp)
                    VALUES 
                        (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (partial_query, pred, conf, user_id))
            
            conn.commit()
            logger.info("Successfully recorded all predictions")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error recording prediction: {str(e)}", exc_info=True)
        finally:
            cur.close()
    except Exception as conn_error:
        logger.error(f"Connection error in record_prediction: {conn_error}")
    # Note: We don't close the connection here since it was passed in

def generate_predictive_refinements(partial_query: str, predictions: List[str]) -> Dict[str, Any]:
    """Generate refinement suggestions based on partial query and predictions."""
    try:
        # Generate related filters and expertise areas
        related_filters = []
        
        # Add a filter for query type if we have predictions
        if predictions:
            related_filters.append({
                "type": "query_type",
                "label": "Query Suggestions",
                "values": predictions[:3]
            })
        
        # Extract potential expertise areas from predictions
        expertise_areas = set()
        for prediction in predictions:
            words = prediction.lower().split()
            for word in words:
                if len(word) > 3 and word != partial_query.lower():
                    expertise_areas.add(word.capitalize())
        
        # Make sure we have some expertise areas even if extraction failed
        if len(expertise_areas) < 3 and partial_query:
            expertise_areas.add(partial_query.capitalize())
        
        # Remove duplicates while preserving order
        expertise_areas = list(dict.fromkeys(expertise_areas))[:5]
        
        # Construct refinements
        refinements = {
            "filters": related_filters,
            "related_queries": predictions[:5],
            "expertise_areas": expertise_areas
        }
        
        logger.info(f"Generated predictive refinements for query: {partial_query}")
        return refinements
    
    except Exception as e:
        logger.error(f"Error in predictive refinements: {e}")
        # Return a minimal valid structure even on error
        return {
            "filters": [],
            "related_queries": predictions[:5] if predictions else [],
            "expertise_areas": []
        }
    
from typing import Any, List, Dict, Optional
import logging
import uuid
import json
import traceback

from ai_services_api.services.search.core.models import PredictionResponse

# Configure logger
logger = logging.getLogger(__name__)

import os
import logging
import psycopg2
from typing import Optional, Tuple
from urllib.parse import urlparse

# Configure logger
logger = logging.getLogger(__name__)

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
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres'),
            'connect_timeout': 10  # Timeout for connection attempt
        }

def get_db_connection(dbname=None):
    """Get a direct database connection."""
    params = get_connection_params()
    if dbname:
        params['dbname'] = dbname 
    
    try:
        conn = psycopg2.connect(**params)
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

# Simplified replacements for pool-related functions
def get_pooled_connection() -> Tuple[Optional[psycopg2.extensions.connection], None, bool, str]:
    """
    Replacement for pooled connection method.
    Returns a direct connection, with pool-related parameters set to None/False.
    
    Returns:
        Tuple of (connection, None, False, connection_id)
    """
    try:
        conn = get_db_connection()
        conn_id = str(id(conn))  # Use connection's memory address as an ID
        return conn, None, False, conn_id
    except Exception as e:
        logger.error(f"Error getting connection: {e}")
        return None, None, False, ''

def return_connection(conn, pool, using_pool, conn_id):
    """
    Replacement for returning connection to pool.
    For direct connections, this simply closes the connection.
    """
    if conn:
        try:
            conn.close()
            logger.info(f"Closed connection {conn_id}")
        except Exception as e:
            logger.error(f"Error closing connection {conn_id}: {e}")

def log_pool_status():
    """
    Placeholder for pool status logging.
    Does nothing for direct connections.
    """
    # No-op function for compatibility
    pass

class DatabaseConnection:
    """
    Context manager for database connections to maintain compatibility.
    Uses direct connection instead of pooling.
    """
    def __init__(self, dbname=None):
        self.dbname = dbname
        self.conn = None

    def __enter__(self):
        """Enter the runtime context for database connection."""
        try:
            self.conn = get_db_connection(self.dbname)
            return self.conn
        except Exception as e:
            logger.error(f"Error creating database connection: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context for database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
        return False  # Propagate any exceptions

def is_publication_title(suggestion: str, partial_query: str) -> bool:
    """
    Check if suggestion appears to be a publication title.
    Publication titles tend to:
    - Be longer phrases
    - Often contain research-related terms
    - Have proper capitalization patterns
    
    Args:
        suggestion: The suggestion text to check
        partial_query: The partial query being typed
        
    Returns:
        Boolean indicating if the suggestion looks like a publication title
    """
    # Convert partial query and suggestion to lowercase for comparison
    partial_query_lower = partial_query.lower()
    suggestion_lower = suggestion.lower()
    
    # First, check if partial query is in the suggestion
    if partial_query_lower not in suggestion_lower:
        return False
    
    # Publication indicators - common in academic paper titles
    pub_indicators = [
        "study", "research", "analysis", "review", "survey", 
        "approach", "framework", "method", "evaluation", "assessment",
        "impact", "effect", "comparison", "investigation", "development"
    ]
    
    # Check for publication indicators
    has_indicator = any(indicator in suggestion_lower for indicator in pub_indicators)
    
    # Check length - publications titles are usually longer
    word_count = len(suggestion.split())
    good_length = word_count >= 3 and word_count <= 20
    
    # Check for patterns like "Title: Subtitle" or quotes
    has_title_pattern = ":" in suggestion or '"' in suggestion or "'" in suggestion
    
    # Publications often have proper nouns (capitalized words not at start)
    words = suggestion.split()
    has_mid_capitals = any(word[0].isupper() for word in words[1:] if len(word) > 2)
    
    # Check for common paper title ending patterns
    has_paper_ending = any(suggestion_lower.endswith(end) for end in 
                          [" study", " analysis", " review", " approach", " framework"])
    
    # Combine factors - need at least two positive indicators
    indicators = [has_indicator, good_length, has_title_pattern, has_mid_capitals, has_paper_ending]
    score = sum(1 for indicator in indicators if indicator)
    
    return score >= 2


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

async def process_advanced_query_prediction(
    partial_query: str, 
    user_id: str, 
    context: Optional[str] = None,
    limit: int = 10
) -> PredictionResponse:
    """
    Process advanced query prediction with enhanced error handling and type safety
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing prediction - Query: '{partial_query}', User: {user_id}")

    try:
        # Validate context first
        valid_contexts = ["name", "theme", "designation", "publication", None]
        if context and context not in valid_contexts[:-1]:
            raise ValueError(f"Invalid context. Must be one of {', '.join(valid_contexts[:-1])}")

        # Initialize services with error handling
        categorizer = await get_search_categorizer()
        redis = await get_redis()
        conn = None
        predictions = []
        confidence_scores = []
        category_info = {}
        session_id = None

        # Decimal-safe score processing
        def safe_convert_score(score):
            return float(score) if isinstance(score, Decimal) else score

        # Handle empty query scenario
        if not partial_query:
            try:
                # Combined suggestions with type safety
                user_history = await get_user_search_history(user_id, limit=5)
                selected_suggestions = await get_selected_suggestions(user_id, limit=3)
                trending_suggestions = await get_trending_suggestions("", limit=3)
                
                # Process all suggestions with Decimal conversion
                all_suggestions = []
                sources = [
                    (user_history, "history", 0.9),
                    (selected_suggestions, "selected", 0.95),
                    (trending_suggestions, "trending", 0.8)
                ]

                for source_data, source_name, base_score in sources:
                    for item in source_data:
                        text = item.get("text", "").strip()
                        if text:
                            score = safe_convert_score(item.get("score", base_score))
                            all_suggestions.append({
                                "text": text,
                                "source": source_name,
                                "score": score,
                                "category": item.get("category", "general")
                            })

                # Add context-specific queries
                if context:
                    context_suggestions = await get_context_popular_queries(context, limit=3)
                    for query in context_suggestions:
                        if query.strip():
                            all_suggestions.append({
                                "text": query.strip(),
                                "source": f"{context}_popular",
                                "score": 0.85,
                                "category": context
                            })

                # Deduplicate and sort safely
                seen = set()
                filtered = []
                for item in sorted(all_suggestions, key=lambda x: x["score"], reverse=True):
                    clean_text = item["text"].lower()
                    if clean_text and clean_text not in seen:
                        seen.add(clean_text)
                        filtered.append(item)

                # Extract predictions with score conversion
                predictions = [item["text"] for item in filtered[:limit]]
                confidence_scores = [safe_convert_score(item["score"]) for item in filtered[:limit]]

                # Category processing
                if filtered:
                    category_counts = {}
                    for item in filtered[:limit]:
                        category = item.get("category", "general")
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    total = len(filtered[:limit])
                    category_info = {
                        "distribution": {k: v/total for k, v in category_counts.items()},
                        "dominant_category": max(category_counts, key=category_counts.get) if category_counts else "general"
                    }

            except Exception as e:
                logger.error(f"Initial suggestions error: {str(e)}", exc_info=True)
                raise

        else:  # Non-empty query processing
            cache_key = f"advanced_idx_predict:{context or ''}:{partial_query}"
            cached_data = None

            try:
                if redis:
                    cached_data = await redis.get(cache_key)
                    if cached_data:
                        cached_data = json.loads(cached_data)
            except Exception as e:
                logger.error(f"Cache error: {str(e)}")

            if not cached_data:
                # Generate new predictions
                predictor = await get_predictor()
                suggestion_objects = await predictor.predict(
                    partial_query, 
                    limit=limit*2,
                    user_id=user_id,
                    is_focus_state=not partial_query
                )

                # Process scores with Decimal safety
                processed = []
                seen_texts = {}
                for s in suggestion_objects:
                    text = s.get("text", "").strip()
                    score = safe_convert_score(s.get("score", 0.5))
                    if text:
                        clean_text = text.lower()
                        if clean_text in seen_texts:
                            if score > seen_texts[clean_text]["score"]:
                                seen_texts[clean_text] = {"text": text, "score": score}
                        else:
                            seen_texts[clean_text] = {"text": text, "score": score}

                # Sort and limit
                sorted_items = sorted(seen_texts.values(), key=lambda x: x["score"], reverse=True)
                predictions = [item["text"] for item in sorted_items[:limit]]
                confidence_scores = [item["score"] for item in sorted_items[:limit]]

                # Categorization with safety
                if categorizer and partial_query:
                    try:
                        query_category = await categorizer.categorize_query(partial_query, user_id)
                        dominant_category = context or query_category.get("category", "general")
                        category_info = {
                            "query_category": query_category.get("category", "general"),
                            "confidence": safe_convert_score(query_category.get("confidence", 0.5)),
                            "dominant_category": dominant_category
                        }

                        # Limited categorization for performance
                        category_counts = {}
                        for suggestion in predictions[:5]:
                            try:
                                cat = await categorizer.categorize_query(suggestion)
                                category = cat.get("category", "general")
                                category_counts[category] = category_counts.get(category, 0) + 1
                            except Exception:
                                continue
                        
                        if category_counts:
                            total = sum(category_counts.values())
                            category_info["distribution"] = {k: v/total for k, v in category_counts.items()}

                    except Exception as e:
                        logger.warning(f"Categorization error: {str(e)}")

                # Personalization with type safety
                try:
                    personalized = await personalize_suggestions(
                        [{"text": p, "score": s} for p, s in zip(predictions, confidence_scores)],
                        user_id, 
                        partial_query
                    )
                    predictions = [item["text"] for item in personalized]
                    confidence_scores = [safe_convert_score(item["score"]) for item in personalized]
                except Exception as e:
                    logger.error(f"Personalization error: {str(e)}")

                # Cache with Decimal-safe serialization
                if redis and predictions:
                    try:
                        cache_data = {
                            "predictions": predictions,
                            "confidence_scores": confidence_scores,
                            "category_info": category_info
                        }
                        await redis.setex(
                            cache_key,
                            900,
                            json.dumps(cache_data, cls=DecimalEncoder)
                        )
                    except Exception as e:
                        logger.error(f"Cache store error: {str(e)}")

        # Generate refinements with error handling
        refinements_list = []
        try:
            refinements = await generate_dynamic_refinements(partial_query, context, predictions, limit)
            if isinstance(refinements, dict):
                refinements_list.extend(refinements.get("related_queries", []))
                refinements_list.extend(refinements.get("expertise_areas", []))
                for f in refinements.get("filters", []):
                    if isinstance(f.get("values"), list):
                        refinements_list.extend(f["values"])
        except Exception as e:
            logger.error(f"Refinement error: {str(e)}")

        # Session handling with collision protection
        try:
            conn = get_db_connection()
            if conn:
                session_id = await get_or_create_session(conn, user_id)
                await record_prediction(
                    conn,
                    session_id,
                    user_id,
                    partial_query,
                    predictions,
                    confidence_scores
                )
        except Exception as e:
            logger.error(f"Analytics error: {str(e)}")
        finally:
            if conn:
                conn.close()

        # Build final response
        response = PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions),
            search_context=context,
            metadata={"category_info": category_info}
        )

        return response

    except Exception as e:
        logger.error(f"Critical prediction error: {str(e)}", exc_info=True)
        return PredictionResponse(
            predictions=[],
            confidence_scores=[],
            user_id=user_id,
            refinements=[],
            total_suggestions=0,
            metadata={"error": str(e)}
        )

async def get_or_create_session(conn, user_id: str, max_retries: int = 3) -> int:
    """Session creation with collision retries"""
    from psycopg2 import errors

    logger = logging.getLogger(__name__)
    for attempt in range(max_retries):
        try:
            session_id = random.randint(10_000_000, 99_999_999)  # 8-digit ID
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO search_sessions 
                        (session_id, user_id, start_timestamp, is_active)
                    VALUES (%s, %s, CURRENT_TIMESTAMP, true)
                    ON CONFLICT (session_id) DO UPDATE
                        SET last_active = CURRENT_TIMESTAMP
                    RETURNING session_id
                """, (session_id, user_id))
                result = await cur.fetchone()
                if result:
                    return result[0]
        except errors.UniqueViolation:
            logger.warning(f"Session ID collision {session_id}, retry {attempt+1}/{max_retries}")
            continue
        except Exception as e:
            logger.error(f"Session error: {str(e)}")
            raise HTTPException(status_code=500, detail="Session management failed")

    raise HTTPException(status_code=500, detail="Failed to create unique session")
    
async def get_search_categorizer():
    """
    Get or create a singleton instance of the SearchCategorizer.
    
    Returns:
        SearchCategorizer instance or None if not available
    """
    global _categorizer
    
    if _categorizer is None:
        try:
            # Get Redis client for caching
            redis_client = await get_redis()
            
            # Import the SearchCategorizer
            from ai_services_api.services.search.gemini.search_categorizer import SearchCategorizer
            
            # Create categorizer instance
            _categorizer = SearchCategorizer(redis_client=redis_client)
            
            # Start background worker
            await _categorizer.start_background_worker()
            
            logger.info("Search categorizer initialized successfully")
        except ImportError:
            logger.warning("SearchCategorizer module not available")
            return None
        except Exception as e:
            logger.error(f"Error initializing search categorizer: {e}")
            return None
    
    return _categorizer

# Initialize global categorizer instance
_categorizer = None

def filter_predictions_by_context(
    predictions: List[str], 
    confidence_scores: List[float], 
    partial_query: str, 
    context: str
) -> Tuple[List[str], List[float]]:
    """
    Strictly filter predictions based on context type.
    For 'name' context, only return person name suggestions.
    For 'publication' context, prioritize publication-like suggestions.
    
    Args:
        predictions: List of predicted query completions
        confidence_scores: Corresponding confidence scores for each prediction
        partial_query: The partial query being typed
        context: Context type ('name', 'theme', 'designation', 'publication')
        
    Returns:
        Tuple of filtered predictions and their confidence scores
    """
    if context != "name" and context != "publication":
        # For other contexts, return original predictions
        return predictions, confidence_scores
    
    filtered_predictions = []
    filtered_scores = []
    
    if context == "name":
        for i, pred in enumerate(predictions):
            original_score = confidence_scores[i] if i < len(confidence_scores) else 0.5
            
            if is_person_name(pred, partial_query):
                filtered_predictions.append(pred)
                filtered_scores.append(original_score * 1.2)  # Boost name matches
        
        # Fallback suggestions when no perfect matches exist
        if not filtered_predictions:
            return generate_name_fallback_suggestions(partial_query)
    
    elif context == "publication":
        # Publication filtering logic
        for i, pred in enumerate(predictions):
            original_score = confidence_scores[i] if i < len(confidence_scores) else 0.5
            
            if is_publication_title(pred, partial_query):
                filtered_predictions.append(pred)
                filtered_scores.append(original_score * 1.2)  # Boost publication matches
        
        # If no matches, return original predictions for publication context
        if not filtered_predictions:
            return predictions, confidence_scores
    
    return filtered_predictions, filtered_scores

def is_person_name(suggestion: str, partial_query: str) -> bool:
    """
    Strict check if suggestion appears to be a person name.
    Enhanced for prefix matching.
    
    Names should:
    - Start with the partial query (case-insensitive)
    - Be 1-3 words 
    - Have each word capitalized
    - Not contain research or technical terms
    """
    # Research and technical terms to exclude
    topic_indicators = [
        "research", "study", "analysis", "method", 
        "framework", "data", "model", "system", 
        "pathway", "protein", "drug", "discovery", 
        "resistance", "structure", "function", 
        "signaling", "antimicrobial", "cancer"
    ]
    
    # Convert partial query and suggestion to lowercase for comparison
    partial_query_lower = partial_query.lower()
    suggestion_lower = suggestion.lower()
    
    # First, check if suggestion starts with the partial query
    if not suggestion_lower.startswith(partial_query_lower):
        return False
    
    words = suggestion.split()
    
    # Check length (allow single word for partial names)
    if not (1 <= len(words) <= 3):
        return False
    
    # Check capitalization (except for partial single words)
    if len(words) > 1 and not all(word.istitle() for word in words):
        return False
    
    # Check for topic indicators
    if any(indicator in suggestion_lower for indicator in topic_indicators):
        return False
    
    # Additional checks to prevent non-name suggestions
    if any(
        last_word.endswith(suffix) 
        for last_word in [words[-1].lower()] 
        for suffix in ['ing', 'tion', 'ism', 'ogy', 'path', 'ence']
    ):
        return False
    
    return True


async def generate_dynamic_refinements(
    partial_query: str,
    context: Optional[str],
    current_predictions: List[str],
    limit: int = 5
) -> Dict[str, Any]:
    """
    Generate refinement suggestions that match the current query prefix.
    
    Args:
        partial_query: The partial query being typed (may be empty)
        context: Optional context type (name, theme, designation, publication)
        current_predictions: Current predictions for reference
        limit: Maximum number of suggestions per category
        
    Returns:
        Dictionary with categorized refinement suggestions
    """
    try:
        # Set up empty refinements structure
        refinements = {
            "filters": [],
            "related_queries": [],
            "expertise_areas": []
        }
        
        # If no query, provide general refinements based on context
        if not partial_query:
            # Add context-specific filter categories
            if context == "name":
                refinements["filters"].append({
                    "type": "name_filter",
                    "label": "Name Search",
                    "values": ["First Name", "Last Name", "Full Name"]
                })
            elif context == "theme":
                refinements["filters"].append({
                    "type": "theme_filter",
                    "label": "Research Areas",
                    "values": ["Neuroscience", "Data Science", "Psychology", "Biochemistry", "Engineering"]
                })
            elif context == "designation":
                refinements["filters"].append({
                    "type": "role_filter",
                    "label": "Roles",
                    "values": ["Professor", "Researcher", "Director", "Scientist", "Student"]
                })
            elif context == "publication":
                refinements["filters"].append({
                    "type": "publication_filter",
                    "label": "Publication Types",
                    "values": ["Journal Articles", "Conference Papers", "Books", "Reviews", "Preprints"]
                })
            else:
                refinements["filters"].append({
                    "type": "search_type",
                    "label": "Search Types",
                    "values": ["People", "Research", "Publications", "Departments", "Expertise"]
                })
            
            # Add some trending topics as expertise areas
            trend_suggestions = await get_trending_suggestions("", limit)
            trend_texts = [s.get("text", "") for s in trend_suggestions]
            refinements["expertise_areas"] = trend_texts
            
            # Return early for empty query case
            return refinements
        
        # For active queries, filter refinements to match the prefix
        
        # First get existing refinements from Gemini or other sources
        try:
            # Use GoogleAutocompletePredictor to get related queries
            predictor = await get_predictor()
            suggestion_objects = await predictor.predict(partial_query, limit=limit*2)
            related_queries = [s["text"] for s in suggestion_objects]
            
            # Filter to only queries that start with the partial query
            prefix_queries = [q for q in related_queries if q.lower().startswith(partial_query.lower())]
            
            # If not enough prefix matches, also include queries that contain the partial query
            if len(prefix_queries) < limit:
                contains_queries = [q for q in related_queries if partial_query.lower() in q.lower() and q not in prefix_queries]
                related_queries = prefix_queries + contains_queries[:limit - len(prefix_queries)]
            else:
                related_queries = prefix_queries[:limit]
            
            refinements["related_queries"] = related_queries
        except Exception as query_error:
            logger.error(f"Error getting related queries: {query_error}")
        
        # Generate expertise areas relevant to the partial query
        try:
            # Extract potential expertise areas from predictions and related queries
            expertise_areas = set()
            
            # Use our current predictions
            for prediction in current_predictions:
                if prediction.lower().startswith(partial_query.lower()):
                    expertise_areas.add(prediction.capitalize())
            
            # Use related queries too
            for query in refinements["related_queries"]:
                words = query.lower().split()
                for word in words:
                    # Only add significant words that match our prefix
                    if len(word) > 3 and word.startswith(partial_query.lower()):
                        expertise_areas.add(word.capitalize())
            
            # Ensure we have at least one expertise area
            if not expertise_areas and partial_query:
                expertise_areas.add(partial_query.capitalize())
            
            # Convert to list and limit
            refinements["expertise_areas"] = list(expertise_areas)[:limit]
        except Exception as expertise_error:
            logger.error(f"Error generating expertise areas: {expertise_error}")
        
        # Add context-specific filters that match the prefix
        if context and partial_query:
            try:
                if context == "name":
                    # For name search, suggest common name prefixes
                    name_prefix = partial_query.capitalize()
                    refinements["filters"].append({
                        "type": "name_starts_with",
                        "label": f"Names starting with {partial_query}",
                        "values": [f"{name_prefix}", f"{name_prefix} (First Name)", f"{name_prefix} (Last Name)"]
                    })
                
                elif context == "theme":
                    # For theme search, suggest theme categories
                    theme_categories = []
                    if partial_query.lower().startswith("n"):
                        theme_categories.append("Neuroscience")
                    if partial_query.lower().startswith("p"):
                        theme_categories.append("Psychology")
                    if partial_query.lower().startswith("c"):
                        theme_categories.append("Computer Science")
                    if partial_query.lower().startswith("b"):
                        theme_categories.append("Biology")
                    if partial_query.lower().startswith("e"):
                        theme_categories.append("Engineering")
                    
                    # Only include filter if we have matching categories
                    if theme_categories:
                        refinements["filters"].append({
                            "type": "theme_category",
                            "label": "Research Areas",
                            "values": theme_categories
                        })
                
                elif context == "designation":
                    # Find designations that match our prefix
                    designations = []
                    common_roles = ["Professor", "Researcher", "Scientist", "Director", "Student", "Lecturer", "Fellow"]
                    
                    for role in common_roles:
                        if role.lower().startswith(partial_query.lower()):
                            designations.append(role)
                    
                    if designations:
                        refinements["filters"].append({
                            "type": "role_filter",
                            "label": "Job Roles",
                            "values": designations
                        })
            except Exception as filter_error:
                logger.error(f"Error generating context filters: {filter_error}")
        
        return refinements
    
    except Exception as e:
        logger.error(f"Error in dynamic refinements: {e}", exc_info=True)
        return {
            "filters": [],
            "related_queries": [],
            "expertise_areas": []
        }

def generate_name_fallback_suggestions(partial_query: str) -> Tuple[List[str], List[float]]:
    """Generate simple name suggestions when no good matches are found"""
    common_first_names = [
        "Abdul", "Abdullah", "Abdel", "Abdala"
    ]
    common_last_names = [
        "Rahman", "Ali", "Ahmed", "Muhammad"
    ]
    
    suggestions = []
    
    # If partial query is one word, try to match with names
    if len(partial_query.split()) == 1:
        # Prioritize combinations that include the partial query
        suggestions = [
            f"{first} {last}" 
            for first in common_first_names 
            for last in common_last_names
            if partial_query.lower() in (first + " " + last).lower()
        ]
        
        # If no matches, fall back to generic combinations
        if not suggestions:
            suggestions = [
                f"{first} {last}" 
                for first in common_first_names[:2] 
                for last in common_last_names[:2]
            ]
    else:
        # If already two words, just return as-is if it looks like a name
        suggestions = [partial_query] if is_person_name(partial_query, partial_query) else []
    
    # If still no suggestions, create some
    if not suggestions:
        suggestions = [f"{partial_query} {last}" for last in common_last_names[:2]]
    
    # Generate decreasing confidence scores
    scores = [0.9 - (i * 0.1) for i in range(len(suggestions))]
    
    return suggestions, scores

def generate_advanced_predictive_refinements(
    partial_query: str, 
    predictions: List[str], 
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate advanced predictive refinements with context-specific filtering.
    
    Args:
        partial_query: The partial query string
        predictions: List of predicted suggestions
        context: Optional context for filtering (name, theme, designation)
    
    Returns:
        Dict of refinement suggestions
    """
    logger.info(f"Generating advanced predictive refinements - Context: {context}")
    
    try:
        # Context-specific keywords for refinement
        context_keywords = {
            "name": {
                "filters": ["Personal Profiles", "Researcher Names", "Expert Identities"],
                "expertise_prefixes": ["Expert in", "Researcher with", "Professional"]
            },
            "theme": {
                "filters": ["Research Areas", "Study Domains", "Academic Fields"],
                "expertise_prefixes": ["Research in", "Specializing in", "Focus on"]
            },
            "designation": {
                "filters": ["Professional Roles", "Job Titles", "Career Levels"],
                "expertise_prefixes": ["Experts as", "Professionals with", "Specialists in"]
            }
        }
        
        # Prepare base refinements structure
        refinements = {
            "filters": [],
            "related_queries": predictions[:5],
            "expertise_areas": []
        }
        
        # Add context-specific filters if context is provided
        if context and context in context_keywords:
            refinements["filters"] = [{
                "type": f"{context}_filter",
                "label": context_keywords[context]["filters"][0],
                "values": context_keywords[context]["filters"]
            }]
            
            # Generate expertise areas
            expertise_areas = set()
            prefix = context_keywords[context]["expertise_prefixes"][0]
            
            for prediction in predictions:
                words = prediction.split()
                for word in words:
                    if len(word) > 3 and word.lower() != partial_query.lower():
                        expertise_areas.add(f"{prefix} {word.capitalize()}")
            
            # Ensure some expertise areas exist
            if not expertise_areas:
                expertise_areas.add(f"{prefix} {partial_query.capitalize()}")
            
            refinements["expertise_areas"] = list(expertise_areas)[:5]
        
        return refinements
    
    except Exception as e:
        logger.error(f"Error in advanced predictive refinements: {e}")
        return {
            "filters": [],
            "related_queries": predictions[:5],
            "expertise_areas": []
        }
