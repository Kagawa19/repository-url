from fastapi import HTTPException
from typing import Any, List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json
import uuid
import asyncio
from redis import asyncio as aioredis

from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.search.core.models import PredictionResponse
from ai_services_api.services.search.core.personalization import personalize_suggestions



# Configure logger
logger = logging.getLogger(__name__)

# Global predictor instance (singleton)
_predictor = None

# Global Redis instance
_redis = None

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

# Record session and prediction in database for analytics
async def get_or_create_session(conn, user_id: str) -> str:
    """Create a session for tracking user interactions."""
    logger.info(f"Getting or creating session for user: {user_id}")
    cur = conn.cursor()
    try:
        session_id = int(str(int(datetime.utcnow().timestamp()))[-8:])
        logger.debug(f"Generated session ID: {session_id}")
        
        cur.execute("""
            INSERT INTO search_sessions 
                (session_id, user_id, start_timestamp, is_active)
            VALUES (%s, %s, CURRENT_TIMESTAMP, true)
            RETURNING session_id
        """, (session_id, user_id))
        
        conn.commit()
        logger.debug(f"Session created successfully with ID: {session_id}")
        return str(session_id)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating session: {str(e)}", exc_info=True)
        logger.debug(f"Session creation failed: {str(e)}")
        raise
    finally:
        cur.close()

async def record_prediction(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """Record prediction in database for analytics."""
    try:
        # Create a background task for recording to avoid blocking the response
        asyncio.create_task(
            _record_prediction_async(conn, session_id, user_id, partial_query, predictions, confidence_scores)
        )
    except Exception as e:
        logger.error(f"Failed to create background task for recording prediction: {e}")

async def _record_prediction_async(conn, session_id: str, user_id: str, partial_query: str, predictions: List[str], confidence_scores: List[float]):
    """Asynchronous helper for recording predictions in the database."""
    logger.info(f"Recording predictions for user {user_id}, session {session_id}")
    
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
    
async def process_query_prediction(
    partial_query: str, 
    user_id: str, 
    context: Optional[str] = None
) -> PredictionResponse:
    logger.info(f"Processing query prediction - Query: '{partial_query}', User: {user_id}, Context: {context}")
    
    valid_contexts = ["name", "theme", "designation", None]
    if context and context not in valid_contexts[:-1]:
        raise ValueError(f"Invalid context. Must be one of {', '.join(valid_contexts[:-1])}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]
    
    redis = await get_redis()
    
    try:
        cache_key = f"google_autocomplete:{context or ''}:{partial_query}"
        
        if redis:
            try:
                cached_result = await redis.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache hit for query: {partial_query}")
                    cached_data = json.loads(cached_result)
                    suggestion_objects = cached_data.get("suggestions", [])
                    confidence_scores = cached_data.get("confidence_scores", [])
                    predictions = [s["text"] for s in suggestion_objects]
                    
                    if context:
                        predictions, confidence_scores = filter_predictions_by_context(
                            predictions, confidence_scores, partial_query, context
                        )
                    
                    try:
                        personalized_suggestions = await personalize_suggestions(
                            [{"text": pred, "score": score} for pred, score in zip(predictions, confidence_scores)],
                            user_id, 
                            partial_query
                        )
                        predictions = [s["text"] for s in personalized_suggestions]
                        confidence_scores = [s.get("score", 0.5) for s in personalized_suggestions]
                    except Exception as personalize_error:
                        logger.error(f"Personalization error for cached results: {personalize_error}")
                    
                    refinements_dict = generate_predictive_refinements(partial_query, predictions)
                    
                    # Convert refinements to a flat list of strings as expected by PredictionResponse
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

                    return PredictionResponse(
                        predictions=predictions,
                        confidence_scores=confidence_scores,
                        user_id=user_id,
                        refinements=refinements_list,
                        total_suggestions=len(predictions)
                    )

            except Exception as cache_error:
                logger.error(f"Cache retrieval error: {cache_error}", exc_info=True)
        
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
        
        predictor = await get_predictor()
        suggestion_objects = await predictor.predict(partial_query, limit=10)
        predictions = [s["text"] for s in suggestion_objects]
        confidence_scores = [s.get("score", 0.5) for s in suggestion_objects]
        
        if context:
            predictions, confidence_scores = filter_predictions_by_context(
                predictions, confidence_scores, partial_query, context
            )
        
        try:
            personalized_suggestions = await personalize_suggestions(
                [{"text": pred, "score": score} for pred, score in zip(predictions, confidence_scores)],
                user_id, 
                partial_query
            )
            predictions = [s["text"] for s in personalized_suggestions]
            confidence_scores = [s.get("score", 0.5) for s in personalized_suggestions]
        except Exception as personalize_error:
            logger.error(f"Personalization error for fresh results: {personalize_error}")
        
        logger.debug(f"Generated {len(predictions)} predictions: {predictions}")
        
        if redis and predictions:
            try:
                cache_data = {
                    "suggestions": [
                        {"text": pred, "score": score} 
                        for pred, score in zip(predictions, confidence_scores)
                    ],
                    "confidence_scores": confidence_scores
                }
                await redis.setex(cache_key, 900, json.dumps(cache_data))
                logger.debug(f"Cached predictions for: {partial_query}")
            except Exception as cache_error:
                logger.error(f"Cache storage error: {cache_error}", exc_info=True)
        
        refinements_dict = generate_predictive_refinements(partial_query, predictions)
        
        # Convert refinements to a flat list of strings as expected by PredictionResponse
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

        if conn and predictions:
            await record_prediction(
                conn,
                session_id,
                user_id,
                partial_query,
                predictions,
                confidence_scores
            )
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions)
        )
    
    except Exception as e:
        logger.exception(f"Unexpected error in process_query_prediction: {e}")
        # Initialize empty lists for fallback response
        predictions = []
        confidence_scores = []
        refinements_list = []  # Empty list of strings for refinements
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions)
        )
async def process_advanced_query_prediction(
    partial_query: str, 
    user_id: str, 
    context: Optional[str] = None,
    limit: int = 10
) -> PredictionResponse:
    """
    Process advanced query prediction with context-specific suggestions using FAISS indexes.
    This function leverages the ExpertSearchIndexManager to get suggestions from indexes
    rather than direct database queries.
    
    Args:
        partial_query: Partial query to predict
        user_id: User identifier
        context: Optional context for prediction (name, theme, designation)
        limit: Maximum number of predictions
    
    Returns:
        PredictionResponse with context-aware predictions
    """
    logger.info(f"Processing advanced query prediction - Query: {partial_query}, Context: {context}")
    
    try:
        # Validate context
        valid_contexts = ["name", "theme", "designation", None]
        if context and context not in valid_contexts[:-1]:
            raise ValueError(f"Invalid context. Must be one of {', '.join(valid_contexts[:-1])}")
        
        conn = None
        session_id = str(uuid.uuid4())[:8]
        redis = await get_redis()
        
        # Try to get from cache first
        predictions = []
        confidence_scores = []
        cache_key = f"advanced_idx_predict:{context or ''}:{partial_query}"
        
        if redis:
            try:
                cached_result = await redis.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache hit for advanced query: {partial_query}")
                    cached_data = json.loads(cached_result)
                    predictions = cached_data.get("predictions", [])
                    confidence_scores = cached_data.get("confidence_scores", [])
            except Exception as cache_error:
                logger.error(f"Cache retrieval error: {cache_error}", exc_info=True)
        
        # If not in cache, use the ExpertSearchIndexManager to get predictions
        if not predictions:
            try:
                # Create search manager
                from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
                search_manager = ExpertSearchIndexManager()
                
                # Use the appropriate index-based search method based on context
                if context == "name":
                    # Use the name index search
                    results = search_manager.search_experts_by_name(
                        partial_query, 
                        k=limit, 
                        active_only=True, 
                        min_score=0.1
                    )
                    
                    # Extract names and scores
                    for result in results:
                        full_name = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
                        if full_name:
                            predictions.append(full_name)
                            confidence_scores.append(float(result.get('score', 0.5)))
                
                elif context == "designation":
                    # Use the designation index search
                    results = search_manager.search_experts_by_designation(
                        partial_query, 
                        k=limit*3, 
                        active_only=True, 
                        min_score=0.1
                    )
                    
                    # Extract unique designations with scores
                    unique_designations = {}
                    for result in results:
                        designation = result.get('designation', '')
                        if designation and designation not in unique_designations:
                            unique_designations[designation] = float(result.get('score', 0.5))
                    
                    # Convert to lists
                    predictions = list(unique_designations.keys())
                    confidence_scores = list(unique_designations.values())
                    
                    # Limit to top results
                    predictions = predictions[:limit]
                    confidence_scores = confidence_scores[:limit]
                
                elif context == "theme":
                    # Use the theme index search
                    results = search_manager.search_experts_by_theme(
                        partial_query, 
                        k=limit*3, 
                        active_only=True, 
                        min_score=0.1
                    )
                    
                    # Extract unique themes with scores
                    unique_themes = {}
                    for result in results:
                        theme = result.get('theme', '')
                        if theme and theme not in unique_themes:
                            unique_themes[theme] = float(result.get('score', 0.5))
                    
                    # Convert to lists
                    predictions = list(unique_themes.keys())
                    confidence_scores = list(unique_themes.values())
                    
                    # Limit to top results
                    predictions = predictions[:limit]
                    confidence_scores = confidence_scores[:limit]
                
                else:
                    # No specific context, use general expert search
                    results = search_manager.search_experts(
                        partial_query, 
                        k=limit*3, 
                        active_only=True, 
                        min_score=0.1
                    )
                    
                    # Collect all types of results
                    names = {}
                    designations = {}
                    themes = {}
                    
                    for result in results:
                        # Add name
                        full_name = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
                        if full_name and full_name not in names:
                            names[full_name] = float(result.get('score', 0.5))
                            
                        # Add designation
                        designation = result.get('designation', '')
                        if designation and designation not in designations:
                            designations[designation] = float(result.get('score', 0.5)) * 0.9  # Slightly lower priority
                            
                        # Add theme
                        theme = result.get('theme', '')
                        if theme and theme not in themes:
                            themes[theme] = float(result.get('score', 0.5)) * 0.8  # Lower priority
                    
                    # Combine all results
                    combined_items = list(names.items()) + list(designations.items()) + list(themes.items())
                    
                    # Sort by score (descending)
                    combined_items.sort(key=lambda x: x[1], reverse=True)
                    
                    # Extract predictions and scores
                    predictions = [item[0] for item in combined_items[:limit]]
                    confidence_scores = [item[1] for item in combined_items[:limit]]
                
                # If predictions are empty, fall back to GoogleAutocompletePredictor
                # but only for generating suggestions, not for final results
                if not predictions:
                    logger.info(f"No index-based predictions found, using fallback for context: {context}")
                    
                    # Get predictions from Google Autocomplete
                    predictor = await get_predictor()
                    suggestion_objects = await predictor.predict(partial_query, limit=limit)
                    ext_predictions = [s["text"] for s in suggestion_objects]
                    ext_confidence_scores = [s.get("score", 0.5) for s in suggestion_objects]
                    
                    # Apply context-specific filtering
                    if context and callable(getattr(search_manager, f"search_experts_by_{context}", None)):
                        # For each predicted term, try to find matching experts
                        for i, term in enumerate(ext_predictions[:3]):  # Check top 3 predictions
                            search_method = getattr(search_manager, f"search_experts_by_{context}")
                            term_results = search_method(term, k=3)
                            
                            if term_results:
                                # Extract the relevant field based on context
                                for result in term_results:
                                    if context == "name":
                                        item = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
                                    else:
                                        item = result.get(context, '')
                                        
                                    if item and item not in predictions:
                                        predictions.append(item)
                                        # Score based on position in the fallback list
                                        confidence_scores.append(0.9 - (0.1 * i))
                    
                    # If still no predictions, use the external ones directly
                    if not predictions:
                        predictions = ext_predictions
                        confidence_scores = [score * 0.7 for score in ext_confidence_scores]  # Lower confidence for external results
                
                # Cache the results
                if redis and predictions:
                    try:
                        cache_data = {
                            "predictions": predictions,
                            "confidence_scores": confidence_scores
                        }
                        await redis.setex(cache_key, 900, json.dumps(cache_data))
                    except Exception as cache_error:
                        logger.error(f"Cache storage error: {cache_error}", exc_info=True)
                        
            except Exception as idx_error:
                logger.error(f"Index search error: {idx_error}", exc_info=True)
                
                # Fallback to GoogleAutocompletePredictor if index search fails
                predictor = await get_predictor()
                suggestion_objects = await predictor.predict(partial_query, limit=limit)
                predictions = [s["text"] for s in suggestion_objects]
                confidence_scores = [s.get("score", 0.5) for s in suggestion_objects]
        
        # Personalize suggestions if possible
        try:
            personalized_suggestions = await personalize_suggestions(
                [{"text": pred, "score": score} for pred, score in zip(predictions, confidence_scores)],
                user_id, 
                partial_query
            )
            predictions = [s["text"] for s in personalized_suggestions]
            confidence_scores = [s.get("score", 0.5) for s in personalized_suggestions]
        except Exception as personalize_error:
            logger.error(f"Personalization error: {personalize_error}")
        
        # Generate advanced context-specific refinements
        refinements_dict = None
        try:
            # Use the ExpertSearchIndexManager to generate refinements
            from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
            search_manager = ExpertSearchIndexManager()
            
            # Get results to generate refinements
            if context:
                search_method = getattr(search_manager, f"search_experts_by_{context}", None)
                if callable(search_method):
                    results = search_method(partial_query, k=5)
                else:
                    results = search_manager.search_experts(partial_query, k=5)
            else:
                results = search_manager.search_experts(partial_query, k=5)
            
            # Generate refinements
            refinements_dict = await search_manager.generate_advanced_search_refinements(
                context or "general",
                partial_query,
                user_id,
                results
            )
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements_dict = {
                "filters": [],
                "related_queries": predictions[:5] if predictions else [],
                "expertise_areas": []
            }
        
        # Convert refinements to a flat list of strings as expected by PredictionResponse
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
        
        # Record analytics if we have a connection
        try:
            conn = get_db_connection()
            if conn and predictions:
                session_id = await get_or_create_session(conn, user_id)
                await record_prediction(
                    conn,
                    session_id,
                    user_id,
                    partial_query,
                    predictions,
                    confidence_scores
                )
        except Exception as record_error:
            logger.error(f"Error recording prediction: {record_error}", exc_info=True)
        finally:
            if conn:
                conn.close()
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions),
            search_context=context
        )
    
    except Exception as e:
        logger.error(f"Error in advanced query prediction: {e}", exc_info=True)
        # Initialize empty lists for fallback response
        predictions = []
        confidence_scores = []
        refinements_list = []
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            user_id=user_id,
            refinements=refinements_list,
            total_suggestions=len(predictions)
        )

def filter_predictions_by_context(
    predictions: List[str], 
    confidence_scores: List[float], 
    partial_query: str, 
    context: str
) -> Tuple[List[str], List[float]]:
    """
    Strictly filter predictions based on context type.
    For 'name' context, only return person name suggestions.
    """
    if context != "name":
        # For non-name contexts, return original predictions
        return predictions, confidence_scores
    
    filtered_predictions = []
    filtered_scores = []
    
    for i, pred in enumerate(predictions):
        original_score = confidence_scores[i] if i < len(confidence_scores) else 0.5
        
        if is_person_name(pred, partial_query):
            filtered_predictions.append(pred)
            filtered_scores.append(original_score * 1.2)  # Boost name matches
    
    # Fallback suggestions when no perfect matches exist
    if not filtered_predictions:
        return generate_name_fallback_suggestions(partial_query)
    
    return filtered_predictions, filtered_scores

def is_person_name(suggestion: str, partial_query: str) -> bool:
    """
    Strict check if suggestion appears to be a person name.
    Names should:
    - Contain the partial query (case-insensitive)
    - Be 2-3 words 
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
    
    # First, check if partial query is in the suggestion
    if partial_query_lower not in suggestion_lower:
        return False
    
    words = suggestion.split()
    
    # Check length
    if not (2 <= len(words) <= 3):
        return False
    
    # Check capitalization
    if not all(word.istitle() for word in words):
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

