# expert_search.py
from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
from datetime import datetime
import json
import uuid

from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from ai_services_api.services.message.core.database import get_db_connection
from ai_services_api.services.search.core.models import ExpertSearchResult, SearchResponse
from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
from datetime import datetime
import json
import uuid


# Configure logger
logger = logging.getLogger(__name__)

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

async def record_search(conn, session_id: str, user_id: str, query: str, results: List[Dict], response_time: float):
    """Record search analytics in the database."""
    logger.info(f"Recording search analytics - Session: {session_id}, User: {user_id}")
    logger.debug(f"Recording search for query: {query} with {len(results)} results")
    
    cur = conn.cursor()
    try:
        # Record search analytics
        cur.execute("""
            INSERT INTO search_analytics
                (search_id, query, user_id, response_time,
                 result_count, search_type, timestamp)
            VALUES
                ((SELECT id FROM search_sessions WHERE session_id = %s),
                %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (
            session_id,
            query,
            user_id,
            response_time,
            len(results),
            'expert_search'
        ))
        
        search_id = cur.fetchone()[0]
        logger.debug(f"Created search analytics record with ID: {search_id}")

        # Record top 5 expert matches
        logger.debug(f"Recording top 5 matches from {len(results)} total results")
        for rank, result in enumerate(results[:5], 1):
            cur.execute("""
                INSERT INTO expert_search_matches
                    (search_id, expert_id, rank_position, similarity_score)
                VALUES (%s, %s, %s, %s)
            """, (
                search_id,
                result["id"],
                rank,
                result.get("score", 0.0)
            ))
            logger.debug(f"Recorded match - Expert ID: {result['id']}, Rank: {rank}")

        conn.commit()
        logger.info(f"Successfully recorded all search data for search ID: {search_id}")
        return search_id
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording search: {str(e)}", exc_info=True)
        logger.debug(f"Search recording failed: {str(e)}")
        raise
    finally:
        cur.close()

async def process_expert_search(query: str, user_id: str, active_only: bool = True, k: int = 5) -> SearchResponse:
    """
    Process expert search with improved Gemini-enhanced refinements
    
    Args:
        query (str): Search query
        user_id (str): User identifier
        active_only (bool): Filter for active experts only
        k (int): Number of results to return
    
    Returns:
        SearchResponse: Search results including optional refinements
    """
    logger.info(f"Processing expert search - Query: {query}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID in case DB connection fails
    
    try:
        # Establish database connection
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Execute search with timing
        start_time = datetime.utcnow()
        results = []
        try:
            search_manager = ExpertSearchIndexManager()
            results = search_manager.search_experts(query, k=k, active_only=active_only)
            logger.info(f"Search found {len(results)} results")
        except Exception as search_error:
            logger.error(f"Search execution error: {search_error}", exc_info=True)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Format results - handle empty results gracefully
        formatted_results = []
        try:
            formatted_results = [
                ExpertSearchResult(
                    id=str(result.get('id', 'unknown')),
                    first_name=result.get('first_name', ''),
                    last_name=result.get('last_name', ''),
                    designation=result.get('designation', ''),
                    theme=result.get('theme', ''),
                    unit=result.get('unit', ''),
                    contact=result.get('contact', ''),
                    is_active=result.get('is_active', True),
                    score=result.get('score'),
                    bio=result.get('bio'),
                    knowledge_expertise=result.get('knowledge_expertise', [])
                ) for result in results
            ]
        except Exception as format_error:
            logger.error(f"Result formatting error: {format_error}", exc_info=True)
        
        # Record search analytics if we have a connection
        if conn and results:
            try:
                await record_search(conn, session_id, user_id, query, results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions using Gemini
        refinements = {}
        try:
            # First try to get refinements from search manager
            if results:
                refinements = search_manager.get_search_refinements(query, results)
            else:
                # If no results or empty refinements, enhance with Gemini
                try:
                    # Use GoogleAutocompletePredictor to get related queries
                    gemini_predictor = GoogleAutocompletePredictor()
                    suggestion_objects = await gemini_predictor.predict(query, limit=5)
                    related_queries = [s["text"] for s in suggestion_objects]
                    
                    # Extract potential expertise areas
                    expertise_areas = set()
                    for related_query in related_queries:
                        words = related_query.lower().split()
                        for word in words:
                            if len(word) > 3 and word != query.lower():
                                expertise_areas.add(word.capitalize())
                    
                    # Ensure we have some expertise areas
                    if len(expertise_areas) < 3:
                        expertise_areas.add(query.capitalize())
                    
                    # Build enhanced refinements
                    refinements = {
                        "filters": [],
                        "related_queries": related_queries,
                        "expertise_areas": list(expertise_areas)[:5]
                    }
                except Exception as gemini_error:
                    logger.error(f"Gemini refinement error: {gemini_error}", exc_info=True)
                    # Fallback to basic refinements
                    refinements = {
                        "filters": [],
                        "related_queries": [],
                        "expertise_areas": [query.capitalize()]
                    }
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements = {
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        
        # Prepare response
        response = SearchResponse(
            total_results=len(formatted_results),
            experts=formatted_results,
            user_id=user_id,
            session_id=session_id,
            refinements=refinements
        )
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in expert search: {e}", exc_info=True)
        
        # Always return a valid response
        return SearchResponse(
            total_results=0,
            experts=[],
            user_id=user_id,
            session_id=session_id,
            refinements={
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        )
    finally:
        # Always close connection if opened
        if conn:
            conn.close()
            logger.debug("Database connection closed")

async def process_expert_name_search(
    name: str, 
    user_id: str, 
    active_only: bool = True, 
    k: int = 5  # Add this parameter with a default value
) -> SearchResponse:
    """
    Process search for experts by name.
    
    Args:
        name: Name to search for
        user_id: User identifier
        active_only: Filter for active experts only
    
    Returns:
        SearchResponse: Search results
    """
    logger.info(f"Processing name search - Name: {name}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID
    
    try:
        # Establish database connection
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Execute search with timing
        start_time = datetime.utcnow()
        results = []
        try:
            search_manager = ExpertSearchIndexManager()
            results = search_manager.search_experts_by_name(name, k=10, active_only=active_only)
            logger.info(f"Name search found {len(results)} results")
        except Exception as search_error:
            logger.error(f"Search execution error: {search_error}", exc_info=True)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Format results - handle empty results gracefully
        formatted_results = []
        try:
            formatted_results = [
                ExpertSearchResult(
                    id=str(result.get('id', 'unknown')),
                    first_name=result.get('first_name', ''),
                    last_name=result.get('last_name', ''),
                    designation=result.get('designation', ''),
                    theme=result.get('theme', ''),
                    unit=result.get('unit', ''),
                    contact=result.get('contact', ''),
                    is_active=result.get('is_active', True),
                    score=result.get('score'),
                    bio=result.get('bio'),
                    knowledge_expertise=result.get('knowledge_expertise', [])
                ) for result in results
            ]
        except Exception as format_error:
            logger.error(f"Result formatting error: {format_error}", exc_info=True)
        
        # Record search analytics if we have a connection
        if conn and results:
            try:
                await record_search(conn, session_id, user_id, f"name:{name}", results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions
        # Replace the entire refinements block with:
        refinements = {}
        try:
            # Use the new refinement generation method
            refinements = await search_manager.generate_advanced_search_refinements(
                search_type='name',  # Explicitly specify as a keyword argument 
                query=name, 
                user_id=user_id, 
                results=results
            )
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements = {
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        
        # Prepare response
        response = SearchResponse(
            total_results=len(formatted_results),
            experts=formatted_results,
            user_id=user_id,
            session_id=session_id,
            refinements=refinements
        )
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in name search: {e}", exc_info=True)
        
        # Always return a valid response
        return SearchResponse(
            total_results=0,
            experts=[],
            user_id=user_id,
            session_id=session_id,
            refinements={
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        )
    finally:
        # Always close connection if opened
        if conn:
            conn.close()
            logger.debug("Database connection closed")

async def process_expert_theme_search(theme: str, user_id: str, active_only: bool = True) -> SearchResponse:
    """
    Process search for experts by theme.
    
    Args:
        theme: Theme to search for
        user_id: User identifier
        active_only: Filter for active experts only
    
    Returns:
        SearchResponse: Search results
    """
    logger.info(f"Processing theme search - Theme: {theme}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID
    
    try:
        # Establish database connection
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Execute search with timing
        start_time = datetime.utcnow()
        results = []
        try:
            search_manager = ExpertSearchIndexManager()
            results = search_manager.search_experts_by_theme(theme, k=10, active_only=active_only)
            logger.info(f"Theme search found {len(results)} results")
        except Exception as search_error:
            logger.error(f"Search execution error: {search_error}", exc_info=True)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Format results - handle empty results gracefully
        formatted_results = []
        try:
            formatted_results = [
                ExpertSearchResult(
                    id=str(result.get('id', 'unknown')),
                    first_name=result.get('first_name', ''),
                    last_name=result.get('last_name', ''),
                    designation=result.get('designation', ''),
                    theme=result.get('theme', ''),
                    unit=result.get('unit', ''),
                    contact=result.get('contact', ''),
                    is_active=result.get('is_active', True),
                    score=result.get('score'),
                    bio=result.get('bio'),
                    knowledge_expertise=result.get('knowledge_expertise', [])
                ) for result in results
            ]
        except Exception as format_error:
            logger.error(f"Result formatting error: {format_error}", exc_info=True)
        
        # Record search analytics if we have a connection
        if conn and results:
            try:
                await record_search(conn, session_id, user_id, f"theme:{theme}", results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions
        # Replace the entire refinements block with:
        refinements = {}
        try:
            # Use the new refinement generation method
            refinements = await search_manager.generate_advanced_search_refinements(
                search_type='theme', 
                query=theme, 
                user_id=user_id, 
                results=results
            )
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements = {
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
                
        # Prepare response
        response = SearchResponse(
            total_results=len(formatted_results),
            experts=formatted_results,
            user_id=user_id,
            session_id=session_id,
            refinements=refinements
        )
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in theme search: {e}", exc_info=True)
        
        # Always return a valid response
        return SearchResponse(
            total_results=0,
            experts=[],
            user_id=user_id,
            session_id=session_id,
            refinements={
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        )
    finally:
        # Always close connection if opened
        if conn:
            conn.close()
            logger.debug("Database connection closed")

async def process_expert_designation_search(designation: str, user_id: str, active_only: bool = True) -> SearchResponse:
    """
    Process search for experts by designation.
    
    Args:
        designation: Designation to search for
        user_id: User identifier
        active_only: Filter for active experts only
    
    Returns:
        SearchResponse: Search results
    """
    logger.info(f"Processing designation search - Designation: {designation}, User: {user_id}")
    
    conn = None
    session_id = str(uuid.uuid4())[:8]  # Default session ID
    
    try:
        # Establish database connection
        try:
            conn = get_db_connection()
            session_id = await get_or_create_session(conn, user_id)
            logger.debug(f"Created session: {session_id}")
        except Exception as db_error:
            logger.error(f"Database connection error: {db_error}", exc_info=True)
            # Continue processing with default session ID
        
        # Execute search with timing
        start_time = datetime.utcnow()
        results = []
        try:
            search_manager = ExpertSearchIndexManager()
            results = search_manager.search_experts_by_designation(designation, k=10, active_only=active_only)
            logger.info(f"Designation search found {len(results)} results")
        except Exception as search_error:
            logger.error(f"Search execution error: {search_error}", exc_info=True)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Format results - handle empty results gracefully
        formatted_results = []
        try:
            formatted_results = [
                ExpertSearchResult(
                    id=str(result.get('id', 'unknown')),
                    first_name=result.get('first_name', ''),
                    last_name=result.get('last_name', ''),
                    designation=result.get('designation', ''),
                    theme=result.get('theme', ''),
                    unit=result.get('unit', ''),
                    contact=result.get('contact', ''),
                    is_active=result.get('is_active', True),
                    score=result.get('score'),
                    bio=result.get('bio'),
                    knowledge_expertise=result.get('knowledge_expertise', [])
                ) for result in results
            ]
        except Exception as format_error:
            logger.error(f"Result formatting error: {format_error}", exc_info=True)
        
        # Record search analytics if we have a connection
        if conn and results:
            try:
                await record_search(conn, session_id, user_id, f"designation:{designation}", results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions
        # Replace the entire refinements block with:
        refinements = {}
        try:
            # Use the new refinement generation method
            refinements = await search_manager.generate_advanced_search_refinements(
                search_type='designation', 
                query=designation, 
                user_id=user_id, 
                results=results
            )
        except Exception as refine_error:
            logger.error(f"Refinements generation error: {refine_error}", exc_info=True)
            refinements = {
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
                
        # Prepare response
        response = SearchResponse(
            total_results=len(formatted_results),
            experts=formatted_results,
            user_id=user_id,
            session_id=session_id,
            refinements=refinements
        )
        
        return response
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Critical error in designation search: {e}", exc_info=True)
        
        # Always return a valid response
        return SearchResponse(
            total_results=0,
            experts=[],
            user_id=user_id,
            session_id=session_id,
            refinements={
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
        )
    finally:
        # Always close connection if opened
        if conn:
            conn.close()
            logger.debug("Database connection closed")