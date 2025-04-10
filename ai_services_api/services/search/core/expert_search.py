# expert_search.py
import asyncio
import hashlib
import os
import time

import redis
from ai_services_api.services.search.core.personalization import safe_background_task
from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
from datetime import datetime
import json
import uuid

# Add this at the top of expert_search.py
from functools import lru_cache
import psycopg2.pool

from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
from ai_services_api.services.message.core.database import get_connection_params, get_db_connection
from ai_services_api.services.search.core.models import ExpertSearchResult, SearchResponse

from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import logging
from datetime import datetime
import json
import uuid



# Configure logger
logger = logging.getLogger(__name__)
import os
import asyncio
import logging
import psycopg2
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
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

# Async context manager for database connections
class AsyncDatabaseConnection:
    """Async context manager for database connections."""
    def __init__(self, dbname=None):
        self.dbname = dbname
        self.conn = None

    async def __aenter__(self):
        """Async enter method to get a database connection."""
        try:
            # Run database connection in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.conn = await loop.run_in_executor(
                None, 
                get_db_connection, 
                self.dbname
            )
            return self.conn
        except Exception as e:
            logger.error(f"Error in async database connection: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit method to close the database connection."""
        if self.conn:
            try:
                # Run connection closing in a thread
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.conn.close)
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
        return False  # Propagate any exceptions

async def get_or_create_session(conn, user_id: str) -> str:
    """
    Create a session for tracking user interactions.
    Fixed to handle missing constraints and tables.

    Args:
        conn: Database connection
        user_id: User identifier
        
    Returns:
        str: Session ID
    """
    logger.info(f"Getting or creating session for user: {user_id}")

    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())[:8]
        logger.debug(f"Generated session ID: {session_id}")
        
        cur = conn.cursor()
        
        # First check if the table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'search_sessions'
            );
        """)
        
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            logger.warning("search_sessions table does not exist, using generated session_id")
            cur.close()
            return session_id
        
        # Try a simple insert without the ON CONFLICT clause
        try:
            cur.execute("""
                INSERT INTO search_sessions 
                    (session_id, user_id, start_timestamp, is_active)
                VALUES (%s, %s, CURRENT_TIMESTAMP, true)
            """, (session_id, user_id))
            
            conn.commit()
            logger.debug(f"Session created successfully with ID: {session_id}")
            return session_id
        except Exception as insert_error:
            logger.error(f"Error inserting session: {insert_error}")
            conn.rollback()
            return session_id
    
    except Exception as e:
        logger.error(f"Error creating/retrieving session: {str(e)}", exc_info=True)
        # Return the generated session_id even if there was an error
        return session_id
    finally:
        if 'cur' in locals():
            cur.close()
def fetch_suggestions_from_db(conn, search_input: str) -> List[Dict]:
    """
    Queries resources and experts tables for matches based on search_input.
    Fixed to properly extract expert data from the label without querying missing columns.

    Args:
        conn: Active database connection
        search_input (str): The user's typed search string

    Returns:
        List[Dict]: Combined list of matching suggestions
    """
    logger.info(f"Starting fetch_suggestions_from_db for query: '{search_input}'")
    
    curr = conn.cursor()

    # Build a basic query with only the essential columns we know exist
    query = """
    SELECT 'expert' AS type, e.id, 
           CONCAT(
               COALESCE(e.first_name, ''), 
               ' ', 
               COALESCE(e.last_name, ''),
               CASE WHEN e.designation IS NOT NULL THEN CONCAT(' - ', e.designation) ELSE '' END
           ) AS label
    FROM experts_expert e
    WHERE 
        e.first_name ILIKE %s OR
        e.last_name ILIKE %s OR
        e.designation ILIKE %s
    
    UNION

    SELECT 'resource' AS type, r.id, r.title AS label
    FROM resources_resource r
    WHERE 
        r.title ILIKE %s
    
    ORDER BY label ASC
    LIMIT 15;
    """

    # Prepare the query parameters
    like_query = f"%{search_input}%"
    curr.execute(query, [like_query, like_query, like_query, like_query])
    results = curr.fetchall()
    
    logger.info(f"fetch_suggestions_from_db found {len(results)} results")
    
    if results:
        logger.info(f"Result structure: type={type(results[0])}, columns={len(results[0])}")
        logger.info(f"Sample result: {results[0]}")
    
    # Convert results to properly structured dictionaries with extracted name data
    suggestions = []
    for row in results:
        suggestion_type, suggestion_id, label = row
        
        suggestion = {
            "type": suggestion_type,
            "id": suggestion_id,
            "label": label
        }
        
        # If this is an expert suggestion, extract data from the label
        if suggestion_type == 'expert':
            # Parse the label to extract first_name, last_name, and designation
            try:
                # Label format: "First Last - Designation"
                label_parts = label.split(' - ')
                name = label_parts[0].strip()
                name_parts = name.split()
                
                # Extract first and last name
                first_name = ""
                last_name = ""
                
                if len(name_parts) == 1:
                    first_name = name_parts[0]
                elif len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = ' '.join(name_parts[1:])
                    
                # Extract designation if available
                designation = label_parts[1].strip() if len(label_parts) > 1 else ""
                
                # Add the extracted data to the suggestion
                suggestion.update({
                    "first_name": first_name,
                    "last_name": last_name,
                    "designation": designation,
                    "theme": "",  # Default empty values for required fields
                    "unit": "",
                    "contact": "",
                    "is_active": True,
                    "bio": "",
                    "knowledge_expertise": [],
                    "score": 0.9  # High score for direct matches
                })
                logger.info(f"Extracted name data: first_name='{first_name}', last_name='{last_name}', designation='{designation}'")
            except Exception as e:
                logger.error(f"Error parsing expert label '{label}': {e}")
                # Provide default values even if parsing fails
                suggestion.update({
                    "first_name": "",
                    "last_name": "",
                    "designation": "",
                    "theme": "",
                    "unit": "",
                    "contact": "",
                    "is_active": True,
                    "bio": "",
                    "knowledge_expertise": [],
                    "score": 0.5
                })
        
        suggestions.append(suggestion)
    
    # Log final suggestion structure
    if suggestions:
        logger.info(f"Converted suggestion structure: {list(suggestions[0].keys())}")
        logger.info(f"Sample suggestion: {suggestions[0]}")
    
    return suggestions

def get_expert_by_id_or_name(self, query: str) -> List[Dict[str, Any]]:
    """
    Try to find an expert by ID first, then fall back to name search.
    """
    logger.info(f"Attempting to find expert by ID or name: '{query}'")
    try:
        # First, try to treat the query as an ID
        if query.isdigit():
            logger.info(f"Query '{query}' appears to be a numeric ID, trying exact ID lookup")
            expert = self.get_expert_by_id(query)
            if expert:
                logger.info(f"Found expert with ID {query}")
                return [expert]
            else:
                logger.info(f"No expert found with ID {query}")
        
        # Extract name parts
        name_parts = query.split()
        logger.info(f"Extracted name parts from '{query}': {name_parts}")
        
        # Connect to database
        conn = self.db.get_connection()
        with conn.cursor() as cur:
            # Search by full name
            if len(name_parts) > 1:
                first_name = name_parts[0]
                last_name = ' '.join(name_parts[1:])
                logger.info(f"Searching for expert with first_name='{first_name}', last_name='{last_name}'")
                
                cur.execute("""
                    SELECT e.id, e.first_name, e.last_name, 
                           COALESCE(e.designation, '') as designation,
                           COALESCE(e.theme, '') as theme,
                           COALESCE(e.unit, '') as unit,
                           COALESCE(e.bio, '') as bio,
                           e.knowledge_expertise,
                           e.is_active
                    FROM experts_expert e
                    WHERE 
                        (LOWER(e.first_name) = LOWER(%s) AND LOWER(e.last_name) = LOWER(%s))
                        OR
                        (LOWER(CONCAT(e.first_name, ' ', e.last_name)) = LOWER(%s))
                """, [first_name, last_name, query])
            else:
                # Search by single name part (first or last)
                logger.info(f"Searching for expert with single name part: '{query}'")
                cur.execute("""
                    SELECT e.id, e.first_name, e.last_name, 
                           COALESCE(e.designation, '') as designation,
                           COALESCE(e.theme, '') as theme,
                           COALESCE(e.unit, '') as unit,
                           COALESCE(e.bio, '') as bio,
                           e.knowledge_expertise,
                           e.is_active
                    FROM experts_expert e
                    WHERE 
                        LOWER(e.first_name) = LOWER(%s) OR
                        LOWER(e.last_name) = LOWER(%s)
                """, [query, query])
            
            # Format results
            rows = cur.fetchall()
            logger.info(f"Database search returned {len(rows)} results")
            
            results = []
            for row in rows:
                expert_id = str(row[0])
                first_name = row[1] or ""
                last_name = row[2] or ""
                logger.info(f"Found expert: {first_name} {last_name} (ID: {expert_id})")
                
                results.append({
                    "id": expert_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "designation": row[3] or "",
                    "theme": row[4] or "",
                    "unit": row[5] or "",
                    "contact": "",  # Default empty value
                    "is_active": row[8] if row[8] is not None else True,
                    "score": 1.0,  # Exact match
                    "bio": row[6] or "",
                    "knowledge_expertise": self._parse_jsonb(row[7]) if row[7] else []
                })
            
            return results
    except Exception as e:
        logger.error(f"Error in expert lookup by ID or name: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()
    
def get_expert_by_id(self, expert_id: str) -> Dict[str, Any]:
    """
    Get an expert directly by ID.
    
    Args:
        expert_id: Expert ID to fetch
        
    Returns:
        Dict with expert details or None if not found
    """
    try:
        conn = self.db.get_connection()
        with conn.cursor() as cur:
            # Query for the expert with this exact ID
            cur.execute("""
                SELECT e.id, e.first_name, e.last_name, 
                       COALESCE(e.designation, '') as designation,
                       COALESCE(e.theme, '') as theme,
                       COALESCE(e.unit, '') as unit,
                       COALESCE(e.bio, '') as bio,
                       e.knowledge_expertise,
                       e.is_active
                FROM experts_expert e
                WHERE e.id = %s
            """, [expert_id])
            
            row = cur.fetchone()
            if row:
                # Format the result as a dictionary
                return {
                    "id": str(row[0]),
                    "first_name": row[1] or "",
                    "last_name": row[2] or "",
                    "designation": row[3] or "",
                    "theme": row[4] or "",
                    "unit": row[5] or "",
                    "contact": "",  # Default empty value
                    "is_active": row[8] if row[8] is not None else True,
                    "score": 1.0,  # Perfect match
                    "bio": row[6] or "",
                    "knowledge_expertise": self._parse_jsonb(row[7]) if row[7] else []
                }
            
            return None
    except Exception as e:
        logger.error(f"Error fetching expert by ID: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()
async def process_expert_search(query: str, user_id: str, active_only: bool = True, k: int = 5) -> SearchResponse:
    """
    Process expert search, returning suggestions for user input and performing full search.
    Enhanced to handle exact name matches through get_expert_by_id_or_name.
    """
    logger.info(f"Processing expert search - Query: '{query}', User: {user_id}")
    
    session_id = str(uuid.uuid4())[:8]  # Default session ID in case DB connection fails
    start_time = datetime.utcnow()
    
    try:
        # Use the DatabaseConnection context manager for clean connection handling
        async with AsyncDatabaseConnection() as conn:
            # Create session
            try:
                session_id = await get_or_create_session(conn, user_id)
                logger.debug(f"Created session: {session_id}")
            except Exception as session_error:
                logger.error(f"Error creating session: {session_error}", exc_info=True)
                # Continue with default session ID
            
            # Execute search with timing
            results = []
            is_complex_query = len(query.split()) > 3 or any(c in query for c in [':', '&', '|', '"', "'"])
            
            try:
                # First try the direct lookup for any query that looks like a name
                # This should happen BEFORE other search methods
                if ' ' in query:  # If there's a space, it might be a full name
                    logger.info(f"Attempting direct name lookup for: '{query}'")
                    # Run this in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    search_manager = ExpertSearchIndexManager()
                    direct_results = await loop.run_in_executor(
                        None,
                        lambda: search_manager.get_expert_by_id_or_name(query)
                    )
                    
                    if direct_results:
                        logger.info(f"Found direct name match for '{query}' - {len(direct_results)} results")
                        results = direct_results
                        # Skip other search methods if we found direct matches
                
                # Only continue with regular search if no direct results found
                if not results:
                    # Search execution (autocomplete vs full search)
                    if query:
                        # If query is relatively short/simple, check for autocomplete suggestions 
                        loop = asyncio.get_event_loop()
                        suggestions = await loop.run_in_executor(
                            None,  # Default executor
                            lambda: fetch_suggestions_from_db(conn, query)
                        )
                        
                        # Filter suggestions to only keep experts
                        expert_suggestions = [s for s in suggestions if s.get("type") == "expert"]
                        
                        # If query is complex, perform full search for experts and resources
                        if is_complex_query:
                            search_manager = ExpertSearchIndexManager()
                            search_results = search_manager.search_experts(query, k=k, active_only=active_only)
                            results = search_results
                        else:
                            # For simple queries, use the suggestions
                            results = expert_suggestions

                logger.info(f"Search found {len(results)} results")
                
                # Append search to user history in background with safe_background_task
                await safe_background_task(
                    _record_search_history_task,
                    user_id, 
                    query, 
                    len(results)
                )
                
            except Exception as search_error:
                logger.error(f"Search execution error: {search_error}", exc_info=True)
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Ensure all results are in a consistent dictionary format
            normalized_results = []
            for result in results:
                if isinstance(result, dict):
                    # Ensure all required fields exist in the dictionary
                    normalized_result = {
                        "id": str(result.get("id", "unknown")),
                        "first_name": result.get("first_name", ""),
                        "last_name": result.get("last_name", ""),
                        "designation": result.get("designation", ""),
                        "theme": result.get("theme", ""),
                        "unit": result.get("unit", ""),
                        "contact": result.get("contact", ""),
                        "is_active": result.get("is_active", True),
                        "score": result.get("score", 1.0),
                        "bio": result.get("bio", ""),
                        "knowledge_expertise": result.get("knowledge_expertise", [])
                    }
                    normalized_results.append(normalized_result)
                elif isinstance(result, (list, tuple)):
                    # Convert tuple/list to dictionary with proper field names
                    # This assumes a specific order of fields in the tuple
                    try:
                        normalized_result = {
                            "id": str(result[1]),  # Assuming ID is at index 1
                            "first_name": "",  # Default values for missing fields
                            "last_name": "",
                            "designation": "",
                            "theme": "",
                            "unit": "",
                            "contact": "",
                            "is_active": True,
                            "score": 1.0,
                            "bio": "",
                            "knowledge_expertise": []
                        }
                        normalized_results.append(normalized_result)
                    except (IndexError, TypeError) as e:
                        logger.error(f"Error normalizing result tuple/list: {e}", exc_info=True)
                else:
                    logger.warning(f"Skipping result of unexpected type: {type(result)}")
            
            # Format normalized results into ExpertSearchResult objects
            formatted_results = []
            try:
                formatted_results = [
                    ExpertSearchResult(
                        id=result["id"],
                        first_name=result["first_name"],
                        last_name=result["last_name"],
                        designation=result["designation"],
                        theme=result["theme"],
                        unit=result["unit"],
                        contact=result["contact"],
                        is_active=result["is_active"],
                        score=result["score"],
                        bio=result["bio"],
                        knowledge_expertise=result["knowledge_expertise"]
                    ) for result in normalized_results
                ]
            except Exception as format_error:
                logger.error(f"Result formatting error: {format_error}", exc_info=True)
            
            # Record search analytics directly using the current connection
            if results:
                try:
                    await record_search(conn, session_id, user_id, query, normalized_results, response_time)
                except Exception as record_error:
                    logger.error(f"Error recording search: {record_error}", exc_info=True)
            
            # Prepare response with formatted results
            response = SearchResponse(
                total_results=len(formatted_results),
                experts=formatted_results,
                user_id=user_id,
                session_id=session_id,
                refinements={}  # We are not generating refinements now
            )
            
            # Log total processing time
            total_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Total processing time: {total_time:.3f}s")
            
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
            refinements={}
        )
async def record_search(conn, session_id: str, user_id: str, query: str, results: list, response_time: float):
    """
    Record search analytics in the database.
    
    Args:
        conn: Database connection
        session_id: Session identifier
        user_id: User identifier
        query: Search query
        results: Search results
        response_time: Time taken to perform the search
    
    Returns:
        int: Search record ID or None if recording failed
    """
    logger.info(f"Recording search analytics - Session: {session_id}, User: {user_id}")
    
    if not conn:
        logger.error("Cannot record search: No database connection provided")
        return None
    
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
        for rank, result in enumerate(results[:5], 1):
            cur.execute("""
                INSERT INTO expert_search_matches
                    (search_id, expert_id, rank_position, similarity_score)
                VALUES (%s, %s, %s, %s)
            """, (
                search_id,
                result.get("id", "unknown"),
                rank,
                result.get("score", 0.0)
            ))
            logger.debug(f"Recorded match - Expert ID: {result.get('id')}, Rank: {rank}")

        conn.commit()
        logger.info(f"Successfully recorded all search data for search ID: {search_id}")
        return search_id
        
    except Exception as e:
        logger.error(f"Error recording search: {str(e)}", exc_info=True)
        conn.rollback()
        return None
    finally:
        cur.close()


async def record_search(conn, session_id: str, user_id: str, query: str, results: list, response_time: float):
    """
    Record search analytics in the database.
    
    Args:
        conn: Database connection
        session_id: Session identifier
        user_id: User identifier
        query: Search query
        results: Search results
        response_time: Time taken to perform the search
    
    Returns:
        int: Search record ID or None if recording failed
    """
    logger.info(f"Recording search analytics - Session: {session_id}, User: {user_id}")
    
    if not conn:
        logger.error("Cannot record search: No database connection provided")
        return None
    
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
        for rank, result in enumerate(results[:5], 1):
            cur.execute("""
                INSERT INTO expert_search_matches
                    (search_id, expert_id, rank_position, similarity_score)
                VALUES (%s, %s, %s, %s)
            """, (
                search_id,
                result.get("id", "unknown"),
                rank,
                result.get("score", 0.0)
            ))
            logger.debug(f"Recorded match - Expert ID: {result.get('id')}, Rank: {rank}")

        conn.commit()
        logger.info(f"Successfully recorded all search data for search ID: {search_id}")
        return search_id
        
    except Exception as e:
        logger.error(f"Error recording search: {str(e)}", exc_info=True)
        conn.rollback()
        return None
    finally:
        cur.close()





import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def get_connection_pool():
    """Get the async connection pool."""
    # Assuming you have a method to fetch connection params.
    params = get_connection_params()
    return await asyncpg.create_pool(**params)





# Optimized version of process_expert_name_search
async def process_expert_name_search(
    name: str, 
    user_id: str, 
    active_only: bool = True, 
    k: int = 5
) -> SearchResponse:
    """
    Process search for experts by name.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        name: Name to search for
        user_id: User identifier
        active_only: Filter for active experts only
        k: Number of results to return
    
    Returns:
        SearchResponse: Search results
    """
    logger.info(f"Processing name search - Name: {name}, User: {user_id}")
    
    session_id = str(uuid.uuid4())[:8]  # Default session ID
    start_time = datetime.utcnow()
    
    try:
        # Use DatabaseConnection context manager for clean connection handling
        async with AsyncDatabaseConnection() as conn:
            try:
                session_id = await get_or_create_session(conn, user_id)
                logger.debug(f"Created session: {session_id}")
            except Exception as session_error:
                logger.error(f"Error creating session: {session_error}", exc_info=True)
                # Continue with default session ID
            
            # Execute search with timing
            results = []
            try:
                search_manager = ExpertSearchIndexManager()
                results = search_manager.search_experts_by_name(name, k=k, active_only=active_only)
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
            
            # Record search analytics directly with current connection
            if results:
                try:
                    await record_search(conn, session_id, user_id, f"name:{name}", results, response_time)
                except Exception as record_error:
                    logger.error(f"Error recording search: {record_error}", exc_info=True)
            
            # Generate refinement suggestions
            refinements = {}
            try:
                # Use the new refinement generation method
                refinements = await search_manager.generate_advanced_search_refinements(
                    search_type='name',
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

# Optimized version of process_expert_theme_search
async def process_expert_theme_search(
        theme: str, 
        user_id: str, 
        active_only: bool = True, 
        k: int = 5
    ) -> SearchResponse:
    """
    Process search for experts by theme.
    Optimized to use the DatabaseConnection context manager.
    
    Args:
        theme: Theme to search for
        user_id: User identifier
        active_only: Filter for active experts only
        k: Number of results to return
    
    Returns:
        SearchResponse: Search results
    """
    logger.info(f"Processing theme search - Theme: {theme}, User: {user_id}")
    
    session_id = str(uuid.uuid4())[:8]  # Default session ID
    start_time = datetime.utcnow()
    
    try:
        # Use DatabaseConnection context manager for clean connection handling
        async with AsyncDatabaseConnection() as conn:
            try:
                session_id = await get_or_create_session(conn, user_id)
                logger.debug(f"Created session: {session_id}")
            except Exception as session_error:
                logger.error(f"Error creating session: {session_error}", exc_info=True)
                # Continue with default session ID
            
            # Execute search with timing
            results = []
            try:
                search_manager = ExpertSearchIndexManager()
                results = search_manager.search_experts_by_theme(theme, k=k, active_only=active_only)
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
            
            # Record search analytics directly with current connection
            if results:
                try:
                    await record_search(conn, session_id, user_id, f"theme:{theme}", results, response_time)
                except Exception as record_error:
                    logger.error(f"Error recording search: {record_error}", exc_info=True)
            
            # Generate refinement suggestions
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




async def _generate_refinements(query: str, results: List[Dict], search_manager) -> Dict:
    """Generate refinements in a separate function for parallelism."""
    try:
        # First try to get refinements from search manager
        if results:
            refinements = search_manager.get_search_refinements(query, results)
            if refinements:
                return refinements
                
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
            
            return refinements
            
        except Exception as gemini_error:
            logger.error(f"Gemini refinement error: {gemini_error}", exc_info=True)
            # Fallback to basic refinements
            return {
                "filters": [],
                "related_queries": [],
                "expertise_areas": [query.capitalize()]
            }
            
    except Exception as e:
        logger.error(f"Refinements generation error: {e}", exc_info=True)
        return {
            "filters": [],
            "related_queries": [],
            "expertise_areas": []
        }
# Properly implement background task with safe_background_task

async def record_search(conn, session_id: str, user_id: str, query: str, results: List[Dict], response_time: float):
    """
    Record search analytics in the database.
    Fixed to handle missing tables and transaction failures.
    
    Args:
        conn: Database connection
        session_id: Session identifier
        user_id: User identifier
        query: Search query
        results: Search results
        response_time: Time taken to perform the search
    
    Returns:
        int: Search record ID or None if recording failed
    """
    logger.info(f"Recording search analytics - Session: {session_id}, User: {user_id}")
    
    if not conn:
        logger.error("Cannot record search: No database connection provided")
        return None
    
    cur = None
    try:
        # First check if tables exist
        cur = conn.cursor()
        
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'search_analytics'
            );
        """)
        
        analytics_table_exists = cur.fetchone()[0]
        
        if not analytics_table_exists:
            logger.warning("search_analytics table does not exist, skipping analytics recording")
            return None
        
        # Get search_id if search_sessions table exists
        search_id = None
        try:
            cur.execute("""
                SELECT id FROM search_sessions WHERE session_id = %s
            """, (session_id,))
            
            row = cur.fetchone()
            if row:
                search_id = row[0]
            else:
                # Use session_id as fallback
                search_id = session_id
                
        except Exception as session_error:
            logger.error(f"Error getting session ID: {session_error}")
            search_id = session_id  # Fallback to using session_id directly
            conn.rollback()  # Rollback failed transaction
            
            # Get a fresh cursor after rollback
            cur.close()
            cur = conn.cursor()
        
        # Record search analytics
        try:
            # Simplified insert that doesn't rely on search_sessions table
            cur.execute("""
                INSERT INTO search_analytics
                    (query, user_id, response_time,
                     result_count, search_type, timestamp)
                VALUES
                    (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (
                query,
                user_id,
                response_time,
                len(results),
                'expert_search'
            ))
            
            search_id = cur.fetchone()[0]
            logger.debug(f"Created search analytics record with ID: {search_id}")
        except Exception as analytics_error:
            logger.error(f"Error recording search analytics: {analytics_error}")
            conn.rollback()
            return None

        # Check if matches table exists before trying to record matches
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'expert_search_matches'
            );
        """)
        
        matches_table_exists = cur.fetchone()[0]
        
        if matches_table_exists and search_id:
            try:
                # Record top 5 expert matches
                for rank, result in enumerate(results[:5], 1):
                    # Ensure we have a valid ID even if it's a string
                    expert_id = str(result.get("id", "unknown"))
                    score = float(result.get("score", 0.0))
                    
                    cur.execute("""
                        INSERT INTO expert_search_matches
                            (search_id, expert_id, rank_position, similarity_score)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        search_id,
                        expert_id,
                        rank,
                        score
                    ))
                    logger.debug(f"Recorded match - Expert ID: {expert_id}, Rank: {rank}")
            except Exception as matches_error:
                logger.error(f"Error recording search matches: {matches_error}")
                conn.rollback()
                return search_id  # Still return search_id since analytics were recorded

        conn.commit()
        logger.info(f"Successfully recorded search data for search ID: {search_id}")
        return search_id
        
    except Exception as e:
        logger.error(f"Error recording search: {str(e)}", exc_info=True)
        if hasattr(conn, 'rollback'):
            conn.rollback()
        return None
    finally:
        if cur:
            cur.close()

async def _record_search_history_task(conn, user_id: str, query: str, result_count: int):
    """
    Record search in user history - implementation for safe_background_task.
    Modified to handle missing tables.
    
    Args:
        conn: Database connection (provided by safe_background_task)
        user_id: User identifier
        query: The search query
        result_count: Number of results found
    """
    try:
        # Connect to Redis for search history (this part should work)
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
        
        # Check if database table exists before trying to use it
        if conn:
            try:
                cur = conn.cursor()
                
                # Check if the table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'user_search_history'
                    );
                """)
                
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    cur.execute("""
                        INSERT INTO user_search_history
                            (user_id, query, result_count, timestamp)
                        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    """, (user_id, query, result_count))
                    conn.commit()
                else:
                    logger.warning("user_search_history table does not exist, skipping DB recording")
                
                cur.close()
            except Exception as db_error:
                logger.error(f"Error recording search history in database: {db_error}")
                if hasattr(conn, 'rollback'):
                    conn.rollback()
        
    except Exception as e:
        logger.error(f"Error recording search history: {e}")


# 

# Wrapper function to launch the background task with proper connection handling
# Async background task wrapper
async def safe_background_task(func, *args, **kwargs):
    """
    Wrapper to safely run background tasks with database connection management.
    
    Args:
        func: Async function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    """
    try:
        # Use async context manager to handle connection
        async with AsyncDatabaseConnection() as conn:
            # Call the function with the connection as the first argument
            return await func(conn, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error in background task {func.__name__}: {e}")

# Example implementation of _update_trending_suggestions_task
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
                    INSERT INTO trending_analytics
                        (query, suggestion, timestamp)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                """, (normalized_query, normalized_suggestion))
                conn.commit()
                cur.close()
            except Exception as db_error:
                logger.error(f"Error recording trending data in database: {db_error}")
                conn.rollback()
        
        logger.info(f"Updated trending data for suggestion: {normalized_suggestion}")
        
    except Exception as e:
        logger.error(f"Error updating trending suggestions: {e}")

# Wrapper function to launch the background task
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

# S

# Modified version of record_search to be used as a background task
async def _record_search_background(conn, session_id: str, user_id: str, query: str, results: List[Dict], response_time: float):
    """
    Background task implementation for recording search analytics.
    Designed to be used with safe_background_task.
    
    Args:
        conn: Database connection (provided by safe_background_task)
        session_id: Session identifier
        user_id: User identifier
        query: The search query
        results: Search results
        response_time: Time taken to perform the search
    """
    await record_search(conn, session_id, user_id, query, results, response_time)

# Wrapper function to start search recording as a background task
async def record_search_async(session_id: str, user_id: str, query: str, results: List[Dict], response_time: float):
    """
    Start search recording as a background task with proper connection management.
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        query: The search query
        results: Search results
        response_time: Time taken to perform the search
    """
    try:
        await safe_background_task(
            _record_search_background, 
            session_id, 
            user_id, 
            query, 
            results, 
            response_time
        )
    except Exception as e:
        logger.error(f"Failed to start search recording task: {e}")

async def process_expert_designation_search(
        designation: str, 
        user_id: str, 
        active_only: bool = True, 
        k: int = 5  # Add this parameter with a default value
    ) -> SearchResponse:
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
                search_type='designation',  # Explicitly specify as a keyword argument 
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

async def process_advanced_search(
    query: Optional[str], 
    user_id: str, 
    active_only: bool = True, 
    k: int = 5,
    name: Optional[str] = None,
    theme: Optional[str] = None, 
    designation: Optional[str] = None,
    publication: Optional[str] = None
) -> SearchResponse:
    """
    Process advanced search with multiple optional parameters.
    
    Args:
        query: General search query
        user_id: User identifier
        active_only: Filter for active experts only
        k: Number of results to return
        name: Expert name to search for
        theme: Theme to search for
        designation: Designation to search for
        publication: Publication to search for
    
    Returns:
        SearchResponse: Combined search results
    """
    logger.info(f"Processing advanced search - User: {user_id}")
    
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
        
        # Execute multiple searches based on provided parameters
        start_time = datetime.utcnow()
        all_results = []
        
        search_manager = ExpertSearchIndexManager()
        
        # Process each search parameter if provided
        if name:
            name_results = search_manager.search_experts_by_name(name, k=k, active_only=active_only)
            all_results.extend(name_results)
            
        if theme:
            theme_results = search_manager.search_experts_by_theme(theme, k=k, active_only=active_only)
            all_results.extend(theme_results)
            
        if designation:
            designation_results = search_manager.search_experts_by_designation(designation, k=k, active_only=active_only)
            all_results.extend(designation_results)
            
        # If query parameter is provided and no specific parameters, do a general search
        if query and not (name or theme or designation or publication):
            query_results = search_manager.search_experts(query, k=k, active_only=active_only)
            all_results.extend(query_results)
        
        # Handle publication search if implemented
        if publication:
            # If you have a publication search implementation
            pass
            
        # If no results from specific searches but we have a general query, use it
        if not all_results and query:
            query_results = search_manager.search_experts(query, k=k, active_only=active_only)
            all_results.extend(query_results)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Remove duplicates by expert ID while preserving the highest score
        unique_results = {}
        for result in all_results:
            expert_id = str(result.get('id', 'unknown'))
            current_score = result.get('score', 0)
            
            if expert_id not in unique_results or current_score > unique_results[expert_id].get('score', 0):
                unique_results[expert_id] = result
        
        # Convert to list and sort by score
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = final_results[:k]  # Limit to requested k results
        
        # Format results
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
                ) for result in final_results
            ]
        except Exception as format_error:
            logger.error(f"Result formatting error: {format_error}", exc_info=True)
        
        # Record search analytics if we have a connection
        if conn and final_results:
            try:
                # Construct a composite query string for analytics
                composite_query = []
                if query:
                    composite_query.append(f"query:{query}")
                if name:
                    composite_query.append(f"name:{name}")
                if theme:
                    composite_query.append(f"theme:{theme}")
                if designation:
                    composite_query.append(f"designation:{designation}")
                if publication:
                    composite_query.append(f"publication:{publication}")
                
                analytics_query = " | ".join(composite_query)
                
                await record_search(conn, session_id, user_id, analytics_query, final_results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions
        refinements = {}
        try:
            # Use the refinement generation method with appropriate context
            # Prioritize which search_type to use for refinements
            if name:
                search_type = 'name'
            elif theme:
                search_type = 'theme'
            elif designation:
                search_type = 'designation'
            elif publication:
                search_type = 'publication'
            else:
                search_type = 'general'
                
            refinements = await search_manager.generate_advanced_search_refinements(
                search_type=search_type,
                query=query or name or theme or designation or publication,
                user_id=user_id,
                results=final_results
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
        logger.error(f"Critical error in advanced search: {e}", exc_info=True)
        
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

async def process_publication_search(
    publication: str, 
    user_id: str, 
    k: int = 5
) -> SearchResponse:
    """
    Process search for experts by publication.
    
    Args:
        publication: Publication to search for
        user_id: User identifier
        k: Number of results to return
    
    Returns:
        SearchResponse: Search results
    """
    logger.info(f"Processing publication search - Publication: {publication}, User: {user_id}")
    
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
        
        # Execute search with timing
        start_time = datetime.utcnow()
        results = []
        
        # Here you would implement the actual publication search logic
        # This is a placeholder implementation - modify with your actual search logic
        try:
            search_manager = ExpertSearchIndexManager()
            # Assuming there's a method for searching by publication
            # If not available, fall back to standard search
            if hasattr(search_manager, 'search_experts_by_publication'):
                results = search_manager.search_experts_by_publication(publication, k=k)
            else:
                # Fallback to general search with publication focused query
                results = search_manager.search_experts(f"publication {publication}", k=k)
            
            logger.info(f"Publication search found {len(results)} results")
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
                await record_search(conn, session_id, user_id, f"publication:{publication}", results, response_time)
            except Exception as record_error:
                logger.error(f"Error recording search: {record_error}", exc_info=True)
        
        # Generate refinement suggestions
        refinements = {}
        try:
            # Use the publication-specific refinement generation if available
            refinements = await search_manager.generate_advanced_search_refinements(
                search_type='publication',  
                query=publication, 
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
        logger.error(f"Critical error in publication search: {e}", exc_info=True)
        
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