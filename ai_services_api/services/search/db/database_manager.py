import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import uuid
from dotenv import load_dotenv
from ai_services_api.services.centralized_repository.database_setup import get_db_connection
from psycopg2.extras import DictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        """Initialize database connection."""
        connection_ctx = get_db_connection()
        self.conn = connection_ctx.__enter__()
        self._cursor = None
        self._connection_ctx = connection_ctx

    def get_cursor(self):
        """Get a database cursor with DictCursor factory."""
        if not self._cursor or self._cursor.closed:
            self._cursor = self.conn.cursor(cursor_factory=DictCursor)
        return self._cursor

    def execute(self, query: str, params: tuple = None) -> List[Tuple[Any, ...]]:
        """Execute a query and return results if any."""
        cursor = self.get_cursor()
        try:
            cursor.execute(query, params)
            self.conn.commit()
            if cursor.description:  # If the query returns results
                return cursor.fetchall()
            return []
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Query execution failed: {str(e)}\nQuery: {query}\nParams: {params}")
            raise

    def start_search_session(self, user_id: str) -> int:
        """Start a new search session and return session_id."""
        try:
            result = self.execute("""
                INSERT INTO search_sessions 
                (user_id, start_timestamp, query_count, successful_searches, is_active)
                VALUES (%s, NOW(), 0, 0, TRUE)
                RETURNING session_id
            """, (user_id,))
            
            if result and result[0]:
                return result[0][0]
            raise Exception("Failed to create search session")
        except Exception as e:
            logger.error(f"Error starting search session: {e}")
            raise

    def update_search_session(self, session_id: int, successful: bool = True) -> None:
        """Update search session metrics."""
        try:
            self.execute("""
                UPDATE search_sessions 
                SET 
                    query_count = query_count + 1,
                    successful_searches = successful_searches + %s,
                    end_timestamp = NOW()
                WHERE session_id = %s
            """, (1 if successful else 0, session_id))
        except Exception as e:
            logger.error(f"Error updating search session: {e}")
            raise

    def end_search_session(self, session_id: int) -> None:
        """End a search session."""
        try:
            self.execute("""
                UPDATE search_sessions 
                SET 
                    is_active = FALSE,
                    end_timestamp = NOW()
                WHERE session_id = %s
            """, (session_id,))
        except Exception as e:
            logger.error(f"Error ending search session: {e}")
            raise

    def get_active_session(self, user_id: str) -> Optional[int]:
        """Get the active session ID for a user if it exists."""
        try:
            result = self.execute("""
                SELECT session_id
                FROM search_sessions
                WHERE user_id = %s AND is_active = TRUE
                ORDER BY start_timestamp DESC
                LIMIT 1
            """, (user_id,))
            
            return result[0][0] if result else None
        except Exception as e:
            logger.error(f"Error getting active session: {e}")
            raise

    def add_expert(self, first_name: str, last_name: str, 
                  knowledge_expertise: List[str] = None,
                  domains: List[str] = None,
                  fields: List[str] = None,
                  subfields: List[str] = None,
                  orcid: str = None) -> str:
        """Add or update an expert in the database."""
        try:
            # Convert empty strings to None
            orcid = orcid if orcid and orcid.strip() else None
            
            cursor = self.get_cursor()
            cursor.execute("""
                INSERT INTO experts_expert 
                (first_name, last_name, knowledge_expertise, domains, fields, subfields, orcid)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (orcid) 
                WHERE orcid IS NOT NULL AND orcid != ''
                DO UPDATE SET
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    knowledge_expertise = EXCLUDED.knowledge_expertise,
                    domains = EXCLUDED.domains,
                    fields = EXCLUDED.fields,
                    subfields = EXCLUDED.subfields
                RETURNING id
            """, (first_name, last_name, 
                  knowledge_expertise or [], 
                  domains or [], 
                  fields or [], 
                  subfields or [], 
                  orcid))
            
            expert_id = cursor.fetchone()[0]
            self.conn.commit()
            logger.info(f"Added/updated expert data for {first_name} {last_name}")
            return expert_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding expert {first_name} {last_name}: {e}")
            raise

    def add_publication(self, doi: str, title: str, abstract: str, summary: str) -> None:
        """Add or update a publication in the database."""
        try:
            self.execute("""
                INSERT INTO resources_resource (doi, title, abstract, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doi) DO UPDATE 
                SET title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    summary = EXCLUDED.summary
            """, (doi, title, abstract, summary))
            logger.info(f"Added/updated publication: {title}")
        except Exception as e:
            logger.error(f"Error adding publication: {e}")
            raise

    def add_author(self, name: str, orcid: Optional[str] = None, 
                  author_identifier: Optional[str] = None) -> int:
        """Add an author and return their ID."""
        try:
            result = self.execute("""
                INSERT INTO authors (name, orcid, author_identifier)
                VALUES (%s, %s, %s)
                ON CONFLICT (orcid) 
                WHERE orcid IS NOT NULL
                DO UPDATE SET
                    name = EXCLUDED.name,
                    author_identifier = EXCLUDED.author_identifier
                RETURNING author_id
            """, (name, orcid, author_identifier))
            
            return result[0][0]
        except Exception as e:
            logger.error(f"Error adding author {name}: {e}")
            raise

    def add_tag(self, tag_name: str, tag_type: str) -> int:
        """Add a tag and return its ID."""
        try:
            result = self.execute("""
                INSERT INTO tags (tag_name)
                VALUES (%s)
                ON CONFLICT (tag_name) DO UPDATE
                SET tag_name = EXCLUDED.tag_name
                RETURNING tag_id
            """, (tag_name,))
            
            return result[0][0]
        except Exception as e:
            logger.error(f"Error adding tag {tag_name}: {e}")
            raise

    def link_author_publication(self, author_id: int, doi: str) -> None:
        """Link an author to a publication."""
        try:
            self.execute("""
                INSERT INTO publication_authors (doi, author_id)
                VALUES (%s, %s)
                ON CONFLICT (doi, author_id) DO NOTHING
            """, (doi, author_id))
        except Exception as e:
            logger.error(f"Error linking author {author_id} to publication {doi}: {e}")
            raise

    def link_publication_tag(self, doi: str, tag_id: int) -> None:
        """Link a tag to a publication."""
        try:
            self.execute("""
                INSERT INTO publication_tags (doi, tag_id)
                VALUES (%s, %s)
                ON CONFLICT (doi, tag_id) DO NOTHING
            """, (doi, tag_id))
        except Exception as e:
            logger.error(f"Error linking tag {tag_id} to publication {doi}: {e}")
            raise

    def update_expert(self, expert_id: str, updates: Dict[str, Any]) -> None:
        """Update expert information."""
        try:
            set_clauses = []
            params = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
            
            params.append(expert_id)
            query = f"""
                UPDATE experts_expert 
                SET {', '.join(set_clauses)}
                WHERE id = %s
            """
            
            self.execute(query, tuple(params))
            logger.info(f"Expert {expert_id} updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating expert {expert_id}: {e}")
            raise

    def get_expert_by_name(self, first_name: str, last_name: str) -> Optional[Tuple]:
        """Get expert by first_name and last_name."""
        try:
            result = self.execute("""
                SELECT id, first_name, last_name, knowledge_expertise, domains, fields, subfields, orcid
                FROM experts_expert
                WHERE first_name = %s AND last_name = %s
            """, (first_name, last_name))
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error retrieving expert {first_name} {last_name}: {e}")
            raise

    def get_recent_queries(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get most recent search queries."""
        try:
            result = self.execute("""
                SELECT query_id, query, timestamp, result_count, search_type
                FROM query_history_ai
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            
            return [{
                'query_id': row[0],
                'query': row[1],
                'timestamp': row[2].isoformat(),
                'result_count': row[3],
                'search_type': row[4]
            } for row in result]
            
        except Exception as e:
            logger.error(f"Error getting recent queries: {e}")
            return []

    def get_term_frequencies(self, expert_id: Optional[int] = None) -> Dict[str, int]:
        """Get term frequency dictionary"""
        try:
            if expert_id:
                result = self.execute("""
                    SELECT term, frequency 
                    FROM term_frequencies 
                    WHERE expert_id = %s AND last_updated >= NOW() - INTERVAL '30 days'
                """, (expert_id,))
            else:
                result = self.execute("""
                    SELECT term, SUM(frequency) as total_frequency
                    FROM term_frequencies 
                    WHERE last_updated >= NOW() - INTERVAL '30 days'
                    GROUP BY term
                """)
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting term frequencies: {e}")
            return {}

    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular search queries."""
        try:
            result = self.execute("""
                SELECT query, COUNT(*) as count
                FROM query_history_ai
                GROUP BY query
                ORDER BY count DESC
                LIMIT %s
            """, (limit,))
            
            return [{
                'query': row[0],
                'count': row[1]
            } for row in result]
            
        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []

    def get_user_queries(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent queries for a specific user."""
        try:
            result = self.execute("""
                SELECT query_id, query, timestamp, result_count, search_type
                FROM query_history_ai
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            
            return [{
                'query_id': row[0],
                'query': row[1],
                'timestamp': row[2].isoformat(),
                'result_count': row[3],
                'search_type': row[4]
            } for row in result]
            
        except Exception as e:
            logger.error(f"Error getting user queries: {e}")
            return []

    def add_query(self, query: str, result_count: int, search_type: str = 'semantic', 
                 user_id: Optional[str] = None) -> Optional[int]:
        """Add a search query to history."""
        try:
            result = self.execute("""
                INSERT INTO query_history_ai (query, result_count, search_type, user_id)
                VALUES (%s, %s, %s, %s)
                RETURNING query_id
            """, (query, result_count, search_type, user_id))
            
            return result[0][0] if result else None
            
        except Exception as e:
            logger.error(f"Error adding query to history: {e}")
            raise
    
    async def record_search_analytics(conn, session_id, query, user_id, response_time, result_count, search_type, category=None):
        """Record search analytics with search_id and category fields."""
        try:
            cur = conn.cursor()
            try:
                # Generate a search_id if not provided (can be session_id or a uuid)
                search_id = int(session_id) if session_id.isdigit() else int(uuid.uuid4().int % 2147483647)
                
                cur.execute("""
                    INSERT INTO search_analytics
                        (search_id, query, user_id, response_time,
                        result_count, search_type, timestamp, category)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s)
                    RETURNING id
                """, (search_id, query, user_id, response_time, result_count, search_type, category))
                
                search_analytics_id = cur.fetchone()[0]
                conn.commit()
                return search_analytics_id
            except Exception as e:
                conn.rollback()
                logger.error(f"Error recording search analytics: {e}")
                logger.error(f"DETAIL: {e.diag.message_detail if hasattr(e, 'diag') else ''}")
                raise
            finally:
                cur.close()
        except Exception as e:
            logger.error(f"Error in record_search_analytics: {e}")
            raise

    def get_search_metrics(self, start_date: str, end_date: str, 
                          search_type: List[str] = None) -> Dict:
        """Get search metrics for a date range."""
        try:
            query = """
            SELECT 
                COUNT(*) as total_searches,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(response_time) as avg_response_time,
                COUNT(CASE WHEN result_count > 0 THEN 1 END)::FLOAT / 
                    COUNT(*) as success_rate
            FROM search_analytics
            WHERE timestamp BETWEEN %s AND %s
            """
            
            if search_type:
                query += " AND search_type = ANY(%s)"
                result = self.execute(query, (start_date, end_date, search_type))
            else:
                result = self.execute(query, (start_date, end_date))
                
            return {
                'total_searches': result[0][0],
                'unique_users': result[0][1],
                'avg_response_time': result[0][2],
                'success_rate': result[0][3]
            }
        except Exception as e:
            logger.error(f"Error getting search metrics: {e}")
            raise

    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Get system performance metrics."""
        try:
            result = self.execute("""
                SELECT 
                    AVG(response_time) as avg_response_time,
                    COUNT(*) as total_queries,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(CASE WHEN result_count = 0 THEN 1 END)::FLOAT / 
                        COUNT(*) as error_rate
                FROM search_analytics
                WHERE timestamp > NOW() - INTERVAL '%s hours'
            """, (hours,))
            
            return {
                'avg_response_time': result[0][0],
                'total_queries': result[0][1],
                'unique_users': result[0][2],
                'error_rate': result[0][3]
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            raise

    def record_expert_search(self, search_id: int, expert_id: str, 
                           rank_position: int) -> None:
        """Record expert search result."""
        try:
            self.execute("""
                INSERT INTO expert_searches 
                (search_id, expert_id, rank_position)
                VALUES (%s, %s, %s)
            """, (search_id, expert_id, rank_position))
        except Exception as e:
            logger.error(f"Error recording expert search: {e}")
            raise

    def record_query_prediction(self, partial_query: str, predicted_query: str,
                              confidence_score: float, user_id: str) -> None:
        """Record query prediction."""
        try:
            self.execute("""
                INSERT INTO query_predictions 
                (partial_query, predicted_query, confidence_score, user_id, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
            """, (partial_query, predicted_query, confidence_score, user_id))
        except Exception as e:
            logger.error(f"Error recording query prediction: {e}")
            raise

    def start_search_session(self, user_id: str) -> int:
        """Start a new search session and return session_id."""
        try:
            result = self.execute("""
                INSERT INTO search_sessions (user_id)
                VALUES (%s)
                RETURNING id
            """, (user_id,))
            return result[0][0]
        except Exception as e:
            logger.error(f"Error starting search session: {e}")
            raise

    def update_search_session(self, session_id: int, successful: bool = True) -> None:
        """Update search session metrics."""
        try:
            self.execute("""
                UPDATE search_sessions 
                SET query_count = query_count + 1,
                    successful_searches = successful_searches + %s,
                    end_timestamp = NOW()
                WHERE id = %s
            """, (1 if successful else 0, session_id))
        except Exception as e:
            logger.error(f"Error updating search session: {e}")
            raise

    def record_click(self, search_id: int, expert_id: str = None) -> None:
        """Record when a user clicks on a search result."""
        try:
            # Update search logs
            self.execute("""
                UPDATE search_logs 
                SET clicked = TRUE 
                WHERE id = %s
            """, (search_id,))
            
            if expert_id:
                # Update expert search if applicable
                self.execute("""
                    UPDATE expert_searches 
                    SET clicked = TRUE,
                    click_timestamp = NOW()
                    WHERE search_id = %s AND expert_id = %s
                """, (search_id, expert_id))
        except Exception as e:
            logger.error(f"Error recording click: {e}")
            raise

    

    def get_expert_metrics(self, expert_id: str = None) -> Dict:
        """Get expert search metrics."""
        try:
            query = """
            SELECT 
                es.expert_id,
                COUNT(*) as total_appearances,
                AVG(es.rank_position) as avg_rank,
                SUM(CASE WHEN es.clicked THEN 1 ELSE 0 END)::FLOAT / 
                    COUNT(*) as click_through_rate
            FROM expert_searches es
            """
            
            if expert_id:
                query += " WHERE expert_id = %s"
                result = self.execute(query, (expert_id,))
            else:
                query += " GROUP BY expert_id"
                result = self.execute(query)
                
            return [
                {
                    'expert_id': row[0],
                    'appearances': row[1],
                    'avg_rank': row[2],
                    'click_rate': row[3]
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error getting expert metrics: {e}")
            raise

    


    def close(self):
        """Close database connection and cursor."""
        try:
            if hasattr(self, '_cursor') and self._cursor and not getattr(self._cursor, 'closed', True):
                self._cursor.close()
                self._cursor = None
                
            if hasattr(self, 'conn') and self.conn:
                if hasattr(self, '_connection_ctx') and self._connection_ctx:
                    self._connection_ctx.__exit__(None, None, None)
                    self._connection_ctx = None
                self.conn = None
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()