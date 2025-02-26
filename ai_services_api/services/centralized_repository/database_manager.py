import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import psycopg2
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Create a database connection with robust connection parameters.
    
    Returns:
        psycopg2.extensions.connection: An open database connection
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Determine connection parameters
        db_url = os.getenv('DATABASE_URL')
        
        if db_url:
            # Parse database URL if available
            from urllib.parse import urlparse
            url = urlparse(db_url)
            conn_params = {
                'host': url.hostname,
                'port': url.port or 5432,
                'dbname': url.path[1:],
                'user': url.username,
                'password': url.password
            }
        else:
            # Fallback to individual environment variables
            conn_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'dbname': os.getenv('DB_NAME', 'aphrc'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres')
            }
        
        # Remove None values
        conn_params = {k: v for k, v in conn_params.items() if v is not None}
        
        # Establish connection
        conn = psycopg2.connect(**conn_params)
        return conn
    
    except (Exception, psycopg2.Error) as e:
        logger.error(f"Error connecting to the database: {e}")
        raise

class DatabaseManager:
    def __init__(self):
        """Initialize database connection and cursor."""
        self.conn = None
        self.cur = None
        try:
            self.conn = get_db_connection()
            self.cur = self.conn.cursor()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def execute(self, query: str, params: tuple = None) -> Any:
        """
        Execute a query and optionally return results if available.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters to include in the query.

        Returns:
            Any: If the query returns results (e.g., SELECT), a list of tuples is returned.
                 Otherwise, None is returned for queries like INSERT, UPDATE, DELETE.
        """
        try:
            self.cur.execute(query, params)
            self.conn.commit()

            if self.cur.description:  # Checks if the query returns results
                return self.cur.fetchall()  # Return results for SELECT queries
            return None  # Return None for non-SELECT queries

        except Exception as e:
            self.conn.rollback()  # Rollback the transaction on error
            logger.error(f"Query execution failed: {str(e)}\nQuery: {query}\nParams: {params}")
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
            
            self.cur.execute("""
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
            
            expert_id = self.cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"Added initial expert data for {first_name} {last_name}")
            return expert_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding expert {first_name} {last_name}: {e}")
            raise
     
    def add_publication(self,
        title: str,
        summary: str,
        source: str = None,
        type: str = None,
        authors: List[str] = None,
        domains: List[str] = None,
        publication_year: Optional[int] = None,
        doi: Optional[str] = None) -> None:
        """Add or update a publication in the database."""
        try:
            # Convert authors list to JSON if provided
            authors_json = json.dumps(authors) if authors is not None else None
            
            # Update or insert in a single operation
            update_result = self.execute("""
                UPDATE resources_resource
                SET summary = COALESCE(%s, summary),
                    doi = COALESCE(%s, doi),
                    type = COALESCE(%s, type),
                    authors = COALESCE(%s, authors),
                    domains = COALESCE(%s, domains),
                    publication_year = COALESCE(%s, publication_year)
                WHERE (doi = %s) OR 
                    (title = %s AND source = %s)
                RETURNING id
            """, (
                summary, doi, type, authors_json, domains, publication_year,
                doi, title, source
            ))
            
            # If no existing record, insert new publication
            if not update_result:
                self.execute("""
                    INSERT INTO resources_resource 
                    (doi, title, summary, source, type, authors, domains, publication_year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    doi, title, summary, source, type, authors_json, domains, publication_year
                ))
                logger.info(f"Added new publication: {title}")
            else:
                logger.info(f"Updated existing publication: {title}")
                
        except Exception as e:
            logger.error(f"Error processing publication '{title}': {e}")
            raise
            
    def get_all_publications(self) -> List[Dict]:
        """
        Retrieve all publications from the database.
        
        Returns:
            List of publication dictionaries
        """
        try:
            result = self.execute("SELECT * FROM resources_resource")
            return [dict(zip([column[0] for column in self.cur.description], row)) for row in result]
        except Exception as e:
            logger.error(f"Error retrieving publications: {e}")
            return []

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

    def add_author(self, author_name: str, orcid: Optional[str] = None, author_identifier: Optional[str] = None) -> int:
        """
        Add an author as a tag or return existing tag ID.
        
        Args:
            author_name (str): Name of the author.
            orcid (str, optional): ORCID identifier of the author.
            author_identifier (str, optional): Other identifier for the author.
        
        Returns:
            int: ID of the added or existing tag.
        """
        try:
            # First, check if the author tag already exists
            result = self.execute("""
                SELECT tag_id FROM tags 
                WHERE tag_name = %s AND tag_type = 'author'
            """, (author_name,))
            
            if result:
                # Author tag already exists, return its ID
                return result[0][0]
            
            # Insert new author tag
            result = self.execute("""
                INSERT INTO tags (tag_name, tag_type, additional_metadata) 
                VALUES (%s, 'author', %s)
                RETURNING tag_id
            """, (author_name, json.dumps({
                'orcid': orcid,
                'author_identifier': author_identifier
            })))
            
            if result:
                tag_id = result[0][0]
                logger.info(f"Added new author tag: {author_name}")
                return tag_id
            
            raise ValueError(f"Failed to add author tag: {author_name}")
        
        except Exception as e:
            logger.error(f"Error adding author tag {author_name}: {e}")
            raise

    def link_author_publication(self, author_id: int, identifier: str) -> None:
        """
        Link an author with a publication using either DOI or title.
        
        Args:
            author_id: ID of the author tag
            identifier: Either DOI or title of the publication
        """
        try:
            # Check if the link already exists
            result = self.execute("""
                SELECT 1 FROM publication_tags 
                WHERE (doi = %s OR title = %s) AND tag_id = %s
            """, (identifier, identifier, author_id))
            
            if result:
                return
            
            # Create new link
            self.execute("""
                INSERT INTO publication_tags (doi, title, tag_id)
                VALUES (%s, %s, %s)
            """, (identifier if '10.' in identifier else None,  # Assume it's a DOI if it starts with '10.'
                identifier if '10.' not in identifier else None,
                author_id))
            
            logger.info(f"Linked publication {identifier} with author tag {author_id}")
        
        except Exception as e:
            logger.error(f"Error linking publication {identifier} with author tag {author_id}: {e}")
            raise

    def close(self):
        """Close database connection."""
        try:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()