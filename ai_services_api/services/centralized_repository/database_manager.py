import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import psycopg2
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection with robust connection parameters."""
    try:
        load_dotenv()
        db_url = os.getenv('DATABASE_URL')
        if db_url:
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
            conn_params = {
                'host': os.getenv('DB_HOST', 'postgres'),
                'port': os.getenv('DB_PORT', '5432'),
                'dbname': os.getenv('DB_NAME', 'aphrc'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'p0stgres')
            }
        conn_params = {k: v for k, v in conn_params.items() if v is not None}
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
        """Execute a query and optionally return results if available."""
        try:
            self.cur.execute(query, params)
            self.conn.commit()
            if self.cur.description:
                return self.cur.fetchall()
            return None
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Query execution failed: {str(e)}\nQuery: {query}\nParams: {params}")
            raise

    def add_expert(self, first_name: str, last_name: str, 
                  knowledge_expertise: List[str] = None,
                  domains: List[str] = None,
                  fields: List[str] = None,
                  subfields: List[str] = None,
                  orcid: str = None,
                  profile_summary: str = None,
                  affiliation: str = None,
                  contact_email: str = None) -> str:
        """Add or update an expert in the database."""
        try:
            orcid = orcid if orcid and orcid.strip() else None
            self.cur.execute("""
                INSERT INTO experts_expert 
                (first_name, last_name, knowledge_expertise, domains, fields, subfields, orcid, profile_summary, affiliation, contact_email)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (orcid) 
                WHERE orcid IS NOT NULL AND orcid != ''
                DO UPDATE SET
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    knowledge_expertise = EXCLUDED.knowledge_expertise,
                    domains = EXCLUDED.domains,
                    fields = EXCLUDED.fields,
                    subfields = EXCLUDED.subfields,
                    profile_summary = EXCLUDED.profile_summary,
                    affiliation = EXCLUDED.affiliation,
                    contact_email = EXCLUDED.contact_email
                RETURNING id
            """, (first_name, last_name, 
                  knowledge_expertise or [], 
                  domains or [], 
                  fields or [], 
                  subfields or [], 
                  orcid, profile_summary, affiliation, contact_email))
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
        doi: Optional[str] = None,
        affiliation: str = None,
        contact_email: str = None) -> None:
        """Add or update a publication or expert in the database."""
        try:
            if type == 'expert':
                name = title or summary
                name_parts = re.split(r'\s+', name.strip(), 1)
                first_name = name_parts[0] if name_parts else 'Unknown'
                last_name = name_parts[1] if len(name_parts) > 1 else 'Unknown'
                
                extracted_domains = domains or []
                if summary:
                    for keyword in ['health', 'education', 'gender', 'climate', 'demography']:
                        if keyword in summary.lower():
                            extracted_domains.append(keyword.capitalize())
                
                expert_id = self.add_expert(
                    first_name=first_name,
                    last_name=last_name,
                    knowledge_expertise=[],
                    domains=extracted_domains,
                    fields=[],
                    subfields=[],
                    orcid=None,
                    profile_summary=summary,
                    affiliation=affiliation or 'APHRC',
                    contact_email=contact_email
                )
                logger.info(f"Added expert: {first_name} {last_name} from {source or 'unknown'}")
                return
            
            title = title[:500] if title else 'Untitled'
            source = source[:500] if source else None
            type = type[:100] if type else None
            
            authors_json = json.dumps(authors) if authors is not None else None
            
            if publication_year is not None:
                publication_year = str(publication_year)
            
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
            
            if not update_result:
                max_id_result = self.execute("SELECT MAX(id) FROM resources_resource")
                next_id = 1
                if max_id_result and max_id_result[0][0] is not None:
                    next_id = max_id_result[0][0] + 1
                    
                self.execute("""
                    INSERT INTO resources_resource 
                    (id, doi, title, summary, source, type, authors, domains, publication_year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    next_id, doi, title, summary, source, type, authors_json, domains, publication_year
                ))
                logger.info(f"Added new publication: {title[:50]}...")
            else:
                logger.info(f"Updated existing publication: {title[:50]}...")
                
        except Exception as e:
            logger.error(f"Error processing publication/expert '{title[:50]}...': {e}")
            raise

    def get_all_publications(self) -> List[Dict]:
        """Retrieve all publications from the database."""
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
                SELECT id, first_name, last_name, knowledge_expertise, domains, fields, subfields, orcid, profile_summary, affiliation, contact_email
                FROM experts_expert
                WHERE first_name = %s AND last_name = %s
            """, (first_name, last_name))
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error retrieving expert {first_name} {last_name}: {e}")
            raise

    def add_author(self, author_name: str, orcid: Optional[str] = None, author_identifier: Optional[str] = None) -> int:
        """Add an author as a tag or return existing tag ID."""
        try:
            result = self.execute("""
                SELECT tag_id FROM tags 
                WHERE tag_name = %s AND tag_type = 'author'
            """, (author_name,))
            
            if result:
                return result[0][0]
            
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
        """Link an author with a publication using either DOI or title."""
        try:
            result = self.execute("""
                SELECT 1 FROM publication_tags 
                WHERE (doi = %s OR title = %s) AND tag_id = %s
            """, (identifier, identifier, author_id))
            
            if result:
                return
            
            self.execute("""
                INSERT INTO publication_tags (doi, title, tag_id)
                VALUES (%s, %s, %s)
            """, (identifier if '10.' in identifier else None,
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