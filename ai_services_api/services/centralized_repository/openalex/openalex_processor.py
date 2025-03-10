import os
import logging
import aiohttp  # type: ignore[import]
import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from urllib.parse import urlparse
import json
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
import requests
from ai_services_api.services.centralized_repository.openalex.expert_processor import ExpertProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_connection_params():
    """Get database connection parameters from environment variables."""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        parsed_url = urlparse(database_url)
        return {
            'host': parsed_url.hostname,
            'port': parsed_url.port,
            'dbname': parsed_url.path[1:],
            'user': parsed_url.username,
            'password': parsed_url.password
        }
    else:
        in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        return {
            'host': '167.86.85.127' if in_docker else 'localhost',
            'port': '5432',
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
        }

def get_db_connection(dbname=None):
    """Create a connection to PostgreSQL database."""
    params = get_connection_params()
    if dbname:
        params['dbname'] = dbname
    
    try:
        conn = psycopg2.connect(**params)
        logger.info(f"Connected to database: {params['dbname']}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise

class OpenAlexProcessor:
    def __init__(self):
        """Initialize the OpenAlex processor."""
        try:
            load_dotenv()
            self.base_url = os.getenv('OPENALEX_API_URL', 'https://api.openalex.org')
            self.session = None
            self.db = DatabaseManager()
            self.expert_processor = ExpertProcessor(self.db, self.base_url)
            logger.info("OpenAlexProcessor initialized")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    async def load_initial_experts(self, expertise_csv: str):
        """Load initial expert data from CSV, skipping existing experts."""
        try:
            if not os.path.exists(expertise_csv):
                raise FileNotFoundError(f"CSV file not found: {expertise_csv}")
            
            logger.info(f"Loading experts from {expertise_csv}")
            
            conn = get_db_connection()
            cur = conn.cursor()

            # Check column type for knowledge_expertise
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'experts_expert'
                AND column_name = 'knowledge_expertise';
            """)
            column_types = dict(cur.fetchall())

            df = pd.read_csv(expertise_csv)
            for _, row in df.iterrows():
                try:
                    first_name = row['First_name']
                    last_name = row['Last_name']
                    
                    # Check if expert already exists
                    cur.execute("""
                        SELECT id FROM experts_expert 
                        WHERE first_name = %s AND last_name = %s
                    """, (first_name, last_name))
                    
                    if cur.fetchone() is not None:
                        logger.info(f"Expert already exists: {first_name} {last_name} - skipping")
                        continue

                    expertise_str = row['Knowledge and Expertise']
                    expertise_list = []
                    if not pd.isna(expertise_str):
                        expertise_list = [exp.strip() for exp in expertise_str.split(',') if exp.strip()]
                    
                    cur.execute("""
                        INSERT INTO experts_expert (
                            first_name, last_name, designation, theme, unit,
                            contact_details, knowledge_expertise
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        first_name, last_name, row['Designation'],
                        row['Theme'], row['Unit'], row['Contact Details'],
                        json.dumps(expertise_list) if column_types['knowledge_expertise'] == 'jsonb' else expertise_list
                    ))
                    conn.commit()
                    logger.info(f"Added expert: {first_name} {last_name}")
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error processing row: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading experts: {e}")
            raise
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    async def _update_single_expert(self, session: aiohttp.ClientSession, 
                                expert_id: int, first_name: str, last_name: str):
        """Update a single expert."""
        try:
            success = await self.update_expert_fields(session, first_name, last_name)
            if success:
                logger.info(f"Updated expert: {first_name} {last_name}")
            else:
                logger.warning(f"Failed to update expert: {first_name} {last_name}")
        except Exception as e:
            logger.error(f"Error processing expert {first_name} {last_name}: {e}")

    async def update_experts_with_openalex(self):
        """Update experts with OpenAlex data."""
        try:
            experts = self.db.execute("""
                SELECT id, first_name, last_name
                FROM experts_expert
                WHERE orcid IS NULL OR orcid = ''
            """)
            
            if not experts:
                logger.info("No experts to update")
                return
            
            logger.info(f"Found {len(experts)} experts to update")
            
            async with aiohttp.ClientSession() as session:
                batch_size = 5
                for i in range(0, len(experts), batch_size):
                    batch = experts[i:i + batch_size]
                    tasks = []
                    
                    for expert_id, first_name, last_name in batch:
                        task = asyncio.create_task(
                            self._update_single_expert(session, expert_id, first_name, last_name)
                        )
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    if i + batch_size < len(experts):
                        await asyncio.sleep(2)

            logger.info("Expert update completed")
                        
        except Exception as e:
            logger.error(f"Error updating experts: {e}")
            raise

    async def update_expert_fields(self, session: aiohttp.ClientSession, 
                           first_name: str, last_name: str) -> bool:
        """Update expert fields with OpenAlex data."""
        try:
            orcid, openalex_id = self.get_expert_openalex_data(first_name, last_name)
            
            if not openalex_id:
                logger.warning(f"No OpenAlex ID found for {first_name} {last_name}")
                return False
            
            domains, fields, subfields = await self.get_expert_domains(
                session, first_name, last_name, openalex_id
            )
            
            try:
                self.db.execute("""
                    UPDATE experts_expert
                    SET orcid = COALESCE(NULLIF(%s, ''), orcid),
                        domains = %s,
                        fields = %s,
                        subfields = %s
                    WHERE first_name = %s AND last_name = %s
                """, (orcid, domains, fields, subfields, first_name, last_name))
                
                logger.info(f"Updated fields for {first_name} {last_name}")
                return True

            except Exception as e:
                logger.error(f"Database update error for {first_name} {last_name}: {e}")
                raise

        except Exception as e:
            logger.error(f"Error updating expert fields for {first_name} {last_name}: {e}")
            return False

    async def get_expert_domains(self, session: aiohttp.ClientSession, 
                           first_name: str, last_name: str, 
                           openalex_id: str) -> Tuple[List[str], List[str], List[str]]:
        """Get expert domains from their works."""
        try:
            works = await self.get_expert_works(session, openalex_id)
            
            domains = set()
            fields = set()
            subfields = set()

            for work in works:
                topics = work.get('topics', [])
                for topic in topics:
                    if topic:
                        domain = topic.get('domain', {}).get('display_name')
                        field = topic.get('field', {}).get('display_name')
                        topic_subfields = [sf.get('display_name') for sf in topic.get('subfields', [])]

                        if domain:
                            domains.add(domain)
                        if field:
                            fields.add(field)
                        subfields.update(sf for sf in topic_subfields if sf)

            return list(domains), list(fields), list(subfields)

        except Exception as e:
            logger.error(f"Error getting domains for {first_name} {last_name}: {e}")
            return [], [], []

    async def get_expert_works(self, session: aiohttp.ClientSession, openalex_id: str, 
                         retries: int = 3, delay: int = 5) -> List[Dict]:
        """Fetch expert works from OpenAlex."""
        url = f"{self.base_url}/works"
        params = {
            'filter': f"authorships.author.id:{openalex_id}",
            'per-page': 50
        }

        for attempt in range(retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
                    elif response.status == 429:
                        wait_time = delay * (attempt + 1)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Failed to fetch works: Status {response.status}")
                        if attempt < retries - 1:
                            await asyncio.sleep(delay)
                            continue
                        break

            except Exception as e:
                logger.error(f"Error fetching works: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    continue
                break

        return []

    

    async def process_publications(self, pub_processor: PublicationProcessor, source: str = 'openalex'):
        """Process publications for experts with ORCID."""
        try:
            experts = self.db.execute("""
                SELECT id, first_name, last_name, orcid
                FROM experts_expert
                WHERE orcid IS NOT NULL AND orcid != ''
                AND first_name <> 'Unknown' AND last_name <> 'Unknown'
            """)
            
            if not experts:
                logger.info("No experts found with ORCID")
                return
                
            publication_count = 0
            TARGET_PUBLICATIONS = 3
                
            async with aiohttp.ClientSession() as session:
                for expert_id, first_name, last_name, orcid in experts:
                    if publication_count >= TARGET_PUBLICATIONS:
                        break
                    
                    try:
                        fetched_works = await self._fetch_expert_publications(
                            session, orcid,
                            per_page=min(5, TARGET_PUBLICATIONS - publication_count)
                        )
                        
                        for work in fetched_works:
                            if publication_count >= TARGET_PUBLICATIONS:
                                break
                                
                            try:
                                # Extract DOI or URL for the work
                                doi = work.get('doi')
                                urls = work.get('alternate_host_venues', [])
                                primary_url = None
                                if urls:
                                    primary_url = urls[0].get('url')
                                if not primary_url:
                                    primary_url = work.get('id', '').replace('https://openalex.org/', 'https://explore.openalex.org/')
                                    
                                # Store either DOI or URL in the doi field
                                work['doi'] = doi if doi else primary_url
                                    
                                self.db.execute("BEGIN")
                                # Process the work and get the resource_id
                                resource_id = self.db.add_publication(
                                    title=work.get('title', ''),
                                    doi=work['doi'],
                                    summary=work.get('abstract', '')[:500],
                                    source=source,
                                    type='publications',
                                    authors=work.get('authors', []),
                                    domains=work.get('domains', []),
                                    publication_year=work.get('publication_year')
                                )
                                
                                if resource_id:
                                    # Link the expert with the resource using expert_resource_links
                                    self.db.execute("""
                                        INSERT INTO expert_resource_links 
                                        (expert_id, resource_id, author_position, confidence_score)
                                        VALUES (%s, %s, %s, %s)
                                        ON CONFLICT (expert_id, resource_id) DO NOTHING
                                    """, (expert_id, resource_id, 1, 1.0))
                                    
                                    publication_count += 1
                                    self.db.execute("COMMIT")
                                    logger.info(f"Processed publication {publication_count}/{TARGET_PUBLICATIONS}")
                                else:
                                    self.db.execute("ROLLBACK")
                            except Exception as e:
                                self.db.execute("ROLLBACK")
                                logger.error(f"Error processing publication: {e}")
                                continue
                    except Exception as e:
                        logger.error(f"Error processing expert {first_name} {last_name}: {e}")
                        continue
            
            logger.info("Publications processing completed")
                    
        except Exception as e:
            logger.error(f"Error processing publications: {e}")
        
        
        
    async def _fetch_expert_publications(self, session: aiohttp.ClientSession, orcid: str,
                                    per_page: int = 5) -> List[Dict[str, Any]]:
        """Fetch publications for an expert."""
        try:
            url = f"{self.base_url}/works"
            params = {
                'filter': f"author.orcid:{orcid}",
                'per-page': per_page
            }
                
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                elif response.status == 429:
                    logger.warning("Rate limit hit, waiting before retry")
                    await asyncio.sleep(5)
                    return []
                else:
                    logger.error(f"Failed to fetch publications: Status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching publications: {e}")
            return []

    def get_expert_openalex_data(self, first_name: str, last_name: str) -> Tuple[str, str]:
        """Get expert's ORCID and OpenAlex ID."""
        try:
            url = f"{self.base_url}/authors"
            params = {
                "search": f"{first_name} {last_name}",
                "filter": "display_name.search:" + f'"{first_name} {last_name}"'
            }
            
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        if results:
                            author = results[0]
                            return author.get('orcid', ''), author.get('id', '')
                    
                    elif response.status_code == 429:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"Rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                        
                except requests.RequestException as e:
                    logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                    if attempt < 2:
                        time.sleep(5)
                    continue
                
        except Exception as e:
            logger.error(f"Error fetching data for {first_name} {last_name}: {e}")
        return '', ''

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            if hasattr(self, 'db'):
                self.db.close()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing resources: {e}")

    async def _validate_expert(self, expert_id: int, first_name: str, last_name: str) -> bool:
        """Validate expert data."""
        try:
            if not all([expert_id, first_name, last_name]):
                logger.warning(f"Invalid expert data: id={expert_id}, name={first_name} {last_name}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating expert data: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()