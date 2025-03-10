from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Dict, Optional
from ai_services_api.services.message.core.database import get_db_connection
from datetime import datetime
import logging

from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
import json

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

class PublicationResponse(BaseModel):
    id: int
    doi: Optional[str] = None
    title: str
    summary: Optional[str] = None
    domains: Optional[List[str]] = None
    topics: Optional[Dict] = None
    type: Optional[str] = None
    authors: Optional[List[Dict]] = None
    publication_year: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@router.get("/", response_model=List[PublicationResponse])
async def get_publications(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    domain: Optional[str] = None,
    year: Optional[str] = None,
    author: Optional[str] = None,
    search: Optional[str] = None,
    type: Optional[str] = None
):
    """
    Retrieve publications with optional filtering.
    """
    conn = None
    cur = None
    start_time = datetime.utcnow()
    
    logger.info(f"Fetching publications with filters - domain: {domain}, year: {year}, author: {author}, search: {search}, type: {type}")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        logger.debug("Database connection established successfully")
        
        # Build query with flexible filtering
        query = """
            SELECT 
                id, doi, title, summary, domains, topics, 
                type, authors, publication_year, source,
                created_at, updated_at
            FROM resources_resource
            WHERE 1=1
        """
        params = []
        
        # Add filters if provided
        if domain:
            query += " AND %s = ANY(domains)"
            params.append(domain)
        
        if year:
            query += " AND publication_year = %s"
            params.append(year)
        
        if author:
            query += " AND authors::jsonb @> %s::jsonb"
            # Search for author name in the JSON array
            author_json = json.dumps([{"name": author}])
            params.append(author_json)
        
        if search:
            query += " AND (title ILIKE %s OR summary ILIKE %s)"
            search_param = f"%{search}%"
            params.extend([search_param, search_param])
        
        if type:
            query += " AND type = %s"
            params.append(type)
        
        # Add pagination
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        logger.debug(f"Executing query: {query} with params: {params}")
        cur.execute(query, tuple(params))
        
        publications = cur.fetchall()
        logger.info(f"Found {len(publications)} publications matching criteria")
        
        # Process the results
        results = []
        for pub in publications:
            # Convert JSON strings to Python objects
            if pub.get('authors') and isinstance(pub['authors'], str):
                try:
                    pub['authors'] = json.loads(pub['authors'])
                except:
                    pub['authors'] = []
            
            if pub.get('topics') and isinstance(pub['topics'], str):
                try:
                    pub['topics'] = json.loads(pub['topics'])
                except:
                    pub['topics'] = {}
            
            results.append(pub)
        
        return results
    
    except Exception as e:
        logger.error(f"Error retrieving publications: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving publications: {str(e)}")
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.debug("Database connections closed")
        logger.info(f"Request completed in {(datetime.utcnow() - start_time).total_seconds()} seconds")

@router.get("/{publication_id}", response_model=PublicationResponse)
async def get_publication_by_id(
    publication_id: int,
    request: Request
):
    """
    Retrieve a specific publication by ID.
    """
    conn = None
    cur = None
    
    logger.info(f"Fetching publication with ID: {publication_id}")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                id, doi, title, summary, domains, topics, 
                type, authors, publication_year, source,
                created_at, updated_at
            FROM resources_resource
            WHERE id = %s
        """
        
        cur.execute(query, (publication_id,))
        publication = cur.fetchone()
        
        if not publication:
            logger.warning(f"Publication with ID {publication_id} not found")
            raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
        
        # Convert JSON strings to Python objects
        if publication.get('authors') and isinstance(publication['authors'], str):
            try:
                publication['authors'] = json.loads(publication['authors'])
            except:
                publication['authors'] = []
        
        if publication.get('topics') and isinstance(publication['topics'], str):
            try:
                publication['topics'] = json.loads(publication['topics'])
            except:
                publication['topics'] = {}
        
        logger.info(f"Successfully retrieved publication: {publication['title']}")
        return publication
    
    except Exception as e:
        logger.error(f"Error retrieving publication: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving publication: {str(e)}")
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.debug("Database connections closed")

@router.get("/domains", response_model=List[str])
async def get_publication_domains(request: Request):
    """
    Retrieve all unique domains from publications.
    """
    conn = None
    cur = None
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT DISTINCT unnest(domains) as domain
            FROM resources_resource
            WHERE domains IS NOT NULL
            ORDER BY domain
        """
        
        cur.execute(query)
        domains = [row['domain'] for row in cur.fetchall()]
        
        logger.info(f"Retrieved {len(domains)} unique domains")
        return domains
    
    except Exception as e:
        logger.error(f"Error retrieving domains: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving domains: {str(e)}")
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()