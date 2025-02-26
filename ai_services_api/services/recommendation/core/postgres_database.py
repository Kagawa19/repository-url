import os
import psycopg2
from psycopg2 import sql
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from urllib.parse import urlparse
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

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
        return {
            'host': os.getenv('POSTGRES_HOST', '167.86.85.127'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
        }

def get_db_connection():
    """Create a connection to PostgreSQL database."""
    params = get_connection_params()
    try:
        conn = psycopg2.connect(**params)
        logger.info(f"Successfully connected to database: {params['dbname']}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to the database: {e}")
        raise

async def normalize_expertise(expertise_list: List[str]) -> Dict[str, Any]:
    """
    Use Gemini to normalize and categorize expertise
    Returns data in JSONB compatible format
    """
    if not expertise_list:
        return {
            "domains": [],
            "fields": [],
            "skills": [],
            "keywords": []
        }

    prompt = f"""
    Analyze these expertise items: {', '.join(expertise_list)}
    Categorize them into:
    1. Broad domains (main research areas)
    2. Specific fields
    3. Technical skills
    4. Related keywords
    
    Return as a JSON structure with these exact keys:
    {{
        "domains": [],
        "fields": [],
        "skills": [],
        "keywords": []
    }}
    """

    try:
        response = model.generate_content(prompt)
        categories = json.loads(response.text)  # Using json.loads instead of eval
        logger.info("Successfully normalized expertise using Gemini")
        return categories
    except Exception as e:
        logger.error(f"Error normalizing expertise: {e}")
        return {
            "domains": expertise_list[:2],
            "fields": expertise_list[2:4],
            "skills": expertise_list[4:],
            "keywords": expertise_list
        }

async def insert_expert(conn, expert_data: Dict[str, Any]):
    """Insert expert with JSONB data handling and analytics tracking."""
    start_time = time.time()
    try:
        with conn.cursor() as cur:
            # Get expertise data
            expertise_list = expert_data.get('knowledge_expertise', [])
            
            # Normalize expertise using Gemini
            normalized_expertise = await normalize_expertise(expertise_list)
            
            # Prepare JSONB data
            expertise_jsonb = json.dumps(expertise_list)
            normalized_jsonb = json.dumps(normalized_expertise)
            
            # Extract name components
            full_name = expert_data.get('display_name', '').split()
            first_name = full_name[0] if full_name else 'Unknown'
            last_name = ' '.join(full_name[1:]) if len(full_name) > 1 else 'Unknown'
            
            # Insert with JSONB handling
            cur.execute("""
                INSERT INTO experts_expert (
                    id,
                    first_name,
                    last_name,
                    knowledge_expertise,
                    domains,
                    fields,
                    subfields,
                    normalized_expertise,
                    last_updated
                ) VALUES (
                    %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    knowledge_expertise = EXCLUDED.knowledge_expertise,
                    domains = EXCLUDED.domains,
                    fields = EXCLUDED.fields,
                    subfields = EXCLUDED.subfields,
                    normalized_expertise = EXCLUDED.normalized_expertise,
                    last_updated = NOW()
            """, (
                expert_data.get('id'),
                first_name,
                last_name,
                expertise_jsonb,
                json.dumps(expert_data.get('domains', [])),
                json.dumps(expert_data.get('fields', [])),
                json.dumps(expert_data.get('subfields', [])),
                normalized_jsonb
            ))
            
            # Record processing metrics
            await record_expert_processing(conn, expert_data.get('id'), {
                'processing_time': time.time() - start_time,
                'domains_count': len(normalized_expertise['domains']),
                'fields_count': len(normalized_expertise['fields']),
                'success': True
            })
            
            conn.commit()
            logger.info(f"Successfully inserted/updated expert: {expert_data.get('id')}")
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting expert data: {e}")
        # Record error in processing metrics
        await record_expert_processing(conn, expert_data.get('id'), {
            'processing_time': time.time() - start_time,
            'success': False,
            'error_message': str(e)
        })
        raise

async def get_expert(conn, expert_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve expert data with JSONB handling"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    first_name,
                    last_name,
                    knowledge_expertise,
                    domains,
                    fields,
                    subfields,
                    normalized_expertise,
                    last_updated
                FROM experts_expert
                WHERE id = %s
            """, (expert_id,))
            
            result = cur.fetchone()
            if result:
                return {
                    'id': result[0],
                    'first_name': result[1],
                    'last_name': result[2],
                    'knowledge_expertise': result[3],
                    'domains': result[4],
                    'fields': result[5],
                    'subfields': result[6],
                    'normalized_expertise': result[7],
                    'last_updated': result[8]
                }
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving expert data: {e}")
        return None

async def update_expert_expertise(conn, expert_id: str, new_expertise: List[str]):
    """Update expert's expertise with JSONB handling"""
    try:
        # Get current expertise for comparison
        current_expert = await get_expert(conn, expert_id)
        old_expertise = current_expert.get('knowledge_expertise', []) if current_expert else []
        
        # Normalize new expertise
        normalized = await normalize_expertise(new_expertise)
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE experts_expert
                SET 
                    knowledge_expertise = %s::jsonb,
                    normalized_expertise = %s::jsonb,
                    last_updated = NOW()
                WHERE id = %s
            """, (
                json.dumps(new_expertise),
                json.dumps(normalized),
                expert_id
            ))
            
            # Record the update
            await record_expertise_update(conn, expert_id, old_expertise, new_expertise)
            
            conn.commit()
            logger.info(f"Successfully updated expertise for expert: {expert_id}")
            return True
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating expert expertise: {e}")
        return False

async def record_expert_processing(conn, expert_id: str, processing_data: Dict[str, Any]):
    """Record processing metrics with JSONB data"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO expert_processing_logs (
                    expert_id,
                    processing_time,
                    domains_count,
                    fields_count,
                    success,
                    error_message,
                    metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """, (
                expert_id,
                processing_data.get('processing_time', 0),
                processing_data.get('domains_count', 0),
                processing_data.get('fields_count', 0),
                processing_data.get('success', True),
                processing_data.get('error_message'),
                json.dumps(processing_data.get('metadata', {}))
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error recording expert processing: {e}")
        conn.rollback()

async def get_expert(conn, expert_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve expert data from PostgreSQL database with normalized expertise"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    first_name,
                    last_name,
                    knowledge_expertise,
                    domains,
                    fields,
                    subfields,
                    normalized_domains,
                    normalized_fields,
                    normalized_skills,
                    keywords,
                    last_updated
                FROM experts_expert
                WHERE id = %s
            """, (expert_id,))
            
            result = cur.fetchone()
            if result:
                return {
                    'id': result[0],
                    'first_name': result[1],
                    'last_name': result[2],
                    'knowledge_expertise': result[3],
                    'domains': result[4],
                    'fields': result[5],
                    'subfields': result[6],
                    'normalized_domains': result[7],
                    'normalized_fields': result[8],
                    'normalized_skills': result[9],
                    'keywords': result[10],
                    'last_updated': result[11]
                }
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving expert data: {e}")
        return None

async def search_experts(conn, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search experts based on expertise, domains, fields, or keywords
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    first_name,
                    last_name,
                    knowledge_expertise,
                    normalized_domains,
                    normalized_fields,
                    normalized_skills
                FROM experts_expert
                WHERE 
                    to_tsvector('english', 
                        array_to_string(knowledge_expertise, ' ') || ' ' ||
                        array_to_string(normalized_domains, ' ') || ' ' ||
                        array_to_string(normalized_fields, ' ') || ' ' ||
                        array_to_string(normalized_skills, ' ')
                    ) @@ plainto_tsquery('english', %s)
                LIMIT %s
            """, (query, limit))
            
            results = cur.fetchall()
            experts = []
            for result in results:
                experts.append({
                    'id': result[0],
                    'first_name': result[1],
                    'last_name': result[2],
                    'knowledge_expertise': result[3],
                    'normalized_domains': result[4],
                    'normalized_fields': result[5],
                    'normalized_skills': result[6]
                })
            return experts
            
    except Exception as e:
        logger.error(f"Error searching experts: {e}")
        return []


async def record_expertise_update(conn, expert_id: str, old_expertise: List[str], 
                                new_expertise: List[str]):
    """Record expertise updates with JSONB handling"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO expertise_update_logs (
                    expert_id,
                    old_expertise,
                    new_expertise,
                    domains_changed,
                    fields_changed,
                    update_metadata
                ) VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb)
            """, (
                expert_id,
                json.dumps(old_expertise),
                json.dumps(new_expertise),
                len(set(new_expertise) - set(old_expertise)),
                len(set(old_expertise) - set(new_expertise)),
                json.dumps({
                    'timestamp': datetime.utcnow().isoformat(),
                    'added_items': list(set(new_expertise) - set(old_expertise)),
                    'removed_items': list(set(old_expertise) - set(new_expertise))
                })
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error recording expertise update: {e}")
        conn.rollback()
