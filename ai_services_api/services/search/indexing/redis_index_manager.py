import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import redis
from dotenv import load_dotenv
import os
import time
import json
from src.utils.db_utils import DatabaseConnector


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ExpertRedisIndexManager:
    def __init__(self):
        """Initialize Redis index manager for experts."""
        try:
            self.db = DatabaseConnector()  # Initialize the database connector
            load_dotenv()
            self.embedding_model = SentenceTransformer(
                os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            )
            self.setup_redis_connections()
            logger.info("ExpertRedisIndexManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ExpertRedisIndexManager: {e}")
            raise


    def setup_redis_connections(self):
        """Setup Redis connections with retry logic."""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                
                # Initialize Redis connections
                self.redis_text = redis.StrictRedis.from_url(
                    self.redis_url, 
                    decode_responses=True,
                    db=0
                )
                self.redis_binary = redis.StrictRedis.from_url(
                    self.redis_url, 
                    decode_responses=False,
                    db=0
                )
                
                # Test connections
                self.redis_text.ping()
                self.redis_binary.ping()
                
                logger.info("Redis connections established successfully")
                return
                
            except redis.ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error("Failed to connect to Redis after maximum retries")
                    raise
                logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)

    


    def _parse_jsonb(self, data):
        """Parse JSONB data safely."""
        if not data:
            return {}
        try:
            if isinstance(data, str):
                return json.loads(data)
            return data
        except:
            return {}

    

    def fetch_resources(self) -> List[Dict[str, Any]]:
        """Fetch all resource data from database."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            conn = None
            cur = None
            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'resources_resource'
                        );
                    """)
                    if not cur.fetchone()[0]:
                        logger.warning("resources_resource table does not exist yet")
                        return []
                    
                    # Fetch all resource records
                    cur.execute("""
                        SELECT 
                            id,
                            title,
                            doi,
                            authors,
                            domains,
                            type,
                            publication_year,
                            summary,
                            source,
                            field,
                            subfield,
                            created_at
                        FROM resources_resource
                        WHERE id IS NOT NULL
                    """)
                    
                    resources = [{
                        'id': row[0],
                        'title': row[1] or '',
                        'doi': row[2],
                        'authors': self._parse_jsonb(row[3]),
                        'domains': row[4] or [],
                        'type': row[5] or 'publication',
                        'publication_year': row[6],
                        'summary': row[7] or '',
                        'source': row[8] or 'unknown',
                        'field': row[9] or '',
                        'subfield': row[10] or '',
                        'created_at': row[11].isoformat() if row[11] else None
                    } for row in cur.fetchall()]
                    
                    logger.info(f"Fetched {len(resources)} resources from database")
                    return resources
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    raise
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()

    def _create_resource_text_content(self, resource: Dict[str, Any]) -> str:
        """Create combined text content for resource embedding with additional safeguards."""
        try:
            # Start with title as the most important information
            text_parts = []
            if resource.get('title'):
                text_parts.append(f"Title: {str(resource['title']).strip()}")
            else:
                text_parts.append("Title: Unknown Resource")

            # Add publication details
            if resource.get('publication_year'):
                text_parts.append(f"Year: {resource['publication_year']}")
                
            if resource.get('doi'):
                text_parts.append(f"DOI: {resource['doi']}")
                
            # Add type and source
            if resource.get('type'):
                text_parts.append(f"Type: {resource['type']}")
                
            if resource.get('source'):
                text_parts.append(f"Source: {resource['source']}")
                
            # Add field and subfield
            if resource.get('field'):
                field_text = f"Field: {resource['field']}"
                if resource.get('subfield'):
                    field_text += f", Subfield: {resource['subfield']}"
                text_parts.append(field_text)
            
            # Handle authors
            authors = resource.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    authors_text = ", ".join([str(author).strip() for author in authors if author])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
                elif isinstance(authors, str):
                    text_parts.append(f"Authors: {authors}")
                elif isinstance(authors, dict):
                    # Handle case where authors might be a dictionary
                    authors_text = ", ".join([str(value).strip() for value in authors.values() if value])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
            
            # Add domains as keywords
            domains = resource.get('domains', [])
            if domains and isinstance(domains, list):
                domains_text = ", ".join([str(domain).strip() for domain in domains if domain])
                if domains_text:
                    text_parts.append(f"Domains: {domains_text}")
            
            # Add summary - most important for semantic search
            if resource.get('summary'):
                summary = str(resource['summary']).strip()
                if summary:
                    text_parts.append(f"Summary: {summary}")
            
            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Resource"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for resource {resource.get('id', 'Unknown')}: {e}")
            return "Error Processing Resource"

    # Here are the methods that need to be modified:

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """Fetch all expert data from database with only required columns."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            conn = None
            cur = None
            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'experts_expert'
                        );
                    """)
                    if not cur.fetchone()[0]:
                        logger.warning("experts_expert table does not exist yet")
                        return []
                    
                    # Updated query to use only required columns
                    cur.execute("""
                        SELECT 
                            id,
                            first_name,
                            last_name,
                            knowledge_expertise
                        FROM experts_expert
                        WHERE id IS NOT NULL
                    """)
                    
                    experts = [{
                        'id': row[0],
                        'first_name': row[1] or '',
                        'last_name': row[2] or '',
                        'knowledge_expertise': self._parse_jsonb(row[3])
                    } for row in cur.fetchall()]
                    
                    logger.info(f"Fetched {len(experts)} experts from database")
                    return experts
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    raise
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()

    def _create_text_content(self, expert: Dict[str, Any]) -> str:
        """Create combined text content for embedding with only required fields."""
        try:
            # Ensure we have at least basic information
            name_parts = []
            if expert.get('first_name'):
                name_parts.append(str(expert['first_name']).strip())
            if expert.get('last_name'):
                name_parts.append(str(expert['last_name']).strip())
            
            # Start with basic identity
            text_parts = []
            if name_parts:
                text_parts.append(f"Name: {' '.join(name_parts)}")
            else:
                text_parts.append("Name: Unknown Expert")

            # Handle knowledge expertise
            expertise = expert.get('knowledge_expertise', {})
            if expertise and isinstance(expertise, dict):
                for key, value in expertise.items():
                    if value:
                        if isinstance(value, list):
                            # Clean list values
                            clean_values = [str(v).strip() for v in value if v is not None]
                            clean_values = [v for v in clean_values if v]  # Remove empty strings
                            if clean_values:
                                text_parts.append(f"{key.title()}: {' | '.join(clean_values)}")
                        elif isinstance(value, (str, int, float)):
                            # Handle single values
                            clean_value = str(value).strip()
                            if clean_value:
                                text_parts.append(f"{key.title()}: {clean_value}")
            elif expertise and isinstance(expertise, list):
                # Handle case where knowledge_expertise is a list
                clean_values = [str(v).strip() for v in expertise if v is not None]
                clean_values = [v for v in clean_values if v]  # Remove empty strings
                if clean_values:
                    text_parts.append(f"Expertise: {' | '.join(clean_values)}")

            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Expert Profile"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for expert {expert.get('id', 'Unknown')}: {e}")
            return "Error Processing Expert Profile"

    def _store_expert_data(self, expert: Dict[str, Any], text_content: str, 
                        embedding: np.ndarray) -> None:
        """Store expert data in Redis with only required fields."""
        base_key = f"expert:{expert['id']}"
        
        pipeline = self.redis_text.pipeline()
        try:
            # Store text content
            pipeline.set(f"text:{base_key}", text_content)
            
            # Store embedding as binary
            self.redis_binary.set(
                f"emb:{base_key}", 
                embedding.astype(np.float32).tobytes()
            )
            
            # Store only required metadata
            metadata = {
                'id': str(expert['id']),  # Ensure id is string
                'name': f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip(),
                'first_name': str(expert.get('first_name', '')),
                'last_name': str(expert.get('last_name', '')),
                'expertise': json.dumps(expert.get('knowledge_expertise', {}))
            }
            pipeline.hset(f"meta:{base_key}", mapping=metadata)
            
            pipeline.execute()
            
        except Exception as e:
            pipeline.reset()
            raise e

    def get_expert_metadata(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve expert metadata from Redis."""
        try:
            metadata = self.redis_text.hgetall(f"meta:expert:{expert_id}")
            if metadata:
                # Parse JSON fields
                if metadata.get('expertise'):
                    metadata['expertise'] = json.loads(metadata['expertise'])
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error retrieving expert metadata: {e}")
            return None

    def create_redis_index(self) -> bool:
        """Create Redis indexes for experts with only required fields."""
        try:
            logger.info("Creating Redis indexes for experts...")
            
            # Step 1: Process experts
            experts = self.fetch_experts()
            
            if not experts:
                logger.warning("No experts found to index")
                return False
            
            success_count = 0
            error_count = 0
            
            # Step 2: Index experts
            logger.info(f"Processing {len(experts)} experts for indexing")
            for expert in experts:
                try:
                    expert_id = expert.get('id', 'Unknown')
                    logger.info(f"Processing expert {expert_id}")
                    
                    # Create text content with additional logging
                    text_content = self._create_text_content(expert)
                    if not text_content or text_content.isspace():
                        logger.warning(f"Empty text content generated for expert {expert_id}")
                        continue

                    # Log the text content for debugging
                    logger.debug(f"Text content for expert {expert_id}: {text_content[:100]}...")
                    
                    # Generate embedding with explicit error handling
                    try:
                        if not isinstance(text_content, str):
                            text_content = str(text_content)
                        embedding = self.embedding_model.encode(text_content)
                        if embedding is None or not isinstance(embedding, np.ndarray):
                            logger.error(f"Invalid embedding generated for expert {expert_id}")
                            continue
                    except Exception as embed_err:
                        logger.error(f"Embedding generation failed for expert {expert_id}: {embed_err}")
                        continue
                    
                    # Store in Redis
                    self._store_expert_data(expert, text_content, embedding)
                    success_count += 1
                    logger.info(f"Successfully indexed expert {expert_id}")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error indexing expert {expert.get('id', 'Unknown')}: {str(e)}")
                    continue
            
            logger.info(f"Indexing complete. Successes: {success_count}, Failures: {error_count}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Fatal error in create_redis_index: {e}")
            return False

    def clear_redis_indexes(self) -> bool:
        """Clear all expert Redis indexes."""
        try:
            patterns = ['text:expert:*', 'emb:expert:*', 'meta:expert:*']
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            logger.info("Cleared all expert Redis indexes")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis indexes: {e}")
            return False

    def _store_resource_data(self, resource: Dict[str, Any], text_content: str, 
                        embedding: np.ndarray) -> None:
        """Store resource data in Redis."""
        base_key = f"resource:{resource['id']}"
        
        pipeline = self.redis_text.pipeline()
        try:
            # Store text content
            pipeline.set(f"text:{base_key}", text_content)
            
            # Store embedding as binary
            self.redis_binary.set(
                f"emb:{base_key}", 
                embedding.astype(np.float32).tobytes()
            )
            
            # Store metadata
            metadata = {
                'id': str(resource['id']),
                'title': str(resource.get('title', '')),
                'doi': str(resource.get('doi', '')),
                'publication_year': str(resource.get('publication_year', '')),
                'type': str(resource.get('type', 'publication')),
                'source': str(resource.get('source', '')),
                'field': str(resource.get('field', '')),
                'subfield': str(resource.get('subfield', '')),
                'created_at': resource.get('created_at', ''),
                'summary_snippet': str(resource.get('summary', ''))[:100] if resource.get('summary') else ''
            }
            
            # Add authors as a JSON string if present
            if resource.get('authors'):
                metadata['authors'] = json.dumps(resource.get('authors', []))
                
            # Add domains as a JSON string if present
            if resource.get('domains'):
                metadata['domains'] = json.dumps(resource.get('domains', []))
            
            pipeline.hset(f"meta:{base_key}", mapping=metadata)
            
            pipeline.execute()
            
        except Exception as e:
            pipeline.reset()
            raise e

    
    

    

    def get_expert_embedding(self, expert_id: str) -> Optional[np.ndarray]:
        """Retrieve expert embedding from Redis."""
        try:
            embedding_bytes = self.redis_binary.get(f"emb:expert:{expert_id}")
            if embedding_bytes:
                return np.frombuffer(embedding_bytes, dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Error retrieving expert embedding: {e}")
            return None

    

    def close(self):
        """Close Redis connections."""
        try:
            if hasattr(self, 'redis_text'):
                self.redis_text.close()
            if hasattr(self, 'redis_binary'):
                self.redis_binary.close()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

    def __del__(self):
        """Ensure connections are closed on deletion."""
        self.close()
