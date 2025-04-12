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
        """Initialize Redis index manager for experts with robust model loading."""
        try:
            self.db = DatabaseConnector()  # Initialize the database connector
            load_dotenv()

            # Define the path to the pre-downloaded model directory
            model_path = '/app/models/sentence-transformers/all-MiniLM-L6-v2'

            try:
                logger.info(f"Loading SentenceTransformer model from: {model_path}")
                self.embedding_model = SentenceTransformer(
                    model_path,
                    local_files_only=True  # Ensures only local files are used
                )
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                logger.warning("Falling back to None; manual embedding will be used.")
                self.embedding_model = None

            # Setup Redis connections
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

    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model is not available."""
        logger.info("Creating fallback embedding")
        # Create a deterministic embedding based on character values
        embedding = np.zeros(384)  # Standard dimension for simple embeddings
        for i, char in enumerate(text):
            embedding[i % len(embedding)] += ord(char) / 1000
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
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

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """Fetch all expert data from database including their resource links."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            conn = None
            cur = None
            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Check if tables exist
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'experts_expert'
                        ) AS expert_table_exists,
                        EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'expert_resource_links'
                        ) AS links_table_exists;
                    """)
                    exists = cur.fetchone()
                    if not exists[0]:  # experts_expert table doesn't exist
                        logger.warning("experts_expert table does not exist yet")
                        return []
                    
                    # Fetch experts with their linked resources in one query
                    query = """
                        SELECT 
                            e.id,
                            e.first_name,
                            e.last_name,
                            e.knowledge_expertise,
                            COALESCE(
                                json_agg(
                                    json_build_object(
                                        'resource_id', erl.resource_id,
                                        'confidence', erl.confidence_score
                                    ) 
                                    ORDER BY erl.confidence_score DESC
                                ) FILTER (WHERE erl.resource_id IS NOT NULL),
                                '[]'::json
                            ) AS linked_resources
                        FROM experts_expert e
                        LEFT JOIN expert_resource_links erl ON e.id = erl.expert_id
                        WHERE e.id IS NOT NULL
                        GROUP BY e.id, e.first_name, e.last_name, e.knowledge_expertise
                    """ if exists[1] else """
                        SELECT 
                            id,
                            first_name,
                            last_name,
                            knowledge_expertise
                        FROM experts_expert
                        WHERE id IS NOT NULL
                    """
                    
                    cur.execute(query)
                    experts = []
                    for row in cur.fetchall():
                        expert = {
                            'id': row[0],
                            'first_name': row[1] or '',
                            'last_name': row[2] or '',
                            'knowledge_expertise': self._parse_jsonb(row[3]),
                            'linked_resources': row[4] if len(row) > 4 else []
                        }
                        experts.append(expert)
                    
                    logger.info(f"Fetched {len(experts)} experts with resource links")
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
        """
        Stores expert data with APHRC classification.
        All experts from experts_expert table are considered APHRC experts.
        """
        # All experts are APHRC (from experts_expert table)
        # Changed key pattern from "expert:aphrc:{expert_id}" to "aphrc_expert:{expert_id}"
        base_key = f"aphrc_expert:{expert['id']}"
        
        pipeline = self.redis_text.pipeline()
        try:
            # Store text content
            pipeline.set(f"text:{base_key}", text_content)
            
            # Store embedding
            self.redis_binary.set(
                f"emb:{base_key}",
                embedding.astype(np.float32).tobytes()
            )
            
            # Enhanced metadata with explicit APHRC marking
            metadata = {
                'id': str(expert['id']),
                'first_name': str(expert.get('first_name', '')),
                'last_name': str(expert.get('last_name', '')),
                'expertise': json.dumps(expert.get('knowledge_expertise', {})),
                'is_aphrc': 'true',  # Explicit marker
                'email': str(expert.get('email', '')),
                'position': str(expert.get('position', '')),
                # ... other existing fields ...
            }
            pipeline.hset(f"meta:{base_key}", mapping=metadata)
            
            # Store resource links (both APHRC and global publications)
            if expert.get('linked_resources'):
                links_key = f"links:expert:{expert['id']}:resources"
                pipeline.delete(links_key)  # Clear existing
                
                for link in expert['linked_resources']:
                    if link.get('resource_id') and link.get('confidence'):
                        pipeline.zadd(
                            links_key,
                            {str(link['resource_id']): float(link['confidence'])}
                        )
                        
                        # Reverse link (resource knows its APHRC experts)
                        res_key = f"links:resource:{link['resource_id']}:experts"
                        pipeline.zadd(
                            res_key,
                            {str(expert['id']): float(link['confidence'])}
                        )
            
            pipeline.execute()
            
        except Exception as e:
            pipeline.reset()
            logger.error(f"Error storing expert {expert.get('id')}: {e}")
            raise
    

    def clear_redis_indexes(self) -> bool:
        """Clear all Redis indexes for both experts and publications."""
        try:
            # Clear both old and new patterns to ensure complete cleanup
            
            # Clear expert indexes - old patterns
            old_expert_patterns = ['text:expert:*', 'emb:expert:*', 'meta:expert:*']
            for pattern in old_expert_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear expert indexes - new patterns
            new_expert_patterns = ['text:aphrc_expert:*', 'emb:aphrc_expert:*', 'meta:aphrc_expert:*']
            for pattern in new_expert_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear publication/resource indexes - old patterns
            old_resource_patterns = [
                'text:resource:*', 'emb:resource:*', 'meta:resource:*',
                'text:publication:*', 'emb:publication:*', 'meta:publication:*'
            ]
            for pattern in old_resource_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear publication/resource indexes - new patterns
            new_resource_patterns = [
                'text:expert_resource:*', 'emb:expert_resource:*', 'meta:expert_resource:*'
            ]
            for pattern in new_resource_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear resource links
            link_patterns = ['links:expert:*', 'links:resource:*', 'expert:*:resources']
            for pattern in link_patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            logger.info("Cleared all expert and publication Redis indexes (both old and new patterns)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis indexes: {e}")
            return False
        

    # 1. Add this method to ExpertRedisIndexManager class:

    def create_expert_redis_index(self) -> bool:
        """Create Redis indexes for experts including their resource links."""
        try:
            logger.info("Starting expert indexing including resource links...")
            
            experts = self.fetch_experts()
            
            if not experts:
                logger.warning("No experts found to index")
                return False
            
            success_count = 0
            error_count = 0
            
            for expert in experts:
                try:
                    expert_id = expert.get('id', 'Unknown')
                    logger.info(f"Processing expert {expert_id} with {len(expert.get('linked_resources', []))} resources")
                    
                    # Create text content
                    text_content = self._create_text_content(expert)
                    if not text_content or text_content.isspace():
                        logger.warning(f"Empty text content for expert {expert_id}")
                        continue

                    # Generate embedding
                    try:
                        embedding = (
                            self.embedding_model.encode(text_content) 
                            if self.embedding_model is not None 
                            else self._create_fallback_embedding(text_content)
                        )
                    except Exception as embed_err:
                        logger.error(f"Embedding failed for expert {expert_id}: {embed_err}")
                        embedding = self._create_fallback_embedding(text_content)
                    
                    # Store all data including links
                    self._store_expert_data(expert, text_content, embedding)
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error indexing expert {expert_id}: {e}")
                    continue
            
            logger.info(
                f"Expert indexing complete. Success: {success_count}, "
                f"Failures: {error_count}, "
                f"Total resource links processed: {sum(len(e.get('linked_resources', [])) for e in experts)}"
            )
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Fatal error in expert indexing: {e}")
            return False

    
    def create_redis_index(self) -> bool:
        """
        Create Redis indexes for experts, publications, and their relationships.
        
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            logger.info("Starting comprehensive Redis indexing process...")
            
            # Track overall success
            overall_success = True
            
            # Step 1: Clear existing indexes (optional, but can help prevent stale data)
            self.clear_redis_indexes()
            
            # Step 2: Index experts
            experts_success = self.create_expert_redis_index()
            
            # Step 3: Index publications/resources
            publications_success = self.create_publications_redis_index()
            
            # Determine final indexing status
            if experts_success and publications_success:
                logger.info("Successfully indexed experts and publications in Redis")
                return True
            else:
                logger.error("Failed to completely index experts and publications")
                return False
            
        except Exception as final_err:
            logger.error(f"Catastrophic error during Redis indexing: {final_err}")
            return False

            
    def fetch_resources(self) -> List[Dict[str, Any]]:
        """Fetch all resources linked to experts either directly or through links."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            conn = None
            cur = None
            try:
                conn = self.db.get_connection()
                with conn.cursor() as cur:
                    # Check if required tables exist
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'resources_resource'
                        ) AS resources_exists,
                        EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'expert_resource_links'
                        ) AS links_exists;
                    """)
                    exists = cur.fetchone()
                    
                    if not exists[0]:  # resources_resource table doesn't exist
                        logger.warning("resources_resource table does not exist yet")
                        return []
                    
                    # Modified query to get all resources linked to experts
                    query = """
                        SELECT r.* 
                        FROM resources_resource r
                        WHERE (
                            -- Resources with direct expert_id that exists in experts_expert
                            (r.expert_id IS NOT NULL AND r.expert_id IN (SELECT id FROM experts_expert))
                            OR
                            -- Resources linked via expert_resource_links
                            (EXISTS (
                                SELECT 1 FROM expert_resource_links erl 
                                WHERE erl.resource_id = r.id 
                                AND erl.expert_id IN (SELECT id FROM experts_expert)
                            ))
                        )
                        ORDER BY r.id
                    """ if exists[1] else """
                        SELECT r.*
                        FROM resources_resource r
                        WHERE r.expert_id IS NOT NULL 
                        AND r.expert_id IN (SELECT id FROM experts_expert)
                        ORDER BY r.id
                    """
                    
                    cur.execute(query)
                    resources = []
                    skipped_count = 0
                    
                    for row in cur.fetchall():
                        try:
                            resource = {
                                'id': row[0],
                                'doi': str(row[1]) if row[1] is not None else '',
                                'title': str(row[2]) if row[2] is not None else '',
                                'abstract': str(row[3]) if row[3] is not None else '',
                                'summary': str(row[4]) if row[4] is not None else '',
                                'domains': self._parse_jsonb(row[5]) if row[5] else [],
                                'topics': self._parse_jsonb(row[6]) if row[6] else {},
                                'description': str(row[7]) if row[7] is not None else '',
                                'expert_id': str(row[8]) if row[8] is not None else '',
                                'type': str(row[9]) if row[9] is not None else 'publication',
                                'subtitles': self._parse_jsonb(row[10]) if row[10] else {},
                                'publishers': self._parse_jsonb(row[11]) if row[11] else {},
                                'collection': str(row[12]) if row[12] is not None else '',
                                'date_issue': str(row[13]) if row[13] is not None else '',
                                'citation': str(row[14]) if row[14] is not None else '',
                                'language': str(row[15]) if row[15] is not None else '',
                                'identifiers': self._parse_jsonb(row[16]) if row[16] else {},
                                'created_at': row[17].isoformat() if row[17] else None,
                                'updated_at': row[18].isoformat() if row[18] else None,
                                'source': str(row[19]) if row[19] is not None else 'unknown',
                                'authors': self._parse_jsonb(row[20]) if row[20] else [],
                                'publication_year': str(row[21]) if row[21] is not None else ''
                            }
                            resources.append(resource)
                        except Exception as row_error:
                            skipped_count += 1
                            logger.error(f"Error processing resource row: {row_error}")
                            continue
                    
                    logger.info(
                        f"Fetched {len(resources)} expert-linked resources from database "
                        f"(skipped {skipped_count} invalid records)"
                    )
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
        
        return []
    def create_publications_redis_index(self) -> bool:
        """Create Redis indexes for publications/resources."""
        try:
            logger.info("Starting publications indexing...")
            
            resources = self.fetch_resources()
            
            if not resources:
                logger.warning("No publications/resources found to index")
                return False
            
            logger.info(f"Processing {len(resources)} publications for indexing")
            success_count = 0
            error_count = 0
            
            for resource in resources:
                try:
                    resource_id = resource.get('id', 'Unknown')
                    logger.info(f"Processing publication {resource_id}")
                    
                    # Create text content
                    text_content = self._create_resource_text_content(resource)
                    if not text_content or text_content.isspace():
                        logger.warning(f"Empty text content generated for publication {resource_id}")
                        continue
                    
                    # Generate embedding with robust fallback
                    try:
                        embedding = (
                            self.embedding_model.encode(text_content) 
                            if self.embedding_model is not None 
                            else self._create_fallback_embedding(text_content)
                        )
                        
                        if embedding is None or not isinstance(embedding, np.ndarray):
                            raise ValueError("Invalid embedding generated")
                    except Exception as embed_err:
                        logger.error(f"Embedding generation failed for publication {resource_id}: {embed_err}")
                        # Mandatory fallback embedding
                        embedding = self._create_fallback_embedding(text_content)
                    
                    # Store in Redis
                    self._store_resource_data(resource, text_content, embedding)
                    success_count += 1
                    logger.info(f"Successfully indexed publication {resource_id}")
                    
                except Exception as resource_index_err:
                    error_count += 1
                    logger.error(f"Error indexing publication {resource.get('id', 'Unknown')}: {resource_index_err}")
            
            logger.info(f"Publication indexing complete. Successes: {success_count}, Failures: {error_count}")
            return success_count > 0
            
        except Exception as resources_fetch_err:
            logger.error(f"Failed to fetch or process publications: {resources_fetch_err}")
            return False

    def get_publications_by_expert_id(self, expert_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve publications associated with a specific expert from Redis.
        
        Args:
            expert_id: The expert's ID
            limit: Maximum number of publications to return
            
        Returns:
            List of publication dictionaries
        """
        try:
            logger.info(f"Retrieving publications for expert {expert_id}")
            publications = []
            
            # 1. First check the dedicated expert-resource links
            try:
                # Check if we have a sorted set for this expert
                links_key = f"links:expert:{expert_id}:resources"
                if self.redis_text.exists(links_key):
                    # Get resource IDs sorted by confidence (highest first)
                    resource_items = self.redis_text.zrevrange(
                        links_key, 0, limit-1, withscores=True
                    )
                    
                    # If we have results from links, process them
                    if resource_items:
                        logger.info(f"Found {len(resource_items)} linked resources for expert {expert_id}")
                        
                        for resource_id, confidence in resource_items:
                            # Get resource metadata
                            meta_key = f"meta:resource:{resource_id}"
                            if not self.redis_text.exists(meta_key):
                                continue
                                
                            meta = self.redis_text.hgetall(meta_key)
                            if meta:
                                # Parse JSON fields
                                publication = {
                                    'id': meta.get('id', ''),
                                    'title': meta.get('title', ''),
                                    'doi': meta.get('doi', ''),
                                    'abstract': meta.get('abstract', ''),
                                    'publication_year': meta.get('publication_year', ''),
                                    'confidence': float(confidence)
                                }
                                
                                # Parse authors
                                try:
                                    authors_json = meta.get('authors', '[]')
                                    publication['authors'] = json.loads(authors_json) if authors_json else []
                                except json.JSONDecodeError:
                                    publication['authors'] = [authors_json] if authors_json else []
                                
                                publications.append(publication)
                        
                        # If we found enough publications through links, return them
                        if len(publications) >= limit:
                            return publications[:limit]
            except Exception as links_error:
                logger.error(f"Error retrieving linked resources for expert {expert_id}: {links_error}")
            
            # 2. If we don't have enough publications from links, look for direct matches
            if len(publications) < limit:
                try:
                    # Scan for resources with matching expert_id
                    remaining = limit - len(publications)
                    cursor = 0
                    direct_matches = []
                    
                    while len(direct_matches) < remaining and (cursor != 0 or not direct_matches):
                        cursor, keys = self.redis_text.scan(cursor, match="meta:resource:*", count=100)
                        for key in keys:
                            # Skip if we already found this resource
                            resource_id = key.split(':')[-1]
                            if any(p.get('id') == resource_id for p in publications):
                                continue
                                
                            try:
                                # Check if this resource is linked to the expert
                                meta = self.redis_text.hgetall(key)
                                if meta.get('expert_id') == str(expert_id):
                                    # Parse JSON fields
                                    publication = {
                                        'id': meta.get('id', ''),
                                        'title': meta.get('title', ''),
                                        'doi': meta.get('doi', ''),
                                        'abstract': meta.get('abstract', ''),
                                        'publication_year': meta.get('publication_year', ''),
                                        'confidence': 0.95  # High confidence for direct expert_id match
                                    }
                                    
                                    # Parse authors
                                    try:
                                        authors_json = meta.get('authors', '[]')
                                        publication['authors'] = json.loads(authors_json) if authors_json else []
                                    except json.JSONDecodeError:
                                        publication['authors'] = [authors_json] if authors_json else []
                                    
                                    direct_matches.append(publication)
                                    
                                    # Stop if we have enough matches
                                    if len(direct_matches) >= remaining:
                                        break
                            except Exception as e:
                                logger.error(f"Error processing key {key}: {e}")
                        
                        if cursor == 0 or len(direct_matches) >= remaining:
                            break
                    
                    # Add direct matches to publications
                    publications.extend(direct_matches)
                    
                    # If we have enough publications now, return them
                    if len(publications) >= limit:
                        return publications[:limit]
                except Exception as direct_error:
                    logger.error(f"Error finding direct matches for expert {expert_id}: {direct_error}")
            
            # 3. If we still don't have enough, try author name matching
            if len(publications) < limit:
                try:
                    # Get expert name for matching
                    expert_meta = self.redis_text.hgetall(f"meta:expert:{expert_id}")
                    if expert_meta:
                        expert_name = f"{expert_meta.get('first_name', '')} {expert_meta.get('last_name', '')}".strip().lower()
                        
                        if expert_name:
                            # Scan for resources with matching author
                            remaining = limit - len(publications)
                            cursor = 0
                            author_matches = []
                            
                            while len(author_matches) < remaining and (cursor != 0 or not author_matches):
                                cursor, keys = self.redis_text.scan(cursor, match="meta:resource:*", count=100)
                                for key in keys:
                                    # Skip if we already found this resource
                                    resource_id = key.split(':')[-1]
                                    if any(p.get('id') == resource_id for p in publications):
                                        continue
                                        
                                    try:
                                        meta = self.redis_text.hgetall(key)
                                        authors_json = meta.get('authors', '[]')
                                        
                                        try:
                                            authors = json.loads(authors_json)
                                            if isinstance(authors, list):
                                                # Check if expert name appears in author list
                                                for author in authors:
                                                    if expert_name in str(author).lower():
                                                        # Parse JSON fields
                                                        publication = {
                                                            'id': meta.get('id', ''),
                                                            'title': meta.get('title', ''),
                                                            'doi': meta.get('doi', ''),
                                                            'abstract': meta.get('abstract', ''),
                                                            'publication_year': meta.get('publication_year', ''),
                                                            'confidence': 0.8  # Good confidence for author name match
                                                        }
                                                        
                                                        publication['authors'] = authors
                                                        author_matches.append(publication)
                                                        break
                                        except json.JSONDecodeError:
                                            # Handle non-JSON authors field
                                            if expert_name in authors_json.lower():
                                                # Parse JSON fields
                                                publication = {
                                                    'id': meta.get('id', ''),
                                                    'title': meta.get('title', ''),
                                                    'doi': meta.get('doi', ''),
                                                    'abstract': meta.get('abstract', ''),
                                                    'publication_year': meta.get('publication_year', ''),
                                                    'confidence': 0.8  # Good confidence for author name match
                                                }
                                                
                                                publication['authors'] = [authors_json] if authors_json else []
                                                author_matches.append(publication)
                                        
                                        # Stop if we have enough matches
                                        if len(author_matches) >= remaining:
                                            break
                                    except Exception as e:
                                        logger.error(f"Error processing key {key} for author match: {e}")
                                
                                if cursor == 0 or len(author_matches) >= remaining:
                                    break
                            
                            # Add author matches to publications
                            publications.extend(author_matches)
                except Exception as author_error:
                    logger.error(f"Error finding author matches for expert {expert_id}: {author_error}")
            
            # Sort by confidence score
            publications.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Limit to requested number
            logger.info(f"Found {len(publications)} publication keys for expert {expert_id}")
            return publications[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving publications for expert {expert_id}: {e}")
            return []
    




    def _normalize_name(self, name: str) -> str:
        """Normalize author name for comparison"""
        if not name or not isinstance(name, str):
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = ' '.join(name.lower().split())
        
        # Remove common suffixes and prefixes
        prefixes = ['dr.', 'dr ', 'prof.', 'prof ', 'professor ', 'mr.', 'mr ', 'mrs.', 'mrs ', 'ms.', 'ms ']
        suffixes = [' phd', ' md', ' jr', ' sr', ' jr.', ' sr.', ' ii', ' iii', ' iv']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        return normalized

    def _store_resource_data(self, resource: Dict[str, Any], text_content: str, embedding: np.ndarray) -> None:
        """Stores resources that are linked to experts in Redis."""
        conn = None
        pipeline = None
        
        try:
            # Get expert_id - can come from either direct field or links
            expert_id = resource.get('expert_id')
            
            # If no direct expert_id, check if resource is linked to any expert
            if not expert_id or str(expert_id).strip().lower() == 'none':
                try:
                    conn = self.db.get_connection()
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT expert_id FROM expert_resource_links 
                            WHERE resource_id = %s LIMIT 1
                        """, (resource['id'],))
                        link_result = cur.fetchone()
                        expert_id = link_result[0] if link_result else None
                except Exception as e:
                    logger.error(f"Error checking resource links for {resource.get('id')}: {e}")
                    raise

            # Validate we have an expert_id
            if not expert_id:
                logger.warning(f"Resource {resource.get('id')} has no expert association - skipping")
                return
                
            # Convert to string and clean
            expert_id = str(expert_id).strip()
            
            # Verify expert exists using same connection
            try:
                if not conn:  # Reuse connection if we have one
                    conn = self.db.get_connection()
                    
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM experts_expert WHERE id = %s", (expert_id,))
                    if not cur.fetchone():
                        logger.warning(f"Expert {expert_id} not found for resource {resource.get('id')}")
                        return

                # Proceed with Redis storage
                base_key = f"expert_resource:{resource['id']}"  # Simplified key pattern
                pipeline = self.redis_text.pipeline()
                
                # Store text content
                pipeline.set(f"text:{base_key}", text_content)
                
                # Store embedding
                self.redis_binary.set(f"emb:{base_key}", embedding.astype(np.float32).tobytes())
                
                # Store metadata
                metadata = {
                    'id': str(resource['id']),
                    'title': str(resource.get('title', '')),
                    'abstract': str(resource.get('abstract', '')),
                    'authors': json.dumps(resource.get('authors', [])),
                    'publication_year': str(resource.get('publication_year', '')),
                    'expert_id': expert_id,
                    'doi': str(resource.get('doi', ''))
                }
                pipeline.hset(f"meta:{base_key}", mapping=metadata)
                
                # Create reverse link from expert to resource
                expert_resources_key = f"expert:{expert_id}:resources"
                pipeline.sadd(expert_resources_key, str(resource['id']))
                
                pipeline.execute()
                
            except Exception as e:
                if pipeline:
                    pipeline.reset()
                logger.error(f"Error storing resource {resource.get('id')}: {e}")
            

    def _create_resource_text_content(self, resource: Dict[str, Any]) -> str:
        """Create comprehensive text content for resource embedding."""
        try:
            # Create a comprehensive text representation that captures all significant details
            text_parts = []
            
            # Essential fields
            if resource.get('title'):
                text_parts.append(f"Title: {str(resource['title']).strip()}")
            
            # Abstract or summary (prioritize abstract)
            if resource.get('abstract'):
                text_parts.append(f"Abstract: {str(resource['abstract']).strip()}")
            elif resource.get('summary'):
                text_parts.append(f"Summary: {str(resource['summary']).strip()}")
            
            # Bibliographic details
            if resource.get('publication_year'):
                text_parts.append(f"Year: {resource['publication_year']}")
            
            if resource.get('doi'):
                text_parts.append(f"DOI: {resource['doi']}")
            
            # Authors
            authors = resource.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    authors_text = ", ".join([str(author).strip() for author in authors if author])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
                elif isinstance(authors, str):
                    text_parts.append(f"Authors: {authors}")
                elif isinstance(authors, dict):
                    # Handle dictionary representation of authors
                    authors_text = ", ".join([str(value).strip() for value in authors.values() if value])
                    if authors_text:
                        text_parts.append(f"Authors: {authors_text}")
            
            # Additional contextual information
            if resource.get('type'):
                text_parts.append(f"Type: {resource['type']}")
            
            if resource.get('source'):
                text_parts.append(f"Source: {resource['source']}")
            
            # Domains and topics
            domains = resource.get('domains', [])
            if domains and isinstance(domains, list):
                domains_text = ", ".join([str(domain).strip() for domain in domains if domain])
                if domains_text:
                    text_parts.append(f"Domains: {domains_text}")
            
            topics = resource.get('topics', {})
            if topics and isinstance(topics, dict):
                topics_text = ", ".join([f"{k}: {v}" for k, v in topics.items() if v])
                if topics_text:
                    text_parts.append(f"Topics: {topics_text}")
            
            # Optional detailed fields
            if resource.get('description'):
                text_parts.append(f"Description: {str(resource['description']).strip()}")
            
            # Identifiers
            identifiers = resource.get('identifiers', {})
            if identifiers and isinstance(identifiers, dict):
                identifiers_text = ", ".join([f"{k}: {v}" for k, v in identifiers.items() if v])
                if identifiers_text:
                    text_parts.append(f"Identifiers: {identifiers_text}")
            
            # Publishers and collection
            publishers = resource.get('publishers', {})
            if publishers and isinstance(publishers, dict):
                publishers_text = ", ".join([str(value).strip() for value in publishers.values() if value])
                if publishers_text:
                    text_parts.append(f"Publishers: {publishers_text}")
            
            if resource.get('collection'):
                text_parts.append(f"Collection: {resource['collection']}")
            
            # Language and citation
            if resource.get('language'):
                text_parts.append(f"Language: {resource['language']}")
            
            if resource.get('citation'):
                text_parts.append(f"Citation: {resource['citation']}")
            
            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Resource"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for resource {resource.get('id', 'Unknown')}: {e}")
            return "Error Processing Resource"
   

    
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

    

    
    