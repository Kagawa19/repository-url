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

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """Fetch all expert data from database."""
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
                    
                    # Updated query to use only existing columns
                    cur.execute("""
                        SELECT 
                            id,
                            email,
                            knowledge_expertise,
                            is_active,
                            is_staff,
                            created_at,
                            updated_at,
                            bio,
                            orcid,
                            first_name,
                            last_name,
                            contact_details,
                            unit,
                            designation,
                            theme
                        FROM experts_expert
                        WHERE id IS NOT NULL
                    """)
                    
                    experts = [{
                        'id': row[0],
                        'email': row[1],
                        'knowledge_expertise': self._parse_jsonb(row[2]),
                        'is_active': row[3],
                        'is_staff': row[4],
                        'created_at': row[5].isoformat() if row[5] else None,
                        'updated_at': row[6].isoformat() if row[6] else None,
                        'bio': row[7] or '',
                        'orcid': row[8],
                        'first_name': row[9] or '',
                        'last_name': row[10] or '',
                        'contact_details': row[11],
                        'unit': row[12] or '',
                        'designation': row[13] or '',
                        'theme': row[14] or ''
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

    def _create_text_content(self, expert: Dict[str, Any]) -> str:
        """Create combined text content for embedding with additional safeguards."""
        try:
            # Ensure we have at least some basic information
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

            # Add other fields with explicit string conversion and cleanup
            fields = {
                'Email': expert.get('email'),
                'Unit': expert.get('unit'),
                'Bio': expert.get('bio'),
                'ORCID': expert.get('orcid'),
                'Designation': expert.get('designation'),
                'Theme': expert.get('theme')
            }
            
            for field, value in fields.items():
                if value:  # Check if value exists and is not None
                    cleaned_value = str(value).strip()
                    if cleaned_value:  # Check if value is not empty after cleaning
                        text_parts.append(f"{field}: {cleaned_value}")

            # Handle knowledge expertise separately
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

            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Expert Profile"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for expert {expert.get('id', 'Unknown')}: {e}")
            return "Error Processing Expert Profile"

    def create_redis_index(self) -> bool:
        """Create Redis indexes for experts with enhanced error handling."""
        try:
            logger.info("Creating Redis indexes for experts...")
            experts = self.fetch_experts()
            
            if not experts:
                logger.warning("No experts found to index")
                return False
            
            success_count = 0
            error_count = 0
            
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

    def _store_expert_data(self, expert: Dict[str, Any], text_content: str, 
                          embedding: np.ndarray) -> None:
        """Store expert data in Redis."""
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
            
            # Store metadata
            metadata = {
                'id': str(expert['id']),  # Ensure id is string
                'email': str(expert.get('email', '')),
                'name': f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip(),
                'unit': str(expert.get('unit', '')),
                'bio': str(expert.get('bio', '')),
                'orcid': str(expert.get('orcid', '')),
                'designation': str(expert.get('designation', '')),
                'theme': str(expert.get('theme', '')),
                'expertise': json.dumps(expert.get('knowledge_expertise', {})),
                'is_active': json.dumps(expert.get('is_active', False)),
                'updated_at': expert.get('updated_at', '')
            }
            pipeline.hset(f"meta:{base_key}", mapping=metadata)
            
            pipeline.execute()
            
        except Exception as e:
            pipeline.reset()
            raise e

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

    def get_expert_metadata(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve expert metadata from Redis."""
        try:
            metadata = self.redis_text.hgetall(f"meta:expert:{expert_id}")
            if metadata:
                # Parse JSON fields
                for field in ['expertise', 'is_active']:
                    if metadata.get(field):
                        metadata[field] = json.loads(metadata[field])
                return metadata
            return None
        except Exception as e:
            logger.error(f"Error retrieving expert metadata: {e}")
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
