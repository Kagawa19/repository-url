import os
import numpy as np
import faiss
import pickle
import redis
import json
import time
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from src.utils.db_utils import DatabaseConnector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ExpertSearchIndexManager:
    def __init__(self):
        """Initialize ExpertSearchIndexManager."""
        self.setup_paths()
        self.setup_redis()
        self.model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
        self.db = DatabaseConnector()

    def setup_paths(self):
        """Setup paths for storing models and mappings."""
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = current_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.models_dir / 'expert_faiss_index.idx'
        self.mapping_path = self.models_dir / 'expert_mapping.pkl'

    def setup_redis(self):
        """Setup Redis connections."""
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_EMBEDDINGS_DB', 1)),
            decode_responses=True
        )
        self.redis_binary = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_EMBEDDINGS_DB', 1)),
            decode_responses=False
        )

        
        

        # First, the corrected ExpertSearchIndexManager methods with original names:

    def create_expert_text(self, expert: Dict[str, Any]) -> str:
        """Create searchable text from expert data."""
        text_parts = [
            f"First Name: {expert['first_name']}",
            f"Last Name: {expert['last_name']}",
            f"Expertise: {' | '.join(expert['knowledge_expertise'])}"
        ]
        return '\n'.join(text_parts)

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """Fetch all experts with retry logic."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
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
                        logger.warning("experts_expert table does not exist")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return []
                    
                    # Fetch only the required columns
                    cur.execute("""
                        SELECT 
                            id,
                            first_name,
                            last_name,
                            knowledge_expertise
                        FROM experts_expert
                        WHERE id IS NOT NULL
                    """)
                    rows = cur.fetchall()
                    
                    experts = []
                    for row in rows:
                        try:
                            expert = {
                                'id': row[0],
                                'first_name': row[1] or '',
                                'last_name': row[2] or '',
                                'knowledge_expertise': row[3] if isinstance(row[3], list) else json.loads(row[3]) if row[3] else []
                            }
                            experts.append(expert)
                        except Exception as e:
                            logger.error(f"Error processing expert data: {e}")
                    
                    return experts
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    return []
            finally:
                if 'conn' in locals():
                    conn.close()

    def store_in_redis(self, key: str, embedding: np.ndarray, metadata: dict):
        """Store expert embedding and metadata in Redis."""
        try:
            pipeline = self.redis_binary.pipeline()
            
            # Handle null values in metadata
            for k, value in metadata.items():
                if value is None:
                    metadata[k] = ''
            
            pipeline.hset(
                f"expert:{key}",
                mapping={
                    'vector': embedding.tobytes(),
                    'metadata': json.dumps({
                        'id': metadata['id'],
                        'first_name': metadata['first_name'],
                        'last_name': metadata['last_name'],
                        'knowledge_expertise': metadata['knowledge_expertise']
                    })
                }
            )
            pipeline.execute()
        except Exception as e:
            logger.error(f"Error storing expert in Redis: {e}")

    def create_faiss_index(self) -> bool:
        """Create FAISS index for expert search."""
        try:
            # Fetch expert data
            experts = self.fetch_experts()
            if not experts:
                logger.warning("No expert data available to create index")
                return False

            # Prepare text for embeddings
            texts = [self.create_expert_text(expert) for expert in experts]

            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            
            # Create and populate FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Store embeddings and metadata
            for i, (expert, embedding) in enumerate(zip(experts, embeddings)):
                # Store in Redis
                self.store_in_redis(
                    str(expert['id']),
                    embedding,
                    {
                        'id': expert['id'],
                        'first_name': expert['first_name'],
                        'last_name': expert['last_name'],
                        'knowledge_expertise': expert['knowledge_expertise']
                    }
                )
                
                # Add to FAISS index
                index.add(embedding.reshape(1, -1).astype(np.float32))

            # Save FAISS index and mapping
            faiss.write_index(index, str(self.index_path))
            with open(self.mapping_path, 'wb') as f:
                pickle.dump({i: expert['id'] for i, expert in enumerate(experts)}, f)

            logger.info(f"Successfully created index with {len(experts)} experts")
            return True

        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False

    def search_experts(self, query: str, k: int = 5, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Search for similar experts using the index.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            active_only (bool): Whether to return only active experts (ignored in this simplified version)
            
        Returns:
            List of expert matches with metadata
        """
        try:
            # Load index and mapping
            index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, 'rb') as f:
                id_mapping = pickle.load(f)

            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search index
            distances, indices = index.search(query_embedding.astype(np.float32), k)
            
            # Fetch results from Redis
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # FAISS may return -1 for not enough matches
                    continue
                    
                expert_id = id_mapping[idx]
                expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                
                if expert_data:
                    metadata = json.loads(expert_data[b'metadata'].decode())
                    metadata['score'] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity score
                    results.append(metadata)
            
            return results

        except Exception as e:
            logger.error(f"Error searching experts: {e}")
            return []

    def _parse_jsonb(self, data):
        """Parse JSONB data safely."""
        if not data:
            return []
        try:
            if isinstance(data, str):
                return json.loads(data)
            return data
        except:
            return []

    # Add this method to the ExpertSearchIndexManager class, and update the fetch_experts method:

    def fetch_experts(self) -> List[Dict[str, Any]]:
        """Fetch all experts with retry logic."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
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
                        logger.warning("experts_expert table does not exist")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return []
                    
                    # Fetch only the required columns
                    cur.execute("""
                        SELECT 
                            id,
                            first_name,
                            last_name,
                            knowledge_expertise
                        FROM experts_expert
                        WHERE id IS NOT NULL
                    """)
                    rows = cur.fetchall()
                    
                    experts = []
                    for row in rows:
                        try:
                            expert = {
                                'id': row[0],
                                'first_name': row[1] or '',
                                'last_name': row[2] or '',
                                'knowledge_expertise': self._parse_jsonb(row[3])
                            }
                            experts.append(expert)
                        except Exception as e:
                            logger.error(f"Error processing expert data: {e}")
                    
                    return experts
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    return []
            finally:
                if 'conn' in locals():
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
def initialize_expert_search():
    """Initialize expert search index."""
    try:
        logger.info("Creating FAISS search index...")
        manager = ExpertSearchIndexManager()
        success = manager.create_faiss_index()
        if not success:
            logger.error("FAISS index creation failed")
            return False
        logger.info("FAISS search index created successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing expert search: {e}")
        return False

if __name__ == "__main__":
    initialize_expert_search()
