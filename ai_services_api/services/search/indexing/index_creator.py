import os
import numpy as np
from ai_services_api.services.search.gemini.gemini_predictor import GoogleAutocompletePredictor
import faiss
import pickle
import redis
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OfflineEmbedder:
    """
    A deterministic embedding generator that works without internet access.
    Uses feature hashing to create consistent embeddings from text.
    """
    
    def __init__(self, embedding_dim=384):
        """
        Initialize the offline embedder.
        
        Args:
            embedding_dim (int): Dimension of the embeddings to generate
        """
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized OfflineEmbedder with dimension {embedding_dim}")
    
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        """
        Generate embeddings using text hashing for consistency.
        
        Args:
            texts: Text or list of texts to encode
            convert_to_numpy (bool): Always returns numpy array
            show_progress_bar (bool): Ignored, for API compatibility
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts")
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        for i, text in enumerate(texts):
            # Generate embedding from text hash
            embeddings[i] = self._text_to_embedding(text)
        
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def _text_to_embedding(self, text):
        """
        Convert text to an embedding vector using feature hashing.
        
        Args:
            text (str): Text to convert
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Create a deterministic seed from the text
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.sha256(text_bytes)
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        # Use numpy's random with the seed for deterministic output
        np.random.seed(seed)
        
        # Generate random values based on text features
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Split text into words and process each word
        words = text.lower().split()
        for word in words:
            word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            word_seed = word_hash % (2**32)
            np.random.seed(word_seed)
            
            # Generate a sparse word vector and add it to the embedding
            word_vec = np.zeros(self.embedding_dim, dtype=np.float32)
            
            # Make it sparse - only activate ~5% of dimensions for each word
            active_dims = np.random.choice(
                self.embedding_dim, 
                size=max(1, int(self.embedding_dim * 0.05)), 
                replace=False
            )
            
            # Set values for active dimensions
            for dim in active_dims:
                # Value between -1 and 1
                word_vec[dim] = (np.random.random() * 2) - 1
            
            # Add the word vector to the overall embedding
            embedding += word_vec
        
        # Add some noise to make embeddings more diverse
        np.random.seed(seed)
        noise = np.random.normal(0, 0.01, self.embedding_dim)
        embedding += noise
        
        return embedding

class ExpertSearchIndexManager:
    def __init__(self):
        """Initialize ExpertSearchIndexManager with offline embedder."""
        self.setup_paths()
        self.setup_redis()
        
        # Use offline embedder that doesn't require internet access
        logger.info("Initializing offline embedding model")
        self.model = OfflineEmbedder(embedding_dim=384)
        
        # Import DatabaseConnector within method to avoid circular imports
        try:
            from src.utils.db_utils import DatabaseConnector
            self.db = DatabaseConnector()
        except ImportError:
            logger.warning("Could not import DatabaseConnector, trying alternative import path")
            try:
                from src.utils.db_utils import DatabaseConnector
                self.db = DatabaseConnector()
            except ImportError:
                logger.error("Failed to import DatabaseConnector")
                raise

    def setup_paths(self):
        """Setup paths for storing models and mappings."""
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = current_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.models_dir / 'expert_faiss_index.idx'
        self.mapping_path = self.models_dir / 'expert_mapping.pkl'

    def setup_redis(self):
        """Setup Redis connections."""
        try:
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
        except Exception as e:
            logger.error(f"Error setting up Redis: {e}")
            raise

        
    def get_expert_by_id_or_name(self, query: str) -> List[Dict[str, Any]]:
        """
        Try to find an expert by ID first, then fall back to name search.
        """
        logger.info(f"Attempting to find expert by ID or name: '{query}'")
        try:
            # First, try to treat the query as an ID
            if query.isdigit():
                logger.info(f"Query '{query}' appears to be a numeric ID, trying exact ID lookup")
                expert = self.get_expert_by_id(query)
                if expert:
                    logger.info(f"Found expert with ID {query}")
                    return [expert]
                else:
                    logger.info(f"No expert found with ID {query}")
            
            # Extract name parts
            name_parts = query.split()
            logger.info(f"Extracted name parts from '{query}': {name_parts}")
            
            # Connect to database
            conn = self.db.get_connection()
            with conn.cursor() as cur:
                # Search by full name
                if len(name_parts) > 1:
                    first_name = name_parts[0]
                    last_name = ' '.join(name_parts[1:])
                    logger.info(f"Searching for expert with first_name='{first_name}', last_name='{last_name}'")
                    
                    cur.execute("""
                        SELECT e.id, e.first_name, e.last_name, 
                            COALESCE(e.designation, '') as designation,
                            COALESCE(e.theme, '') as theme,
                            COALESCE(e.unit, '') as unit,
                            COALESCE(e.bio, '') as bio,
                            e.knowledge_expertise,
                            e.is_active
                        FROM experts_expert e
                        WHERE 
                            (LOWER(e.first_name) = LOWER(%s) AND LOWER(e.last_name) = LOWER(%s))
                            OR
                            (LOWER(CONCAT(e.first_name, ' ', e.last_name)) = LOWER(%s))
                    """, [first_name, last_name, query])
                else:
                    # Search by single name part (first or last)
                    logger.info(f"Searching for expert with single name part: '{query}'")
                    cur.execute("""
                        SELECT e.id, e.first_name, e.last_name, 
                            COALESCE(e.designation, '') as designation,
                            COALESCE(e.theme, '') as theme,
                            COALESCE(e.unit, '') as unit,
                            COALESCE(e.bio, '') as bio,
                            e.knowledge_expertise,
                            e.is_active
                        FROM experts_expert e
                        WHERE 
                            LOWER(e.first_name) = LOWER(%s) OR
                            LOWER(e.last_name) = LOWER(%s)
                    """, [query, query])
                
                # Format results
                rows = cur.fetchall()
                logger.info(f"Database search returned {len(rows)} results")
                
                results = []
                for row in rows:
                    expert_id = str(row[0])
                    first_name = row[1] or ""
                    last_name = row[2] or ""
                    logger.info(f"Found expert: {first_name} {last_name} (ID: {expert_id})")
                    
                    results.append({
                        "id": expert_id,
                        "first_name": first_name,
                        "last_name": last_name,
                        "designation": row[3] or "",
                        "theme": row[4] or "",
                        "unit": row[5] or "",
                        "contact": "",  # Default empty value
                        "is_active": row[8] if row[8] is not None else True,
                        "score": 1.0,  # Exact match
                        "bio": row[6] or "",
                        "knowledge_expertise": self._parse_jsonb(row[7]) if row[7] else []
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error in expert lookup by ID or name: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
        
    def get_expert_by_id(self, expert_id: str) -> Dict[str, Any]:
        """
        Get an expert directly by ID.
        
        Args:
            expert_id: Expert ID to fetch
            
        Returns:
            Dict with expert details or None if not found
        """
        try:
            conn = self.db.get_connection()
            with conn.cursor() as cur:
                # Query for the expert with this exact ID
                cur.execute("""
                    SELECT e.id, e.first_name, e.last_name, 
                        COALESCE(e.designation, '') as designation,
                        COALESCE(e.theme, '') as theme,
                        COALESCE(e.unit, '') as unit,
                        COALESCE(e.bio, '') as bio,
                        e.knowledge_expertise,
                        e.is_active
                    FROM experts_expert e
                    WHERE e.id = %s
                """, [expert_id])
                
                row = cur.fetchone()
                if row:
                    # Format the result as a dictionary
                    return {
                        "id": str(row[0]),
                        "first_name": row[1] or "",
                        "last_name": row[2] or "",
                        "designation": row[3] or "",
                        "theme": row[4] or "",
                        "unit": row[5] or "",
                        "contact": "",  # Default empty value
                        "is_active": row[8] if row[8] is not None else True,
                        "score": 1.0,  # Perfect match
                        "bio": row[6] or "",
                        "knowledge_expertise": self._parse_jsonb(row[7]) if row[7] else []
                    }
                
                return None
        except Exception as e:
            logger.error(f"Error fetching expert by ID: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()

    def create_expert_text(self, expert: Dict[str, Any]) -> str:
        """
        Create searchable text from expert data, focusing only on knowledge expertise.
        
        Args:
            expert (Dict[str, Any]): Expert data dictionary
            
        Returns:
            str: Searchable text representation of the expert
        """
        # Start with basic identity for minimal context
        text_parts = [
            f"First Name: {expert['first_name']}",
            f"Last Name: {expert['last_name']}"
        ]
        
        # Focus primarily on knowledge expertise with repetition for emphasis
        if expert.get('knowledge_expertise') and isinstance(expert['knowledge_expertise'], list):
            expertise = expert['knowledge_expertise']
            if expertise:
                # Add expertise in different formats for better semantic matching
                text_parts.append(f"Knowledge Expertise: {' | '.join(expertise)}")
                text_parts.append(f"Expert in: {' | '.join(expertise)}")
                text_parts.append(f"Specializes in: {' | '.join(expertise)}")
                # Repeat each expertise area individually for stronger emphasis
                for area in expertise:
                    text_parts.append(f"Expertise Area: {area}")
        
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

    def store_in_redis(self, key: str, embedding: np.ndarray, metadata: dict):
        """Store expert embedding and metadata in Redis."""
        try:
            pipeline = self.redis_binary.pipeline()
            
            # Handle null values in metadata
            for k, value in metadata.items():
                if value is None:
                    metadata[k] = ''
            
            # Ensure knowledge_expertise is always serialized as a list
            if 'knowledge_expertise' in metadata and not isinstance(metadata['knowledge_expertise'], list):
                if metadata['knowledge_expertise']:
                    metadata['knowledge_expertise'] = [metadata['knowledge_expertise']]
                else:
                    metadata['knowledge_expertise'] = []
            
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

            logger.info(f"Fetched {len(experts)} experts for indexing")

            # Prepare text for embeddings
            texts = [self.create_expert_text(expert) for expert in experts]
            logger.info(f"Created text representations for {len(texts)} experts")

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            logger.info(f"Generated embeddings with shape {embeddings.shape}")
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {dimension}")
            index = faiss.IndexFlatL2(dimension)
            
            # Store embeddings and metadata
            logger.info("Storing embeddings and metadata in Redis...")
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
                
                # Log progress
                if (i+1) % 10 == 0 or i == len(experts) - 1:
                    logger.info(f"Stored {i+1}/{len(experts)} experts")

            # Save FAISS index and mapping
            logger.info(f"Saving FAISS index to {str(self.index_path)}")
            faiss.write_index(index, str(self.index_path))
            
            logger.info(f"Saving ID mapping to {str(self.mapping_path)}")
            with open(self.mapping_path, 'wb') as f:
                pickle.dump({i: expert['id'] for i, expert in enumerate(experts)}, f)

            logger.info(f"Successfully created index with {len(experts)} experts")
            return True

        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False

    def search_experts(self, query: str, k: int = 5, active_only: bool = False, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar experts using the index with improved filtering.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            active_only (bool): Whether to return only active experts
            min_score (float): Minimum similarity score threshold
            
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
            
            # Search index - get more candidates for filtering
            max_candidates = min(k * 3, index.ntotal)
            distances, indices = index.search(query_embedding.astype(np.float32), max_candidates)
            
            # Fetch and filter results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # FAISS may return -1 for not enough matches
                    continue
                    
                expert_id = id_mapping[idx]
                expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                
                if expert_data:
                    try:
                        metadata = json.loads(expert_data[b'metadata'].decode())
                        
                        # For L2 distance, lower is better, so convert to similarity score
                        distance = float(distances[0][i])
                        score = float(1.0 / (1.0 + distance))
                        
                        # Always include the result but boost scores for expertise matches
                        metadata['score'] = score
                        
                        # Expertise relevance boost: check if query terms appear in expertise
                        if self._has_expertise_match(query, metadata.get('knowledge_expertise', [])):
                            # Boost score for direct expertise matches
                            metadata['score'] = min(1.0, score * 1.5)
                        
                        results.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing expert {expert_id}: {e}")
            
            # Sort by score (highest first) and return top-k
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:k]

        except Exception as e:
            logger.error(f"Error searching experts: {e}")
            return []

    def _has_expertise_match(self, query: str, expertise_list: List[str]) -> bool:
        """
        Check if query terms match any expertise areas with a more lenient approach.
        
        Args:
            query (str): Search query
            expertise_list (List[str]): List of expertise areas
            
        Returns:
            bool: True if there's a match, False otherwise
        """
        if not expertise_list:
            return False
            
        # Normalize query for comparison
        query_lower = query.lower()
        query_terms = set(term.lower() for term in query.split())
        
        for expertise in expertise_list:
            if not expertise:
                continue
                
            expertise_lower = expertise.lower()
            
            # Check for direct substring match first (most reliable)
            if query_lower in expertise_lower or expertise_lower in query_lower:
                return True
                
            # Also check for individual term matches
            expertise_terms = set(term.lower() for term in expertise.split())
            
            # Check for any overlap between query terms and expertise terms
            if query_terms.intersection(expertise_terms):
                return True
                
        return False

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

    async def generate_advanced_search_refinements(
        self, 
        search_type: str, 
        query: str, 
        user_id: str, 
        results: List[Dict[str, Any]] = []
    ) -> Dict[str, Any]:
        """
        Generate advanced search refinements based on search type.
        
        Args:
            search_type (str): Type of search ('name', 'theme', 'designation', 'publication', 'general')
            query (str): Original search query
            user_id (str): User identifier
            results (List[Dict[str, Any]], optional): Current search results
        
        Returns:
            Dict[str, Any]: Refinement suggestions
        """
        logger.info(f"Generating advanced refinements for {search_type} search: {query}")
        
        try:
            # If we have results, try to get refinements from search manager
            search_manager = ExpertSearchIndexManager()
            if results:
                return search_manager.get_search_refinements(query, results)
            
            # If no results or empty refinements, enhance with Gemini
            try:
                # Use GoogleAutocompletePredictor to get related queries
                gemini_predictor = GoogleAutocompletePredictor()
                
                # Include context in the query for better suggestions
                context_query = query
                if search_type == 'name':
                    context_query = f"expert named {query}"
                elif search_type == 'theme':
                    context_query = f"research theme {query}"
                elif search_type == 'designation':
                    context_query = f"job role {query}"
                elif search_type == 'publication':
                    context_query = f"publication on {query}"
                
                suggestion_objects = await gemini_predictor.predict(context_query, limit=5)
                related_queries = [s["text"] for s in suggestion_objects]
                
                # Extract potential expertise areas
                expertise_areas = set()
                for related_query in related_queries:
                    words = related_query.lower().split()
                    for word in words:
                        if len(word) > 3 and word != query.lower():
                            expertise_areas.add(word.capitalize())
                
                # Ensure we have some expertise areas
                if len(expertise_areas) < 3:
                    expertise_areas.add(query.capitalize())
                
                # Build enhanced refinements with search type specific filters
                filters = []
                
                # Add search type specific filters
                if search_type == 'name':
                    filters.append({
                        "type": "expertise_filter",
                        "label": "Expert Expertise",
                        "values": list(expertise_areas)[:3]
                    })
                elif search_type == 'theme':
                    filters.append({
                        "type": "theme_filter",
                        "label": "Research Areas",
                        "values": list(expertise_areas)[:3]
                    })
                elif search_type == 'designation':
                    filters.append({
                        "type": "role_filter",
                        "label": "Job Roles",
                        "values": [f"{query.capitalize()} Specialist", f"Senior {query.capitalize()}", f"Junior {query.capitalize()}"]
                    })
                elif search_type == 'publication':
                    filters.append({
                        "type": "publication_filter",
                        "label": "Publication Types",
                        "values": ["Journal Articles", "Conference Papers", "Book Chapters", "Reports"]
                    })
                # For general searches, include multiple filter types
                else:
                    filters.append({
                        "type": "search_type",
                        "label": "Search Categories",
                        "values": ["Experts", "Publications", "Research Areas"]
                    })
                
                # Build enhanced refinements
                refinements = {
                    "filters": filters,
                    "related_queries": related_queries,
                    "expertise_areas": list(expertise_areas)[:5]
                }
                
                return refinements
            
            except Exception as gemini_error:
                logger.error(f"Gemini refinement error: {gemini_error}", exc_info=True)
                # Fallback to basic refinements
                refinements = {
                    "filters": [],
                    "related_queries": [],
                    "expertise_areas": [query.capitalize()]
                }
                
            return refinements
        
        except Exception as e:
            logger.error(f"Error generating advanced refinements: {e}", exc_info=True)
            return {
                "filters": [],
                "related_queries": [],
                "expertise_areas": []
            }
            
    
    def get_search_refinements(self, query: str, current_results: List[Dict]) -> Dict:
        """
        Generate search refinement suggestions based on current query and results.
        
        Args:
            query (str): Original search query
            current_results (List[Dict]): Current search results
        
        Returns:
            Dict: Refinement suggestions with multiple categories
        """
        try:
            # Import MLPredictor here to avoid potential circular import
            from ai_services_api.services.search.ml.ml_predictor import MLPredictor
            
            # Create ML Predictor instance
            ml_predictor = MLPredictor()
            
            refinements = {
                "filters": self._suggest_filters(query, current_results),
                "related_queries": self._suggest_related_queries(query, ml_predictor),
                "expertise_areas": self._extract_expertise_areas(current_results)
            }
            
            return refinements
        except Exception as e:
            logger.error(f"Error generating search refinements: {e}")
            return {}

    def _suggest_related_queries(self, query: str, ml_predictor) -> List[str]:
        """Generate related query suggestions."""
        try:
            # Use MLPredictor to get related queries
            related_queries = ml_predictor.predict(query, user_id="system", limit=5)
            
            # Enhance suggestions with query variations
            variations = [
                f"expert in {query}",
                f"research about {query}",
                f"top {query} specialists"
            ]
            
            # Combine and remove duplicates
            seen = set()
            unique_suggestions = []
            for suggestion in list(related_queries) + variations:
                if suggestion.lower() not in seen and suggestion.lower() != query.lower():
                    seen.add(suggestion.lower())
                    unique_suggestions.append(suggestion)
            
            return unique_suggestions[:5]
        
        except Exception as e:
            logger.error(f"Error generating related queries: {e}")
            return []

    def _suggest_filters(self, query: str, results: List[Dict]) -> List[Dict]:
        """Generate filter suggestions based on current results."""
        filters = []
        
        # Expertise areas filter
        expertise_areas = set()
        for result in results:
            expertise = result.get('knowledge_expertise', [])
            
            # Normalize expertise to list
            if expertise is None:
                continue
            
            if isinstance(expertise, str):
                expertise = [expertise]
            elif isinstance(expertise, dict):
                expertise = list(expertise.keys())
            
            # Ensure we only add non-empty strings
            expertise = [str(e).strip() for e in expertise if e and str(e).strip()]
            
            expertise_areas.update(expertise)
        
        if expertise_areas:
            filters.append({
                "type": "expertise",
                "label": "Expertise Areas",
                "values": list(expertise_areas)[:5]
            })
        
        # Unit/Department filter
        departments = set()
        for result in results:
            dept = result.get('unit', '')
            if dept and isinstance(dept, str) and dept.strip():
                departments.add(dept.strip())
        
        if departments:
            filters.append({
                "type": "department",
                "label": "Departments",
                "values": list(departments)[:5]
            })
        
        # Additional potential filters
        filters.append({
            "type": "active_status",
            "label": "Availability",
            "values": ["Active Experts", "All Experts"]
        })
        
        return filters
    
    def search_experts_by_name(self, name: str, k: int = 5, active_only: bool = False, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for experts by name using direct database prefix matching for autocomplete.
        
        Args:
            name (str): Name pattern to search for
            k (int): Number of results to return
            active_only (bool): Whether to return only active experts (maintained for compatibility)
            min_score (float): Minimum similarity score threshold (maintained for compatibility)
        
        Returns:
            List[Dict[str, Any]]: List of matching experts
        """
        try:
            # Split the name into parts for more flexible matching
            name_parts = name.lower().strip().split()
            if not name_parts:
                return []
                
            # Get database connection
            conn = self.db.get_connection()
            matching_experts = []
            
            try:
                with conn.cursor() as cur:
                    # Add active_only filter if requested
                    active_filter = "AND is_active = true" if active_only else ""
                    
                    # First try exact prefix match (most relevant for autocomplete)
                    if len(name_parts) == 1:
                        # Single word pattern - match against first or last name start
                        cur.execute(f"""
                            SELECT 
                                id, first_name, last_name, knowledge_expertise
                            FROM experts_expert
                            WHERE 
                                (LOWER(first_name) LIKE %s OR 
                                LOWER(last_name) LIKE %s)
                                {active_filter}
                            ORDER BY 
                                CASE 
                                    WHEN LOWER(first_name) LIKE %s THEN 1
                                    WHEN LOWER(last_name) LIKE %s THEN 2
                                    ELSE 3
                                END,
                                first_name, last_name
                            LIMIT %s
                        """, (
                            f"{name_parts[0]}%", 
                            f"{name_parts[0]}%",
                            f"{name_parts[0]}%", 
                            f"{name_parts[0]}%",
                            k
                        ))
                    else:
                        # Multiple word pattern - try matching first and last name combination
                        first_part = name_parts[0]
                        last_part = name_parts[-1]
                        
                        cur.execute(f"""
                            SELECT 
                                id, first_name, last_name, knowledge_expertise
                            FROM experts_expert
                            WHERE 
                                (LOWER(first_name) LIKE %s AND LOWER(last_name) LIKE %s)
                                {active_filter}
                            ORDER BY 
                                first_name, last_name
                            LIMIT %s
                        """, (
                            f"{first_part}%", 
                            f"{last_part}%",
                            k
                        ))
                    
                    # Process results
                    rows = cur.fetchall()
                    
                    # Convert results to dictionaries with calculated scores
                    for row in rows:
                        expert_id, first_name, last_name, knowledge_expertise = row
                        full_name = f"{first_name or ''} {last_name or ''}".strip()
                        
                        # Calculate match score based on how well it matches the pattern
                        match_score = 0.95  # High score for direct database matches
                        
                        matching_experts.append({
                            'id': expert_id,
                            'first_name': first_name or '',
                            'last_name': last_name or '',
                            'knowledge_expertise': self._parse_jsonb(knowledge_expertise),
                            'score': match_score
                        })
                    
                    # If no exact prefix matches found and we have at least 3 characters, try more relaxed matching
                    if not matching_experts and len(name) >= 3:
                        # For single-part patterns, try substring matching anywhere in name
                        if len(name_parts) == 1:
                            cur.execute(f"""
                                SELECT 
                                    id, first_name, last_name, knowledge_expertise
                                FROM experts_expert
                                WHERE 
                                    (LOWER(first_name) LIKE %s OR 
                                    LOWER(last_name) LIKE %s)
                                    {active_filter}
                                ORDER BY 
                                    first_name, last_name
                                LIMIT %s
                            """, (
                                f"%{name_parts[0]}%", 
                                f"%{name_parts[0]}%",
                                k
                            ))
                            
                            rows = cur.fetchall()
                            
                            # Add these with lower scores
                            for row in rows:
                                expert_id, first_name, last_name, knowledge_expertise = row
                                full_name = f"{first_name or ''} {last_name or ''}".strip()
                                
                                matching_experts.append({
                                    'id': expert_id,
                                    'first_name': first_name or '',
                                    'last_name': last_name or '',
                                    'knowledge_expertise': self._parse_jsonb(knowledge_expertise),
                                    'score': 0.7  # Lower score for substring matches
                                })
            
            finally:
                conn.close()
            
            # Sort by score (most relevant first)
            matching_experts.sort(key=lambda x: x.get('score', 0), reverse=True)
            return matching_experts[:k]
            
        except Exception as e:
            logger.error(f"Error in name search: {e}")
            # Fall back to vector search if database query fails
            try:
                return self._fallback_vector_name_search(name, k, active_only, min_score)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
        
    def _fallback_vector_name_search(self, name: str, k: int, active_only: bool, min_score: float):
        """Fallback to vector search if direct database query fails"""
        # Original vector search implementation as backup
        try:
            # Load index and mapping
            index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, 'rb') as f:
                id_mapping = pickle.load(f)
            
            # Create a name-focused query 
            name_focused_query = f"Name: {name} Expert named {name}"
            
            # Generate query embedding
            query_embedding = self.model.encode([name_focused_query], convert_to_numpy=True)
            
            # Search index
            max_candidates = min(k * 5, index.ntotal)
            distances, indices = index.search(query_embedding.astype(np.float32), max_candidates)
            
            # Process results
            results = []
            name_lower = name.lower()
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                    
                expert_id = id_mapping[idx]
                expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                
                if expert_data:
                    try:
                        metadata = json.loads(expert_data[b'metadata'].decode())
                        first_name = metadata.get('first_name', '').lower()
                        last_name = metadata.get('last_name', '').lower()
                        
                        # Only include results with some name relevance
                        if name_lower in first_name or name_lower in last_name:
                            distance = float(distances[0][i])
                            base_score = float(1.0 / (1.0 + distance))
                            metadata['score'] = base_score
                            results.append(metadata)
                            
                    except Exception as e:
                        continue
            
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Fallback vector search failed: {e}")
            return []

    def search_experts_by_theme(self, theme: str, k: int = 5, active_only: bool = False, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for experts by theme using direct database matching of expertise.
        
        Args:
            theme (str): Theme to search for
            k (int): Number of results to return
            active_only (bool): Whether to return only active experts
            min_score (float): Minimum similarity score threshold
        
        Returns:
            List[Dict[str, Any]]: List of matching experts
        """
        try:
            # Normalize the theme search term
            theme_lower = theme.lower().strip()
            if not theme_lower:
                return []
                
            # Get database connection
            conn = self.db.get_connection()
            matching_experts = []
            
            try:
                with conn.cursor() as cur:
                    # Add active_only filter if requested
                    active_filter = "AND is_active = true" if active_only else ""
                    
                    # Search for theme in knowledge expertise using JSON operators
                    # This uses PostgreSQL's JSONB capabilities to search inside the expertise
                    cur.execute(f"""
                        SELECT 
                            id, first_name, last_name, knowledge_expertise, theme
                        FROM experts_expert
                        WHERE 
                            (
                                -- Check in knowledge_expertise (assuming it's JSONB array)
                                (knowledge_expertise ? %s) OR
                                -- Check for partial matches in expertise items
                                (EXISTS (
                                    SELECT FROM jsonb_array_elements_text(knowledge_expertise) 
                                    WHERE value ILIKE %s
                                )) OR
                                -- Check in theme field directly (if it exists)
                                (theme ILIKE %s)
                            )
                            {active_filter}
                        LIMIT %s
                    """, (
                        theme_lower,  # Exact match in expertise
                        f"%{theme_lower}%",  # Partial match in expertise
                        f"%{theme_lower}%",  # Partial match in theme
                        k*3  # Get more to sort better
                    ))
                    
                    rows = cur.fetchall()
                    
                    # Process results with relevance scoring
                    for row in rows:
                        expert_id, first_name, last_name, knowledge_expertise, expert_theme = row
                        expertise_list = self._parse_jsonb(knowledge_expertise)
                        
                        # Calculate theme match score based on where it matched
                        match_score = 0.5  # Base score
                        
                        # Check direct theme match
                        if expert_theme and theme_lower in expert_theme.lower():
                            match_score = 0.9  # High score for theme match
                        
                        # Check expertise matches
                        if isinstance(expertise_list, list):
                            for area in expertise_list:
                                area_lower = str(area).lower()
                                if theme_lower == area_lower:
                                    match_score = max(match_score, 0.95)  # Highest for exact match
                                elif theme_lower in area_lower:
                                    match_score = max(match_score, 0.8)   # Good for partial match
                                
                        matching_experts.append({
                            'id': expert_id,
                            'first_name': first_name or '',
                            'last_name': last_name or '',
                            'knowledge_expertise': expertise_list,
                            'theme': expert_theme or '',
                            'score': match_score
                        })
            
            finally:
                conn.close()
            
            # Sort by score (most relevant first)
            matching_experts.sort(key=lambda x: x.get('score', 0), reverse=True)
            return matching_experts[:k]
            
        except Exception as e:
            logger.error(f"Error in theme search: {e}")
            # Fall back to vector search if database query fails
            try:
                return self._fallback_vector_theme_search(theme, k, active_only, min_score)
            except Exception as fallback_error:
                logger.error(f"Fallback theme search also failed: {fallback_error}")
                return []

    def _fallback_vector_theme_search(self, theme: str, k: int, active_only: bool, min_score: float):
        """Fallback to vector search if direct database query fails"""
        # Original vector search implementation for themes
        try:
            # Load index and mapping
            index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, 'rb') as f:
                id_mapping = pickle.load(f)
            
            # Create a theme-focused query
            theme_focused_query = f"Theme: {theme} Expert specializing in {theme}"
            
            # Generate query embedding
            query_embedding = self.model.encode([theme_focused_query], convert_to_numpy=True)
            
            # Search index
            max_candidates = min(k * 5, index.ntotal)
            distances, indices = index.search(query_embedding.astype(np.float32), max_candidates)
            
            # Process results
            results = []
            theme_lower = theme.lower()
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                    
                expert_id = id_mapping[idx]
                expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                
                if expert_data:
                    try:
                        metadata = json.loads(expert_data[b'metadata'].decode())
                        
                        # Check for theme relevance
                        knowledge_areas = metadata.get('knowledge_expertise', [])
                        
                        if isinstance(knowledge_areas, list):
                            for area in knowledge_areas:
                                area_lower = str(area).lower()
                                if theme_lower in area_lower:
                                    distance = float(distances[0][i])
                                    base_score = float(1.0 / (1.0 + distance))
                                    metadata['score'] = base_score * 1.5  # Boost for expertise match
                                    results.append(metadata)
                                    break
                        
                    except Exception as e:
                        continue
            
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Fallback vector theme search failed: {e}")
            return []

    def search_experts_by_publication(self, publication: str, k: int = 5, active_only: bool = False, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for experts by publication using the expert-resource link table.
        
        Args:
            publication (str): Publication title or identifier to search for
            k (int): Number of results to return
            active_only (bool): Whether to return only active experts
            min_score (float): Minimum confidence score threshold
            
        Returns:
            List[Dict[str, Any]]: List of matching experts with their confidence scores
        """
        try:
            # Get database connection
            conn = self.db.get_connection()
            matching_experts = []
            
            try:
                with conn.cursor() as cur:
                    # Add active_only filter if requested
                    active_filter = "AND e.is_active = true" if active_only else ""
                    
                    # Search for publications matching the query and get linked experts
                    # Cast JSONB fields to text before using ILIKE
                    cur.execute(f"""
                        SELECT 
                            e.id,
                            e.first_name,
                            e.last_name,
                            e.knowledge_expertise,
                            erl.confidence_score,
                            r.title,
                            r.authors::text
                        FROM expert_resource_links erl
                        JOIN experts_expert e ON erl.expert_id = e.id
                        JOIN resources_resource r ON erl.resource_id = r.id
                        WHERE 
                            (
                                r.title ILIKE %s OR
                                r.authors::text ILIKE %s OR
                                r.abstract ILIKE %s OR
                                r.summary ILIKE %s
                            )
                            {active_filter}
                        ORDER BY erl.confidence_score DESC
                        LIMIT %s
                    """, (
                        f"%{publication}%",  # Title match
                        f"%{publication}%",  # Authors match
                        f"%{publication}%",  # Abstract match
                        f"%{publication}%",  # Summary match
                        k*3  # Get more to filter better
                    ))
                    
                    rows = cur.fetchall()
                    
                    # Process results with relevance scoring
                    seen_experts = set()  # To avoid duplicates
                    for row in rows:
                        expert_id, first_name, last_name, knowledge_expertise, confidence_score, title, authors = row
                        
                        # Skip if we've already seen this expert (keep highest score)
                        if expert_id in seen_experts:
                            continue
                        seen_experts.add(expert_id)
                        
                        # Skip if below minimum score
                        if confidence_score < min_score:
                            continue
                        
                        # Calculate match score based on confidence and publication relevance
                        match_score = float(confidence_score)
                        
                        # Boost score if publication appears in title
                        if title and publication.lower() in title.lower():
                            match_score = min(1.0, match_score * 1.2)
                        
                        matching_experts.append({
                            'id': expert_id,
                            'first_name': first_name or '',
                            'last_name': last_name or '',
                            'knowledge_expertise': self._parse_jsonb(knowledge_expertise),
                            'score': match_score,
                            'publication_match': {
                                'title': title,
                                'authors': authors
                            }
                        })
            
            finally:
                conn.close()
            
            # Sort by score (most relevant first)
            matching_experts.sort(key=lambda x: x.get('score', 0), reverse=True)
            return matching_experts[:k]
            
        except Exception as e:
            logger.error(f"Error in publication search: {e}")
            # Fall back to vector search if database query fails
            try:
                return self._fallback_vector_publication_search(publication, k, active_only, min_score)
            except Exception as fallback_error:
                logger.error(f"Fallback publication search also failed: {fallback_error}")
                return []
    def _fallback_vector_publication_search(self, publication: str, k: int, active_only: bool, min_score: float):
        """Fallback to vector search if direct database query fails"""
        try:
            # Load index and mapping
            index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, 'rb') as f:
                id_mapping = pickle.load(f)
            
            # Create a publication-focused query
            publication_focused_query = f"Publication: {publication} Expert who published {publication}"
            
            # Generate query embedding
            query_embedding = self.model.encode([publication_focused_query], convert_to_numpy=True)
            
            # Search index
            max_candidates = min(k * 5, index.ntotal)
            distances, indices = index.search(query_embedding.astype(np.float32), max_candidates)
            
            # Process results
            results = []
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                    
                expert_id = id_mapping[idx]
                expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                
                if expert_data:
                    try:
                        metadata = json.loads(expert_data[b'metadata'].decode())
                        
                        # For L2 distance, lower is better, so convert to similarity score
                        distance = float(distances[0][i])
                        base_score = float(1.0 / (1.0 + distance))
                        
                        # Add publication context to metadata
                        metadata['score'] = base_score
                        metadata['publication_match'] = {
                            'title': publication,
                            'authors': 'Unknown (vector search fallback)'
                        }
                        
                        results.append(metadata)
                    except Exception as e:
                        continue
            
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Fallback vector publication search failed: {e}")
            return []

    def search_experts_by_designation(self, designation: str, k: int = 5, active_only: bool = False, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for experts by designation using direct database matching.
        
        Args:
            designation (str): Designation to search for
            k (int): Number of results to return
            active_only (bool): Whether to return only active experts
            min_score (float): Minimum similarity score threshold
        
        Returns:
            List[Dict[str, Any]]: List of matching experts
        """
        try:
            # Normalize the designation search term
            designation_lower = designation.lower().strip()
            if not designation_lower:
                return []
                
            # Get database connection
            conn = self.db.get_connection()
            matching_experts = []
            
            try:
                with conn.cursor() as cur:
                    # Add active_only filter if requested
                    active_filter = "AND is_active = true" if active_only else ""
                    
                    # Search for designation in relevant fields
                    # Using only the columns that actually exist in your schema
                    cur.execute(f"""
                        SELECT 
                            id, first_name, last_name, knowledge_expertise, 
                            designation, role_id, unit
                        FROM experts_expert
                        WHERE 
                            (
                                designation ILIKE %s OR
                                search_text ILIKE %s OR
                                CAST(role_id AS TEXT) = %s
                            )
                            {active_filter}
                        LIMIT %s
                    """, (
                        f"%{designation_lower}%",  # Partial match in designation
                        f"%{designation_lower}%",  # Partial match in search_text
                        designation_lower,         # Exact match for role_id
                        k*3                        # Get more to sort better
                    ))
                    
                    rows = cur.fetchall()
                    
                    # Process results with relevance scoring
                    for row in rows:
                        expert_id, first_name, last_name, knowledge_expertise, expert_designation, role_id, unit = row
                        
                        # Calculate designation match score based on where it matched
                        match_score = 0.5  # Base score
                        
                        # Check exact matches in different fields
                        if expert_designation and expert_designation.lower() == designation_lower:
                            match_score = 0.95  # Highest for exact designation match
                        elif expert_designation and designation_lower in expert_designation.lower():
                            match_score = 0.9   # High for partial designation match
                        
                        # Include role_id and unit info if they matched
                        role_match = False
                        if role_id and str(role_id) == designation_lower:
                            match_score = max(match_score, 0.85)
                            role_match = True
                        
                        unit_match = False
                        if unit and designation_lower in unit.lower():
                            match_score = max(match_score, 0.8)
                            unit_match = True
                                
                        matching_experts.append({
                            'id': expert_id,
                            'first_name': first_name or '',
                            'last_name': last_name or '',
                            'knowledge_expertise': self._parse_jsonb(knowledge_expertise),
                            'designation': expert_designation or '',
                            'role_id': role_id,
                            'unit': unit or '',
                            'score': match_score
                        })
            
            finally:
                conn.close()
            
            # Sort by score (most relevant first)
            matching_experts.sort(key=lambda x: x.get('score', 0), reverse=True)
            return matching_experts[:k]
            
        except Exception as e:
            logger.error(f"Error in designation search: {e}")
            # Fall back to vector search if database query fails
            try:
                return self._fallback_vector_designation_search(designation, k, active_only, min_score)
            except Exception as fallback_error:
                logger.error(f"Fallback designation search also failed: {fallback_error}")
                return []

    def _fallback_vector_designation_search(self, designation: str, k: int, active_only: bool, min_score: float):
        """Fallback to vector search if direct database query fails"""
        # Original vector search implementation for designation
        try:
            # Load index and mapping
            index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, 'rb') as f:
                id_mapping = pickle.load(f)
            
            # Create a designation-focused query
            designation_focused_query = f"Designation: {designation} Expert with role {designation}"
            
            # Generate query embedding
            query_embedding = self.model.encode([designation_focused_query], convert_to_numpy=True)
            
            # Search index
            max_candidates = min(k * 5, index.ntotal)
            distances, indices = index.search(query_embedding.astype(np.float32), max_candidates)
            
            # Process results
            results = []
            designation_lower = designation.lower()
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                    
                expert_id = id_mapping[idx]
                expert_data = self.redis_binary.hgetall(f"expert:{expert_id}")
                
                if expert_data:
                    try:
                        metadata = json.loads(expert_data[b'metadata'].decode())
                        
                        # Use the designation field from metadata
                        expert_designation = metadata.get('designation', '').lower()
                        
                        if designation_lower in expert_designation:
                            distance = float(distances[0][i])
                            base_score = float(1.0 / (1.0 + distance))
                            metadata['score'] = base_score * 1.2  # Boost for direct matches
                            results.append(metadata)
                            
                    except Exception as e:
                        continue
            
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Fallback vector designation search failed: {e}")
            return []
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
