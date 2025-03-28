import psycopg2
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse
from contextlib import contextmanager
import redis
import time

import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_gemini_api_key():
    """
    Load Gemini API key from environment variables or .env file.
    
    Returns:
        str: Gemini API key
    Raises:
        ValueError if API key is not found
    """
    # Try to load from .env file
    load_dotenv()
    
    # Possible environment variable names
    api_key_vars = [
        'GEMINI_API_KEY', 
        'GOOGLE_API_KEY', 
        'AI_API_KEY'
    ]
    
    # Try each possible variable name
    for var_name in api_key_vars:
        api_key = os.getenv(var_name)
        if api_key:
            logging.info(f"Gemini API key found via {var_name}")
            return api_key
    
    # If no API key found
    logging.error("No Gemini API key found in environment variables")
    raise ValueError("Gemini API key is not set. Please set it in your .env file or environment variables.")

def _setup_gemini():
    """
    Configure Gemini API with loaded API key.
    
    Returns:
        Configured Gemini model
    """
    # Load API key
    api_key = load_gemini_api_key()
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Select and return a model
    try:
        # Preferred model names
        preferred_models = [
            'gemini-1.5-pro-latest',
            'models/gemini-1.5-pro-latest',
            'models/gemini-1.0-pro',
            'gemini-pro'
        ]
        
        # Find the first available model
        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(model_name)
                logging.info(f"Using Gemini model: {model_name}")
                return model
            except Exception as model_error:
                logging.warning(f"Could not initialize model {model_name}: {model_error}")
        
        raise ValueError("No suitable Gemini model could be found")
    
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {e}")
        raise

def get_db_connection_params():
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
    
    return {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
    }

@contextmanager
def get_db_connection():
    """Get database connection with proper error handling and cleanup."""
    params = get_db_connection_params()
    conn = None
    try:
        conn = psycopg2.connect(**params)
        logging.info(f"Connected to database: {params['dbname']} at {params['host']}")
        yield conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed")

class DatabaseClassificationManager:
    """
    Manages database operations for classification results.
    """
    
    @staticmethod
    def update_resource_classification(
        resource_id: int, 
        domains: List[str], 
        topics: Dict[str, List[str]]
    ) -> bool:
        """
        Update the classification for a specific resource.
        
        Args:
            resource_id: ID of the resource to update
            domains: List of domains 
            topics: Dictionary of topics by domain
        
        Returns:
            Boolean indicating successful update
        """
        try:
            # Validate inputs
            if not domains or len(domains) == 0:
                logging.warning(f"Empty domains provided for resource {resource_id}, using 'Uncategorized'")
                domains = ["Uncategorized"]
                
            if not topics or len(topics) == 0:
                logging.warning(f"Empty topics provided for resource {resource_id}, using default")
                topics = {"Uncategorized": ["General"]}
            
            # Log what we're about to update
            logging.info(f"Updating resource {resource_id} with domains: {domains}")
            logging.info(f"Using topics: {topics}")
            
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare topics as JSONB
                    topics_jsonb = json.dumps(topics)
                    logging.info(f"Converted topics to JSON: {topics_jsonb}")
                    
                    # SQL to update domains and topics
                    update_query = """
                    UPDATE resources_resource 
                    SET 
                        domains = %s, 
                        topics = %s, 
                        updated_at = CURRENT_TIMESTAMP 
                    WHERE id = %s
                    """
                    
                    # Log the query and parameters
                    logging.info(f"Executing query: {update_query}")
                    logging.info(f"With parameters: domains={domains}, topics_jsonb={topics_jsonb}, resource_id={resource_id}")
                    
                    # Execute the update
                    cur.execute(update_query, (
                        domains,  # PostgreSQL text array 
                        topics_jsonb,  # JSONB 
                        resource_id
                    ))
                    
                    # Commit the transaction
                    conn.commit()
                    
                    # Check if a row was actually updated
                    if cur.rowcount > 0:
                        logging.info(f"Successfully updated classification for resource {resource_id}")
                        # Verify the update worked by querying the record
                        verify_query = "SELECT domains, topics FROM resources_resource WHERE id = %s"
                        cur.execute(verify_query, (resource_id,))
                        result = cur.fetchone()
                        if result:
                            logging.info(f"Verification - Updated values: domains={result[0]}, topics={result[1]}")
                        return True
                    else:
                        logging.warning(f"No resource found with ID {resource_id}")
                        return False
        
        except Exception as e:
            logging.error(f"Error updating resource classification: {e}")
            # Log the full exception traceback for debugging
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    @staticmethod
    def get_unclassified_resources(limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve unclassified resources from the database.
        
        Args:
            limit: Maximum number of resources to retrieve
        
        Returns:
            List of unclassified resources
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Query to find resources without classifications
                    query = """
                    SELECT id, title, abstract 
                    FROM resources_resource 
                    WHERE 
                        domains IS NULL 
                        OR array_length(domains, 1) = 0 
                        OR topics IS NULL 
                    LIMIT %s
                    """
                    
                    cur.execute(query, (limit,))
                    
                    # Fetch results
                    resources = []
                    for row in cur.fetchall():
                        resources.append({
                            'id': row[0],
                            'title': row[1],
                            'abstract': row[2]
                        })
                    
                    logging.info(f"Retrieved {len(resources)} unclassified resources")
                    return resources
        
        except Exception as e:
            logging.error(f"Error retrieving unclassified resources: {e}")
            return []
    
    @staticmethod
    def get_domain_classification_stats() -> Dict[str, int]:
        """
        Get statistics on domain classifications.
        
        Returns:
            Dictionary of domain counts
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Query to count resources per domain
                    query = """
                    SELECT 
                        unnest(domains) as domain, 
                        COUNT(*) as resource_count 
                    FROM resources_resource 
                    WHERE domains IS NOT NULL 
                      AND array_length(domains, 1) > 0
                    GROUP BY domain 
                    ORDER BY resource_count DESC
                    """
                    
                    cur.execute(query)
                    
                    # Convert results to dictionary
                    domain_stats = dict(cur.fetchall())
                    
                    logging.info("Retrieved domain classification statistics")
                    return domain_stats
        
        except Exception as e:
            logging.error(f"Error retrieving domain classification stats: {e}")
            return {}

class ResourceIndexManager:
    """Manages resource indexing and embedding for classification."""
    
    def __init__(self):
        """Initialize the resource index manager."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize embedding model with offline handling
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            
            # Set self.embedding_model to None initially
            self.embedding_model = None
            
            try:
                # First try loading with standard mode
                from sentence_transformers import SentenceTransformer
                logger.info(f"Attempting to load model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
            except (OSError, IOError) as e:
                logger.warning(f"Failed to load model online: {e}. Trying offline mode...")
                try:
                    # Try with local_files_only=True (offline)
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(model_name, local_files_only=True)
                except Exception as offline_error:
                    logger.error(f"Failed to load model in offline mode: {offline_error}")
                    # Keep self.embedding_model as None - we'll use fallback methods
                    logger.warning("Using fallback text encoding method")
            
            # Setup Redis connections
            self.setup_redis_connections()
            
            # Initialize database manager
            self.db_manager = DatabaseClassificationManager()
            
            logger.info("ResourceIndexManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ResourceIndexManager: {e}")
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
        if not isinstance(text, str):
            text = str(text)
            
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

    def _create_resource_text_content(self, resource: Dict[str, Any]) -> str:
        """Create combined text content for resource embedding."""
        try:
            # Start with title as the most important information
            text_parts = []
            if resource.get('title'):
                text_parts.append(f"Title: {str(resource['title']).strip()}")
            else:
                text_parts.append("Title: Unknown Resource")
                
            # Add abstract if available
            if resource.get('abstract'):
                text_parts.append(f"Abstract: {str(resource['abstract']).strip()}")
            
            # Join all parts and ensure we have content
            final_text = '\n'.join(text_parts)
            if not final_text.strip():
                return "Unknown Resource"
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error creating text content for resource {resource.get('id', 'Unknown')}: {e}")
            return "Error Processing Resource"

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
                'abstract': str(resource.get('abstract', ''))[:500] if resource.get('abstract') else ''
            }
            
            pipeline.hset(f"meta:{base_key}", mapping=metadata)
            pipeline.execute()
            
        except Exception as e:
            pipeline.reset()
            raise e

    def get_resource_embedding(self, resource_id: str) -> Optional[np.ndarray]:
        """Retrieve resource embedding from Redis."""
        try:
            embedding_bytes = self.redis_binary.get(f"emb:resource:{resource_id}")
            if embedding_bytes:
                return np.frombuffer(embedding_bytes, dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Error retrieving resource embedding: {e}")
            return None

    def get_resource_metadata(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve resource metadata from Redis."""
        try:
            metadata = self.redis_text.hgetall(f"meta:resource:{resource_id}")
            return metadata if metadata else None
        except Exception as e:
            logger.error(f"Error retrieving resource metadata: {e}")
            return None

    def clear_resource_indexes(self) -> bool:
        """Clear all resource Redis indexes."""
        try:
            patterns = ['text:resource:*', 'emb:resource:*', 'meta:resource:*']
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = self.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            logger.info("Cleared all resource Redis indexes")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis indexes: {e}")
            return False

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


class DomainClassifier:
    """
    Handles domain classification using both embedding-based similarity
    and Gemini API for classification.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = _setup_gemini()
        
        # Set embedding_model to None initially
        self.embedding_model = None
        
        # Initialize embedding model with offline handling
        try:
            # First try standard loading
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
        except (OSError, IOError) as e:
            logger.warning(f"Failed to load embedding model: {e}. Trying offline mode...")
            try:
                # Try offline mode
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name, local_files_only=True)
            except Exception as offline_error:
                logger.error(f"Failed to load model in offline mode: {offline_error}")
                # Keep self.embedding_model as None - will use fallback methods
                logger.warning("Using fallback text analysis methods without embeddings")
        
        self.db_manager = DatabaseClassificationManager()
        self.resource_manager = ResourceIndexManager()
        self.domain_structure = self.get_existing_domain_structure()
        
        logger.info("DomainClassifier initialized successfully")
    
    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model is not available."""
        if not isinstance(text, str):
            text = str(text)
            
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
    
    def get_existing_domain_structure(self) -> Dict[str, List[str]]:
        """
        Retrieve existing domain structure, first from database stats, 
        then fallback to default structure.
        
        Returns:
            A dictionary of domains with their associated topics
        """
        # Try to get domains from database stats
        domain_stats = self.db_manager.get_domain_classification_stats()
        
        if domain_stats:
            # If we have domain stats, use them as the basis for our structure
            return {
                domain: [f"{domain} Topic {i+1}" for i in range(3)]
                for domain in domain_stats.keys()
            }
        
        # Fallback to default structure
        return {
            "Science": ["Physics", "Biology", "Chemistry"],
            "Social Sciences": ["Sociology", "Psychology", "Anthropology"],
            "Humanities": ["Literature", "History", "Philosophy"],
            "Health Sciences": ["Public Health", "Medicine", "Epidemiology"],
            "Technology": ["AI", "Machine Learning", "Data Science"]
        }
    
    def get_unclassified_publications(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve unclassified publications from the database.
        
        Args:
            limit: Number of publications to retrieve
        
        Returns:
            A list of unclassified publication dictionaries
        """
        return self.db_manager.get_unclassified_resources(limit)
    
    def _generate_content(self, prompt, temperature=0.3, max_tokens=2048, max_retries=3, retry_delay=60):
        """Generate content with rate limiting and retries."""
        for attempt in range(max_retries):
            try:
                # Your existing generation code here
                response = self.model.generate_content(prompt, ...)
                return response.text.strip() if response and response.text else None
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    logging.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Error generating content: {e}")
                    return None
    
    def _create_embedding_for_publication(self, publication: Dict[str, Any]) -> np.ndarray:
        """Create embedding for publication text content with fallback method."""
        try:
            # Create text content for embedding
            text_content = ""
            if publication.get('title'):
                text_content += f"Title: {publication['title']} "
            if publication.get('abstract'):
                text_content += f"Abstract: {publication['abstract']}"
            
            if not text_content:
                # Return a zero embedding if no content
                return np.zeros(384)
                
            # Generate embedding
            if self.embedding_model is not None:
                # Use SentenceTransformer if available
                return self.embedding_model.encode(text_content)
            else:
                # Fallback to simple character-based hashing if model not available
                logger.warning("Using fallback embedding method")
                return self._create_fallback_embedding(text_content)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return a zero embedding on error
            return np.zeros(384)

    def _find_similar_domains(self, publication_embedding: np.ndarray, top_n=3) -> List[Tuple[str, float]]:
        """
        Find similar classified publications and their domains using embeddings.
        
        Returns:
            List of (domain_name, similarity_score) tuples
        """
        try:
            # Get all domain names from structure
            all_domains = list(self.domain_structure.keys())
            
            # Check if we can use proper embeddings
            if self.embedding_model is None:
                # If no embedding model, return random domain with low confidence
                import random
                domain = random.choice(all_domains)
                return [(domain, 0.5)]
            
            # Generate embeddings for domain names
            domain_embeddings = {}
            for domain in all_domains:
                try:
                    domain_embeddings[domain] = self.embedding_model.encode(f"Domain: {domain}")
                except:
                    # Fallback to character-based embedding
                    domain_embeddings[domain] = self._create_fallback_embedding(f"Domain: {domain}")
            
            # Calculate similarities
            similarities = []
            for domain, embedding in domain_embeddings.items():
                similarity = np.dot(publication_embedding, embedding) / (
                    np.linalg.norm(publication_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((domain, float(similarity)))
            
            # Sort by similarity (highest first) and return top N
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding similar domains: {e}")
            # Return a random domain with low confidence
            import random
            return [(random.choice(all_domains), 0.5)]
    
    def classify_single_publication(
        self, 
        publication: Dict[str, Any], 
        domain_structure: Dict[str, List[str]]
    ) -> Tuple[Optional[str], Optional[List[str]], bool]:
        """
        Classify a single publication into a domain and topics using both
        embedding similarity and Gemini API.
        
        Args:
            publication: Publication to classify
            domain_structure: Current domain structure
        
        Returns:
            Tuple of (domain, topics, success)
        """
        try:
            # First, try embedding-based approach
            publication_embedding = self._create_embedding_for_publication(publication)
            
            if publication_embedding is not None:
                # Store the embedding in Redis for future use
                text_content = f"Title: {publication.get('title', '')} Abstract: {publication.get('abstract', '')}"
                self.resource_manager._store_resource_data(
                    publication, 
                    text_content, 
                    publication_embedding
                )
                
                # Find similar domains
                similar_domains = self._find_similar_domains(publication_embedding)
                
                if similar_domains:
                    # Get top domain from embedding similarity
                    top_domain, similarity_score = similar_domains[0]
                    
                    # If confidence is high enough, use the embedding result
                    if similarity_score > 0.7:
                        topics = domain_structure.get(top_domain, [])[:3]
                        logger.info(f"Publication {publication.get('id')} classified via embedding with score {similarity_score}")
                        return top_domain, topics, True
            
            # Fallback to Gemini API for classification
            # Prepare prompt for classification
            prompt = f"""
            Classify the following publication into an appropriate domain and provide 3 relevant topics:
            
            Title: {publication.get('title', '')}
            Abstract: {publication.get('abstract', '')}
            
            Available domains: {', '.join(domain_structure.keys())}
            
            Provide the response in this JSON format:
            {{
                "domain": "Chosen Domain",
                "topics": ["Topic 1", "Topic 2", "Topic 3"]
            }}
            """
            
            # Generate classification using Gemini
            classification_str = self._generate_content(prompt)
            
            if classification_str:
                try:
                    # Parse the generated classification
                    classification = json.loads(classification_str)
                    domain = classification.get('domain')
                    topics = classification.get('topics', [])
                    
                    # Validate and fallback if needed
                    if not domain or domain not in domain_structure:
                        domain = list(domain_structure.keys())[0]
                    
                    if not topics:
                        topics = domain_structure.get(domain, [])[:3]
                    
                    logger.info(f"Publication {publication.get('id')} classified via Gemini API")
                    return domain, topics, True
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse classification: {e}")
            
            # Fallback classification logic
            if "AI" in publication.get('title', '').lower():
                return "Technology", ["AI", "Machine Learning", "Data Science"], True
            elif "health" in publication.get('title', '').lower():
                return "Health Sciences", ["Public Health", "Medicine", "Epidemiology"], True
            
            # Default to first domain if no match
            first_domain = list(domain_structure.keys())[0]
            return first_domain, domain_structure[first_domain][:3], True
        
        except Exception as e:
            logger.error(f"Error classifying publication: {e}")
            return None, None, False
    
    def update_publication_classification(
        self, 
        pub_id: str, 
        domain: str, 
        topics: List[str]
    ) -> bool:
        """
        Update publication classification in the database.
        
        Args:
            pub_id: Publication ID
            domain: Classified domain
            topics: Classified topics
        
        Returns:
            Boolean indicating successful update
        """
        try:
            # Convert pub_id to integer if needed
            pub_id_int = int(pub_id)
            
            # Prepare topics dictionary
            topics_dict = {domain: topics}
            
            # Use database manager to update classification
            return self.db_manager.update_resource_classification(
                resource_id=pub_id_int,
                domains=[domain],
                topics=topics_dict
            )
        except Exception as e:
            logger.error(f"Error updating publication classification: {e}")
            return False


def classify_publications(
    batch_size: int = 5, 
    publications_per_batch: int = 1,
    domain_batch_size: int = 3
) -> bool:
    """
    Classify publications using both embedding similarity and Gemini API.
    
    Args:
        batch_size: Number of batches to process
        publications_per_batch: Publications per batch
        domain_batch_size: Sample size for domain generation
    
    Returns:
        Boolean indicating successful classification
    """
    try:
        # Initialize classifier
        classifier = DomainClassifier()
        
        # Refresh domain structure
        domain_structure = classifier.domain_structure
        
        logging.info(f"Starting classification with {len(domain_structure)} domains")
        
        # Process publications in small batches
        total_processed = 0
        total_classified = 0
        processed_publications = set()
        
        for batch in range(batch_size):
            logging.info(f"Processing batch {batch + 1}/{batch_size}")
            
            # Get publications for this batch
            publications = classifier.get_unclassified_publications(publications_per_batch)
            
            if not publications:
                logging.info("No more unclassified publications")
                break
            
            # Process each publication in the batch
            for publication in publications:
                pub_id = publication.get('id')
                
                # Skip if already processed
                if pub_id in processed_publications:
                    logging.info(f"Publication {pub_id} already processed")
                    continue
                
                # Classify the publication
                domain, topics, success = classifier.classify_single_publication(
                    publication, domain_structure)
                
                if success and domain and topics:
                    # Update the publication classification
                    if classifier.update_publication_classification(pub_id, domain, topics):
                        # Increment the domain count
                        total_classified += 1
                        processed_publications.add(pub_id)
                
                total_processed += 1
            
            # Every few batches, refresh domain structure
            if batch > 0 and batch % 5 == 0:
                logging.info("Refreshing domain structure...")
                domain_structure = classifier.get_existing_domain_structure()
        
        logging.info(f"Classification complete: {total_classified}/{total_processed} publications classified")
        return True
        
    except Exception as e:
        logging.error(f"Error in classify_publications: {e}")
        return False