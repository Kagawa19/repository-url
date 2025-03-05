from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import logging
import sys
import numpy as np
import importlib
import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'search_indexes_dag',
    default_args=default_args,
    description='Create and update search indexes',
    schedule_interval=None,
)

# Set up temporary directory
TEMP_DIR = os.path.join(os.getenv('AIRFLOW_HOME', '/opt/airflow'), 'temp')

# Simple embedding class that doesn't require external dependencies
class SimpleEmbedder:
    """A simple embedding class that provides random but consistent embeddings."""
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized SimpleEmbedder with dimension {embedding_dim}")
    
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        """Generate embeddings using text hashing for consistency."""
        if not isinstance(texts, list):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts")
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        for i, text in enumerate(texts):
            # Use text hash as random seed for consistency
            seed = hash(text) % 2**32
            np.random.seed(seed)
            embeddings[i] = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings

def monkey_patch_expert_search_index_manager():
    """
    Apply monkey patches to ExpertSearchIndexManager to avoid SentenceTransformer dependency.
    This function patches the class before it's instantiated.
    """
    try:
        # Import the module containing ExpertSearchIndexManager
        module = importlib.import_module('ai_services_api.services.search.indexing.index_creator')
        
        # Store the original __init__ method
        original_init = module.ExpertSearchIndexManager.__init__
        
        # Define a new __init__ method that doesn't use SentenceTransformer
        def patched_init(self):
            """Patched initialization that uses SimpleEmbedder instead of SentenceTransformer."""
            logger.info("Using patched ExpertSearchIndexManager.__init__")
            
            # Set up paths (similar to what the original would do)
            if hasattr(self, 'setup_paths'):
                self.setup_paths()
            else:
                # Fallback path setup if setup_paths doesn't exist
                import pathlib
                current_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
                self.models_dir = os.path.join(TEMP_DIR, 'models')
                os.makedirs(self.models_dir, exist_ok=True)
                self.index_path = os.path.join(self.models_dir, 'expert_faiss_index.idx')
                self.mapping_path = os.path.join(self.models_dir, 'expert_mapping.pkl')
            
            # Set up Redis connections (similar to what the original would do)
            if hasattr(self, 'setup_redis'):
                self.setup_redis()
            
            # Use our SimpleEmbedder instead of SentenceTransformer
            logger.info("Setting up SimpleEmbedder in patched init")
            self.model = SimpleEmbedder(embedding_dim=384)
            
            # Set up database connector if needed
            if hasattr(module, 'DatabaseConnector'):
                self.db = module.DatabaseConnector()
            
            logger.info("Patched initialization complete")
        
        # Apply the patch
        module.ExpertSearchIndexManager.__init__ = patched_init
        logger.info("Successfully monkey-patched ExpertSearchIndexManager.__init__")
        
        return True
    except Exception as e:
        logger.error(f"Failed to apply monkey patch: {e}", exc_info=True)
        return False

def monkey_patch_redis_index_manager():
    """
    Apply monkey patches to ExpertRedisIndexManager to avoid SentenceTransformer dependency.
    """
    try:
        # Import the module containing ExpertRedisIndexManager
        module = importlib.import_module('ai_services_api.services.search.indexing.redis_index_manager')
        
        # Store the original __init__ method
        original_init = module.ExpertRedisIndexManager.__init__
        
        # Define a new __init__ method that doesn't use SentenceTransformer
        def patched_init(self):
            """Patched initialization that uses SimpleEmbedder instead of SentenceTransformer."""
            logger.info("Using patched ExpertRedisIndexManager.__init__")
            
            try:
                # Initialize database connector if needed
                if hasattr(module, 'DatabaseConnector'):
                    self.db = module.DatabaseConnector()
                
                # Set up Redis connections without the SentenceTransformer
                if hasattr(self, 'setup_redis_connections'):
                    self.setup_redis_connections()
                
                # Use our SimpleEmbedder instead of SentenceTransformer
                logger.info("Setting up SimpleEmbedder in Redis index manager")
                self.embedding_model = SimpleEmbedder(embedding_dim=384)
                
                logger.info("Patched Redis index manager initialization complete")
            except Exception as internal_e:
                logger.error(f"Error in patched init: {internal_e}", exc_info=True)
                raise
        
        # Apply the patch
        module.ExpertRedisIndexManager.__init__ = patched_init
        logger.info("Successfully monkey-patched ExpertRedisIndexManager.__init__")
        
        return True
    except Exception as e:
        logger.error(f"Failed to apply monkey patch for Redis manager: {e}", exc_info=True)
        return False

def create_faiss_index_task():
    """
    Task to create a FAISS index using the patched ExpertSearchIndexManager.
    """
    try:
        # Make sure the temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Apply monkey patches
        logger.info("Applying monkey patches to ExpertSearchIndexManager")
        if not monkey_patch_expert_search_index_manager():
            raise Exception("Failed to apply necessary patches")
            
        # Import the patched class and create an instance
        from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
        
        # Create the patched index manager
        logger.info("Creating instance of patched ExpertSearchIndexManager")
        index_creator = ExpertSearchIndexManager()
        
        # Create the FAISS index
        logger.info("Starting FAISS index creation")
        success = index_creator.create_faiss_index()
        
        if not success:
            logger.error("FAISS index creation failed")
            raise Exception("FAISS index creation failed")
            
        logger.info("FAISS index creation completed successfully")
        return "FAISS index created successfully"
        
    except Exception as e:
        logger.error(f"Error in create_faiss_index_task: {e}", exc_info=True)
        raise

def create_redis_index_task():
    """
    Task to create Redis indexes using patched ExpertRedisIndexManager.
    """
    try:
        # Make sure the temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Apply monkey patches
        logger.info("Applying monkey patches to ExpertRedisIndexManager")
        if not monkey_patch_redis_index_manager():
            raise Exception("Failed to apply necessary patches")
        
        # Import the patched class
        from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager
        
        # Create the patched index manager
        logger.info("Creating instance of patched ExpertRedisIndexManager")
        redis_creator = ExpertRedisIndexManager()
        
        # Clear existing Redis indexes
        logger.info("Clearing existing Redis indexes")
        clear_success = redis_creator.clear_redis_indexes()
        if not clear_success:
            logger.warning("Failed to clear Redis indexes, continuing anyway")
        
        # Create new Redis indexes
        logger.info("Creating new Redis indexes")
        create_success = redis_creator.create_redis_index()
        
        if not create_success:
            logger.error("Redis index creation failed")
            raise Exception("Redis index creation failed")
        
        logger.info("Redis index creation completed successfully")
        return "Redis indexes created successfully"
        
    except Exception as e:
        logger.error(f"Error in create_redis_index_task: {e}", exc_info=True)
        raise

# Define the FAISS index creation task
faiss_index_task = PythonOperator(
    task_id='create_faiss_search_index',
    python_callable=create_faiss_index_task,
    dag=dag,
)

# Define the Redis index creation task
redis_index_task = PythonOperator(
    task_id='create_redis_search_indexes',
    python_callable=create_redis_index_task,
    dag=dag,
)

# Set task dependencies
faiss_index_task >> redis_index_task