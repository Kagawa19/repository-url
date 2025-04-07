from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import logging
import os

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
clear_indexes_dag = DAG(
    'clear_search_indexes_dag',
    default_args=default_args,
    description='Clear search indexes from FAISS and Redis',
    schedule_interval=None,
)

def clear_faiss_index_task():
    """
    Task to clear the FAISS index.
    """
    try:
        # Import the ExpertSearchIndexManager
        from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
        import os
        from pathlib import Path
        
        # Create an instance of the index manager
        logger.info("Creating instance of ExpertSearchIndexManager")
        index_manager = ExpertSearchIndexManager()
        
        # Get paths to index files
        index_path = index_manager.index_path
        mapping_path = index_manager.mapping_path
        
        # Check if files exist and remove them
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info(f"Removed FAISS index file: {index_path}")
        else:
            logger.info(f"FAISS index file not found at: {index_path}")
            
        if os.path.exists(mapping_path):
            os.remove(mapping_path)
            logger.info(f"Removed ID mapping file: {mapping_path}")
        else:
            logger.info(f"ID mapping file not found at: {mapping_path}")
        
        # Clear Redis keys related to experts
        try:
            logger.info("Clearing expert-related Redis keys")
            for key_pattern in ["expert:*"]:
                cursor = 0
                while True:
                    cursor, keys = index_manager.redis_client.scan(cursor, match=key_pattern, count=100)
                    if keys:
                        index_manager.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            logger.info("Successfully cleared expert-related Redis keys")
        except Exception as redis_err:
            logger.error(f"Error clearing Redis keys for experts: {redis_err}")
        
        logger.info("FAISS index clearing completed successfully")
        return "FAISS index cleared successfully"
        
    except Exception as e:
        logger.error(f"Error in clear_faiss_index_task: {e}", exc_info=True)
        raise

def clear_redis_index_task():
    """
    Task to clear Redis indexes using ExpertRedisIndexManager.
    """
    try:
        # Set environment variables for offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # Import the Redis index manager
        from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager
        
        # Create the Redis index manager
        logger.info("Creating instance of ExpertRedisIndexManager")
        redis_manager = ExpertRedisIndexManager()
        
        # Clear existing Redis indexes
        logger.info("Clearing Redis indexes")
        if hasattr(redis_manager, 'clear_redis_indexes'):
            clear_success = redis_manager.clear_redis_indexes()
            if not clear_success:
                logger.error("Failed to clear Redis indexes")
                raise Exception("Redis index clearing failed")
            
            logger.info("Redis indexes cleared successfully")
            return "Redis indexes cleared successfully"
        else:
            # If the clear_redis_indexes method doesn't exist, attempt to clear manually
            logger.info("clear_redis_indexes method not found, attempting manual clearing")
            
            # Clear expert indexes
            expert_patterns = ['text:expert:*', 'emb:expert:*', 'meta:expert:*']
            for pattern in expert_patterns:
                cursor = 0
                while True:
                    cursor, keys = redis_manager.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        redis_manager.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            # Clear resource/publication indexes
            resource_patterns = [
                'text:resource:*', 'emb:resource:*', 'meta:resource:*',
                'text:publication:*', 'emb:publication:*', 'meta:publication:*'
            ]
            for pattern in resource_patterns:
                cursor = 0
                while True:
                    cursor, keys = redis_manager.redis_text.scan(cursor, match=pattern, count=100)
                    if keys:
                        redis_manager.redis_text.delete(*keys)
                    if cursor == 0:
                        break
            
            logger.info("Manual clearing of Redis indexes completed")
            return "Redis indexes cleared manually"
        
    except Exception as e:
        logger.error(f"Error in clear_redis_index_task: {e}", exc_info=True)
        raise

# Define the FAISS index clearing task
clear_faiss_task = PythonOperator(
    task_id='clear_faiss_search_index',
    python_callable=clear_faiss_index_task,
    dag=clear_indexes_dag,
)

# Define the Redis index clearing task
clear_redis_task = PythonOperator(
    task_id='clear_redis_search_indexes',
    python_callable=clear_redis_index_task,
    dag=clear_indexes_dag,
)

# Set task dependencies - can run in parallel if desired
# If you want them to run sequentially, uncomment the line below
clear_faiss_task >> clear_redis_task