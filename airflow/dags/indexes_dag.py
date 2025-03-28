from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import logging
import sys

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

def create_faiss_index_task():
    """
    Task to create a FAISS index using ExpertSearchIndexManager.
    """
    try:
        # Import the ExpertSearchIndexManager directly
        from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
        
        # Create an instance of the index manager
        logger.info("Creating instance of ExpertSearchIndexManager")
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
    Task to create Redis indexes using ExpertRedisIndexManager.
    """
    try:
        # Set environment variables for offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # Import the Redis index manager directly
        from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager
        
        # Create the Redis index manager
        logger.info("Creating instance of ExpertRedisIndexManager")
        redis_creator = ExpertRedisIndexManager()
        
        # Clear existing Redis indexes
        logger.info("Clearing existing Redis indexes")
        if hasattr(redis_creator, 'clear_redis_indexes'):
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