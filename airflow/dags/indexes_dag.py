from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

# Lazy import to reduce initial load time
def lazy_import(module_path, class_name):
    """
    Lazily import a class to reduce initial import time
    """
    def import_class():
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    return import_class

# Configure logging
logger = logging.getLogger(__name__)

# Prepare arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_environment_variables():
    """Load environment variables needed for processing."""
    logger.info("Environment variables loaded successfully")

def create_faiss_index_task():
    """
    Create FAISS search index with minimal dependencies
    """
    load_environment_variables()
    
    # Import here to reduce initial load time
    from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
    
    try:
        # Create FAISS search index 
        index_creator = ExpertSearchIndexManager()
        
        # Create FAISS index 
        if not index_creator.create_faiss_index():
            logger.error("FAISS search index creation failed")
            raise Exception("FAISS search index creation failed")
        
        # Verify index creation
        if not index_creator.verify_index():
            logger.error("FAISS index verification failed")
            raise Exception("FAISS index verification failed")
            
        logger.info("FAISS search index creation complete!")
        
    except Exception as e:
        logger.error(f"FAISS search index creation failed: {e}")
        raise
    finally:
        # Ensure proper cleanup
        if 'index_creator' in locals():
            index_creator.close()

def create_redis_index_task():
    """
    Create Redis search index with minimal dependencies
    """
    load_environment_variables()
    
    # Import here to reduce initial load time
    from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager
    
    try:
        # Create Redis search index
        redis_creator = ExpertRedisIndexManager()
        
        # Clear existing indexes
        if not redis_creator.clear_redis_indexes():
            logger.error("Failed to clear existing Redis indexes")
            raise Exception("Redis index clearing failed")
        
        # Create Redis index
        if not redis_creator.create_redis_index():
            logger.error("Redis search index creation failed")
            raise Exception("Redis search index creation failed")
        
        # Verify Redis index
        if not redis_creator.verify_index():
            logger.error("Redis index verification failed")
            raise Exception("Redis index verification failed")
            
        logger.info("Redis search index creation complete!")
        
    except Exception as e:
        logger.error(f"Redis search index creation failed: {e}")
        raise
    finally:
        # Ensure proper cleanup
        if 'redis_creator' in locals():
            redis_creator.close()

# Create DAG
dag = DAG(
    'search_indexes_dag',
    default_args=default_args,
    description='Monthly search indexes initialization',
    schedule_interval='@monthly',
    catchup=False
)

# Define tasks
create_faiss_index_task = PythonOperator(
    task_id='create_faiss_search_index',
    python_callable=create_faiss_index_task,
    dag=dag
)

create_redis_index_task = PythonOperator(
    task_id='create_redis_search_index',
    python_callable=create_redis_index_task,
    dag=dag
)

# Set task dependencies
create_faiss_index_task >> create_redis_index_task