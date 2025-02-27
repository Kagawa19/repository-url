from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow_utils import setup_logging, load_environment_variables

# Import search index processors
from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager

# Configure logging
logger = setup_logging()

def create_faiss_index_task():
    """
    Create FAISS search index
    """
    load_environment_variables()
    
    try:
        # Create FAISS search index
        index_creator = ExpertSearchIndexManager()
        if not index_creator.create_faiss_index():
            logger.error("FAISS search index creation failed")
            raise Exception("FAISS search index creation failed")
        
        logger.info("FAISS search index creation complete!")
    
    except Exception as e:
        logger.error(f"FAISS search index creation failed: {e}")
        raise

def create_redis_index_task():
    """
    Create Redis search index
    """
    load_environment_variables()
    
    try:
        # Create Redis search index
        redis_creator = ExpertRedisIndexManager()
        if not (redis_creator.clear_redis_indexes() and redis_creator.create_redis_index()):
            logger.error("Redis search index creation failed")
            raise Exception("Redis search index creation failed")
        
        logger.info("Redis search index creation complete!")
    
    except Exception as e:
        logger.error(f"Redis search index creation failed: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'search_indexes_dag',
    default_args=default_args,
    description='Monthly search indexes initialization',
    schedule_interval='@monthly',
    catchup=False
)

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