from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2

# Database connection utilities - based on your schema
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
    
    in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
    return {
        'host': '167.86.85.127' if in_docker else 'localhost',
        'port': '5432',
        'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
    }

@contextmanager
def get_db_connection(dbname=None):
    """Get database connection with proper error handling and connection cleanup."""
    params = get_db_connection_params()
    if dbname:
        params['dbname'] = dbname
    
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

@contextmanager
def get_db_cursor(autocommit=False):
    """Get database cursor with transaction management."""
    with get_db_connection() as conn:
        conn.autocommit = autocommit
        cur = conn.cursor()
        try:
            yield cur, conn
        except Exception as e:
            if not autocommit:
                conn.rollback()
                logging.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            cur.close()

# Import search index processors
from ai_services_api.services.search.indexing.index_creator import ExpertSearchIndexManager
from ai_services_api.services.search.indexing.redis_index_manager import ExpertRedisIndexManager

# Configure logging
def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Load environment variables
def load_environment_variables():
    """Load environment variables needed for processing."""
    # You can add specific environment variable loading logic here if needed
    logging.info("Environment variables loaded successfully")

# Configure logging
logger = setup_logging()

def create_faiss_index_task():
    """
    Create FAISS search index with proper database connection handling
    """
    load_environment_variables()
    
    try:
        # Create FAISS search index with custom db connection
        index_creator = ExpertSearchIndexManager(db_connection_method=get_db_connection)
        
        # Fetch experts from the database
        with get_db_cursor() as (cur, conn):
            cur.execute("""
                SELECT id, first_name, last_name, knowledge_expertise, domains, fields, 
                       normalized_domains, normalized_fields, normalized_skills, keywords
                FROM experts_expert 
                WHERE is_active = true
            """)
            experts = cur.fetchall()
            
            if not experts:
                logger.warning("No active experts found in the database")
            else:
                logger.info(f"Found {len(experts)} active experts for indexing")
        
        # Create FAISS index based on experts
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
    Create Redis search index with proper database connection handling
    """
    load_environment_variables()
    
    try:
        # Create Redis search index with custom db connection
        redis_creator = ExpertRedisIndexManager(db_connection_method=get_db_connection)
        
        # Clear existing indexes
        if not redis_creator.clear_redis_indexes():
            logger.error("Failed to clear existing Redis indexes")
            raise Exception("Redis index clearing failed")
        
        # Fetch experts and their publications
        with get_db_cursor() as (cur, conn):
            cur.execute("""
                SELECT e.id, e.first_name, e.last_name, e.knowledge_expertise, 
                       e.domains, e.fields, e.normalized_domains, e.normalized_fields,
                       COUNT(erl.resource_id) as publication_count
                FROM experts_expert e
                LEFT JOIN expert_resource_links erl ON e.id = erl.expert_id
                WHERE e.is_active = true
                GROUP BY e.id
            """)
            experts_with_pubs = cur.fetchall()
            
            if not experts_with_pubs:
                logger.warning("No active experts found for Redis indexing")
            else:
                logger.info(f"Found {len(experts_with_pubs)} experts for Redis indexing")
        
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