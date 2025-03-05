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

# Import graph initializer
from ai_services_api.services.recommendation.graph_initializer import GraphDatabaseInitializer

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

def initialize_graph_task():
    """
    Initialize graph database with proper connection handling
    """
    load_environment_variables()
    
    try:
        # Initialize graph database with custom connection method
        graph_initializer = GraphDatabaseInitializer(db_connection_method=get_db_connection)
        
        # Verify database connectivity before proceeding
        with get_db_cursor() as (cur, conn):
            # Check if required tables exist
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'experts_expert'
                )
            """)
            experts_table_exists = cur.fetchone()[0]
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'resources_resource'
                )
            """)
            resources_table_exists = cur.fetchone()[0]
            
            if not experts_table_exists or not resources_table_exists:
                logger.error("Required database tables do not exist")
                raise Exception("Required database tables not found")
                
            # Check if there are experts to process
            cur.execute("SELECT COUNT(*) FROM experts_expert")
            expert_count = cur.fetchone()[0]
            
            if expert_count == 0:
                logger.warning("No experts found in database. Graph initialization may be incomplete.")
        
        # Proceed with graph initialization
        logger.info("Starting graph database initialization...")
        graph_success = graph_initializer.initialize_graph()
        
        if not graph_success:
            logger.error("Graph initialization failed")
            raise Exception("Graph initialization failed")
        
        # Verify graph initialization
        if not graph_initializer.verify_graph():
            logger.warning("Graph verification failed. Graph may be incomplete.")
        
        logger.info("Graph initialization complete!")
        
    except Exception as e:
        logger.error(f"Graph initialization failed: {e}")
        raise
    finally:
        # Ensure resources are properly cleaned up
        if 'graph_initializer' in locals():
            graph_initializer.close()

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
    'graph_initialization_dag',
    default_args=default_args,
    description='Monthly graph database initialization',
    schedule_interval='@monthly',
    catchup=False
)

initialize_graph_task = PythonOperator(
    task_id='initialize_graph_database',
    python_callable=initialize_graph_task,
    dag=dag
)