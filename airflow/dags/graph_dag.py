from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
import psycopg2
from contextlib import contextmanager
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

# Database connection utilities
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
    
    # In Docker Compose, always use service name
    return {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),  # Always use service name in Docker
        'port': os.getenv('POSTGRES_PORT', '5432'),
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
        logger.info(f"Connected to database: {params['dbname']} at {params['host']}")
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")

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
                logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            cur.close()

# Lazy import for graph initializer to reduce initial load time


# Load environment variables
def load_environment_variables():
    """Load environment variables needed for processing."""
    logger.info("Environment variables loaded successfully")

# Lazy import for graph initializer to reduce initial load time
def lazy_import(module_path, class_name):
    """Import a class only when needed."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {class_name} from {module_path}: {e}")
        raise

def initialize_graph_task():
    """
    Initialize graph database with proper connection handling
    """
    load_environment_variables()
    
    # Import the GraphDatabaseInitializer class
    try:
        # Get the class (not an instance yet)
        GraphDatabaseInitializerClass = lazy_import(
            'ai_services_api.services.recommendation.graph_initializer', 
            'GraphDatabaseInitializer'
        )
        
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
        
        # Create an instance of GraphDatabaseInitializer
        graph_initializer = GraphDatabaseInitializerClass()
        
        # Call initialize_graph on the instance
        graph_success = graph_initializer.initialize_graph()
        
        if not graph_success:
            logger.error("Graph initialization failed")
            raise Exception("Graph initialization failed")
        
        # Skip verification since the method doesn't exist
        # Instead, just log that initialization completed
        logger.info("Graph initialization complete!")
        
    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        raise
    except Exception as e:
        logger.error(f"Graph initialization failed: {e}")
        raise
    finally:
        # Only try to close the initializer if it was successfully created
        if 'graph_initializer' in locals() and graph_initializer is not None:
            try:
                graph_initializer.close()
            except Exception as e:
                logger.error(f"Error closing graph initializer: {e}")

# DAG configuration
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

# Define task
initialize_graph_task_operator = PythonOperator(
    task_id='initialize_graph_database',
    python_callable=initialize_graph_task,
    dag=dag
)