from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
from contextlib import contextmanager
from urllib.parse import urlparse
import psycopg2
import json

# Import utility functions and email notification functions
from airflow_utils import setup_logging, load_environment_variables, success_email, failure_email

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database connection utilities - consistent with your schema
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

# Simple database wrapper to match original API
class DatabaseManager:
    """Database manager for consistent API with original."""
    
    def get_all_publications(self):
        """Get all publications from the database."""
        publications = []
        with get_db_cursor() as (cur, conn):
            try:
                cur.execute("""
                    SELECT id, title, abstract, summary, domains, source
                    FROM resources_resource
                    WHERE title IS NOT NULL
                """)
                results = cur.fetchall()
                
                for row in results:
                    pub_id, title, abstract, summary, domains, source = row
                    publications.append({
                        'id': pub_id,
                        'title': title,
                        'abstract': abstract or summary,
                        'domains': domains if isinstance(domains, list) else [],
                        'source': source
                    })
                
                return publications
            except Exception as e:
                logger.error(f"Error fetching publications: {e}")
                return []
    
    def execute(self, query, params=None):
        """Execute a database query."""
        with get_db_cursor() as (cur, conn):
            try:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    return cur.fetchall()
                
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                conn.rollback()
                raise

def check_environment():
    """Check if required environment variables are set."""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY environment variable is not set. Using fallback classification.")
        logger.warning("For full functionality, ensure GEMINI_API_KEY is set in your environment or .env file")
        return False
    return True

def classify_publications_task(**context):
    """
    Perform corpus analysis and classify all publications
    """
    load_environment_variables()
    
    # Check environment variables first
    has_api_key = check_environment()
    
    try:
        # Create database manager and summarizer
        db = DatabaseManager()
        
        # Import inside the function to ensure proper environment loading
        from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
        summarizer = TextSummarizer()
        
        # Log API key status
        logger.info(f"Classification running with Gemini API available: {hasattr(summarizer, 'can_use_gemini') and summarizer.can_use_gemini}")
        
        # Analyze existing publications
        logger.info("Analyzing existing publications for field classification...")
        existing_publications = db.get_all_publications()
        
        if not existing_publications:
            logger.warning("No publications found for corpus analysis. Skipping classification.")
            return
        
        # Perform corpus analysis to identify fields
        logger.info(f"Performing corpus content analysis on {len(existing_publications)} publications...")
        field_structure = summarizer.analyze_content_corpus(existing_publications)
        logger.info(f"Discovered field structure: {field_structure}")
        
        # Get publications that need classification
        with get_db_cursor() as (cur, conn):
            try:
                # First check if fields column exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'resources_resource' AND column_name IN ('field', 'subfield')
                """)
                columns = [col[0] for col in cur.fetchall()]
                
                if 'field' in columns and 'subfield' in columns:
                    cur.execute("""
                        SELECT id, title, abstract, summary, domains, source
                        FROM resources_resource
                        WHERE (field IS NULL OR subfield IS NULL)
                    """)
                else:
                    # Fields don't exist, need to add them
                    logger.info("Field/subfield columns not found in resources_resource table. Adding them...")
                    cur.execute("""
                        ALTER TABLE resources_resource 
                        ADD COLUMN IF NOT EXISTS field VARCHAR(255),
                        ADD COLUMN IF NOT EXISTS subfield VARCHAR(255)
                    """)
                    conn.commit()
                    
                    # Now get all publications
                    cur.execute("""
                        SELECT id, title, abstract, summary, domains, source
                        FROM resources_resource
                    """)
                
                results = cur.fetchall()
            except Exception as e:
                logger.error(f"Error querying publications for classification: {e}")
                return
        
        if not results:
            logger.info("No publications found requiring classification.")
            return
        
        # Process each publication
        total_classified = 0
        total_publications = len(results)
        
        for idx, row in enumerate(results, 1):
            try:
                publication_id = row[0]
                title = row[1] or ""
                abstract = row[2] or row[3] or ""  # Use abstract or summary
                domains = row[4] if row[4] else []
                source = row[5] or "unknown"
                
                logger.info(f"Classifying Publication {idx}/{total_publications}: {title}")
                
                # Classify the publication
                field, subfield = _classify_publication(
                    title, abstract, domains, field_structure
                )
                
                # Update the resource with field classification
                db.execute("""
                    UPDATE resources_resource
                    SET field = %s, subfield = %s
                    WHERE id = %s
                """, (field, subfield, publication_id))
                
                logger.info(f"Classified {source} publication - {title}: {field}/{subfield}")
                total_classified += 1
            
            except Exception as e:
                logger.error(f"Error classifying publication {row[1] if len(row) > 1 else 'unknown'}: {e}")
                continue
        
        logger.info(f"Classification complete! Classified {total_classified} publications.")
        
        # Log API key status for debugging
        if not has_api_key:
            logger.warning("Classification completed with fallback mode due to missing GEMINI_API_KEY.")
            logger.warning("For improved classification accuracy, please set GEMINI_API_KEY in your environment.")
    
    except Exception as e:
        logger.error(f"Error in publication classification: {e}")
        raise

def _classify_publication(title, abstract, domains, field_structure):
    """
    Classify a single publication based on the generated field structure.
    """
    logger.info(f"Attempting to classify: {title}")
    
    if not field_structure:
        logger.warning("No field structure available. Using generic classification.")
        return "Unclassified", "General"
    
    # Convert domains to list if it's a string or None
    if isinstance(domains, str):
        try:
            domains = json.loads(domains)
        except:
            domains = [domains]
    elif domains is None:
        domains = []
    
    # Try to match based on content
    for field, subfields in field_structure.items():
        if any(keyword.lower() in (title + " " + abstract).lower() for keyword in subfields):
            classification_result = (field, subfields[0])
            logger.info(f"Matched classification: {classification_result}")
            return classification_result
    
    # If no match found, use the first field structure entry
    first_field = list(field_structure.keys())[0]
    default_classification = (first_field, field_structure[first_field][0])
    
    logger.info(f"No direct match. Using default classification: {default_classification}")
    return default_classification

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['briankimu97@gmail.com'],  # Your email address
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': True,  # Set to True to receive success emails
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'corpus_classification_dag',
    default_args=default_args,
    description='Monthly corpus analysis and publication classification',
    schedule_interval='@monthly',
    catchup=False
)

# Define the classification task with email notification
classify_publications_task_operator = PythonOperator(
    task_id='classify_publications',
    python_callable=classify_publications_task,
    on_success_callback=success_email,
    on_failure_callback=failure_email,
    provide_context=True,  # Important to provide context to callback functions
    dag=dag
)