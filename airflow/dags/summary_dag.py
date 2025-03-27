from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import os
import logging
import json
import time
import requests
from typing import List, Dict, Any, Optional
import psycopg2
from contextlib import contextmanager
from urllib.parse import urlparse
from dotenv import load_dotenv
import sys

# Import TextSummarizer directly
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and make sure GEMINI_API_KEY is set
def load_environment_variables():
    """Load environment variables from .env file and inject default API key if needed"""
    try:
        # Load from .env file if available
        load_dotenv()
        
        # Check if GEMINI_API_KEY exists, if not, set it directly
        if 'GEMINI_API_KEY' not in os.environ:
            # Default key from your .env
            default_api_key = "AIzaSyDr8ltNsPp5WXEV-gb1VuM-MDCU9jhACb4"
            os.environ['GEMINI_API_KEY'] = default_api_key
            logger.info(f"Injected default GEMINI_API_KEY")
        
        # Log keys for debugging (first few characters only)
        gemini_key = os.environ.get('GEMINI_API_KEY', '')
        logger.info(f"GEMINI_API_KEY available: {'Yes' if gemini_key else 'No'}, length: {len(gemini_key)}")
        if gemini_key:
            logger.info(f"Key starts with: {gemini_key[:4]}...")
        
        logger.info("Environment variables loaded")
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        raise

# Database connection utilities (kept from original DAG)
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
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
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

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['your-email@example.com'],  # Replace with your email
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'resource_summary_generator_dag',
    default_args=default_args,
    description='Generate summaries for resources using Gemini API',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
)

def fetch_resources_without_summary(batch_size=20) -> List[Dict[str, Any]]:
    """
    Fetch resources from the database that don't have summaries yet.
    
    Args:
        batch_size: Number of resources to process in one batch
        
    Returns:
        List of resource records
    """
    with get_db_cursor() as (cur, conn):
        # Query for resources without summaries, prioritizing those with abstracts
        cur.execute("""
            SELECT 
                id, title, abstract, description, domains, topics, type,
                authors, publication_year
            FROM resources_resource
            WHERE summary IS NULL OR summary = ''
            ORDER BY 
                CASE WHEN abstract IS NOT NULL AND abstract != '' THEN 0 ELSE 1 END,
                id
            LIMIT %s
        """, (batch_size,))
        
        columns = [desc[0] for desc in cur.description]
        resources = []
        
        for row in cur.fetchall():
            resource = dict(zip(columns, row))
            resources.append(resource)
        
        logger.info(f"Fetched {len(resources)} resources without summaries")
        return resources

def update_resource_summary(resource_id: int, summary: str, content_type: str = None) -> bool:
    """
    Update the summary field for a resource.
    
    Args:
        resource_id: ID of the resource to update
        summary: Generated summary text
        content_type: Optional content type for the resource
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_cursor() as (cur, conn):
            # Update with content type if provided
            if content_type:
                cur.execute("""
                    UPDATE resources_resource
                    SET summary = %s, content_type = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (summary, content_type, resource_id))
            else:
                cur.execute("""
                    UPDATE resources_resource
                    SET summary = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (summary, resource_id))
            
            conn.commit()
            logger.info(f"Updated summary for resource ID: {resource_id}")
            return True
    except Exception as e:
        logger.error(f"Failed to update summary for resource ID {resource_id}: {e}")
        return False

def generate_resource_summaries():
    """
    Main task function to generate summaries for resources.
    """
    try:
        # Make sure environment variables are loaded and GEMINI_API_KEY is set
        load_environment_variables()
        
        # Create TextSummarizer instance
        summarizer = TextSummarizer()
        logger.info(f"TextSummarizer initialized")
        
        # Get resources without summaries
        resources = fetch_resources_without_summary(batch_size=50)
        
        if not resources:
            logger.info("No resources found that need summaries")
            return "No resources to process"
        
        # Process each resource
        success_count = 0
        fail_count = 0
        
        for resource in resources:
            resource_id = resource.get('id')
            title = resource.get('title', '')
            abstract = resource.get('abstract', '')
            domains = resource.get('domains', [])
            
            # Fallback to description if abstract is empty
            if not abstract and resource.get('description'):
                abstract = resource.get('description')
            
            try:
                # Skip if not enough content to summarize
                if len(title.strip()) < 10:
                    logger.warning(f"Resource ID {resource_id} has insufficient title for summarization")
                    continue
                
                # Generate summary using the updated method
                logger.info(f"Generating summary for resource ID: {resource_id}")
                summary, content_type = summarizer.summarize(title, abstract)
                
                if not summary or summary.startswith("Failed"):
                    logger.error(f"Failed to generate summary for resource ID: {resource_id}")
                    fail_count += 1
                    continue
                
                # Attempt to classify field and subfield
                try:
                    field, subfield = summarizer.classify_field_and_subfield(title, abstract, domains)
                    # You could save field and subfield if needed
                except Exception as e:
                    logger.warning(f"Field classification failed for resource {resource_id}: {e}")
                    field, subfield = "Unclassified", "General"
                
                # Update the database
                if update_resource_summary(resource_id, summary, content_type):
                    success_count += 1
                else:
                    fail_count += 1
                
                # Add a small delay to prevent hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing resource ID {resource_id}: {e}")
                fail_count += 1
        
        result_message = f"Summary generation complete. Success: {success_count}, Failed: {fail_count}"
        logger.info(result_message)
        return result_message
        
    except Exception as e:
        logger.error(f"Error in generate_resource_summaries task: {e}", exc_info=True)
        raise

# Define the task to generate summaries
generate_summaries_task = PythonOperator(
    task_id='generate_resource_summaries',
    python_callable=generate_resource_summaries,
    dag=dag,
    provide_context=True,
    execution_timeout=timedelta(hours=2)
)