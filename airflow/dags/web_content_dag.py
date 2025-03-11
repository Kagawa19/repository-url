from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import os

# Import email notification functions
from airflow_utils import setup_logging, load_environment_variables, success_email, failure_email

# Configure logging
logger = logging.getLogger(__name__)

# Lazy import function to reduce initial load time
def lazy_import(module_path, class_name):
    """
    Lazily import a class to reduce initial load time
    """
    def import_class():
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    return import_class

# Prepare arguments for DAG
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

def load_environment_variables():
    """
    Load environment variables with sensible defaults
    
    Returns:
        dict: Configuration parameters for web content processing
    """
    # Load configuration with sensible defaults
    config = {
        'max_pages': int(os.getenv('MAX_WEB_PAGES', '1000')),
        'max_workers': int(os.getenv('MAX_WORKERS', '4')),
        'batch_size': int(os.getenv('BATCH_SIZE', '50')),
        'timeout': int(os.getenv('PROCESSING_TIMEOUT', '3600'))  # 1 hour default
    }
    
    logger.info(f"Web Content Processing Configuration: {config}")
    return config

def process_web_content_task():
    """
    Process web content with minimal dependencies and optimized loading
    """
    # Load configuration
    config = load_environment_variables()
    
    # Lazy import of WebContentProcessor to reduce initial load time
    WebContentProcessor = lazy_import(
        'ai_services_api.services.centralized_repository.web_content.services.processor', 
        'WebContentProcessor'
    )()
    
    try:
        # Process content with configured parameters
        results = WebContentProcessor.process_content(
            max_workers=config['max_workers'],
            max_pages=config['max_pages'],
            batch_size=config['batch_size']
        )
        
        # Log processing results with detailed breakdown
        logger.info(f"""Web Content Processing Results:
            Total Pages Processed: {results.get('processed_pages', 0)}
            Pages Updated: {results.get('updated_pages', 0)}
            PDF Chunks Processed: {results.get('processed_chunks', 0)}
            PDF Chunks Updated: {results.get('updated_chunks', 0)}
            Processing Time: {results.get('processing_time', 'N/A')}
        """)
        
        # Additional success logging
        if results.get('processed_pages', 0) > 0:
            logger.info("Web content processing completed successfully")
        else:
            logger.warning("No web content was processed")
        
    except Exception as e:
        # Comprehensive error logging
        logger.error(f"Web content processing failed: {str(e)}")
        raise
    finally:
        # Ensure proper cleanup
        if 'WebContentProcessor' in locals():
            try:
                WebContentProcessor.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

# Create DAG
dag = DAG(
    'web_content_processing_dag',
    default_args=default_args,
    description='Monthly web content processing and indexing',
    schedule_interval='@monthly',
    catchup=False,
    max_active_runs=1  # Ensure only one run at a time
)

# Define task with email notification
process_web_content_operator = PythonOperator(
    task_id='process_web_content',
    python_callable=process_web_content_task,
    on_success_callback=success_email,
    on_failure_callback=failure_email,
    provide_context=True,  # Important to provide context to callback functions
    dag=dag,
    # Set a generous timeout to allow for comprehensive processing
    execution_timeout=timedelta(hours=2)
)