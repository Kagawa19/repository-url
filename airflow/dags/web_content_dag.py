from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import time
from airflow_utils import setup_logging, load_environment_variables

# Import web content processor
from ai_services_api.services.web_content.services.processor import WebContentProcessor

# Configure logging
logger = setup_logging()

def process_web_content_task(**kwargs):
    """
    Process web content with optimized batch processing
    """
    load_environment_variables()
    
    try:
        # Get configuration from DAG run configuration or use defaults
        max_workers = kwargs.get('max_workers', 4)
        batch_size = kwargs.get('batch_size', 10)
        
        # Start timing the process
        start_time = time.time()
        
        # Create web content processor
        processor = WebContentProcessor(
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        try:
            # Process web content
            results = processor.process_content()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log detailed processing results
            logger.info(f"""Web Content Processing Results:
                Pages Processed: {results['processed_pages']}
                Pages Updated: {results['updated_pages']}
                PDF Chunks Processed: {results['processed_chunks']}
                PDF Chunks Updated: {results['updated_chunks']}
                Processing Time: {processing_time:.2f} seconds
                Average Time Per Page: {processing_time/max(results['processed_pages'], 1):.2f} seconds
            """)
            
            return results
        
        finally:
            # Ensure cleanup happens even if processing fails
            processor.cleanup()
    
    except Exception as e:
        logger.error(f"Error processing web content: {str(e)}")
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
    'web_content_processing_dag',
    default_args=default_args,
    description='Monthly web content processing and indexing',
    schedule_interval='@monthly',
    catchup=False
)

process_web_content_task = PythonOperator(
    task_id='process_web_content',
    python_callable=process_web_content_task,
    provide_context=True,
    op_kwargs={
        'max_workers': 4,
        'batch_size': 10
    },
    dag=dag
)