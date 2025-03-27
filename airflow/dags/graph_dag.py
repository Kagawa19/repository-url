from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_environment_variables():
    """Load environment variables from .env file."""
    try:
        # Try to load from .env file in different potential locations
        dotenv_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            "/opt/airflow/.env"
        ]
        
        for dotenv_path in dotenv_paths:
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                logger.info(f"Loaded environment variables from {dotenv_path}")
                break
        else:
            logger.warning("No .env file found, using system environment variables")
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")

def check_environment():
    """Check if required environment variables are set."""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY environment variable is not set.")
        logger.warning("For full functionality, ensure GEMINI_API_KEY is set in your environment or .env file")
        return False
    return True

def classify_publications_task(**context):
    """
    Task to classify publications with domains and topics
    """
    # Load environment variables
    load_environment_variables()
    
    # Check environment variables first
    has_api_key = check_environment()
    if not has_api_key:
        logger.warning("Running classification without Gemini API key. Results may be less accurate.")
    
    try:
        # Import the domain classification service
        # Make sure this file is in the same directory as your DAG or in the PYTHONPATH
        from domain_classification_service import run_classification_service, print_domain_structure
        
        # Run the classification service with batch limits based on environment
        batch_limit = int(os.getenv('CLASSIFICATION_BATCH_LIMIT', '5'))
        
        # Get execution date from context for logging
        execution_date = context.get('execution_date', 'unknown')
        logger.info(f"Running classification task on {execution_date} with batch limit {batch_limit}")
        
        # Run the classification service
        run_classification_service(batch_limit=batch_limit)
        
        # Print domain structure report
        logger.info("Domain Structure Report:")
        domain_report = print_domain_structure()
        
        return True
    except Exception as e:
        logger.error(f"Error in publication classification task: {e}")
        raise

# Email notification functions
def success_email(context):
    """Send email notification on task success."""
    task_instance = context['task_instance']
    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    execution_date = context['execution_date']
    
    logger.info(f"Success: Task {task_id} in DAG {dag_id} succeeded on {execution_date}")
    # In a real setup, send an actual email here

def failure_email(context):
    """Send email notification on task failure."""
    task_instance = context['task_instance']
    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    execution_date = context['execution_date']
    exception = context.get('exception')
    
    logger.error(f"Failure: Task {task_id} in DAG {dag_id} failed on {execution_date}")
    logger.error(f"Exception: {exception}")
    # In a real setup, send an actual email here

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['briankimu97@gmail.com'],  # Replace with your email address
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'publication_domain_classification',
    default_args=default_args,
    description='Classify publications into domains and topics',
    # Schedule options:
    # - '@daily' for daily runs
    # - '@weekly' for weekly runs
    # - '0 0 * * *' for once a day at midnight
    # - '0 */12 * * *' for every 12 hours
    # - '0 0 * * 1-5' for weekdays at midnight
    schedule_interval='@daily',  # Run daily to keep up with new publications
    catchup=False
)

# Define the classification task
classify_task = PythonOperator(
    task_id='classify_publications',
    python_callable=classify_publications_task,
    on_success_callback=success_email,
    on_failure_callback=failure_email,
    provide_context=True,
    dag=dag
)

# Add a task to generate a report (you can expand this in the future)
# report_task = PythonOperator(
#     task_id='generate_classification_report',
#     python_callable=generate_report_task,
#     provide_context=True,
#     dag=dag
# )

# Define task dependencies
# classify_task >> report_task