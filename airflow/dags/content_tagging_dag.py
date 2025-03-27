from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base import BaseHook
from airflow.exceptions import AirflowException

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamically add project root to Python path
# Adjust these paths to match your project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import classification functions
try:
    from ai_services_api.services.centralized_repository.domain_classification_service import (
        classify_publications, 
        print_domain_structure
    )
except ImportError as e:
    logger.error(f"Could not import classification functions: {e}")
    classify_publications = None
    print_domain_structure = None

def load_environment_variables():
    """
    Load environment variables from multiple potential locations.
    Prioritizes .env files and provides comprehensive logging.
    """
    try:
        # Potential .env file locations
        env_paths = [
            os.path.join(PROJECT_ROOT, '.env'),
            os.path.join(os.path.dirname(__file__), '.env'),
            '/opt/airflow/.env'
        ]
        
        loaded = False
        for path in env_paths:
            if os.path.exists(path):
                load_dotenv(path)
                logger.info(f"Loaded environment variables from {path}")
                loaded = True
                break
        
        if not loaded:
            logger.warning("No .env file found. Using system environment variables.")
        
        # Validate critical environment variables
        critical_vars = ['GEMINI_API_KEY', 'DATABASE_URL']
        missing_vars = [var for var in critical_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing critical environment variables: {missing_vars}")
            raise ValueError(f"Missing environment variables: {missing_vars}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        raise

def validate_dependencies():
    """
    Validate that all required dependencies are installed and configured.
    """
    try:
        # Check for required libraries
        required_libs = [
            'google.generativeai',
            'psycopg2',
            'dotenv'
        ]
        
        missing_libs = []
        for lib in required_libs:
            try:
                __import__(lib.split('.')[0])
            except ImportError:
                missing_libs.append(lib)
        
        if missing_libs:
            error_msg = f"Missing libraries: {', '.join(missing_libs)}"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        return True
    except Exception as e:
        logger.error(f"Dependency validation failed: {e}")
        raise

def classification_task(**context):
    """
    Main task for classification with comprehensive error handling.
    """
    try:
        # Validate environment and dependencies before processing
        load_environment_variables()
        validate_dependencies()
        
        # Validate import of classification function
        if classify_publications is None:
            raise ImportError("classify_publications function is not available")
        
        # Configure task-specific parameters
        batch_size = context['dag_run'].conf.get('batch_size', 5)
        publications_per_batch = context['dag_run'].conf.get('publications_per_batch', 1)
        
        logger.info(f"Starting classification with batch_size={batch_size}, "
                    f"publications_per_batch={publications_per_batch}")
        
        # Execute classification
        result = classify_publications(
            batch_size=batch_size,
            publications_per_batch=publications_per_batch
        )
        
        if not result:
            raise AirflowException("Classification process failed")
        
        return "Classification completed successfully"
    
    except Exception as e:
        logger.error(f"Classification task failed: {e}")
        raise AirflowException(f"Classification task failed: {e}")

def domain_structure_task(**context):
    """
    Task to print and log domain structure.
    """
    try:
        # Validate environment and dependencies
        load_environment_variables()
        validate_dependencies()
        
        # Validate import of domain structure function
        if print_domain_structure is None:
            raise ImportError("print_domain_structure function is not available")
        
        # Execute domain structure printing
        result = print_domain_structure()
        
        logger.info(f"Domain structure task result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Domain structure task failed: {e}")
        raise AirflowException(f"Domain structure task failed: {e}")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'domain_classification_workflow',
    default_args=default_args,
    description='Workflow for classifying publications and managing domain structure',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    max_active_runs=1  # Ensure only one run at a time
)

# Define tasks using PythonOperator
load_env_task = PythonOperator(
    task_id='load_environment_variables',
    python_callable=load_environment_variables,
    provide_context=True,
    dag=dag,
)

classification_operator = PythonOperator(
    task_id='run_publication_classification',
    python_callable=classification_task,
    provide_context=True,
    dag=dag,
)

domain_structure_operator = PythonOperator(
    task_id='print_domain_structure',
    python_callable=domain_structure_task,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
load_env_task >> classification_operator >> domain_structure_operator