from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow_utils import setup_logging, load_environment_variables

# Import graph initializer
from ai_services_api.services.recommendation.graph_initializer import GraphDatabaseInitializer

# Configure logging
logger = setup_logging()

def initialize_graph_task():
    """
    Initialize graph database
    """
    load_environment_variables()
    
    try:
        # Initialize graph database
        graph_initializer = GraphDatabaseInitializer()
        graph_success = graph_initializer.initialize_graph()
        
        if not graph_success:
            logger.error("Graph initialization failed")
            raise Exception("Graph initialization failed")
        
        logger.info("Graph initialization complete!")
    
    except Exception as e:
        logger.error(f"Graph initialization failed: {e}")
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