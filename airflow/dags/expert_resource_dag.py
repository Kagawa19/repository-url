from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import os
import logging
import json
import time
from typing import List, Dict, Any, Optional
import psycopg2
from contextlib import contextmanager
from urllib.parse import urlparse
from dotenv import load_dotenv

# Import the matcher
from ai_services_api.services.centralized_repository.expert_matching.matcher import Matcher, EnhancedMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
def load_environment_variables():
    """Load environment variables from .env file"""
    try:
        # Load from .env file if available
        load_dotenv()
        logger.info("Environment variables loaded")
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        raise

# Import email notification utilities
try:
    from airflow_utils import success_email, failure_email
except ImportError:
    # Fallback email notification functions if not imported
    def success_email(context):
        logging.info("Task succeeded. Success email would be sent.")
    
    def failure_email(context):
        logging.error("Task failed. Failure email would be sent.")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['your-email@example.com'],  # Replace with your email
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'expert_resource_matching_dag',
    default_args=default_args,
    description='Match experts with resources and update database links',
    schedule_interval='@weekly',  # Run once a week
    catchup=False,
    max_active_runs=1  # Ensure only one run at a time
)

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
    
    # In Docker Compose, use service name
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

def generate_matching_stats():
    """Generate statistics about expert-resource matches"""
    try:
        stats = {}
        with get_db_cursor() as (cur, conn):
            # Get count of experts with resources
            cur.execute("""
                SELECT COUNT(DISTINCT expert_id) 
                FROM expert_resource_links
            """)
            stats['experts_with_resources'] = cur.fetchone()[0]
            
            # Get count of resources with experts
            cur.execute("""
                SELECT COUNT(DISTINCT resource_id) 
                FROM expert_resource_links
            """)
            stats['resources_with_experts'] = cur.fetchone()[0]
            
            # Get total links count
            cur.execute("SELECT COUNT(*) FROM expert_resource_links")
            stats['total_links'] = cur.fetchone()[0]
            
            # Get average confidence score
            cur.execute("""
                SELECT AVG(confidence_score) 
                FROM expert_resource_links
            """)
            avg_confidence = cur.fetchone()[0]
            stats['avg_confidence'] = round(avg_confidence, 3) if avg_confidence else 0
            
            # Get distribution of confidence scores
            cur.execute("""
                SELECT 
                    CASE 
                        WHEN confidence_score >= 0.9 THEN 'high (0.9-1.0)'
                        WHEN confidence_score >= 0.7 THEN 'medium (0.7-0.9)'
                        ELSE 'low (<0.7)'
                    END as confidence_level,
                    COUNT(*) as count
                FROM expert_resource_links
                GROUP BY confidence_level
                ORDER BY confidence_level
            """)
            
            confidence_dist = {}
            for row in cur.fetchall():
                confidence_dist[row[0]] = row[1]
            
            stats['confidence_distribution'] = confidence_dist
            
            # Get top 5 experts by resource count
            cur.execute("""
                SELECT expert_id, COUNT(*) as count
                FROM expert_resource_links
                GROUP BY expert_id
                ORDER BY count DESC
                LIMIT 5
            """)
            
            top_experts = []
            for row in cur.fetchall():
                top_experts.append({'expert_id': row[0], 'resource_count': row[1]})
            
            stats['top_experts'] = top_experts
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generating matching stats: {e}")
        return {'error': str(e)}

def run_expert_resource_matching(**context):
    """
    Main task function to match experts to resources.
    This function will use the enhanced matcher.
    """
    try:
        # Make sure environment variables are loaded
        load_environment_variables()
        
        # Use the enhanced matcher via the original Matcher interface
        matcher = Matcher()
        matcher.link_matched_experts_to_db(use_enhanced=True)
        
        # Generate matching statistics
        stats = generate_matching_stats()
        
        # Log statistics
        logger.info("Expert-Resource Matching Statistics:")
        logger.info(f"Experts with resources: {stats.get('experts_with_resources', 0)}")
        logger.info(f"Resources with experts: {stats.get('resources_with_experts', 0)}")
        logger.info(f"Total links: {stats.get('total_links', 0)}")
        logger.info(f"Average confidence score: {stats.get('avg_confidence', 0)}")
        
        # Store stats in XCom for other tasks
        context['ti'].xcom_push(key='matching_stats', value=stats)
        
        return "Expert-resource matching completed successfully"
        
    except Exception as e:
        logger.error(f"Error in run_expert_resource_matching task: {e}", exc_info=True)
        raise

def run_direct_matching(**context):
    """
    Alternative task to run the original matching algorithm.
    This can be used for comparison or backup.
    """
    try:
        # Make sure environment variables are loaded
        load_environment_variables()
        
        # Use the original matcher
        matcher = Matcher()
        matcher.link_matched_experts_to_db(use_enhanced=False)
        
        return "Direct expert-resource matching completed successfully"
        
    except Exception as e:
        logger.error(f"Error in run_direct_matching task: {e}", exc_info=True)
        raise

def generate_matching_report(**context):
    """
    Generate a report of the matching results.
    """
    try:
        # Get statistics from XCom
        stats = context['ti'].xcom_pull(task_ids='run_expert_resource_matching', key='matching_stats')
        
        if not stats:
            logger.warning("No statistics available from matching task")
            return "No matching statistics available"
        
        # Format report
        report = [
            "Expert-Resource Matching Report",
            "================================",
            f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total experts with resources: {stats.get('experts_with_resources', 0)}",
            f"Total resources with experts: {stats.get('resources_with_experts', 0)}",
            f"Total links created: {stats.get('total_links', 0)}",
            f"Average confidence score: {stats.get('avg_confidence', 0)}",
            "",
            "Confidence Score Distribution:",
        ]
        
        # Add confidence distribution
        conf_dist = stats.get('confidence_distribution', {})
        for level, count in conf_dist.items():
            report.append(f"  - {level}: {count} links")
        
        report.append("")
        report.append("Top Experts by Resource Count:")
        
        # Add top experts
        top_experts = stats.get('top_experts', [])
        for i, expert in enumerate(top_experts, 1):
            report.append(f"  {i}. Expert ID {expert['expert_id']}: {expert['resource_count']} resources")
        
        # Join report into a string
        report_text = "\n".join(report)
        
        # Log report
        logger.info("Expert-Resource Matching Report Generated:\n" + report_text)
        
        # Save report to a file (optional)
        report_dir = os.getenv('REPORT_DIR', '/tmp/reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(
            report_dir, 
            f"expert_resource_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        return f"Expert-resource matching report generated: {report_path}"
        
    except Exception as e:
        logger.error(f"Error generating matching report: {e}", exc_info=True)
        raise

# Define the tasks
enhanced_matching_task = PythonOperator(
    task_id='run_expert_resource_matching',
    python_callable=run_expert_resource_matching,
    dag=dag,
    on_success_callback=success_email,
    on_failure_callback=failure_email,
    provide_context=True,
    execution_timeout=timedelta(hours=1)
)

report_task = PythonOperator(
    task_id='generate_matching_report',
    python_callable=generate_matching_report,
    dag=dag,
    on_failure_callback=failure_email,
    provide_context=True,
    execution_timeout=timedelta(minutes=30)
)

# Define task dependencies
enhanced_matching_task >> report_task

# Optional: Also define the direct matching task (commented out by default)
# direct_matching_task = PythonOperator(
#     task_id='run_direct_matching',
#     python_callable=run_direct_matching,
#     dag=dag,
#     on_failure_callback=failure_email,
#     provide_context=True,
#     execution_timeout=timedelta(hours=1)
# )