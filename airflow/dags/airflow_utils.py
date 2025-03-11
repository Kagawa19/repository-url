import os
import logging
from dotenv import load_dotenv
from airflow.utils.email import send_email

# Import necessary classes and processors
from ai_services_api.services.centralized_repository.openalex.openalex_processor import OpenAlexProcessor
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.orcid.orcid_processor import OrcidProcessor
from ai_services_api.services.centralized_repository.knowhub.knowhub_scraper import KnowhubScraper
from ai_services_api.services.centralized_repository.website.website_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.nexus.researchnexus_scraper import ResearchNexusScraper
from ai_services_api.services.centralized_repository.openalex.expert_processor import ExpertProcessor

def setup_logging():
    """Configure logging for Airflow DAGs"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()

def success_email(context):
    """
    Send success email notification when a task succeeds
    
    Args:
        context: Task context dictionary provided by Airflow
    """
    task_instance = context['task_instance']
    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    execution_date = context['execution_date']
    
    subject = f"Airflow Success: {dag_id}.{task_id}"
    html_content = f"""
    <h3>Task {task_id} completed successfully!</h3>
    <p><strong>DAG</strong>: {dag_id}</p>
    <p><strong>Task</strong>: {task_id}</p>
    <p><strong>Execution Date</strong>: {execution_date}</p>
    <p><strong>Log URL</strong>: <a href="{task_instance.log_url}">View Log</a></p>
    """
    
    to_emails = ["briankimu97@gmail.com"]  # Your email address
    send_email(to=to_emails, subject=subject, html_content=html_content)

def failure_email(context):
    """
    Send failure email notification when a task fails
    
    Args:
        context: Task context dictionary provided by Airflow
    """
    task_instance = context['task_instance']
    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    execution_date = context['execution_date']
    exception = context.get('exception')
    
    subject = f"Airflow Failure: {dag_id}.{task_id}"
    html_content = f"""
    <h3>Task {task_id} failed!</h3>
    <p><strong>DAG</strong>: {dag_id}</p>
    <p><strong>Task</strong>: {task_id}</p>
    <p><strong>Execution Date</strong>: {execution_date}</p>
    <p><strong>Exception</strong>: {exception}</p>
    <p><strong>Log URL</strong>: <a href="{task_instance.log_url}">View Log</a></p>
    """
    
    to_emails = ["briankimu97@gmail.com"]  # Your email address
    send_email(to=to_emails, subject=subject, html_content=html_content)