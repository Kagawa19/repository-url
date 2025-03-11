from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import logging
import json

# Import email notification functions
from airflow_utils import setup_logging, load_environment_variables, success_email, failure_email

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import necessary processors
from ai_services_api.services.centralized_repository.openalex.openalex_processor import OpenAlexProcessor
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.orcid.orcid_processor import OrcidProcessor
from ai_services_api.services.centralized_repository.knowhub.knowhub_scraper import KnowhubScraper
from ai_services_api.services.centralized_repository.website.website_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.nexus.researchnexus_scraper import ResearchNexusScraper

def process_publications_task():
    """Process publications from all sources."""
    load_environment_variables()
    
    try:
        # Initialize processors
        openalex_processor = OpenAlexProcessor()
        text_summarizer = TextSummarizer()
        publication_processor = PublicationProcessor(openalex_processor.db, text_summarizer)
        
        # Process experts' fields
        logger.info("Updating experts with OpenAlex data...")
        openalex_processor.update_experts_with_openalex()
        logger.info("Expert data enrichment complete!")
        
        # Process publications from different sources
        sources = [
            {
                'name': 'openalex', 
                'processor': openalex_processor,
                'method': openalex_processor.process_publications
            },
            {
                'name': 'orcid', 
                'processor': OrcidProcessor(),
                'method': OrcidProcessor().process_publications
            },
            {
                'name': 'knowhub', 
                'processor': KnowhubScraper(summarizer=text_summarizer),
                'method': lambda: KnowhubScraper(text_summarizer).fetch_all_content(limit=2)
            },
            {
                'name': 'researchnexus', 
                'processor': ResearchNexusScraper(summarizer=text_summarizer),
                'method': lambda: ResearchNexusScraper(text_summarizer).fetch_content(limit=2)
            },
            {
                'name': 'website', 
                'processor': WebsiteScraper(summarizer=text_summarizer),
                'method': lambda: WebsiteScraper(text_summarizer).fetch_content(limit=2)
            }
        ]
        
        # Process each source
        for source_config in sources:
            try:
                logger.info(f"Processing publications from {source_config['name']} source")
                
                # Different sources have slightly different processing methods
                if source_config['name'] in ['knowhub', 'researchnexus', 'website']:
                    content = source_config['method']()
                    
                    # For sources that return multiple content types or items
                    if isinstance(content, dict):
                        for content_type, items in content.items():
                            for item in items:
                                publication_processor.process_single_work(item, source=source_config['name'])
                    elif isinstance(content, list):
                        for item in content:
                            publication_processor.process_single_work(item, source=source_config['name'])
                else:
                    # For processors like OpenAlex and ORCID
                    source_config['method'](publication_processor, source=source_config['name'])
                
                logger.info(f"Completed processing publications from {source_config['name']} source")
            
            except Exception as source_error:
                logger.error(f"Error processing publications from {source_config['name']} source: {source_error}")
                # Continue with other sources even if one fails
                continue
        
        logger.info("Publication processing complete!")
        
    except Exception as e:
        logger.error(f"Publication processing failed: {e}")
        raise
    finally:
        # Ensure all processors are closed
        if 'openalex_processor' in locals():
            openalex_processor.close()
        if 'publication_processor' in locals():
            publication_processor.close()

# DAG definition
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

dag = DAG(
    'publications_processing_dag',
    default_args=default_args,
    description='Monthly publications processing from multiple sources',
    schedule_interval='@monthly',
    catchup=False
)

process_publications_operator = PythonOperator(
    task_id='process_publications',
    python_callable=process_publications_task,
    on_success_callback=success_email,
    on_failure_callback=failure_email,
    provide_context=True,  # Important to provide context to callback functions
    dag=dag
)