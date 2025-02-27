from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow_utils import setup_logging, load_environment_variables

# Import necessary processors
from ai_services_api.services.centralized_repository.openalex.openalex_processor import OpenAlexProcessor
from ai_services_api.services.centralized_repository.publication_processor import PublicationProcessor
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.orcid.orcid_processor import OrcidProcessor
from ai_services_api.services.centralized_repository.knowhub.knowhub_scraper import KnowhubScraper
from ai_services_api.services.centralized_repository.website.website_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.nexus.researchnexus_scraper import ResearchNexusScraper

# Configure logging
logger = setup_logging()

def process_publications_task():
    """
    Process publications from all sources without classification
    """
    load_environment_variables()
    
    try:
        # Initialize processors
        openalex_processor = OpenAlexProcessor()
        publication_processor = PublicationProcessor(openalex_processor.db, TextSummarizer())
        
        # Process experts' fields
        openalex_processor.update_experts_with_openalex()
        logger.info("Expert data enrichment complete!")
        
        # Process OpenAlex publications
        openalex_processor.process_publications(publication_processor, source='openalex')
        
        # Process ORCID publications
        orcid_processor = OrcidProcessor()
        orcid_processor.process_publications(publication_processor, source='orcid')
        orcid_processor.close()
        
        # Process KnowHub content
        knowhub_scraper = KnowhubScraper(summarizer=TextSummarizer())
        all_content = knowhub_scraper.fetch_all_content(limit=2)
        
        for content_type, items in all_content.items():
            if items:
                for item in items:
                    publication_processor.process_single_work(item, source='knowhub')
        
        # Process ResearchNexus publications
        research_nexus_scraper = ResearchNexusScraper(summarizer=TextSummarizer())
        research_nexus_publications = research_nexus_scraper.fetch_content(limit=2)
        
        for pub in research_nexus_publications:
            publication_processor.process_single_work(pub, source='researchnexus')
        
        # Process Website publications
        website_scraper = WebsiteScraper(summarizer=TextSummarizer())
        website_publications = website_scraper.fetch_content(limit=2)
        
        for pub in website_publications:
            publication_processor.process_single_work(pub, source='website')
        
        logger.info("Publication processing complete!")
        
    except Exception as e:
        logger.error(f"Publication processing failed: {e}")
        raise
    finally:
        # Close processors
        if 'openalex_processor' in locals():
            openalex_processor.close()
        if 'orcid_processor' in locals():
            orcid_processor.close()
        if 'knowhub_scraper' in locals():
            knowhub_scraper.close()
        if 'research_nexus_scraper' in locals():
            research_nexus_scraper.close()
        if 'website_scraper' in locals():
            website_scraper.close()

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
    'publications_processing_dag',
    default_args=default_args,
    description='Monthly publications processing from multiple sources',
    schedule_interval='@monthly',
    catchup=False
)

process_publications_task = PythonOperator(
    task_id='process_publications',
    python_callable=process_publications_task,
    dag=dag
)