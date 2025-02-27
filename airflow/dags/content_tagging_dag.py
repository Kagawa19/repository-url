from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow_utils import setup_logging, load_environment_variables

# Import necessary processors
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager

# Configure logging
logger = setup_logging()

def classify_publications_task():
    """
    Perform corpus analysis and classify all publications
    """
    load_environment_variables()
    
    try:
        # Create database manager and summarizer
        db = DatabaseManager()
        summarizer = TextSummarizer()
        
        # Analyze existing publications
        logger.info("Analyzing existing publications for field classification...")
        existing_publications = db.get_all_publications()
        
        if not existing_publications:
            logger.warning("No publications found for corpus analysis. Skipping classification.")
            return
        
        # Perform corpus analysis to identify fields
        logger.info("Performing corpus content analysis...")
        field_structure = summarizer.analyze_content_corpus(existing_publications)
        logger.info(f"Discovered field structure: {field_structure}")
        
        # Get publications that need classification
        results = db.execute("""
            SELECT id, title, summary, domains, source
            FROM resources_resource
            WHERE (field IS NULL OR subfield IS NULL)
        """)
        
        if not results:
            logger.info("No publications found requiring classification.")
            return
        
        # Process each publication
        total_classified = 0
        total_publications = len(results)
        
        for idx, row in enumerate(results, 1):
            try:
                publication_id, title, abstract, domains, source = row
                
                logger.info(f"Classifying Publication {idx}/{total_publications}: {title}")
                
                # Classify the publication
                field, subfield = _classify_publication(
                    title, abstract or "", domains or [], field_structure
                )
                
                # Update the resource with field classification
                db.execute("""
                    UPDATE resources_resource
                    SET field = %s, subfield = %s
                    WHERE id = %s
                """, (field, subfield, publication_id))
                
                logger.info(f"Classified {source} publication - {title}: {field}/{subfield}")
                total_classified += 1
            
            except Exception as e:
                logger.error(f"Error classifying publication {row[1]}: {e}")
                continue
        
        logger.info(f"Classification complete! Classified {total_classified} publications.")
    
    except Exception as e:
        logger.error(f"Error in publication classification: {e}")
        raise

def _classify_publication(title, abstract, domains, field_structure):
    """
    Classify a single publication based on the generated field structure.
    """
    logger.info(f"Attempting to classify: {title}")
    
    if not field_structure:
        logger.warning("No field structure available. Using generic classification.")
        return "Unclassified", "General"
    
    for field, subfields in field_structure.items():
        if any(keyword.lower() in (title + " " + abstract).lower() for keyword in subfields):
            classification_result = (field, subfields[0])
            logger.info(f"Matched classification: {classification_result}")
            return classification_result
    
    first_field = list(field_structure.keys())[0]
    default_classification = (first_field, field_structure[first_field][0])
    
    logger.info(f"No direct match. Using default classification: {default_classification}")
    return default_classification

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
    'corpus_classification_dag',
    default_args=default_args,
    description='Monthly corpus analysis and publication classification',
    schedule_interval='@monthly',
    catchup=False
)

classify_publications_task = PythonOperator(
    task_id='classify_publications',
    python_callable=classify_publications_task,
    dag=dag
)