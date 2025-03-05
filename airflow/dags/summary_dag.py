from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import logging
import json
import time
import requests
from typing import List, Dict, Any, Optional
import psycopg2
from contextlib import contextmanager
from urllib.parse import urlparse
from dotenv import load_dotenv
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and make sure GEMINI_API_KEY is set
def load_environment_variables():
    """Load environment variables from .env file and inject default API key if needed"""
    try:
        # Load from .env file if available
        load_dotenv()
        
        # Check if GEMINI_API_KEY exists, if not, set it directly
        if 'GEMINI_API_KEY' not in os.environ:
            # Default key from your .env
            default_api_key = "AIzaSyDr8ltNsPp5WXEV-gb1VuM-MDCU9jhACb4"
            os.environ['GEMINI_API_KEY'] = default_api_key
            logger.info(f"Injected default GEMINI_API_KEY")
        
        # Log keys for debugging (first few characters only)
        gemini_key = os.environ.get('GEMINI_API_KEY', '')
        logger.info(f"GEMINI_API_KEY available: {'Yes' if gemini_key else 'No'}, length: {len(gemini_key)}")
        if gemini_key:
            logger.info(f"Key starts with: {gemini_key[:4]}...")
        
        logger.info("Environment variables loaded")
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        raise

# Load environment variables at module level
load_environment_variables()

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'resource_summary_generator_dag',
    default_args=default_args,
    description='Generate summaries for resources using Gemini API',
    schedule_interval='@daily',  # Run once a day
    catchup=False,
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
    
    # In Docker Compose, always use service name
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

# Monkey patch TextSummarizer to inject API key
def patch_text_summarizer():
    """Monkey patch TextSummarizer to always have a valid API key."""
    try:
        from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
        
        # Store the original _setup_gemini method
        original_setup_gemini = TextSummarizer._setup_gemini
        
        # Create a patched version that always returns a valid configuration
        def patched_setup_gemini(self):
            try:
                # Try to use the original method first
                return original_setup_gemini(self)
            except ValueError as e:
                # If the original method fails because of missing API key
                if "GEMINI_API_KEY" in str(e):
                    import google.generativeai as genai
                    
                    # Use our default key
                    default_api_key = "AIzaSyDr8ltNsPp5WXEV-gb1VuM-MDCU9jhACb4"
                    genai.configure(api_key=default_api_key)
                    logger.info("Using default API key in patched _setup_gemini method")
                    
                    # Set up the model with the default configuration
                    generation_config = {
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "top_k": 64,
                        "max_output_tokens": 1024,
                    }
                    
                    # Return the configured model
                    return genai.GenerativeModel(
                        model_name="gemini-pro",
                        generation_config=generation_config
                    )
                else:
                    # Re-raise if it's not about the API key
                    raise
        
        # Apply the patch
        TextSummarizer._setup_gemini = patched_setup_gemini
        logger.info("Successfully patched TextSummarizer._setup_gemini")
        
        return True
    except ImportError as e:
        logger.warning(f"Could not patch TextSummarizer: {e}")
        return False

class DirectGeminiSummarizer:
    """Direct implementation of summarization using Gemini API."""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY', "AIzaSyDr8ltNsPp5WXEV-gb1VuM-MDCU9jhACb4")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def summarize_text(self, content: str, max_retries=3, retry_delay=2) -> Optional[str]:
        """
        Generate a summary using Gemini API.
        
        Args:
            content: The text content to summarize
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated summary text or None if failed
        """
        prompt = f"""
        Please provide a concise summary (150-250 words) of the following content.
        Focus on the main points, findings, or conclusions.
        Use clear, professional language and maintain the original meaning.
        
        CONTENT TO SUMMARIZE:
        {content}
        
        SUMMARY:
        """
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 800,
            }
        }
        
        url = f"{self.api_url}?key={self.api_key}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    
                    # Extract the generated text from the response
                    if "candidates" in response_json and len(response_json["candidates"]) > 0:
                        candidate = response_json["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip()
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    continue
                
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                
            except Exception as e:
                logger.error(f"Error during API request (attempt {attempt+1}/{max_retries}): {e}")
                
            # Wait before retrying
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        return None

def get_summarizer():
    """
    Get the best available summarizer, falling back through multiple options.
    """
    # Try to patch the TextSummarizer first
    patch_text_summarizer()
    
    try:
        # First try to use the patched TextSummarizer
        from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
        summarizer = TextSummarizer()
        logger.info("Using patched TextSummarizer")
        return summarizer
    except Exception as e:
        logger.warning(f"Failed to use TextSummarizer: {e}")
        
        # Fall back to direct implementation
        logger.info("Falling back to DirectGeminiSummarizer")
        return DirectGeminiSummarizer()

def prepare_resource_content(resource: Dict[str, Any]) -> str:
    """
    Prepare content for summarization based on available fields.
    
    Args:
        resource: Dictionary containing resource information
        
    Returns:
        Text content ready for summarization
    """
    content_parts = []
    
    # Add title
    if resource.get('title'):
        content_parts.append(f"Title: {resource['title']}")
    
    # If abstract is available, use it as primary content
    if resource.get('abstract') and resource['abstract'].strip():
        content_parts.append(f"Abstract: {resource['abstract']}")
    else:
        # No abstract, so include more fields for context
        
        # Add description if available
        if resource.get('description') and resource['description'].strip():
            content_parts.append(f"Description: {resource['description']}")
        
        # Add authors
        if resource.get('authors'):
            try:
                if isinstance(resource['authors'], str):
                    authors_data = json.loads(resource['authors'])
                else:
                    authors_data = resource['authors']
                
                if isinstance(authors_data, list):
                    authors_text = ", ".join(authors_data)
                    content_parts.append(f"Authors: {authors_text}")
                elif isinstance(authors_data, dict):
                    authors_text = ", ".join(str(v) for v in authors_data.values() if v)
                    content_parts.append(f"Authors: {authors_text}")
            except (json.JSONDecodeError, TypeError):
                # If we can't parse JSON, use as is
                if isinstance(resource['authors'], str):
                    content_parts.append(f"Authors: {resource['authors']}")
        
        # Add year
        if resource.get('publication_year'):
            content_parts.append(f"Year: {resource['publication_year']}")
        
        # Add domains/topics if available
        if resource.get('domains'):
            try:
                if isinstance(resource['domains'], list):
                    domains_text = ", ".join(resource['domains'])
                    content_parts.append(f"Domains: {domains_text}")
                elif isinstance(resource['domains'], str):
                    domains_text = resource['domains']
                    content_parts.append(f"Domains: {domains_text}")
            except TypeError:
                pass
        
        # Add topics if available
        if resource.get('topics'):
            try:
                if isinstance(resource['topics'], str):
                    topics_data = json.loads(resource['topics'])
                else:
                    topics_data = resource['topics']
                
                if isinstance(topics_data, list):
                    topics_text = ", ".join(topics_data)
                    content_parts.append(f"Topics: {topics_text}")
                elif isinstance(topics_data, dict):
                    topics_text = ", ".join(str(v) for v in topics_data.values() if v)
                    content_parts.append(f"Topics: {topics_text}")
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Combine all parts into one text
    return "\n\n".join(content_parts)

def fetch_resources_without_summary(batch_size=20) -> List[Dict[str, Any]]:
    """
    Fetch resources from the database that don't have summaries yet.
    
    Args:
        batch_size: Number of resources to process in one batch
        
    Returns:
        List of resource records
    """
    with get_db_cursor() as (cur, conn):
        # Query for resources without summaries, prioritizing those with abstracts
        cur.execute("""
            SELECT 
                id, title, abstract, description, domains, topics, type,
                authors, publication_year
            FROM resources_resource
            WHERE summary IS NULL OR summary = ''
            ORDER BY 
                CASE WHEN abstract IS NOT NULL AND abstract != '' THEN 0 ELSE 1 END,
                id
            LIMIT %s
        """, (batch_size,))
        
        columns = [desc[0] for desc in cur.description]
        resources = []
        
        for row in cur.fetchall():
            resource = dict(zip(columns, row))
            resources.append(resource)
        
        logger.info(f"Fetched {len(resources)} resources without summaries")
        return resources

def update_resource_summary(resource_id: int, summary: str) -> bool:
    """
    Update the summary field for a resource.
    
    Args:
        resource_id: ID of the resource to update
        summary: Generated summary text
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_cursor() as (cur, conn):
            cur.execute("""
                UPDATE resources_resource
                SET summary = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (summary, resource_id))
            conn.commit()
            logger.info(f"Updated summary for resource ID: {resource_id}")
            return True
    except Exception as e:
        logger.error(f"Failed to update summary for resource ID {resource_id}: {e}")
        return False

def generate_resource_summaries():
    """
    Main task function to generate summaries for resources.
    """
    try:
        # Make sure environment variables are loaded and GEMINI_API_KEY is set
        load_environment_variables()
        
        # Print Python path for debugging
        logger.info(f"Python path: {sys.path}")
        
        # Get the appropriate summarizer
        summarizer = get_summarizer()
        logger.info(f"Using summarizer: {summarizer.__class__.__name__}")
        
        # Get resources without summaries
        resources = fetch_resources_without_summary(batch_size=50)
        
        if not resources:
            logger.info("No resources found that need summaries")
            return "No resources to process"
        
        # Process each resource
        success_count = 0
        fail_count = 0
        
        for resource in resources:
            resource_id = resource.get('id')
            
            try:
                # Prepare content for summarization
                content = prepare_resource_content(resource)
                
                # Skip if not enough content to summarize
                if len(content.strip()) < 50:
                    logger.warning(f"Resource ID {resource_id} has insufficient content for summarization")
                    continue
                
                # Generate summary
                logger.info(f"Generating summary for resource ID: {resource_id}")
                summary = summarizer.summarize_text(content)
                
                if not summary:
                    logger.error(f"Failed to generate summary for resource ID: {resource_id}")
                    fail_count += 1
                    continue
                
                # Update the database
                if update_resource_summary(resource_id, summary):
                    success_count += 1
                else:
                    fail_count += 1
                
                # Add a small delay to prevent hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing resource ID {resource_id}: {e}")
                fail_count += 1
        
        result_message = f"Summary generation complete. Success: {success_count}, Failed: {fail_count}"
        logger.info(result_message)
        return result_message
        
    except Exception as e:
        logger.error(f"Error in generate_resource_summaries task: {e}", exc_info=True)
        raise

# Define the task to generate summaries
generate_summaries_task = PythonOperator(
    task_id='generate_resource_summaries',
    python_callable=generate_resource_summaries,
    dag=dag,
)

# Task dependencies can be added here if there are multiple tasks