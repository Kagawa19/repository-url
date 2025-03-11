#!/usr/bin/env python3
import os
import psycopg2
import psycopg2.extras
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import concurrent.futures
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": "5432"
}

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env file")

genai.configure(api_key=GEMINI_API_KEY)

def list_available_models():
    """
    List available Gemini models
    """
    try:
        # Convert generator to list for easier manipulation
        models = list(genai.list_models())
        logger.info("Available Gemini Models:")
        for m in models:
            logger.info(f"- {m.name}")
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []

def get_best_model():
    """
    Find the best available text generation model
    Prioritize fastest models
    """
    models = list_available_models()
    
    # Priority list of model names to try (fastest first)
    model_priorities = [
        'gemini-1.5-flash-latest',  # Fastest model
        'gemini-1.5-flash',
        'gemini-1.5-pro-latest',
        'gemini-1.5-pro'
    ]
    
    for priority_model in model_priorities:
        for model in models:
            if priority_model in model.name:
                logger.info(f"Selected model: {model.name}")
                return model.name
    
    # Fallback
    if models:
        fallback_model = models[0].name
        logger.warning(f"No preferred model found. Using fallback model: {fallback_model}")
        return fallback_model
    
    logger.error("No available models found!")
    return None

def generate_summary(text, model_name, max_length=300):
    """
    Generate a summary using Gemini API
    
    Args:
        text (str): Text to summarize
        model_name (str): Name of the Gemini model to use
        max_length (int): Maximum length of summary
    
    Returns:
        str: Generated summary
    """
    try:
        # Ensure text is not too long
        text = text[:2000]  # Limit input to prevent overwhelming the API
        
        # Use selected Gemini model
        model = genai.GenerativeModel(model_name)
        
        # Craft a prompt that encourages concise summarization
        prompt = f"""Provide a concise, informative summary of the following text. 
        The summary should be clear, capture the key points, and be no more than {max_length} words. 
        If the text is very short or lacks substantial content, create a brief descriptive summary:

        TEXT: {text}
        
        SUMMARY:"""
        
        # Generate summary
        response = model.generate_content(prompt)
        
        # Extract and clean the summary
        summary = response.text.strip()
        
        # Truncate to max length if necessary
        return summary[:max_length]
    
    except Exception as e:
        logger.error(f"Error generating summary with {model_name}: {e}")
        return ""

def fetch_resources_without_summary(batch_size=100):
    """
    Fetch resources that don't have a summary
    
    Args:
        batch_size (int): Number of resources to fetch in one batch
    
    Returns:
        list: List of resources without summaries
    """
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Fetch resources without a summary
            cur.execute("""
                SELECT id, title, abstract, description, type
                FROM resources_resource
                WHERE summary IS NULL OR summary = ''
                LIMIT %s
            """, (batch_size,))
            
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching resources: {e}")
        return []
    finally:
        conn.close()

def update_resource_summary(resource_id, summary):
    """
    Update a resource with its generated summary
    
    Args:
        resource_id (int): ID of the resource
        summary (str): Generated summary
    """
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE resources_resource
                SET summary = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (summary, resource_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating summary for resource {resource_id}: {e}")
        conn.rollback()
    finally:
        conn.close()

def process_resource(resource, model_name):
    """
    Process a single resource to generate and update its summary
    
    Args:
        resource (dict): Resource dictionary
        model_name (str): Name of the Gemini model to use
    
    Returns:
        tuple: (resource_id, summary)
    """
    # Determine the best text source for summarization
    text_source = None
    if resource['abstract'] and len(resource['abstract']) > 50:
        text_source = resource['abstract']
    elif resource['description'] and len(resource['description']) > 50:
        text_source = resource['description']
    else:
        text_source = f"{resource['title']} - {resource['type']} resource"
    
    # Generate summary
    summary = generate_summary(text_source, model_name)
    
    # Update database
    if summary:
        update_resource_summary(resource['id'], summary)
    
    return (resource['id'], summary)

def main():
    """
    Main function to generate summaries for resources without summaries
    """
    try:
        # Get the best available model
        model_name = get_best_model()
        if not model_name:
            logger.error("No suitable Gemini model found. Exiting.")
            return
        
        while True:
            # Fetch resources without summaries
            resources = fetch_resources_without_summary()
            
            # Break if no more resources
            if not resources:
                logger.info("No more resources without summaries.")
                break
            
            logger.info(f"Processing {len(resources)} resources")
            
            # Use concurrent processing to speed up summary generation
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Use tqdm for progress tracking
                list(tqdm(
                    executor.map(lambda r: process_resource(r, model_name), resources), 
                    total=len(resources), 
                    desc="Generating Summaries"
                ))
            
            # Short pause to prevent overwhelming the API
            import time
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()