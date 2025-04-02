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
import time

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

def get_best_model():
    """
    Find the best available text generation model
    Prioritize fastest models
    """
    try:
        # Convert generator to list for easier manipulation
        models = list(genai.list_models())
        
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
    except Exception as e:
        logger.error(f"Error getting best model: {e}")
        return None

def get_existing_domains():
    """
    Retrieve existing domains from the database
    
    Returns:
        list: List of existing domains
    """
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor() as cur:
            query = """
            SELECT DISTINCT unnest(domains) as domain
            FROM resources_resource
            WHERE domains IS NOT NULL
              AND array_length(domains, 1) > 0
            """
            cur.execute(query)
            domains = [row[0] for row in cur.fetchall()]
            logger.info(f"Retrieved {len(domains)} existing domains")
            return domains
    except Exception as e:
        logger.error(f"Error retrieving existing domains: {e}")
        return []
    finally:
        conn.close()

def classify_resource(text, model_name, existing_domains):
    """
    Classify a resource into domains and topics using Gemini API
    
    Args:
        text (str): Text to classify (title, abstract, etc.)
        model_name (str): Name of the Gemini model to use
        existing_domains (list): List of existing domains
    
    Returns:
        tuple: (domains, topics_dict, confidence) - classification result
    """
    try:
        # Ensure text is not too long
        if len(text) > 5000:
            text = text[:5000]  # Limit input to prevent token limits
        
        # Use selected Gemini model
        model = genai.GenerativeModel(model_name)
        
        # Format domains for the prompt
        domains_str = ", ".join(existing_domains) if existing_domains else "No existing domains yet"
        
        # Craft a prompt for classification
        prompt = f"""Analyze the following text and classify it into appropriate domains.
        
        TEXT: {text}
        
        EXISTING DOMAINS: {domains_str}
        
        Your task is to:
        1. Identify the most relevant domains for this text. You can use existing domains or suggest new ones if necessary.
        2. For each domain, suggest 2-3 specific topics.
        3. Provide a confidence score (0-100) for your classification.
        
        Format your response as valid JSON:
        {{
            "domains": ["Domain1", "Domain2"],
            "topics": {{
                "Domain1": ["Topic1", "Topic2", "Topic3"],
                "Domain2": ["Topic1", "Topic2"]
            }},
            "confidence": 85
        }}
        
        Include ONLY the JSON in your response.
        """
        
        # Generate classification
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        try:
            # Extract JSON from response (in case there's additional text)
            import re
            json_match = re.search(r'({.*})', result, re.DOTALL)
            if json_match:
                result = json_match.group(1)
            
            # Parse JSON response
            classification = json.loads(result)
            domains = classification.get("domains", [])
            topics_dict = classification.get("topics", {})
            confidence = classification.get("confidence", 0)
            
            logger.info(f"Classification result: domains={domains}, confidence={confidence}")
            return domains, topics_dict, confidence
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.debug(f"Raw response: {result}")
            return [], {}, 0
    
    except Exception as e:
        logger.error(f"Error classifying resource with {model_name}: {e}")
        return [], {}, 0

def fetch_resources_without_classification(batch_size=100):
    """
    Fetch resources that don't have domain classifications
    
    Args:
        batch_size (int): Number of resources to fetch in one batch
    
    Returns:
        list: List of resources without classifications
    """
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            query = """
                SELECT id, title, abstract, description, type
                FROM resources_resource
                WHERE domains IS NULL 
                   OR array_length(domains, 1) = 0
                   OR domains = '{}'
                LIMIT %s
            """
            logger.info(f"Executing query with batch size: {batch_size}")
            
            params = (batch_size,)
            cur.execute(query, params)
            
            results = cur.fetchall()
            logger.info(f"Found {len(results)} resources without domain classifications")
            
            return results
    except Exception as e:
        logger.error(f"Unexpected error fetching resources: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        conn.close()

def update_resource_classification(resource_id, domains, topics_dict):
    """
    Update a resource with its classification
    
    Args:
        resource_id (int): ID of the resource
        domains (list): List of classified domains
        topics_dict (dict): Dictionary of topics by domain
    """
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor() as cur:
            # Convert topics dictionary to JSON string
            topics_json = json.dumps(topics_dict)
            
            cur.execute("""
                UPDATE resources_resource
                SET domains = %s, topics = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (domains, topics_json, resource_id))
        conn.commit()
        logger.info(f"Updated classification for resource {resource_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating classification for resource {resource_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def process_resource(resource, model_name, existing_domains):
    """
    Process a single resource to classify and update its domains/topics
    
    Args:
        resource (dict): Resource dictionary
        model_name (str): Name of the Gemini model to use
        existing_domains (list): List of existing domains
    
    Returns:
        tuple: (resource_id, success)
    """
    # Determine the best text source for classification
    text_source = ""
    if resource['title']:
        text_source += f"Title: {resource['title']}\n"
    
    if resource['abstract'] and len(resource['abstract']) > 50:
        text_source += f"Abstract: {resource['abstract']}\n"
    elif resource['description'] and len(resource['description']) > 50:
        text_source += f"Description: {resource['description']}\n"
    
    if resource['type']:
        text_source += f"Type: {resource['type']}"
    
    # Skip if not enough text to classify
    if len(text_source.strip()) < 10:
        logger.warning(f"Resource {resource['id']} has insufficient text for classification")
        return resource['id'], False
    
    # Classify the resource
    domains, topics_dict, confidence = classify_resource(text_source, model_name, existing_domains)
    
    # Only update if we have domains and the confidence is reasonable
    if domains and confidence >= 50:
        success = update_resource_classification(resource['id'], domains, topics_dict)
        return resource['id'], success
    else:
        logger.warning(f"Low confidence classification for resource {resource['id']}: {confidence}")
        return resource['id'], False

def main():
    """
    Main function to classify resources without domain classifications
    """
    try:
        # Get the best available model
        model_name = get_best_model()
        if not model_name:
            logger.error("No suitable Gemini model found. Exiting.")
            return
        
        # Get existing domains initially
        existing_domains = get_existing_domains()
        domains_refresh_counter = 0
        
        while True:
            # Fetch resources without domain classifications
            resources = fetch_resources_without_classification()
            
            # Break if no more resources
            if not resources:
                logger.info("No more resources without domain classifications.")
                break
            
            logger.info(f"Processing {len(resources)} resources")
            
            # Use concurrent processing to speed up classification
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for resource in resources:
                    future = executor.submit(
                        process_resource, 
                        resource, 
                        model_name, 
                        existing_domains
                    )
                    futures.append(future)
                
                # Process results with progress bar
                for future in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures), 
                    desc="Classifying Resources"
                ):
                    resource_id, success = future.result()
                    
                    # Increment domains refresh counter for each successful classification
                    if success:
                        domains_refresh_counter += 1
            
            # Refresh domains list periodically to include new domains
            if domains_refresh_counter >= 20:
                logger.info("Refreshing domains list...")
                existing_domains = get_existing_domains()
                domains_refresh_counter = 0
            
            # Short pause to prevent overwhelming the API
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()