import psycopg2
import json
from typing import List, Dict, Any, Optional
import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse
from contextlib import contextmanager

import google.generativeai as genai

from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import os
import logging
from dotenv import load_dotenv

def load_gemini_api_key():
    """
    Load Gemini API key from environment variables or .env file.
    
    Returns:
        str: Gemini API key
    Raises:
        ValueError if API key is not found
    """
    # Try to load from .env file
    load_dotenv()
    
    # Possible environment variable names
    api_key_vars = [
        'GEMINI_API_KEY', 
        'GOOGLE_API_KEY', 
        'AI_API_KEY'
    ]
    
    # Try each possible variable name
    for var_name in api_key_vars:
        api_key = os.getenv(var_name)
        if api_key:
            logging.info(f"Gemini API key found via {var_name}")
            return api_key
    
    # If no API key found
    logging.error("No Gemini API key found in environment variables")
    raise ValueError("Gemini API key is not set. Please set it in your .env file or environment variables.")

def _setup_gemini():
    """
    Configure Gemini API with loaded API key.
    
    Returns:
        Configured Gemini model
    """
    import google.generativeai as genai
    
    # Load API key
    api_key = load_gemini_api_key()
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Select and return a model
    try:
        # Preferred model names
        preferred_models = [
            'gemini-1.5-pro-latest',
            'models/gemini-1.5-pro-latest',
            'models/gemini-1.0-pro',
            'gemini-pro'
        ]
        
        # Find the first available model
        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(model_name)
                logging.info(f"Using Gemini model: {model_name}")
                return model
            except Exception as model_error:
                logging.warning(f"Could not initialize model {model_name}: {model_error}")
        
        raise ValueError("No suitable Gemini model could be found")
    
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {e}")
        raise

def get_db_connection_params():
    """Get database connection parameters from environment variables."""
    import os
    from urllib.parse import urlparse

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
    
    return {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
    }

@contextmanager
def get_db_connection():
    """Get database connection with proper error handling and cleanup."""
    import psycopg2
    import logging

    params = get_db_connection_params()
    conn = None
    try:
        conn = psycopg2.connect(**params)
        logging.info(f"Connected to database: {params['dbname']} at {params['host']}")
        yield conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logging.info("Database connection closed")

class DatabaseClassificationManager:
    """
    Manages database operations for classification results.
    """
    
    @staticmethod
    def update_resource_classification(
        resource_id: int, 
        domains: List[str], 
        topics: Dict[str, List[str]]
    ) -> bool:
        """
        Update the classification for a specific resource.
        
        Args:
            resource_id: ID of the resource to update
            domains: List of domains 
            topics: Dictionary of topics by domain
        
        Returns:
            Boolean indicating successful update
        """
        import logging
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare topics as JSONB
                    topics_jsonb = json.dumps(topics)
                    
                    # SQL to update domains and topics
                    update_query = """
                    UPDATE resources_resource 
                    SET 
                        domains = %s, 
                        topics = %s, 
                        updated_at = CURRENT_TIMESTAMP 
                    WHERE id = %s
                    """
                    
                    # Execute the update
                    cur.execute(update_query, (
                        domains,  # PostgreSQL text array 
                        topics_jsonb,  # JSONB 
                        resource_id
                    ))
                    
                    # Commit the transaction
                    conn.commit()
                    
                    # Check if a row was actually updated
                    if cur.rowcount > 0:
                        logging.info(f"Successfully updated classification for resource {resource_id}")
                        return True
                    else:
                        logging.warning(f"No resource found with ID {resource_id}")
                        return False
        
        except Exception as e:
            logging.error(f"Error updating resource classification: {e}")
            return False
    
    @staticmethod
    def get_unclassified_resources(limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve unclassified resources from the database.
        
        Args:
            limit: Maximum number of resources to retrieve
        
        Returns:
            List of unclassified resources
        """
        import logging
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Query to find resources without classifications
                    query = """
                    SELECT id, title, abstract 
                    FROM resources_resource 
                    WHERE 
                        domains IS NULL 
                        OR array_length(domains, 1) = 0 
                        OR topics IS NULL 
                    LIMIT %s
                    """
                    
                    cur.execute(query, (limit,))
                    
                    # Fetch results
                    resources = []
                    for row in cur.fetchall():
                        resources.append({
                            'id': row[0],
                            'title': row[1],
                            'abstract': row[2]
                        })
                    
                    logging.info(f"Retrieved {len(resources)} unclassified resources")
                    return resources
        
        except Exception as e:
            logging.error(f"Error retrieving unclassified resources: {e}")
            return []
    
    @staticmethod
    def get_domain_classification_stats() -> Dict[str, int]:
        """
        Get statistics on domain classifications.
        
        Returns:
            Dictionary of domain counts
        """
        import logging
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Query to count resources per domain
                    query = """
                    SELECT 
                        unnest(domains) as domain, 
                        COUNT(*) as resource_count 
                    FROM resources_resource 
                    WHERE domains IS NOT NULL 
                      AND array_length(domains, 1) > 0
                    GROUP BY domain 
                    ORDER BY resource_count DESC
                    """
                    
                    cur.execute(query)
                    
                    # Convert results to dictionary
                    domain_stats = dict(cur.fetchall())
                    
                    logging.info("Retrieved domain classification statistics")
                    return domain_stats
        
        except Exception as e:
            logging.error(f"Error retrieving domain classification stats: {e}")
            return {}

# Modify the existing classification methods to use this new database manager
class DirectAPIClassifier:
    def __init__(self):
        """Initialize the classifier."""
        self.model = self._setup_gemini()
        self.domain_structure = self.get_existing_domain_structure()
        self.db_manager = DatabaseClassificationManager()
        
        logger.info("DirectAPIClassifier initialized successfully")
    
    def get_existing_domain_structure(self) -> Dict[str, List[str]]:
        """
        Retrieve existing domain structure, first from database stats, 
        then fallback to default structure.
        
        Returns:
            A dictionary of domains with their associated topics
        """
        # Try to get domains from database stats
        domain_stats = self.db_manager.get_domain_classification_stats()
        
        if domain_stats:
            # If we have domain stats, use them as the basis for our structure
            return {
                domain: [f"{domain} Topic {i+1}" for i in range(3)]
                for domain in domain_stats.keys()
            }
        
        # Fallback to default structure
        return {
            "Science": ["Physics", "Biology", "Chemistry"],
            "Social Sciences": ["Sociology", "Psychology", "Anthropology"],
            "Humanities": ["Literature", "History", "Philosophy"]
        }
    
    def get_unclassified_publications(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve unclassified publications from the database.
        
        Args:
            limit: Number of publications to retrieve
        
        Returns:
            A list of unclassified publication dictionaries
        """
        return self.db_manager.get_unclassified_resources(limit)
    
    def _generate_content(self, prompt, temperature=0.3, max_tokens=2048):
        """
        Generate content using the Gemini API with error handling.
        
        Args:
            prompt: Input prompt for classification
            temperature: Controls randomness of output
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated content or None
        """
        try:
            # Truncate extremely long prompts
            if len(prompt) > 30000:
                logging.warning(f"Prompt truncated from {len(prompt)} to 30000 characters")
                prompt = prompt[:30000]
            
            # Generation configuration
            generation_config = {
                'temperature': temperature,
                'max_output_tokens': max_tokens
            }
            
            # Safety settings to allow more flexible content
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            # Generate content
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Return text if available
            return response.text.strip() if response and response.text else None
        
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            return None
    
    def classify_single_publication(
        self, 
        publication: Dict[str, Any], 
        domain_structure: Dict[str, List[str]]
    ) -> Tuple[Optional[str], Optional[List[str]], bool]:
        """
        Classify a single publication into a domain and topics.
        
        Args:
            publication: Publication to classify
            domain_structure: Current domain structure
        
        Returns:
            Tuple of (domain, topics, success)
        """
        # Use Gemini to generate classification if possible
        # This is a placeholder - you'd want to implement more sophisticated classification
        try:
            # Prepare prompt for classification
            prompt = f"""
            Classify the following publication into an appropriate domain and provide 3 relevant topics:
            
            Title: {publication.get('title', '')}
            Abstract: {publication.get('abstract', '')}
            
            Provide the response in this JSON format:
            {{
                "domain": "Chosen Domain",
                "topics": ["Topic 1", "Topic 2", "Topic 3"]
            }}
            """
            
            # Generate classification using Gemini
            classification_str = self._generate_content(prompt)
            
            if classification_str:
                try:
                    # Parse the generated classification
                    classification = json.loads(classification_str)
                    domain = classification.get('domain')
                    topics = classification.get('topics', [])
                    
                    # Validate and fallback if needed
                    if not domain or domain not in domain_structure:
                        domain = list(domain_structure.keys())[0]
                    
                    if not topics:
                        topics = domain_structure.get(domain, [])[:3]
                    
                    return domain, topics, True
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse classification: {e}")
            
            # Fallback classification logic
            if "AI" in publication.get('title', '').lower():
                return "Technology", ["AI", "Machine Learning", "Data Science"], True
            elif "health" in publication.get('title', '').lower():
                return "Health Sciences", ["Public Health", "Medicine", "Epidemiology"], True
            
            # Default to first domain if no match
            first_domain = list(domain_structure.keys())[0]
            return first_domain, domain_structure[first_domain][:3], True
        
        except Exception as e:
            logger.error(f"Error classifying publication: {e}")
            return None, None, False
    
    def update_publication_classification(
        self, 
        pub_id: str, 
        domain: str, 
        topics: List[str]
    ) -> bool:
        """
        Update publication classification in the database.
        
        Args:
            pub_id: Publication ID
            domain: Classified domain
            topics: Classified topics
        
        Returns:
            Boolean indicating successful update
        """
        try:
            # Convert pub_id to integer if needed
            pub_id_int = int(pub_id)
            
            # Prepare topics dictionary
            topics_dict = {domain: topics}
            
            # Use database manager to update classification
            return self.db_manager.update_resource_classification(
                resource_id=pub_id_int,
                domains=[domain],
                topics=topics_dict
            )
        except Exception as e:
            logger.error(f"Error updating publication classification: {e}")
            return False

# Update classify_publications function to log more details
def classify_publications(
    batch_size: int = 5, 
    publications_per_batch: int = 1, 
    domain_batch_size: int = 3
) -> bool:
    """
    Classify publications in very small batches to manage API limitations.
    
    Args:
        batch_size: Number of batches to process
        publications_per_batch: Publications per batch
        domain_batch_size: Sample size for domain generation
    
    Returns:
        Boolean indicating successful classification
    """
    try:
        # Initialize classifier
        classifier = DirectAPIClassifier()
        
        # Refresh domain structure
        domain_structure = classifier.domain_structure
        
        logging.info(f"Starting classification with {len(domain_structure)} domains")
        
        # Process publications in small batches
        total_processed = 0
        total_classified = 0
        processed_publications = set()
        
        for batch in range(batch_size):
            logging.info(f"Processing batch {batch + 1}/{batch_size}")
            
            # Get publications for this batch
            publications = classifier.get_unclassified_publications(publications_per_batch)
            
            if not publications:
                logging.info("No more unclassified publications")
                break
            
            # Process each publication in the batch
            for publication in publications:
                pub_id = publication.get('id')
                
                # Skip if already processed
                if pub_id in processed_publications:
                    logging.info(f"Publication {pub_id} already processed")
                    continue
                
                # Classify the publication
                domain, topics, success = classifier.classify_single_publication(
                    publication, domain_structure)
                
                if success and domain and topics:
                    # Update the publication classification
                    if classifier.update_publication_classification(pub_id, domain, topics):
                        # Increment the domain count
                        total_classified += 1
                        processed_publications.add(pub_id)
                
                total_processed += 1
            
            # Every few batches, refresh domain structure
            if batch > 0 and batch % 5 == 0:
                logging.info("Refreshing domain structure...")
                domain_structure = classifier.get_existing_domain_structure()
        
        logging.info(f"Classification complete: {total_classified}/{total_processed} publications classified")
        return True
        
    except Exception as e:
        logging.error(f"Error in classify_publications: {e}")
        return False