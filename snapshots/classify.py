#!/usr/bin/env python3
import os
import psycopg2
import psycopg2.extras
import numpy as np
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import concurrent.futures

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

class ResourceClassifier:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Classification tracking
        self.classification_metadata = {
            'version': '1.0',
            'total_resources': 0,
            'last_processed_timestamp': None,
            'field_distribution': {}
        }

    def fetch_resources_for_analysis(self, sample_size=5000):
        """
        Fetch a representative sample of resources for classification
        
        Args:
            sample_size (int): Number of resources to sample
        
        Returns:
            list: List of resource dictionaries
        """
        conn = psycopg2.connect(**DB_PARAMS)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Fetch a stratified random sample across different resource types
                cur.execute("""
                    WITH ranked_resources AS (
                        SELECT 
                            *,
                            ROW_NUMBER() OVER (PARTITION BY type ORDER BY RANDOM()) as rn
                        FROM resources_resource
                        WHERE abstract IS NOT NULL AND LENGTH(TRIM(abstract)) > 100
                    )
                    SELECT id, title, abstract, type FROM ranked_resources
                    WHERE rn <= %s
                """, (sample_size // 6,))
                
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching resources: {e}")
            return []
        finally:
            conn.close()

    def classify_resource_batch(self, resources):
        """
        Classify a batch of resources using Gemini
        
        Args:
            resources (list): List of resource dictionaries
        
        Returns:
            list: Classification results
        """
        def classify_single_resource(resource):
            try:
                # Combine title and abstract for comprehensive classification
                text = f"Title: {resource.get('title', '')} \nAbstract: {resource.get('abstract', '')}"
                
                # Prompt for classification
                prompt = f"""Carefully analyze the following research resource and classify it into one primary domain and one specific topic. 
                Provide a JSON response with two keys: 'domain' and 'topic'.

                Available Domains to choose from:
                1. Social Sciences
                2. Health Sciences
                3. Economic Development
                4. Policy & Governance
                5. Education
                6. Environmental Studies

                Text to classify:
                {text}

                IMPORTANT: 
                - Choose ONLY ONE domain
                - Choose ONLY ONE topic
                - If truly uncertain, return the domain and topic most closely matching the content
                - Ensure your response is a valid JSON object
                """
                
                # Generate classification
                response = self.model.generate_content(prompt)
                
                # Parse the response
                try:
                    classification = json.loads(response.text.strip())
                    classification['resource_id'] = resource['id']
                    return classification
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON for resource {resource['id']}")
                    return None
            
            except Exception as e:
                logger.error(f"Error classifying resource {resource.get('id')}: {e}")
                return None
        
        # Use concurrent processing for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Process resources in batches
            classifications = list(filter(None, list(executor.map(classify_single_resource, resources))))
        
        return classifications

    def update_resource_classifications(self, classifications):
        """
        Update resources with their classified domains and topics
        
        Args:
            classifications (list): List of classification results
        """
        conn = psycopg2.connect(**DB_PARAMS)
        try:
            with conn.cursor() as cur:
                # Prepare batch update
                batch_update = [(
                    classification['domain'], 
                    json.dumps([classification['topic']]), 
                    classification['resource_id']
                ) for classification in classifications]
                
                # Batch update domains and topics
                cur.executemany("""
                    UPDATE resources_resource
                    SET domains = ARRAY[%s],
                        topics = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, batch_update)
                
            conn.commit()
            logger.info(f"Updated {len(classifications)} resource classifications")
        
        except Exception as e:
            logger.error(f"Error updating resource classifications: {e}")
            conn.rollback()
        finally:
            conn.close()

    def analyze_classification_distribution(self, classifications):
        """
        Analyze the distribution of domains and topics
        
        Args:
            classifications (list): List of classification results
        
        Returns:
            dict: Distribution of domains and topics
        """
        # Count domain and topic distributions
        domain_counts = {}
        topic_counts = {}
        
        for classification in classifications:
            domain = classification.get('domain', 'Unclassified')
            topic = classification.get('topic', 'Unclassified')
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            'domain_distribution': domain_counts,
            'topic_distribution': topic_counts
        }

    def update_classification_metadata(self, distribution):
        """
        Update classification metadata in the database
        
        Args:
            distribution (dict): Classification distribution
        """
        conn = psycopg2.connect(**DB_PARAMS)
        try:
            with conn.cursor() as cur:
                # Create metadata table if not exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS resource_classification_metadata (
                        id SERIAL PRIMARY KEY,
                        total_resources INTEGER,
                        last_processed_timestamp TIMESTAMP,
                        classification_version VARCHAR(50),
                        field_distribution JSONB,
                        last_trained_model_path VARCHAR(255)
                    )
                """)
                
                # Insert metadata
                cur.execute("""
                    INSERT INTO resource_classification_metadata 
                    (total_resources, last_processed_timestamp, classification_version, field_distribution)
                    VALUES (%s, CURRENT_TIMESTAMP, %s, %s)
                """, (
                    sum(distribution['domain_distribution'].values()),
                    '1.0',
                    json.dumps(distribution)
                ))
                
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating classification metadata: {e}")
            conn.rollback()
        finally:
            conn.close()

    def main(self):
        """
        Main method to run resource classification
        """
        try:
            # Fetch resources for analysis
            resources = self.fetch_resources_for_analysis()
            
            if not resources:
                logger.warning("No resources found for classification")
                return
            
            # Classify resources in batches
            classifications = self.classify_resource_batch(resources)
            
            # Update resources with classifications
            self.update_resource_classifications(classifications)
            
            # Analyze classification distribution
            distribution = self.analyze_classification_distribution(classifications)
            
            # Log distribution
            logger.info("Domain Distribution:")
            for domain, count in distribution['domain_distribution'].items():
                logger.info(f"{domain}: {count}")
            
            logger.info("\nTopic Distribution:")
            for topic, count in distribution['topic_distribution'].items():
                logger.info(f"{topic}: {count}")
            
            # Update classification metadata
            self.update_classification_metadata(distribution)
        
        except Exception as e:
            logger.error(f"An error occurred during classification: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    classifier = ResourceClassifier()
    classifier.main()