#!/usr/bin/env python3
import os
import logging
from datetime import datetime
import json
import zlib
import psycopg2

from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.text_processor import safe_str, truncate_text

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection settings
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": "5432"
}

def generate_unique_id(resource):
    """
    Generate a unique integer ID for a resource
    Uses a combination of attributes to create a deterministic integer
    """
    # Create a string representation of unique identifiers
    id_string = f"{resource.get('doi', '')}{resource.get('title', '')}"
    
    # If no doi or title, use more fields
    if not id_string.strip():
        id_string = f"{resource.get('source', '')}{resource.get('identifiers', {}).get('handle', '')}"
    
    # Generate a consistent integer hash
    unique_id = abs(zlib.crc32(id_string.encode('utf-8')))
    
    # Ensure the ID is within a reasonable range
    return unique_id % 2147483647  # Max signed 32-bit integer

def insert_resources_to_database(resources):
    """
    Insert resources into the resources_resource table
    
    Args:
        resources (list): List of resource dictionaries to insert
    
    Returns:
        int: Number of successfully inserted resources
    """
    if not resources:
        logger.warning("No resources to save.")
        return 0
    
    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    successful_inserts = 0
    total_resources = len(resources)
    
    try:
        # Process in batches
        batch_size = 50
        for i in range(0, total_resources, batch_size):
            batch = resources[i:i+batch_size]
            
            try:
                # Begin transaction for this batch
                cur.execute("BEGIN;")
                
                for resource in batch:
                    # Generate a unique ID
                    unique_id = generate_unique_id(resource)
                    
                    # Prepare values, ensuring JSON fields are properly serialized
                    values = (
                        unique_id,  # Use generated unique integer ID
                        resource.get('doi'),
                        resource.get('title', 'Untitled Resource'),
                        resource.get('abstract'),
                        resource.get('summary'),
                        json.dumps(resource.get('authors', [])) if resource.get('authors') else None,
                        resource.get('description'),
                        resource.get('type', 'other'),
                        resource.get('source', 'knowhub'),
                        resource.get('date_issue'),
                        resource.get('citation'),
                        resource.get('language', 'en'),
                        json.dumps(resource.get('identifiers', {})) if resource.get('identifiers') else None,
                        resource.get('collection', 'knowhub'),
                        json.dumps(resource.get('publishers', {})) if resource.get('publishers') else None,
                        json.dumps(resource.get('subtitles', {})) if resource.get('subtitles') else None,
                        resource.get('publication_year')
                    )
                    
                    # Perform insert with custom unique ID handling
                    cur.execute("""
                        INSERT INTO resources_resource 
                        (id, doi, title, abstract, summary, authors, description, 
                         type, source, date_issue, citation, language, 
                         identifiers, collection, publishers, subtitles, publication_year)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET 
                        doi = COALESCE(EXCLUDED.doi, resources_resource.doi),
                        title = COALESCE(EXCLUDED.title, resources_resource.title),
                        abstract = COALESCE(EXCLUDED.abstract, resources_resource.abstract),
                        summary = COALESCE(EXCLUDED.summary, resources_resource.summary)
                    """, values)
                    
                    successful_inserts += 1
                
                # Commit this batch
                conn.commit()
                logger.info(f"Inserted batch: {successful_inserts}/{total_resources} resources")
            
            except Exception as batch_error:
                conn.rollback()
                logger.error(f"Error inserting batch: {batch_error}")
                
                # Log detailed error information
                import traceback
                traceback.print_exc()
        
        return successful_inserts
    
    except Exception as e:
        logger.error(f"Database insertion error: {e}")
        return successful_inserts
    
    finally:
        cur.close()
        conn.close()

def main():
    """
    Main function to scrape and save resources from all KnowHub collections
    """
    start_time = datetime.utcnow()
    
    # Create KnowhubScraper instance
    try:
        # Initialize summarizer (optional)
        summarizer = TextSummarizer()
        
        # Create scraper instance
        with KnowhubScraper(summarizer=summarizer) as scraper:
            # Fetch all collections
            logger.info("Starting to fetch all collections from KnowHub")
            
            # Get all content from different collections
            all_content = scraper.fetch_all_collections()
            
            # Combine resources from all collections
            all_resources = []
            for collection_type, resources in all_content.items():
                logger.info(f"Processing {len(resources)} resources from {collection_type}")
                all_resources.extend(resources)
            
            # Insert resources into database
            logger.info(f"Attempting to insert {len(all_resources)} total resources")
            inserted_count = insert_resources_to_database(all_resources)
            
            # Print summary
            elapsed_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Scraping completed. Total resources: {len(all_resources)}")
            logger.info(f"Successfully inserted: {inserted_count} resources")
            logger.info(f"Total time taken: {elapsed_time:.2f} seconds")
            
            # Print collection-wise breakdown
            for collection_type, resources in all_content.items():
                logger.info(f"{collection_type.capitalize()}: {len(resources)} resources")
    
    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()