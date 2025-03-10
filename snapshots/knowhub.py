#!/usr/bin/env python3
import os
import json
import requests
import psycopg2
import time
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection settings
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": "5432"
}

def setup_database():
    """Verify the resources_resource table exists and its structure"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    try:
        # Check table existence
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'resources_resource'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            logger.error("resources_resource table does not exist. Please create it first.")
            return False
        
        # Check columns
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'resources_resource'
        """)
        columns = [col[0] for col in cur.fetchall()]
        
        logger.info(f"Columns in resources_resource: {columns}")
        return True
    
    except Exception as e:
        logger.error(f"Database check error: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def generate_unique_id(resource):
    """
    Generate a unique identifier for a resource
    Uses a hash of key identifying information
    """
    # Create a string representation of unique identifiers
    id_string = f"{resource.get('doi', '')}{resource.get('title', '')}"
    
    # If no doi or title, use more fields
    if not id_string.strip():
        id_string = f"{resource.get('source', '')}{resource.get('identifiers', {}).get('handle', '')}"
    
    # Generate a hash
    return hashlib.md5(id_string.encode()).hexdigest()

def insert_resources_to_database(resources):
    """
    Insert resources into the resources_resource table
    
    Args:
        resources (list): List of resource dictionaries to insert
    
    Returns:
        int: Number of successfully inserted resources
    """
    if not resources:
        logger.warning("No resources to insert.")
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
                        unique_id,  # Use generated unique ID as primary key
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
                        json.dumps(resource.get('subtitles', {})) if resource.get('subtitles') else None
                    )
                    
                    # Perform insert with custom unique ID handling
                    cur.execute("""
                        INSERT INTO resources_resource 
                        (id, doi, title, abstract, summary, authors, description, 
                         type, source, date_issue, citation, language, 
                         identifiers, collection, publishers, subtitles)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    start_time = time.time()
    
    # Verify database setup
    if not setup_database():
        return
    
    # Sample resource data (replace with your actual data collection method)
    sample_resources = [
        {
            'title': 'Sample Research Paper',
            'doi': '10.1000/sample123',
            'abstract': 'This is a sample abstract about an important research topic.',
            'authors': [{'name': 'John Doe'}, {'name': 'Jane Smith'}],
            'type': 'journal_article',
            'source': 'knowhub',
            'date_issue': '2023-01-15',
            'identifiers': {'handle': 'sample-handle'},
            'collection': 'publications'
        }
        # Add more resources here
    ]
    
    try:
        # Insert resources
        inserted_count = insert_resources_to_database(sample_resources)
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"Completed. Inserted {inserted_count} resources in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()