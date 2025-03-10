#!/usr/bin/env python3
import os
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import psycopg2
import zlib
from datetime import datetime, timezone
import re
from urllib.parse import urljoin
import concurrent.futures
import time
import hashlib
import threading

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

class KnowhubScraper:
    def __init__(self):
        # Base URL for KnowHub
        self.base_url = 'https://knowhub.aphrc.org'
        
        # Endpoints for different collections
        self.endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        
        # Track seen handles to avoid duplicates
        self.seen_handles = set()
        
        # Create a requests session with retry and backoff
        self.session = self._create_requests_session()

    def _create_requests_session(self):
        """Create a requests session with retry and backoff strategies"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=5,  # Total number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1  # Exponential backoff
        )
        
        # Mount retry adapter for both HTTP and HTTPS
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Set headers to mimic a browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        
        return session

    def _fetch_page(self, url, timeout=30):
        """
        Fetch a page with robust error handling and multiple retry strategies
        
        Args:
            url (str): URL to fetch
            timeout (int): Timeout in seconds
        
        Returns:
            str or None: Page content or None if failed
        """
        try:
            # Attempt to fetch the page with a longer timeout
            response = self.session.get(
                url, 
                timeout=timeout,
                verify=False  # Disable SSL verification if needed
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            return response.text
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {url}: {e}")
            
            # Additional specific error logging
            if isinstance(e, requests.exceptions.ConnectionError):
                logger.error("Connection error - check network connectivity")
            elif isinstance(e, requests.exceptions.Timeout):
                logger.error("Request timed out - check server responsiveness")
            elif isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"HTTP error occurred: {e.response.status_code}")
            
            return None

    def _extract_item_details(self, item_url):
        """Extract detailed information for a single item"""
        try:
            # Fetch the item page
            page_content = self._fetch_page(item_url)
            if not page_content:
                return None
            
            # Parse the page
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract title
            title_elem = soup.find(['h1', 'h2'], class_=lambda c: c and 'item-title' in c)
            if not title_elem:
                title_elem = soup.find(['h1', 'h2'])
            
            if not title_elem:
                return None
            
            title = title_elem.get_text().strip()
            
            # Determine content type based on URL
            content_type = 'other'
            if '123456789/1' in item_url:
                content_type = 'publications'
            elif '123456789/2' in item_url:
                content_type = 'documents'
            elif '123456789/3' in item_url:
                content_type = 'reports'
            elif '123456789/4' in item_url:
                content_type = 'multimedia'
            
            # Extract metadata
            metadata_section = soup.find('div', class_=lambda c: c and any(x in c for x in ['item-page-field-wrapper', 'item-metadata']))
            
            # Default metadata dictionary
            metadata = {
                'abstract': '',
                'authors': [],
                'date': None,
                'keywords': []
            }
            
            if metadata_section:
                # Extract abstract
                abstract_elem = metadata_section.find('span', class_=['abstract', 'description'])
                if abstract_elem:
                    metadata['abstract'] = abstract_elem.get_text().strip()
                
                # Extract authors
                author_elems = metadata_section.find_all('span', class_=['author', 'creator'])
                metadata['authors'] = [
                    author.get_text().strip() 
                    for author in author_elems 
                    if author.get_text().strip()
                ]
                
                # Extract date
                date_elem = metadata_section.find('span', class_=['date', 'issued'])
                if date_elem:
                    metadata['date'] = date_elem.get_text().strip()
                
                # Extract keywords
                keyword_elems = metadata_section.find_all('span', class_=['subject', 'keyword'])
                metadata['keywords'] = [
                    kw.get_text().strip() 
                    for kw in keyword_elems 
                    if kw.get_text().strip()
                ]
            
            # Construct resource dictionary
            resource = {
                'title': title,
                'abstract': metadata['abstract'],
                'authors': [{'name': author} for author in metadata['authors']],
                'type': content_type,
                'source': 'knowhub',
                'collection': content_type,
                'date_issue': metadata['date'],
                'identifiers': {'handle': item_url.split('/')[-1]},
                'keywords': metadata['keywords']
            }
            
            return resource
        
        except Exception as e:
            logger.error(f"Error extracting details from {item_url}: {e}")
            return None

    def _extract_collection_items(self, collection_url):
        """Extract items from a collection page"""
        try:
            # Fetch the collection page
            page_content = self._fetch_page(collection_url)
            if not page_content:
                logger.warning(f"Could not fetch content from {collection_url}")
                return []
            
            # Parse the page
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Find all item links
            item_links = soup.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
            
            # Extract unique item links
            unique_item_links = set()
            for link in item_links:
                href = link.get('href')
                if href and href not in self.seen_handles:
                    full_url = urljoin(self.base_url, href)
                    unique_item_links.add(full_url)
                    self.seen_handles.add(href)
            
            # Parallel processing of items with more robust error handling
            items = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit extraction tasks with timeout
                future_to_url = {
                    executor.submit(self._extract_item_details, url): url 
                    for url in unique_item_links
                }
                
                # Collect results with timeout
                for future in concurrent.futures.as_completed(future_to_url, timeout=60):
                    url = future_to_url[future]
                    try:
                        item = future.result()
                        if item:
                            items.append(item)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
            
            return items
        
        except Exception as e:
            logger.error(f"Error extracting items from {collection_url}: {e}")
            return []

    def fetch_all_collections(self):
        """Fetch items from all collections"""
        all_collections = {}
        
        # Fetch items from each collection
        for collection_name, collection_url in self.endpoints.items():
            logger.info(f"Fetching items from {collection_name} collection")
            
            # Extract items from the collection
            collection_items = self._extract_collection_items(collection_url)
            
            # Store collected items
            all_collections[collection_name] = collection_items
            
            logger.info(f"Found {len(collection_items)} items in {collection_name}")
        
        return all_collections

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
                        None,  # summary
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
                        abstract = COALESCE(EXCLUDED.abstract, resources_resource.abstract)
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
    start_time = datetime.now(timezone.utc)
    
    try:
        # Create scraper instance
        scraper = KnowhubScraper()
        
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
        elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
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