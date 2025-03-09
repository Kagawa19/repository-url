#!/usr/bin/env python3
import os
import json
import requests
import psycopg2
import time
import re
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowhub_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database connection settings
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": "localhost",
    "port": "5432"
}

class FastKnowhubScraper:
    """High-performance scraper for KnowHub with concurrent processing."""
    
    def __init__(self, max_workers: int = 8, connection_timeout: int = 5):
        """Initialize the scraper with parallel processing capabilities.
        
        Args:
            max_workers: Maximum number of concurrent worker threads
            connection_timeout: Timeout for initial connection in seconds
        """
        self.base_url = os.getenv('KNOWHUB_BASE_URL', 'https://knowhub.aphrc.org')
        self.max_workers = max_workers
        self.timeout = (connection_timeout, 30)  # (connect timeout, read timeout)
        
        # URL structure - try both standard DSpace handles and collection IDs
        # We'll try both approaches for maximum compatibility
        self.handle_endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        
        self.collection_endpoints = {
            'publications': f"{self.base_url}/communities/3/collections",
            'documents': f"{self.base_url}/collections/12", 
            'reports': f"{self.base_url}/collections/14",
            'multimedia': f"{self.base_url}/collections/15"
        }
        
        self.browse_endpoints = {
            'publications': f"{self.base_url}/browse?type=dateissued",
            'documents': f"{self.base_url}/browse?type=doctype&sort_by=1&order=DESC", 
            'reports': f"{self.base_url}/browse?type=doctype&value=Technical+Report",
            'multimedia': f"{self.base_url}/browse?type=dateissued"
        }
        
        # Create session with retry capabilities
        self.session = self._create_session()
        
        # Track processed URLs to avoid duplicates
        self.processed_urls = set()
        
        logger.info(f"FastKnowhubScraper initialized with {max_workers} workers")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry capability and browser-like headers."""
        session = requests.Session()
        
        # Configure retry strategy for robustness
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers to appear like a regular browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a request with proper error handling and logging."""
        try:
            logger.info(f"Requesting: {url}")
            response = self.session.get(url, timeout=self.timeout, verify=False)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout accessing {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error accessing {url}: {e}")
            return None
    
    def _fetch_listing_page(self, url: str) -> List[Dict[str, Any]]:
        """Fetch and parse a listing page to extract item links."""
        response = self._make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        
        # Try multiple selector patterns to find items
        selectors = [
            'div.artifact-description a[href*="handle"]',
            'div.ds-artifact-item a[href*="handle"]', 
            'h4 a[href*="handle"]',
            'table.ds-table a[href*="handle"]',
            'div.recent-submissions a[href*="handle"]',
            'ul.ds-artifact-list a[href*="handle"]',
            'div.item a[href*="handle"]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Found {len(elements)} items using selector: {selector}")
                for elem in elements:
                    href = elem.get('href')
                    if href and '/handle/' in href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in self.processed_urls:
                            title = elem.get_text().strip()
                            if len(title) > 5:  # Skip very short titles that might be navigation
                                links.append({
                                    'url': full_url,
                                    'title': title
                                })
        
        # If we found no links, try a more general approach
        if not links:
            all_links = soup.find_all('a', href=True)
            for a in all_links:
                href = a.get('href', '')
                if '/handle/' in href:
                    full_url = urljoin(self.base_url, href)
                    if full_url not in self.processed_urls:
                        title = a.get_text().strip()
                        if len(title) > 5 and not title.lower() in ['next', 'previous', 'home']:
                            links.append({
                                'url': full_url,
                                'title': title
                            })
        
        return links
    
    def _process_item_page(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an individual item page to extract all metadata."""
        url = item_data['url']
        title = item_data['title']
        
        # Skip if already processed
        if url in self.processed_urls:
            return {}
        
        self.processed_urls.add(url)
        
        # Get item detail page
        response = self._make_request(url)
        if not response:
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Determine resource type from URL
        resource_type = 'other'
        for type_key, handle_url in self.handle_endpoints.items():
            if handle_url in url:
                resource_type = type_key
                break
        
        # Extract handle ID
        handle = None
        handle_match = re.search(r'/handle/([0-9]+/[0-9]+)', url)
        if handle_match:
            handle = handle_match.group(1)
        
        # Extract metadata with multiple strategies
        metadata = self._extract_metadata(soup)
        
        # Add URL, title and handle if they don't exist in metadata
        if not metadata.get('title'):
            metadata['title'] = title
        
        metadata['url'] = url
        metadata['handle'] = handle
        metadata['resource_type'] = resource_type
        
        return metadata
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract all metadata from an item page with multiple strategies."""
        metadata = {
            'title': '',
            'abstract': '',
            'authors': [],
            'publication_date': None,
            'publication_year': None,
            'keywords': [],
            'doi': None,
            'source': 'knowhub'
        }
        
        # Strategy 1: Look for metadata in a table
        metadata_table = soup.find('table', class_='ds-table')
        if metadata_table:
            rows = metadata_table.find_all('tr')
            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) >= 2:
                    field = cells[0].get_text().strip().lower()
                    value = cells[1].get_text().strip()
                    
                    if 'title' in field:
                        metadata['title'] = value
                    elif 'author' in field or 'creator' in field:
                        metadata['authors'] = [name.strip() for name in value.split(';')]
                    elif 'date' in field or 'issued' in field:
                        metadata['publication_date'] = value
                        # Extract year
                        year_match = re.search(r'\b(19|20)\d{2}\b', value)
                        if year_match:
                            metadata['publication_year'] = int(year_match.group(0))
                    elif 'abstract' in field or 'description' in field:
                        metadata['abstract'] = value
                    elif 'subject' in field or 'keyword' in field:
                        metadata['keywords'] = [k.strip() for k in value.split(';')]
                    elif 'doi' in field:
                        doi_match = re.search(r'10\.\d{4,}/\S+', value)
                        if doi_match:
                            metadata['doi'] = doi_match.group(0)
        
        # Strategy 2: Look for dublin core metadata
        dc_elements = soup.find_all('meta', attrs={'name': re.compile(r'^DC\.|^DCTERMS\.')})
        for elem in dc_elements:
            field = elem.get('name', '').lower()
            value = elem.get('content', '').strip()
            
            if value:
                if 'dc.title' in field:
                    metadata['title'] = value
                elif 'dc.creator' in field or 'dc.contributor.author' in field:
                    if value not in metadata['authors']:
                        metadata['authors'].append(value)
                elif 'dc.date.issued' in field or 'dcterms.issued' in field:
                    metadata['publication_date'] = value
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', value)
                    if year_match:
                        metadata['publication_year'] = int(year_match.group(0))
                elif 'dc.description' in field or 'dc.description.abstract' in field:
                    metadata['abstract'] = value
                elif 'dc.subject' in field:
                    if value not in metadata['keywords']:
                        metadata['keywords'].append(value)
                elif 'dc.identifier.doi' in field:
                    metadata['doi'] = value
        
        # Strategy 3: Look for metadata in divs
        # Title (if not found yet)
        if not metadata['title']:
            title_elem = soup.find(['h1', 'h2'], class_=['page-title', 'ds-div-head'])
            if title_elem:
                metadata['title'] = title_elem.get_text().strip()
        
        # Abstract
        if not metadata['abstract']:
            abstract_elem = soup.find('div', class_=['abstract', 'ds-static-div'])
            if abstract_elem:
                metadata['abstract'] = abstract_elem.get_text().strip()
        
        # Format authors as JSON
        if metadata['authors']:
            metadata['authors_json'] = json.dumps([{"name": author} for author in metadata['authors']])
        else:
            metadata['authors_json'] = None
        
        # Format keywords as JSON
        if metadata['keywords']:
            metadata['identifiers_json'] = json.dumps({"keywords": metadata['keywords']})
        else:
            metadata['identifiers_json'] = None
        
        return metadata
    
    def fetch_all_resources(self, max_items_per_type: int = 100) -> int:
        """Fetch resources from all endpoints and store them in database.
        
        Args:
            max_items_per_type: Maximum number of items to fetch for each resource type
            
        Returns:
            Number of items successfully stored in the database
        """
        logger.info("Starting to fetch all KnowHub resources")
        
        # Check database table first
        if not check_resources_table():
            logger.error("Database table check failed")
            return 0
        
        total_stored = 0
        
        # Try all endpoint types for each resource type
        for resource_type, handle_url in self.handle_endpoints.items():
            logger.info(f"Processing {resource_type}...")
            
            # We'll try all three URL types (handle, collection, browse)
            urls_to_try = [
                self.handle_endpoints.get(resource_type),
                self.collection_endpoints.get(resource_type),
                self.browse_endpoints.get(resource_type)
            ]
            
            # Track items found for this type
            item_links = []
            
            # Try each URL until we get results
            for url in urls_to_try:
                if not url:
                    continue
                    
                links = self._fetch_listing_page(url)
                
                if links:
                    logger.info(f"Found {len(links)} items at {url}")
                    item_links.extend(links)
                    
                    # If we have enough links, stop trying other URLs
                    if len(item_links) >= max_items_per_type:
                        item_links = item_links[:max_items_per_type]
                        break
            
            if not item_links:
                logger.warning(f"No {resource_type} items found in any endpoint")
                continue
            
            # Process item pages concurrently
            logger.info(f"Processing {len(item_links)} {resource_type} detail pages concurrently")
            items_processed = self._process_items_concurrently(item_links, resource_type)
            total_stored += items_processed
        
        logger.info(f"Completed fetching all resources. Total stored: {total_stored}")
        return total_stored
    
    def _process_items_concurrently(self, item_links: List[Dict[str, Any]], resource_type: str) -> int:
        """Process multiple item pages concurrently using a thread pool.
        
        Args:
            item_links: List of dictionaries with item URLs and titles
            resource_type: Type of resource being processed
            
        Returns:
            Number of items successfully stored in the database
        """
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self._process_item_page, item): item['url']
                for item in item_links
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    metadata = future.result()
                    if metadata:
                        # Ensure resource type is set
                        metadata['resource_type'] = resource_type
                        
                        # Store in database
                        result_id = insert_single_resource(metadata)
                        if result_id:
                            processed_count += 1
                            logger.info(f"[{i}/{len(item_links)}] Stored {resource_type} item: {metadata.get('title', '')[:50]}...")
                        else:
                            logger.warning(f"[{i}/{len(item_links)}] Failed to store item: {url}")
                    else:
                        logger.warning(f"[{i}/{len(item_links)}] No metadata extracted: {url}")
                
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        logger.info(f"Processed {len(item_links)} {resource_type} items, stored {processed_count}")
        return processed_count
    
    def run(self, max_items_per_type: int = 100) -> int:
        """Main method to run the scraper and store results.
        
        Args:
            max_items_per_type: Maximum number of items to fetch for each resource type
            
        Returns:
            Total number of items stored in the database
        """
        start_time = time.time()
        
        try:
            total_stored = self.fetch_all_resources(max_items_per_type)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Scraping completed in {elapsed_time:.2f} seconds")
            logger.info(f"Total items stored: {total_stored}")
            
            return total_stored
            
        except Exception as e:
            logger.error(f"Error in scraper execution: {e}", exc_info=True)
            return 0
        finally:
            self.close()
    
    def close(self):
        """Close resources."""
        if self.session:
            self.session.close()
            logger.info("Session closed")

def check_resources_table() -> bool:
    """Check if resources_resource table exists and has necessary columns."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if resources_resource table exists
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
        
        # Check if 'source' column exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'resources_resource' AND column_name = 'source'
            );
        """)
        source_column_exists = cur.fetchone()[0]
        
        if not source_column_exists:
            logger.info("'source' column does not exist in resources_resource table. Adding it...")
            cur.execute("""
                ALTER TABLE resources_resource 
                ADD COLUMN IF NOT EXISTS source VARCHAR(50);
            """)
            conn.commit()
        
        logger.info("resources_resource table is ready.")
        return True
        
    except Exception as e:
        logger.error(f"Error checking resources table: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def insert_single_resource(item: Dict[str, Any]) -> Optional[int]:
    """Insert a single KnowHub resource into the resources_resource table."""
    if not item:
        return None
    
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if item with this URL already exists
        if item.get("url"):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get("url"),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Insert the resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, 
             publication_year, source, identifiers)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (
            next_id,
            item.get("title"),
            item.get("abstract"),
            item.get("url"),
            item.get("resource_type"),
            item.get("authors_json"),
            item.get("publication_year"),
            item.get("source", "knowhub"),
            item.get("identifiers_json")
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            logger.info(f"Inserted resource with ID {result[0]}: {item.get('title')}")
            return result[0]
        else:
            logger.warning(f"Failed to insert resource: {item.get('title')}")
            return None
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error inserting resource '{item.get('title')}': {e}")
        return None
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    """Main function to run the scraper."""
    start_time = time.time()
    
    try:
        # Initialize and run the scraper
        scraper = FastKnowhubScraper(max_workers=8, connection_timeout=5)
        total_stored = scraper.run(max_items_per_type=100)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Done! Stored {total_stored} resources in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()