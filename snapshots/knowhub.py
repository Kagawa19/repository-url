#!/usr/bin/env python3
import os
import json
import requests
import psycopg2
import time
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any
import logging
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='knowhub_scraper.log'
)

# Database connection settings
DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "aphrc"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "host": "localhost",
    "port": "5432"
}

class TextSummarizer:
    """Simple placeholder for text summarization functionality."""
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Placeholder for text summarization."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

class KnowhubScraper:
    def __init__(self, summarizer: Optional[TextSummarizer] = None):
        """Initialize KnowhubScraper with advanced connection capabilities."""
        self.base_url = os.getenv('KNOWHUB_BASE_URL', 'https://knowhub.aphrc.org')
        
        # OPTION 1: Use alternative URL structure if the handles aren't working
        self.publications_url = f"{self.base_url}/communities/3/collections"
        
        # Map endpoints using collection IDs instead of handles
        self.endpoints = {
            'documents': f"{self.base_url}/collections/12",
            'reports': f"{self.base_url}/collections/14", 
            'multimedia': f"{self.base_url}/collections/15"
        }
        
        self.summarizer = summarizer or TextSummarizer()
        
        # Create a session with retry capabilities
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,                 # Maximum number of retries
            backoff_factor=1,        # Exponential backoff
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET"]  # Only retry GET requests
        )
        
        # Mount adapter with retry strategy for both http and https
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Add browser-like headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def try_alternative_url(self, resource_type: str) -> str:
        """Try alternative URL formats for KnowHub."""
        # OPTION 2: Try direct browse URL
        if resource_type == 'publications':
            return f"{self.base_url}/browse?type=dateissued"
        elif resource_type == 'documents':
            return f"{self.base_url}/browse?type=doctype&sort_by=1&order=DESC"
        elif resource_type == 'reports':
            return f"{self.base_url}/browse?type=doctype&value=Technical+Report"
        else:
            return f"{self.base_url}/browse"
    
    def fetch_items_by_type(self, resource_type: str) -> List[Dict[str, Any]]:
        """Fetch items from KnowHub based on resource type with fallback mechanisms."""
        if resource_type == 'publications':
            url = self.publications_url
        else:
            url = self.endpoints.get(resource_type)
            if not url:
                logging.warning(f"Unknown resource type: {resource_type}")
                return []
        
        items = []
        try:
            logging.info(f"Fetching {resource_type} from {url}")
            response = self.session.get(url, timeout=(5, 25))  # (connect timeout, read timeout)
            
            if response.status_code == 200:
                items = self._extract_items_from_response(response, resource_type)
            else:
                logging.warning(f"Received status code {response.status_code} from {url}")
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logging.warning(f"Connection issue with primary URL for {resource_type}: {e}")
            
            # Try alternative URL if primary fails
            alt_url = self.try_alternative_url(resource_type)
            logging.info(f"Trying alternative URL for {resource_type}: {alt_url}")
            
            try:
                alt_response = self.session.get(alt_url, timeout=(5, 25))
                if alt_response.status_code == 200:
                    items = self._extract_items_from_response(alt_response, resource_type)
                else:
                    logging.warning(f"Alternative URL returned status code {alt_response.status_code}")
            except requests.exceptions.RequestException as e2:
                logging.error(f"Alternative URL also failed for {resource_type}: {e2}")
                
        except Exception as e:
            logging.error(f"Unexpected error when processing {resource_type}: {e}")
            
        return items
    
    def _extract_items_from_response(self, response, resource_type: str) -> List[Dict[str, Any]]:
        """Extract items from HTML response with multiple parsing strategies."""
        items = []
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Log page details for debugging
        logging.info(f"Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Different parsing strategies
        strategies = [
            self._parse_standard_layout,
            self._parse_artifact_layout,
            self._parse_generic_links,
            self._parse_last_resort
        ]
        
        # Try each strategy until one succeeds
        for strategy in strategies:
            logging.info(f"Trying parsing strategy: {strategy.__name__}")
            extracted_items = strategy(soup, resource_type)
            if extracted_items:
                logging.info(f"Successfully extracted {len(extracted_items)} items with {strategy.__name__}")
                items = extracted_items
                break
        
        return items
    
    def _parse_standard_layout(self, soup, resource_type: str) -> List[Dict[str, Any]]:
        """Parse standard layout based on the screenshot."""
        items = []
        
        # Look for items in a standard layout
        recent_submissions = soup.select('div.recent-submissions li, div.artifact-description')
        
        for item in recent_submissions:
            title_element = item.select_one('h4 a, h3 a, a.artifact-title, a[href*="handle"]')
            if not title_element:
                continue
                
            title = title_element.text.strip()
            item_url = title_element.get('href')
            
            if item_url and not item_url.startswith('http'):
                item_url = f"{self.base_url}{item_url}"
            
            authors_element = item.select_one('.item-authors, .authors')
            authors = authors_element.text.strip() if authors_element else ""
            
            description_element = item.select_one('.item-description, .description')
            description = description_element.text.strip() if description_element else ""
            
            date_element = item.select_one('.item-date, .date, .dateissued')
            publication_date = date_element.text.strip() if date_element else None
            
            items.append({
                "title": title,
                "url": item_url,
                "author": authors,
                "description": self.summarizer.summarize(description),
                "publication_date": publication_date,
                "resource_type": resource_type,
                "source": "knowhub"
            })
            
        return items
    
    def _parse_artifact_layout(self, soup, resource_type: str) -> List[Dict[str, Any]]:
        """Parse artifact layout."""
        items = []
        
        # This attempts to find items that match the structure in typical DSpace installations
        artifact_items = soup.select('.ds-artifact-item, .artifact-item')
        
        for item in artifact_items:
            title_element = item.select_one('.ds-artifact-title a, .artifact-title a')
            if not title_element:
                continue
                
            title = title_element.text.strip()
            item_url = title_element.get('href')
            
            if item_url and not item_url.startswith('http'):
                item_url = f"{self.base_url}{item_url}"
            
            authors_element = item.select_one('.ds-artifact-authors, .artifact-authors')
            authors = authors_element.text.strip() if authors_element else ""
            
            description_element = item.select_one('.ds-artifact-abstract, .artifact-abstract')
            description = description_element.text.strip() if description_element else ""
            
            date_element = item.select_one('.ds-artifact-date, .artifact-date')
            publication_date = date_element.text.strip() if date_element else None
            
            items.append({
                "title": title,
                "url": item_url,
                "author": authors,
                "description": self.summarizer.summarize(description),
                "publication_date": publication_date,
                "resource_type": resource_type,
                "source": "knowhub"
            })
            
        return items
    
    def _parse_generic_links(self, soup, resource_type: str) -> List[Dict[str, Any]]:
        """Parse generic links that could be publications."""
        items = []
        
        # Look for links that might be publications
        links = soup.select('a[href*="handle"], a[href*="item"]')
        
        for link in links:
            title = link.text.strip()
            
            # Ignore navigation, pagination links, or very short titles
            if len(title) < 10 or title.lower() in ['next', 'previous', 'home', 'browse'] or 'page' in title.lower():
                continue
                
            item_url = link.get('href')
            if item_url and not item_url.startswith('http'):
                item_url = f"{self.base_url}{item_url}"
            
            # Create minimal item data
            items.append({
                "title": title,
                "url": item_url,
                "author": "",
                "description": "",
                "publication_date": None,
                "resource_type": resource_type,
                "source": "knowhub"
            })
        
        return items
    
    def _parse_last_resort(self, soup, resource_type: str) -> List[Dict[str, Any]]:
        """Last resort parsing strategy that tries to find anything useful."""
        items = []
        
        # Try to find any table rows that might contain items
        tr_elements = soup.select('table tr')
        
        for tr in tr_elements:
            if len(tr.select('td, th')) < 2:  # Skip rows with too few cells
                continue
                
            link = tr.select_one('a[href*="handle"], a[href*="item"]')
            if not link:
                continue
                
            title = link.text.strip()
            if len(title) < 5:  # Skip very short titles
                continue
                
            item_url = link.get('href')
            if item_url and not item_url.startswith('http'):
                item_url = f"{self.base_url}{item_url}"
            
            # Try to extract other information from table cells
            cells = tr.select('td')
            
            # Create item with available data
            items.append({
                "title": title,
                "url": item_url,
                "author": cells[1].text.strip() if len(cells) > 1 else "",
                "description": "",
                "publication_date": cells[2].text.strip() if len(cells) > 2 else None,
                "resource_type": resource_type,
                "source": "knowhub"
            })
        
        return items
    
    def fetch_item_details(self, item_url: str) -> Dict[str, Any]:
        """Fetch detailed information about an item from its page."""
        try:
            logging.info(f"Fetching details from {item_url}")
            response = self.session.get(item_url, timeout=(5, 25))
            
            if response.status_code != 200:
                logging.warning(f"Failed to fetch item details, status code: {response.status_code}")
                return {}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract detailed information
            abstract = ""
            abstract_element = soup.select_one('.item-abstract, .abstract, meta[name="DCTERMS.abstract"]')
            if abstract_element:
                if abstract_element.name == 'meta':
                    abstract = abstract_element.get('content', '')
                else:
                    abstract = abstract_element.text.strip()
            
            # Extract more accurate author information
            authors = []
            author_elements = soup.select('.item-author, .author, meta[name="DC.creator"]')
            for author_el in author_elements:
                if author_el.name == 'meta':
                    authors.append(author_el.get('content', ''))
                else:
                    authors.append(author_el.text.strip())
            
            # Extract publication date
            publication_date = None
            date_element = soup.select_one('.item-date, .date, meta[name="DCTERMS.issued"]')
            if date_element:
                if date_element.name == 'meta':
                    publication_date = date_element.get('content', '')
                else:
                    publication_date = date_element.text.strip()
            
            return {
                "abstract": abstract,
                "authors": ", ".join(authors) if authors else "",
                "publication_date": publication_date
            }
            
        except Exception as e:
            logging.error(f"Error fetching item details: {e}")
            return {}

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
            logging.error("resources_resource table does not exist. Please create it first.")
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
            logging.info("'source' column does not exist in resources_resource table. Adding it...")
            cur.execute("""
                ALTER TABLE resources_resource 
                ADD COLUMN IF NOT EXISTS source VARCHAR(50);
            """)
            conn.commit()
        
        logging.info("resources_resource table is ready.")
        return True
        
    except Exception as e:
        logging.error(f"Error checking resources table: {e}")
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
                logging.info(f"Resource with URL {item.get('url')} already exists with ID {result[0]}")
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Prepare authors JSON if needed
        authors_json = None
        if item.get("author"):
            # Simple conversion of author string to JSON format
            author_names = [name.strip() for name in item.get("author", "").split(",") if name.strip()]
            if author_names:
                authors_json = json.dumps([{"name": name} for name in author_names])
        
        # Extract publication_year from publication_date if available
        publication_year = None
        if item.get("publication_date"):
            try:
                # Try to extract a year from the date string
                date_str = item.get("publication_date", "")
                # Look for 4-digit year pattern
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    publication_year = int(year_match.group(0))
            except Exception as e:
                logging.warning(f"Error extracting publication year: {e}")
        
        # Insert the resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, 
             publication_year, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (
            next_id,
            item.get("title"),
            item.get("description"),
            item.get("url"),
            item.get("resource_type"),
            authors_json,
            publication_year,
            item.get("source", "knowhub")
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            logging.info(f"Inserted resource with ID {result[0]}: {item.get('title')}")
            return result[0]
        else:
            logging.warning(f"Failed to insert resource: {item.get('title')}")
            return None
        
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Error inserting resource '{item.get('title')}': {e}")
        return None
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    """Main function to scrape KnowHub and insert data into the database."""
    import time
    start_time = time.time()
    
    try:
        # Check if resources_resource table exists
        if not check_resources_table():
            return
        
        # Initialize scraper with text summarizer
        scraper = KnowhubScraper(TextSummarizer())
        
        # Track insertion statistics
        total_resources = 0
        inserted_count = 0
        
        # Process each resource type separately and insert as we go
        for resource_type in ['publications'] + list(scraper.endpoints.keys()):
            logging.info(f"\nProcessing {resource_type}...")
            resources = scraper.fetch_items_by_type(resource_type)
            
            if not resources:
                logging.warning(f"No {resource_type} found.")
                continue
                
            logging.info(f"Found {len(resources)} {resource_type}. Inserting into database...")
            
            # Insert resources one by one for fault tolerance
            for item in resources:
                total_resources += 1
                
                # Fetch additional details if URL is available
                if item.get("url"):
                    try:
                        details = scraper.fetch_item_details(item["url"])
                        # Update item with additional details
                        if details.get("abstract"):
                            item["description"] = details["abstract"]
                        if details.get("authors"):
                            item["author"] = details["authors"]
                        if details.get("publication_date"):
                            item["publication_date"] = details["publication_date"]
                    except Exception as e:
                        logging.warning(f"Error fetching details for {item['url']}: {e}")
                
                # Insert the individual resource
                result_id = insert_single_resource(item)
                if result_id:
                    inserted_count += 1
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)
                
            logging.info(f"Completed {resource_type}: {inserted_count} inserted so far.")
        
        if total_resources == 0:
            logging.warning("No resources found in KnowHub.")
        
        elapsed_time = time.time() - start_time
        logging.info(f"Done! Inserted {inserted_count} resources out of {total_resources} in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()