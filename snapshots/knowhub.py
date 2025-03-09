#!/usr/bin/env python3
import os
import json
import requests
import psycopg2
import time
import re
import subprocess
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any, Set, Union
import logging
from urllib.parse import urljoin, quote
import urllib3
from datetime import datetime

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

class RobustKnowhubScraper:
    """Multi-strategy scraper for KnowHub with fallback mechanisms."""
    
    def __init__(self, use_curl: bool = True, use_oai: bool = True):
        """Initialize the scraper with multiple fallback options.
        
        Args:
            use_curl: Whether to use cURL as a fallback
            use_oai: Whether to attempt to use OAI-PMH as a fallback
        """
        self.base_url = os.getenv('KNOWHUB_BASE_URL', 'https://knowhub.aphrc.org')
        self.use_curl = use_curl
        self.use_oai = use_oai
        self.timeout = 60  # Long timeout for problematic servers
        
        # URL structure - using the handle URL pattern
        self.endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        
        # OAI-PMH endpoints typically used in DSpace
        self.oai_endpoint = f"{self.base_url}/oai/request"
        
        # Create session with extended headers
        self.session = self._create_session()
        
        # Track processed items
        self.processed_urls = set()
        self.processed_handles = set()
        
        logger.info(f"RobustKnowhubScraper initialized with cURL={use_curl}, OAI-PMH={use_oai}")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with enhanced headers."""
        session = requests.Session()
        
        # Set extensive browser-like headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        
        return session
    
    def _make_request(self, url: str) -> Optional[str]:
        """Make a request with multiple fallback methods."""
        logger.info(f"Requesting: {url}")
        
        # Method 1: Standard requests
        try:
            response = self.session.get(url, timeout=self.timeout, verify=False)
            if response.status_code == 200:
                logger.info(f"Request successful with status code {response.status_code}")
                return response.text
            logger.warning(f"Request failed with status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Request error: {e}")
        
        # Method 2: cURL if enabled
        if self.use_curl:
            logger.info("Trying cURL fallback...")
            try:
                html = self._get_with_curl(url)
                if html:
                    logger.info("cURL request successful")
                    return html
                logger.warning("cURL request failed")
            except Exception as e:
                logger.warning(f"cURL error: {e}")
        
        logger.error(f"All request methods failed for {url}")
        return None
    
    def _get_with_curl(self, url: str) -> Optional[str]:
        """Use cURL to fetch a URL, which sometimes works better for problematic sites."""
        try:
            result = subprocess.run(
                ['curl', '-k', '-L', '-s', url],
                capture_output=True, text=True, timeout=self.timeout
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout
            
            logger.warning(f"cURL failed with return code {result.returncode}")
            if result.stderr:
                logger.warning(f"cURL stderr: {result.stderr[:200]}...")
            
            return None
        except subprocess.SubprocessError as e:
            logger.error(f"cURL subprocess error: {e}")
            return None
    
    def _try_oai_pmh(self, resource_type: str) -> List[Dict[str, Any]]:
        """Try to fetch metadata using OAI-PMH protocol."""
        items = []
        
        if not self.use_oai:
            return items
        
        # Map resource type to a set string for OAI-PMH
        set_spec = None
        if resource_type == 'publications':
            set_spec = "col_123456789_1"
        elif resource_type == 'documents':
            set_spec = "col_123456789_2"
        elif resource_type == 'reports':
            set_spec = "col_123456789_3"
        elif resource_type == 'multimedia':
            set_spec = "col_123456789_4"
        
        if not set_spec:
            return items
        
        # Construct OAI-PMH URL
        url = f"{self.oai_endpoint}?verb=ListRecords&metadataPrefix=oai_dc"
        if set_spec:
            url += f"&set={set_spec}"
        
        logger.info(f"Trying OAI-PMH: {url}")
        
        # Make request
        content = self._make_request(url)
        if not content:
            return items
        
        # Parse XML response
        try:
            # Clean potential problematic characters
            content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Define XML namespaces
            namespaces = {
                'oai': 'http://www.openarchives.org/OAI/2.0/',
                'dc': 'http://purl.org/dc/elements/1.1/',
                'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
            }
            
            # Extract records
            records = root.findall('.//oai:record', namespaces)
            logger.info(f"Found {len(records)} OAI-PMH records")
            
            for record in records:
                try:
                    # Get header
                    header = record.find('./oai:header', namespaces)
                    if header is None:
                        continue
                    
                    # Get identifier
                    identifier = header.find('./oai:identifier', namespaces)
                    if identifier is None or not identifier.text:
                        continue
                    
                    # Extract handle from identifier
                    handle_match = re.search(r'oai:[^:]+:([0-9]+/[0-9]+)', identifier.text)
                    handle = handle_match.group(1) if handle_match else None
                    
                    if not handle or handle in self.processed_handles:
                        continue
                    
                    self.processed_handles.add(handle)
                    
                    # Get metadata
                    metadata_elem = record.find('./oai:metadata/oai_dc:dc', namespaces)
                    if metadata_elem is None:
                        continue
                    
                    # Extract Dublin Core elements
                    dc_data = {}
                    
                    # Title
                    title_elem = metadata_elem.find('./dc:title', namespaces)
                    title = title_elem.text if title_elem is not None and title_elem.text else "Unknown Title"
                    
                    # Creator/Author
                    authors = []
                    for creator in metadata_elem.findall('./dc:creator', namespaces):
                        if creator.text:
                            authors.append(creator.text.strip())
                    
                    # Description/Abstract
                    abstract = ""
                    for desc in metadata_elem.findall('./dc:description', namespaces):
                        if desc.text:
                            abstract += desc.text.strip() + " "
                    abstract = abstract.strip()
                    
                    # Date
                    pub_date = None
                    pub_year = None
                    date_elem = metadata_elem.find('./dc:date', namespaces)
                    if date_elem is not None and date_elem.text:
                        pub_date = date_elem.text.strip()
                        # Extract year
                        year_match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                        if year_match:
                            pub_year = int(year_match.group(0))
                    
                    # Subject/Keywords
                    keywords = []
                    for subject in metadata_elem.findall('./dc:subject', namespaces):
                        if subject.text:
                            keywords.append(subject.text.strip())
                    
                    # Type
                    type_elem = metadata_elem.find('./dc:type', namespaces)
                    doc_type = type_elem.text if type_elem is not None and type_elem.text else resource_type
                    
                    # Identifier (URL, DOI)
                    url = None
                    doi = None
                    for identifier in metadata_elem.findall('./dc:identifier', namespaces):
                        if identifier.text:
                            id_text = identifier.text.strip()
                            if id_text.startswith('http'):
                                url = id_text
                            elif 'doi' in id_text.lower() or id_text.startswith('10.'):
                                doi_match = re.search(r'10\.\d{4,}/\S+', id_text)
                                if doi_match:
                                    doi = doi_match.group(0)
                    
                    # If no URL found, construct one
                    if not url:
                        url = f"{self.base_url}/handle/{handle}"
                    
                    # Create item
                    item = {
                        'title': title,
                        'authors': authors,
                        'authors_json': json.dumps([{"name": author} for author in authors]) if authors else None,
                        'abstract': abstract,
                        'publication_date': pub_date,
                        'publication_year': pub_year,
                        'url': url,
                        'doi': doi,
                        'handle': handle,
                        'resource_type': resource_type,
                        'identifiers_json': json.dumps({"handle": handle, "keywords": keywords}) if keywords else None,
                        'source': 'knowhub'
                    }
                    
                    items.append(item)
                    
                except Exception as e:
                    logger.error(f"Error processing OAI-PMH record: {e}")
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            logger.debug(f"XML content preview: {content[:200]}...")
        except Exception as e:
            logger.error(f"OAI-PMH processing error: {e}")
        
        return items
    
    def _extract_items_from_html(self, content: str, resource_type: str) -> List[Dict[str, Any]]:
        """Extract items from HTML content."""
        items = []
        
        if not content:
            return items
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Try multiple selectors to find items
        selectors = [
            'div.artifact-description',
            'div.ds-artifact-item',
            'table.ds-table tr',
            'div.item',
            'li.ds-artifact-item',
            'div.recent-submissions > ul > li'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Found {len(elements)} items using selector: {selector}")
                
                for element in elements:
                    try:
                        # Extract link and title
                        link_elem = element.select_one('a[href*="handle"]')
                        if not link_elem or not link_elem.get('href'):
                            continue
                        
                        href = link_elem.get('href')
                        url = urljoin(self.base_url, href)
                        
                        if url in self.processed_urls:
                            continue
                        
                        self.processed_urls.add(url)
                        
                        # Extract handle from URL
                        handle_match = re.search(r'/handle/([0-9]+/[0-9]+)', href)
                        if not handle_match:
                            continue
                        
                        handle = handle_match.group(1)
                        
                        if handle in self.processed_handles:
                            continue
                        
                        self.processed_handles.add(handle)
                        
                        # Extract title
                        title = link_elem.get_text().strip()
                        if not title or len(title) < 5:
                            continue
                        
                        # Get item details
                        item_details = self._get_item_details(url, handle, title, resource_type)
                        if item_details:
                            items.append(item_details)
                    
                    except Exception as e:
                        logger.warning(f"Error processing item element: {e}")
                
                # If we found items, no need to try other selectors
                if items:
                    break
        
        return items
    
    def _get_item_details(self, url: str, handle: str, title: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed metadata for an item."""
        logger.info(f"Getting details for: {url}")
        
        # Make request for item page
        content = self._make_request(url)
        if not content:
            # If we can't get the item page, create a minimal record
            return {
                'title': title,
                'url': url,
                'handle': handle,
                'abstract': '',
                'authors': [],
                'authors_json': None,
                'publication_date': None,
                'publication_year': None,
                'resource_type': resource_type,
                'identifiers_json': json.dumps({"handle": handle}),
                'source': 'knowhub'
            }
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract metadata
            metadata = {}
            
            # Try to get a better title
            title_elem = soup.select_one('h1.page-title, h2.page-title, h1.ds-div-head, h2.ds-div-head')
            if title_elem:
                metadata['title'] = title_elem.get_text().strip()
            else:
                metadata['title'] = title
            
            # Abstract
            abstract_elem = soup.select_one('div.simple-item-view-description div.item-page-field-wrapper-value, div.item-summary-view-metadata p.item-view-abstract')
            if abstract_elem:
                metadata['abstract'] = abstract_elem.get_text().strip()
            else:
                metadata['abstract'] = ''
            
            # Authors
            authors = []
            author_elems = soup.select('div.simple-item-view-authors div.item-page-field-wrapper-value a, div.item-summary-view-metadata p.item-view-authors a')
            for author_elem in author_elems:
                author_name = author_elem.get_text().strip()
                if author_name and author_name not in authors:
                    authors.append(author_name)
            
            metadata['authors'] = authors
            metadata['authors_json'] = json.dumps([{"name": author} for author in authors]) if authors else None
            
            # Date
            date_elem = soup.select_one('div.simple-item-view-date div.item-page-field-wrapper-value, div.item-summary-view-metadata p.item-view-date')
            if date_elem:
                date_str = date_elem.get_text().strip()
                metadata['publication_date'] = date_str
                
                # Extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    metadata['publication_year'] = int(year_match.group(0))
            else:
                metadata['publication_date'] = None
                metadata['publication_year'] = None
            
            # Keywords/Subjects
            keywords = []
            keyword_elems = soup.select('div.simple-item-view-subject div.item-page-field-wrapper-value a, div.item-summary-view-metadata p.item-view-subject a')
            for keyword_elem in keyword_elems:
                keyword = keyword_elem.get_text().strip()
                if keyword and keyword not in keywords:
                    keywords.append(keyword)
            
            # Additional metadata for identifiers
            identifiers = {
                "handle": handle,
                "keywords": keywords
            }
            
            metadata['identifiers_json'] = json.dumps(identifiers) if keywords else json.dumps({"handle": handle})
            
            # DOI
            doi_elem = soup.select_one('div.simple-item-view-doi div.item-page-field-wrapper-value, div.item-summary-view-metadata p.item-view-doi')
            if doi_elem:
                doi_text = doi_elem.get_text().strip()
                doi_match = re.search(r'10\.\d{4,}/\S+', doi_text)
                if doi_match:
                    metadata['doi'] = doi_match.group(0)
            else:
                metadata['doi'] = None
            
            # Combine with base metadata
            item = {
                'title': metadata.get('title', title),
                'url': url,
                'handle': handle,
                'abstract': metadata.get('abstract', ''),
                'authors': metadata.get('authors', []),
                'authors_json': metadata.get('authors_json'),
                'publication_date': metadata.get('publication_date'),
                'publication_year': metadata.get('publication_year'),
                'doi': metadata.get('doi'),
                'resource_type': resource_type,
                'identifiers_json': metadata.get('identifiers_json'),
                'source': 'knowhub'
            }
            
            return item
            
        except Exception as e:
            logger.error(f"Error extracting item details: {e}")
            
            # Return minimal metadata
            return {
                'title': title,
                'url': url,
                'handle': handle,
                'abstract': '',
                'authors': [],
                'authors_json': None,
                'publication_date': None,
                'publication_year': None,
                'resource_type': resource_type,
                'identifiers_json': json.dumps({"handle": handle}),
                'source': 'knowhub'
            }
    
    def _process_browse_pages(self, resource_type: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """Process browse pages that list many items."""
        items = []
        
        # Try browse URLs
        browse_url = None
        if resource_type == 'publications':
            browse_url = f"{self.base_url}/browse?type=dateissued"
        elif resource_type == 'documents':
            browse_url = f"{self.base_url}/browse?type=doctype&sort_by=1&order=DESC"
        elif resource_type == 'reports':
            browse_url = f"{self.base_url}/browse?type=doctype&value=Technical+Report"
        elif resource_type == 'multimedia':
            browse_url = f"{self.base_url}/browse?type=dateissued"
        
        if not browse_url:
            return items
        
        content = self._make_request(browse_url)
        if not content:
            return items
        
        # Extract items from browse page
        browse_items = self._extract_items_from_html(content, resource_type)
        items.extend(browse_items)
        
        # Try to follow pagination links if needed
        if len(items) < max_items:
            soup = BeautifulSoup(content, 'html.parser')
            next_links = soup.select('a[href*="next"]')
            
            for link in next_links:
                if len(items) >= max_items:
                    break
                
                href = link.get('href')
                if href:
                    next_url = urljoin(self.base_url, href)
                    next_content = self._make_request(next_url)
                    
                    if next_content:
                        next_items = self._extract_items_from_html(next_content, resource_type)
                        items.extend(next_items)
        
        # Limit to max_items
        return items[:max_items]
    
    def fetch_resources(self, resource_type: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """Fetch resources using multiple strategies.
        
        Args:
            resource_type: Type of resource to fetch
            max_items: Maximum number of items to fetch
            
        Returns:
            List of resource items
        """
        logger.info(f"Fetching {resource_type} with max_items={max_items}")
        all_items = []
        
        # Strategy 1: Try OAI-PMH
        if self.use_oai:
            logger.info(f"Trying OAI-PMH for {resource_type}")
            oai_items = self._try_oai_pmh(resource_type)
            if oai_items:
                logger.info(f"Found {len(oai_items)} items via OAI-PMH")
                all_items.extend(oai_items)
                
                # If we have enough items, return
                if len(all_items) >= max_items:
                    return all_items[:max_items]
        
        # Strategy 2: Try browse pages
        if len(all_items) < max_items:
            logger.info(f"Trying browse pages for {resource_type}")
            browse_items = self._process_browse_pages(resource_type, max_items - len(all_items))
            if browse_items:
                logger.info(f"Found {len(browse_items)} items via browse pages")
                all_items.extend(browse_items)
        
        # Strategy 3: Try direct endpoint as a last resort
        if len(all_items) < max_items and resource_type in self.endpoints:
            logger.info(f"Trying direct endpoint for {resource_type}")
            endpoint_url = self.endpoints[resource_type]
            content = self._make_request(endpoint_url)
            
            if content:
                endpoint_items = self._extract_items_from_html(content, resource_type)
                if endpoint_items:
                    logger.info(f"Found {len(endpoint_items)} items via direct endpoint")
                    all_items.extend(endpoint_items)
        
        # Make sure we have no duplicates (by URL)
        unique_items = {}
        for item in all_items:
            url = item.get('url')
            if url and url not in unique_items:
                unique_items[url] = item
        
        return list(unique_items.values())[:max_items]
    
    def run(self, max_items_per_type: int = 50) -> int:
        """Run scraper for all resource types.
        
        Args:
            max_items_per_type: Maximum items to fetch per resource type
            
        Returns:
            Total number of items stored in database
        """
        start_time = time.time()
        total_stored = 0
        
        try:
            # Check database table
            if not check_resources_table():
                logger.error("Database table check failed")
                return 0
            
            # Process each resource type
            for resource_type in self.endpoints:
                logger.info(f"\nProcessing {resource_type}...")
                
                # Fetch items
                items = self.fetch_resources(resource_type, max_items_per_type)
                
                if not items:
                    logger.warning(f"No {resource_type} items found")
                    continue
                
                logger.info(f"Found {len(items)} {resource_type} items, inserting into database...")
                
                # Insert items
                inserted = 0
                for item in items:
                    result_id = insert_single_resource(item)
                    if result_id:
                        inserted += 1
                    
                    # Small delay to avoid database issues
                    time.sleep(0.1)
                
                total_stored += inserted
                logger.info(f"Inserted {inserted} out of {len(items)} {resource_type} items")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Done! Stored {total_stored} resources in {elapsed_time:.2f} seconds")
            return total_stored
            
        except Exception as e:
            logger.error(f"Error running scraper: {e}", exc_info=True)
            return total_stored
        
        finally:
            self.close()
    
    def close(self):
        """Close resources."""
        if hasattr(self, 'session') and self.session:
            self.session.close()

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
    """Insert a single resource into the database."""
    if not item:
        return None
    
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if resource already exists by URL
        if item.get('url'):
            cur.execute("SELECT id FROM resources_resource WHERE url = %s", (item.get('url'),))
            result = cur.fetchone()
            if result:
                logger.info(f"Resource with URL '{item.get('url')}' already exists with ID {result[0]}")
                return result[0]
        
        # Get next available ID
        cur.execute("SELECT MAX(id) FROM resources_resource")
        max_id = cur.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Insert resource
        cur.execute("""
            INSERT INTO resources_resource 
            (id, title, abstract, url, type, authors, 
             publication_year, source, identifiers, doi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (
            next_id,
            item.get('title'),
            item.get('abstract'),
            item.get('url'),
            item.get('resource_type'),
            item.get('authors_json'),
            item.get('publication_year'),
            item.get('source', 'knowhub'),
            item.get('identifiers_json'),
            item.get('doi')
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            logger.info(f"Inserted resource with ID {result[0]}: {item.get('title')[:50]}...")
            return result[0]
        else:
            logger.warning(f"Failed to insert resource: {item.get('title')[:50]}...")
            return None
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error inserting resource '{item.get('title', 'Unknown')}': {e}")
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
        # Initialize and run scraper
        scraper = RobustKnowhubScraper(use_curl=True, use_oai=True)
        total_stored = scraper.run(max_items_per_type=50)
        
        elapsed_time = time.time() - start_time
        print(f"Done! Stored {total_stored} resources in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()