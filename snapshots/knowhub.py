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
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
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
        self.seen_urls = set()
        
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
            # Skip if URL was already processed
            if url in self.seen_urls:
                logger.debug(f"Skipping already processed URL: {url}")
                return None
                
            # Add URL to seen URLs
            self.seen_urls.add(url)
            
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
                'keywords': [],
                'identifiers': {},
                'urls': []
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
                
                # Extract DOI if available
                doi_elem = metadata_section.find('span', class_=['identifier', 'doi'])
                if doi_elem:
                    metadata['identifiers']['doi'] = doi_elem.get_text().strip()
            
            # Look for file download links
            file_section = soup.find('div', class_=lambda c: c and 'item-files' in c)
            if file_section:
                download_links = file_section.find_all('a', href=lambda href: href and ('.pdf' in href or '/bitstream/' in href))
                for link in download_links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(self.base_url, href)
                        metadata['urls'].append({
                            'type': 'download',
                            'url': full_url,
                            'label': link.get_text().strip() or 'Download'
                        })
            
            # Extract handle and other identifiers
            handle = None
            match = re.search(r'/handle/([^?/]+)', item_url)
            if match:
                handle = match.group(1)
                metadata['identifiers']['handle'] = handle
                
            # Construct resource dictionary
            resource = {
                'title': title,
                'abstract': metadata['abstract'],
                'authors': [{'name': author} for author in metadata['authors']],
                'type': content_type,
                'source': 'knowhub',
                'collection': content_type,
                'date_issue': metadata['date'],
                'identifiers': metadata['identifiers'],
                'keywords': metadata['keywords'],
                'urls': metadata['urls']
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

    def _get_browse_links(self, collection_url):
        """Extract browse links from a collection page"""
        browse_links = []
        
        try:
            page_content = self._fetch_page(collection_url)
            if not page_content:
                return browse_links
                
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Find browse links (By Issue Date, Authors, Titles, Subjects)
            browse_sections = soup.find_all('a', href=re.compile(r'/browse\?type='))
            
            for link in browse_sections:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    browse_links.append(full_url)
                    
            # Also check for sub-collection links that appear in the "Recent Submissions" section
            recent_submissions = soup.find(string="Recent Submissions")
            if recent_submissions:
                # Navigate to the container of recent submissions
                parent = recent_submissions.parent
                if parent:
                    container = parent.parent
                    if container:
                        # Find all links within this container that point to collections
                        sub_links = container.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
                        for link in sub_links:
                            href = link.get('href')
                            text = link.get_text().strip()
                            
                            # Skip links with no text or "View more" links or already seen links
                            if (href and 
                                href not in self.seen_handles and 
                                text and 
                                not text.startswith('View more') and
                                not re.search(r'/handle/123456789/\d+/browse', href)):
                                
                                # Check if this is likely a sub-collection (not an item)
                                # Collection links usually have shorter texts like "Reports" or "Policy Brief"
                                if len(text) < 80:  # Somewhat arbitrary cutoff for collection names
                                    full_url = urljoin(self.base_url, href)
                                    browse_links.append(full_url)
                                    logger.info(f"Found sub-collection: {text} at {full_url}")
            
            # Look for browse links in the sidebar
            sidebar = soup.find('div', class_=lambda c: c and 'sidebar' in c)
            if sidebar:
                sub_links = sidebar.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
                for link in sub_links:
                    href = link.get('href')
                    text = link.get_text().strip()
                    
                    if (href and 
                        href not in self.seen_handles and 
                        text and 
                        not text.startswith('View more')):
                        
                        full_url = urljoin(self.base_url, href)
                        browse_links.append(full_url)
                        logger.info(f"Found sidebar link: {text} at {full_url}")
            
            # Add specific browse pages we know exist
            collection_id = re.search(r'/handle/123456789/(\d+)', collection_url)
            if collection_id:
                collection_num = collection_id.group(1)
                browse_types = ['dateissued', 'author', 'title', 'subject']
                
                for browse_type in browse_types:
                    browse_url = f"{self.base_url}/handle/123456789/{collection_num}/browse?type={browse_type}"
                    if browse_url not in browse_links:
                        browse_links.append(browse_url)
                        logger.info(f"Added standard browse link: {browse_url}")
                
            return browse_links
                    
        except Exception as e:
            logger.error(f"Error extracting browse links from {collection_url}: {e}")
            return browse_links

    def _get_view_more_link(self, collection_url):
        """Extract 'View more' link from a collection page"""
        try:
            page_content = self._fetch_page(collection_url)
            if not page_content:
                return None
                
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Look for "View more" link
            view_more = soup.find('a', text=re.compile(r'View more', re.IGNORECASE))
            
            if view_more and view_more.get('href'):
                return urljoin(self.base_url, view_more.get('href'))
                
            return None
                
        except Exception as e:
            logger.error(f"Error extracting view more link from {collection_url}: {e}")
            return None

    def _get_pagination_links(self, page_url):
        """Extract pagination links from a page"""
        next_link = None
        
        try:
            page_content = self._fetch_page(page_url)
            if not page_content:
                return next_link
                
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Find pagination next link
            pagination = soup.find('ul', class_='pagination')
            if pagination:
                next_page = pagination.find('a', text=re.compile(r'Next|Next Page|›|>>', re.IGNORECASE))
                if next_page and next_page.get('href'):
                    next_link = urljoin(self.base_url, next_page.get('href'))
            
            # Check for alternate pagination formats
            if not next_link:
                # Look for offset-based pagination
                parsed_url = urlparse(page_url)
                query_params = parse_qs(parsed_url.query)
                
                if 'offset' in query_params:
                    current_offset = int(query_params['offset'][0])
                    limit = 20  # Default limit, may need to be adjusted
                    
                    # Check if there are more items to fetch
                    if page_content.find('No results found') == -1:
                        new_params = query_params.copy()
                        new_params['offset'] = [str(current_offset + limit)]
                        new_query = urlencode(new_params, doseq=True)
                        
                        next_link = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{new_query}"
            
            return next_link
                
        except Exception as e:
            logger.error(f"Error extracting pagination links from {page_url}: {e}")
            return None

    def _process_browse_page(self, browse_url):
        """Process a browse page and extract all items"""
        all_items = []
        current_url = browse_url
        
        logger.info(f"Processing browse page: {browse_url}")
        
        # Process the initial page
        items = self._extract_collection_items(current_url)
        logger.info(f"Found {len(items)} items on initial browse page")
        all_items.extend(items)
        
        # Process pagination if any
        while True:
            next_page_url = self._get_pagination_links(current_url)
            if not next_page_url:
                logger.info(f"No more pagination found for {current_url}")
                break
                
            logger.info(f"Processing next page: {next_page_url}")
            items = self._extract_collection_items(next_page_url)
            
            if not items:
                # No more items found, break the loop
                logger.info(f"No items found on next page, stopping pagination")
                break
                
            logger.info(f"Found {len(items)} items on pagination page")
            all_items.extend(items)
            current_url = next_page_url
            
            # Short delay to avoid overloading the server
            time.sleep(1)
        
        # Check if this is a collection that might have subcollections
        if '/handle/123456789/' in browse_url and not re.search(r'/browse\?', browse_url):
            logger.info(f"Checking for subcollections in {browse_url}")
            sub_links = self._get_browse_links(browse_url)
            
            for sub_link in sub_links:
                # Don't process links we've already seen
                if sub_link in self.seen_urls:
                    continue
                    
                logger.info(f"Processing subcollection: {sub_link}")
                sub_items = self._process_browse_page(sub_link)
                all_items.extend(sub_items)
        
        return all_items

    def _process_view_more_page(self, view_more_url):
        """Process a 'View more' page and extract all items"""
        return self._process_browse_page(view_more_url)

    def _get_subcollections_from_page(self, collection_url):
        """Extract subcollection links from a collection page by inspecting its structure"""
        subcollections = []
        
        try:
            page_content = self._fetch_page(collection_url)
            if not page_content:
                return subcollections
                
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Look for elements that typically contain subcollection lists in DSpace
            # Often these are in divs with class names like 'list-group', 'collection-list', etc.
            potential_containers = [
                soup.find('div', class_='list-group'),
                soup.find('div', class_='collection-list'),
                soup.find('div', class_='community-browser'),
                soup.find('div', class_='recent-submissions'),
                soup.find('div', class_=lambda c: c and 'recent' in c.lower()),
                soup.find('div', class_=lambda c: c and 'community' in c.lower()),
                soup.find('div', class_=lambda c: c and 'collection' in c.lower()),
            ]
            
            # Find links in all potential containers
            for container in potential_containers:
                if container:
                    links = container.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
                    for link in links:
                        href = link.get('href')
                        text = link.get_text().strip()
                        
                        # Skip "View more" links and empty links
                        if (href and 
                            text and
                            not text.startswith('View more') and
                            not text in ['By Issue Date', 'Authors', 'Titles', 'Subjects'] and
                            not re.search(r'/handle/123456789/\d+/browse', href)):
                            
                            full_url = urljoin(self.base_url, href)
                            if full_url not in subcollections:
                                subcollections.append(full_url)
                                logger.info(f"Found subcollection: {text} at {full_url}")
            
            # Also check for the "Recent Submissions" section specifically
            recent_submissions = soup.find(string="Recent Submissions")
            if recent_submissions:
                parent = recent_submissions.parent
                if parent:
                    container = parent.parent
                    if container:
                        links = container.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
                        for link in links:
                            href = link.get('href')
                            text = link.get_text().strip()
                            
                            # Skip empty links, "View more" and browse links
                            if (href and 
                                text and
                                not text.startswith('View more') and
                                not text in ['By Issue Date', 'Authors', 'Titles', 'Subjects'] and
                                not re.search(r'/handle/123456789/\d+/browse', href)):
                                
                                full_url = urljoin(self.base_url, href)
                                if full_url not in subcollections:
                                    subcollections.append(full_url)
                                    logger.info(f"Found subcollection from Recent Submissions: {text} at {full_url}")
            
            # Add direct browse links for this collection
            match = re.search(r'/handle/123456789/(\d+)', collection_url)
            if match:
                collection_num = match.group(1)
                browse_types = ['dateissued', 'author', 'title', 'subject']
                
                for browse_type in browse_types:
                    browse_url = f"{self.base_url}/handle/123456789/{collection_num}/browse?type={browse_type}"
                    if browse_url not in subcollections:
                        subcollections.append(browse_url)
                        logger.info(f"Added standard browse link: {browse_url}")
            
            # Add discover links that should retrieve most content
            if not subcollections:  # If no subcollections found, use these as a fallback
                discover_links = [
                    f"{self.base_url}/discover?filtertype=dateIssued&filter_relational_operator=equals&filter=%5B2020+TO+2024%5D",
                    f"{self.base_url}/discover?filtertype=dateIssued&filter_relational_operator=equals&filter=%5B2013+TO+2019%5D",
                    f"{self.base_url}/discover?filtertype=has_content_in_original_bundle&filter_relational_operator=equals&filter=true"
                ]
                
                for link in discover_links:
                    if link not in subcollections:
                        subcollections.append(link)
                        logger.info(f"Added discover link: {link}")
                        
            return subcollections
        
        except Exception as e:
            logger.error(f"Error extracting subcollections from {collection_url}: {e}")
            return subcollections

    def _extract_all_collection_items(self, collection_url, collection_name):
        """Extract all items from a collection including all browse pages"""
        all_items = []
        
        logger.info(f"Processing main collection page for: {collection_name}")
        
        # First extract items from the main collection page
        items = self._extract_collection_items(collection_url)
        all_items.extend(items)
        logger.info(f"Found {len(items)} items on main {collection_name} page")
        
        # Get subcollection links by analyzing the page structure
        subcollections = self._get_subcollections_from_page(collection_url)
        logger.info(f"Found {len(subcollections)} subcollection/browse links for {collection_name}")
        
        # Process each subcollection
        for subcollection_url in subcollections:
            if subcollection_url in self.seen_urls:
                logger.info(f"Skipping already processed subcollection: {subcollection_url}")
                continue
                
            logger.info(f"Processing subcollection: {subcollection_url}")
            sub_items = self._process_browse_page(subcollection_url)
            all_items.extend(sub_items)
            logger.info(f"Found {len(sub_items)} items in subcollection: {subcollection_url}")
        
        # Debug info before deduplication
        logger.info(f"Total items before deduplication: {len(all_items)}")
        
        # Create a more sophisticated deduplication key
        unique_items = {}
        duplicate_count = 0
        
        for item in all_items:
            handle = item.get('identifiers', {}).get('handle', '')
            title = item.get('title', '')
            abstract_start = item.get('abstract', '')[:50] if item.get('abstract') else ''
            
            # This key should be more robust to identify truly unique items
            unique_key = f"{handle}|{title}|{abstract_start}"
            
            if unique_key and unique_key not in unique_items:
                unique_items[unique_key] = item
            else:
                duplicate_count += 1
        
        logger.info(f"Removed {duplicate_count} duplicates from {collection_name}")
        logger.info(f"Total unique items found in {collection_name}: {len(unique_items)}")
        
        # Reset the seen_handles set after processing each collection
        prev_seen_count = len(self.seen_handles)
        self.seen_handles = set()
        logger.info(f"Reset seen_handles after collection (cleared {prev_seen_count} entries)")
        
        return list(unique_items.values())

    def explore_collection_structure(self, collection_url, collection_name):
        """
        Explore the structure of a collection before full retrieval
        
        Args:
            collection_url (str): URL of the collection to explore
            collection_name (str): Name of the collection
        
        Returns:
            dict: Detailed structure information about the collection
        """
        structure_info = {
            'collection_name': collection_name,
            'total_browse_links': 0,
            'browse_link_types': [],
            'subcollections': [],
            'view_more_link': None,
            'pagination_info': {
                'has_pagination': False,
                'pagination_type': None
            },
            'sample_items': []
        }
        
        try:
            # Fetch browse links
            browse_links = self._get_browse_links(collection_url)
            structure_info['total_browse_links'] = len(browse_links)
            
            # Categorize browse links
            structure_info['browse_link_types'] = [
                urlparse(link).path.split('/')[-1] 
                for link in browse_links
            ]
            
            # Check for subcollections
            subcollections = self._get_subcollections_from_page(collection_url)
            structure_info['subcollections'] = [
                {'url': sub, 'text': urlparse(sub).path.split('/')[-1]} 
                for sub in subcollections
            ]
            
            # Check for view more link
            view_more_link = self._get_view_more_link(collection_url)
            if view_more_link:
                structure_info['view_more_link'] = view_more_link
            
            # Check pagination
            first_browse_link = browse_links[0] if browse_links else collection_url
            next_page_link = self._get_pagination_links(first_browse_link)
            if next_page_link:
                structure_info['pagination_info'] = {
                    'has_pagination': True,
                    'pagination_type': 'offset' if 'offset' in next_page_link else 'numbered'
                }
            
            # Get a small sample of items from the main page
            sample_items = self._extract_collection_items(collection_url)[:5]
            structure_info['sample_items'] = [
                {
                    'title': item.get('title', 'Unknown'),
                    'type': item.get('type', 'Unknown'),
                    'date': item.get('date_issue', 'Unknown')
                } 
                for item in sample_items
            ]
            
            # Log the exploration results
            logger.info(f"Collection Structure Exploration for {collection_name}:")
            logger.info(f"Total Browse Links: {structure_info['total_browse_links']}")
            logger.info(f"Browse Link Types: {structure_info['browse_link_types']}")
            logger.info(f"Subcollections: {len(structure_info['subcollections'])}")
            logger.info(f"View More Link: {'Yes' if view_more_link else 'No'}")
            logger.info(f"Pagination: {structure_info['pagination_info']}")
            logger.info(f"Sample Items: {len(structure_info['sample_items'])}")
            
            return structure_info
        
        except Exception as e:
            logger.error(f"Error exploring collection structure for {collection_name}: {e}")
            return structure_info

    

    def fetch_all_collections(self, explore_first=True):
        """
        Fetch items from all collections with optional structure exploration
        
        Args:
            explore_first (bool): Whether to first explore the collection structure
        
        Returns:
            dict: Collected resources from all collections
        """
        all_collections = {}
        collection_structures = {}
        
        # First, explore each collection's structure
        if explore_first:
            for collection_name, collection_url in self.endpoints.items():
                logger.info(f"Exploring structure of {collection_name} collection")
                collection_structure = self.explore_collection_structure(collection_url, collection_name)
                collection_structures[collection_name] = collection_structure
        
        # Now fetch items from each collection
        for collection_name, collection_url in self.endpoints.items():
            logger.info(f"Fetching all items from {collection_name} collection")
            
            # If we have explored the structure, we can make more informed decisions
            if explore_first and collection_name in collection_structures:
                structure_info = collection_structures[collection_name]
                
                # You could add custom logic here based on structure_info
                # For example, skip certain collections or modify retrieval strategy
                if structure_info['total_browse_links'] == 0:
                    logger.warning(f"No browse links found for {collection_name}. Skipping full retrieval.")
                    continue
            
            # Extract items from the collection including all browse pages
            collection_items = self._extract_all_collection_items(collection_url, collection_name)
            
            # Store collected items
            all_collections[collection_name] = collection_items
            
            logger.info(f"Found {len(collection_items)} total items in {collection_name}")
        
        return all_collections, collection_structures

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
                        # Store URLs in the identifiers JSON field since urls column doesn't exist
                        identifiers = resource.get('identifiers', {}).copy()
                        if resource.get('urls'):
                            identifiers['download_urls'] = resource.get('urls')
                        
                        values = (
                            unique_id,  # Use generated unique integer ID
                            resource.get('identifiers', {}).get('doi'),
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
                            json.dumps(identifiers) if identifiers else None,
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
                            identifiers = COALESCE(EXCLUDED.identifiers, resources_resource.identifiers)
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
        all_content, collection_structures = scraper.fetch_all_collections(explore_first=True)
        
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
        
        # Optionally print collection structures
        for collection_type, structure in collection_structures.items():
            logger.info(f"Structure of {collection_type.capitalize()} collection:")
            logger.info(json.dumps(structure, indent=2))
    
    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")
        import traceback
        traceback.print_exc()