import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Set, Union, Any
import json
import hashlib
from datetime import datetime, timedelta
import re
import time
from urllib.parse import urljoin
import concurrent.futures
import threading
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import diskcache
import psycopg2
from dataclasses import dataclass, field, asdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.text_processor import safe_str, truncate_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a metrics collector
class ScraperMetrics:
    def __init__(self):
        self.started_at = datetime.utcnow()
        self.request_count = 0
        self.request_success = 0
        self.request_failures = 0
        self.items_processed = 0
        self.items_successful = 0
        self.items_failed = 0
        self.processing_times = []
        self.error_types = {}
        self._lock = threading.Lock()
    
    def record_request(self, success: bool, url: str = None, error: Exception = None):
        with self._lock:
            self.request_count += 1
            if success:
                self.request_success += 1
            else:
                self.request_failures += 1
                if error:
                    error_type = type(error).__name__
                    self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def record_item_processed(self, success: bool, processing_time: float, error: Exception = None):
        with self._lock:
            self.items_processed += 1
            if success:
                self.items_successful += 1
            else:
                self.items_failed += 1
                if error:
                    error_type = type(error).__name__
                    self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
            self.processing_times.append(processing_time)
    
    def get_summary(self) -> Dict:
        with self._lock:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            return {
                'duration': (datetime.utcnow() - self.started_at).total_seconds(),
                'requests': {
                    'total': self.request_count,
                    'success': self.request_success,
                    'failure': self.request_failures,
                    'success_rate': (self.request_success / self.request_count * 100) if self.request_count else 0
                },
                'items': {
                    'total': self.items_processed,
                    'success': self.items_successful,
                    'failure': self.items_failed,
                    'success_rate': (self.items_successful / self.items_processed * 100) if self.items_processed else 0
                },
                'processing_time': {
                    'average': avg_processing_time,
                    'min': min(self.processing_times) if self.processing_times else 0,
                    'max': max(self.processing_times) if self.processing_times else 0
                },
                'errors': self.error_types
            }

# Adaptive rate limiter
class AdaptiveRateLimiter:
    def __init__(self, initial_delay: float = 1.0, min_delay: float = 0.5, max_delay: float = 5.0, backoff_factor: float = 1.5):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.success_count = 0
        self.failure_count = 0
        self._lock = threading.Lock()
        self.last_request_time = 0

    def wait(self):
        """Wait the appropriate amount of time before the next request."""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            # If we need to wait, do so
            if elapsed < self.current_delay:
                time.sleep(self.current_delay - elapsed)
            
            # Update the last request time
            self.last_request_time = time.time()

    def record_success(self):
        """Record a successful request and potentially decrease delay."""
        with self._lock:
            self.success_count += 1
            self.failure_count = 0
            
            # After 5 consecutive successes, we reduce the delay
            if self.success_count >= 5:
                self.current_delay = max(self.current_delay / 1.2, self.min_delay)
                self.success_count = 0

    def record_failure(self):
        """Record a failed request and increase the delay."""
        with self._lock:
            self.failure_count += 1
            self.success_count = 0
            
            # Increase the delay more aggressively with consecutive failures
            self.current_delay = min(self.current_delay * (self.backoff_factor ** self.failure_count), self.max_delay)

@dataclass
class ScraperConfig:
    """Configuration for the KnowhubScraper."""
    base_url: str = os.getenv('KNOWHUB_BASE_URL', 'https://knowhub.aphrc.org')
    cache_dir: str = os.getenv('KNOWHUB_CACHE_DIR', '/tmp/knowhub_cache')
    cache_ttl: int = int(os.getenv('KNOWHUB_CACHE_TTL', '86400'))  # 24 hours in seconds
    request_timeout: int = int(os.getenv('KNOWHUB_REQUEST_TIMEOUT', ''))
    max_workers: int = int(os.getenv('KNOWHUB_MAX_WORKERS', '4'))
    max_retries: int = int(os.getenv('KNOWHUB_MAX_RETRIES', '6'))
    initial_rate_limit_delay: float = float(os.getenv('KNOWHUB_INITIAL_RATE_LIMIT', '1.0'))
    batch_size: int = int(os.getenv('KNOWHUB_BATCH_SIZE', '20'))
    html_selectors: Dict[str, List[str]] = field(default_factory=lambda: {
        'item': [
            ['div', 'article'], 
            ['ds-artifact-item', 'item-wrapper', 'row artifact-description']
        ],
        'title': [
            ['h4', 'h3', 'h2'], 
            ['artifact-title', 'item-title']
        ],
        'metadata': [
            ['div'], 
            ['item-metadata', 'artifact-info']
        ]
    })

    def to_dict(self) -> Dict:
        return asdict(self)


class KnowhubScraper:
    # Add to the KnowhubScraper class, joining with the methods you already have

# Modify the __init__ method to enhance logging and debug info
    def __init__(self, summarizer: Optional[TextSummarizer] = None, config: Optional[ScraperConfig] = None,
                db_connection_string: Optional[str] = None):
        """Initialize KnowhubScraper with enhanced capabilities."""
        # Initialize configuration
        self.config = config or ScraperConfig()
        
        # Get database connection string from environment if not provided
        self.db_connection_string = db_connection_string or os.getenv(
            'KNOWHUB_DB_CONNECTION_STRING', 
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        )
        
        # Base URLs and endpoints
        self.base_url = self.config.base_url
        self.publications_url = f"{self.base_url}/handle/123456789/1"
        
        # Update endpoints to match exact type names
        self.endpoints = {
            'publications': f"{self.base_url}/handle/123456789/1",
            'documents': f"{self.base_url}/handle/123456789/2",
            'reports': f"{self.base_url}/handle/123456789/3",
            'multimedia': f"{self.base_url}/handle/123456789/4"
        }
        
        # Initialize HTTP session with retry capabilities
        self.session = self._create_session()
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # Initialize rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            initial_delay=self.config.initial_rate_limit_delay,
            min_delay=0.5,
            max_delay=5.0,
            backoff_factor=1.5
        )
        
        # Initialize cache
        self.cache = diskcache.Cache(self.config.cache_dir)
        
        # Initialize summarizer (with lazy loading)
        self._summarizer = summarizer
        
        # Track seen publications
        self.seen_handles = set()
        
        # Initialize metrics collector
        self.metrics = ScraperMetrics()
        
        # Create thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info("KnowhubScraper initialized with configuration: %s", self.config.to_dict())
        logger.info(f"Using publications URL: {self.publications_url}")
        logger.info(f"Additional endpoints: {', '.join(self.endpoints.keys())}")
        logger.info(f"Database connection string: {self.db_connection_string[:20]}...")
        logger.info(f"Using hierarchical collection scraping approach")

    # Update the main method for the script
    
    # Modified wrapper method to use the hierarchical collection 
    # approach
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry capabilities."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    @property
    def summarizer(self) -> TextSummarizer:
        """Lazy-load summarizer when needed."""
        if self._summarizer is None:
            logger.info("Lazy-loading TextSummarizer")
            self._summarizer = TextSummarizer()
        return self._summarizer

    
    def fetch_all_content(self, limit: int = None, auto_save: bool = False) -> Dict[str, List[Dict]]:
        """
        Main method to fetch all content - redirects to hierarchical collection approach.
        Maintained for compatibility with existing code.
        """
        logger.info(f"fetch_all_content called, redirecting to hierarchical collection approach")
        return self.fetch_all_collections()

    # Helper method to normalize strings consistently
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace, etc."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # Remove unwanted characters
        text = re.sub(r'[\r\n\t]', ' ', text)
        
        return text
        
    # Improved normalize_author_name method
    def _normalize_author_name(self, author_name: str) -> str:
        """Normalize author names to a consistent format."""
        if not author_name:
            return ""
        
        # Basic normalization
        author_name = self._normalize_text(author_name)
        
        # Remove trailing commas
        author_name = re.sub(r',\s*$', '', author_name)
        
        # Fix capitalization (handling special cases like "McName")
        parts = re.split(r'[\s,;-]+', author_name)
        normalized_parts = []
        
        for part in parts:
            if not part:
                continue
                
            # Handle special cases like "McDowell"
            if part.lower().startswith("mc") and len(part) > 2:
                normalized_part = "Mc" + part[2].upper() + part[3:].lower()
            else:
                normalized_part = part.capitalize()
                
            normalized_parts.append(normalized_part)
        
        # Rejoin with spaces
        return " ".join(normalized_parts)

    # Improved normalize_keyword method
    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keywords."""
        if not keyword:
            return ""
        
        # Basic normalization
        keyword = self._normalize_text(keyword)
        
        # Convert to lowercase
        keyword = keyword.lower()
        
        # Remove trailing punctuation
        keyword = re.sub(r'[;,.\s]+$', '', keyword)
        
        return keyword

    def fetch_all_collections(self) -> Dict[str, List[Dict]]:
        """
        Fetch content by traversing the hierarchical collection structure instead of using pagination.
        This is the key method to use instead of fetch_all_content for KnowHub.
        """
        all_content = {}
        
        logger.info("Fetching ALL content from all endpoints using collection structure")
        
        # Cache key for the operation
        cache_key = f"all_collections_{datetime.utcnow().strftime('%Y%m%d')}"
        
        # Try to get from cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Retrieved all collection content from cache")
            return cached_data
        
        # First fetch the main collections from each endpoint
        for endpoint_name, url in self.endpoints.items():
            logger.info(f"Fetching content from {endpoint_name} endpoint: {url}")
            
            # Fetch top-level subcollections
            subcollections = self._fetch_subcollections(url)
            if subcollections:
                logger.info(f"Found {len(subcollections)} subcollections for {endpoint_name}")
                
                # Process each subcollection
                endpoint_content = []
                
                # Process each subcollection in parallel
                future_to_subcollection = {
                    self.executor.submit(self._process_subcollection, sub_url, sub_name): sub_name
                    for sub_name, sub_url in subcollections.items()
                }
                
                for future in concurrent.futures.as_completed(future_to_subcollection):
                    sub_name = future_to_subcollection[future]
                    try:
                        items = future.result()
                        if items:
                            logger.info(f"Added {len(items)} items from subcollection '{sub_name}'")
                            endpoint_content.extend(items)
                    except Exception as e:
                        logger.error(f"Error processing subcollection '{sub_name}': {e}")
                
                # Direct items at this level
                direct_items = self._extract_direct_items(url)
                if direct_items:
                    logger.info(f"Found {len(direct_items)} direct items at {endpoint_name} level")
                    endpoint_content.extend(direct_items)
                
                all_content[endpoint_name] = endpoint_content
                logger.info(f"Total {len(endpoint_content)} items collected for {endpoint_name}")
            else:
                # If no subcollections, try to extract items directly
                direct_items = self._extract_direct_items(url)
                if direct_items:
                    logger.info(f"Found {len(direct_items)} direct items for {endpoint_name}")
                    all_content[endpoint_name] = direct_items
                else:
                    logger.warning(f"No content found for {endpoint_name}")
                    all_content[endpoint_name] = []
        
        # Add "Recent Submissions" for each endpoint, which can contain items not in subcollections
        for endpoint_name, endpoint_url in self.endpoints.items():
            recent_submissions_url = f"{endpoint_url}/recent-submissions"
            logger.info(f"Checking recent submissions for {endpoint_name}: {recent_submissions_url}")
            
            recent_items = self._extract_direct_items(recent_submissions_url)
            if recent_items:
                logger.info(f"Found {len(recent_items)} recent items for {endpoint_name}")
                # Add only new items (not already in the collection)
                existing_handles = {item.get('identifiers', {}).get('handle') for item in all_content.get(endpoint_name, [])}
                new_items = [item for item in recent_items 
                            if item.get('identifiers', {}).get('handle') not in existing_handles]
                
                if new_items:
                    logger.info(f"Adding {len(new_items)} new recent items to {endpoint_name}")
                    if endpoint_name not in all_content:
                        all_content[endpoint_name] = []
                    all_content[endpoint_name].extend(new_items)
        
        # Check "By Issue Date" browse pages for each endpoint to ensure we get all items
        for endpoint_name, endpoint_url in self.endpoints.items():
            date_browse_url = f"{endpoint_url}/browse?type=dateissued"
            logger.info(f"Checking date browsing for {endpoint_name}: {date_browse_url}")
            
            date_items = self._extract_items_from_browse(date_browse_url)
            if date_items:
                logger.info(f"Found {len(date_items)} items by date for {endpoint_name}")
                # Add only new items
                existing_handles = {item.get('identifiers', {}).get('handle') for item in all_content.get(endpoint_name, [])}
                new_items = [item for item in date_items 
                            if item.get('identifiers', {}).get('handle') not in existing_handles]
                
                if new_items:
                    logger.info(f"Adding {len(new_items)} new date-browsed items to {endpoint_name}")
                    if endpoint_name not in all_content:
                        all_content[endpoint_name] = []
                    all_content[endpoint_name].extend(new_items)
        
        # Save to cache
        self.cache.set(cache_key, all_content, expire=self.config.cache_ttl)
        
        # Calculate and log total items
        total_items = sum(len(items) for items in all_content.values())
        logger.info(f"Completed fetching all content from collections. Total items: {total_items}")
        
        return all_content

    def _fetch_subcollections(self, url: str) -> Dict[str, str]:
        """Extract subcollection URLs from a collection page."""
        html_content = self._fetch_page(url)
        if not html_content:
            return {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        subcollections = {}
        
        # Look for subcommunities section
        subcommunity_headers = soup.find_all(string=re.compile('Sub-communities', re.IGNORECASE))
        for header in subcommunity_headers:
            section = header.find_parent(['div', 'section'])
            if section:
                # Find links with counts in brackets [X]
                for link in section.find_all('a', href=True):
                    text = link.get_text().strip()
                    if text and re.search(r'\[\d+\]$', text):
                        name = re.sub(r'\[\d+\]$', '', text).strip()
                        subcollections[name] = urljoin(self.base_url, link['href'])
        
        # Look for collections section
        collection_headers = soup.find_all(string=re.compile('Collections', re.IGNORECASE))
        for header in collection_headers:
            section = header.find_parent(['div', 'section'])
            if section:
                for link in section.find_all('a', href=True):
                    text = link.get_text().strip()
                    if text:
                        # Extract name and count if available
                        if re.search(r'\[\d+\]$', text):
                            name = re.sub(r'\[\d+\]$', '', text).strip()
                        else:
                            name = text
                        subcollections[name] = urljoin(self.base_url, link['href'])
        
        # If none found, look for any links that might be collections
        if not subcollections:
            # Look for links to handles that might be collections
            handle_links = soup.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
            for link in handle_links:
                text = link.get_text().strip()
                if text and not text.lower() in ['home', 'communities & collections', 'by issue date', 'authors', 'titles', 'subjects']:
                    subcollections[text] = urljoin(self.base_url, link['href'])
        
        return subcollections

    def _process_subcollection(self, url: str, name: str) -> List[Dict]:
        """Process all items in a subcollection."""
        logger.info(f"Processing subcollection: {name} at {url}")
        
        # First check if there are further subcollections
        sub_subcollections = self._fetch_subcollections(url)
        
        all_items = []
        
        # If there are sub-subcollections, process them
        if sub_subcollections:
            logger.info(f"Found {len(sub_subcollections)} sub-subcollections in '{name}'")
            for sub_name, sub_url in sub_subcollections.items():
                try:
                    items = self._process_subcollection(sub_url, f"{name} > {sub_name}")
                    all_items.extend(items)
                except Exception as e:
                    logger.error(f"Error processing sub-subcollection '{sub_name}': {e}")
        
        # Extract direct items from this collection
        direct_items = self._extract_direct_items(url)
        if direct_items:
            logger.info(f"Found {len(direct_items)} direct items in subcollection '{name}'")
            all_items.extend(direct_items)
        
        # Check for browse by date to ensure we get all items
        date_browse_url = f"{url}/browse?type=dateissued"
        date_items = self._extract_items_from_browse(date_browse_url)
        if date_items:
            # Add only new items
            existing_handles = {item.get('identifiers', {}).get('handle') for item in all_items}
            new_items = [item for item in date_items 
                        if item.get('identifiers', {}).get('handle') not in existing_handles]
            
            if new_items:
                logger.info(f"Adding {len(new_items)} new items from date browse in '{name}'")
                all_items.extend(new_items)
        
        logger.info(f"Completed subcollection '{name}' with {len(all_items)} total items")
        return all_items

    def _extract_direct_items(self, url: str) -> List[Dict]:
        """Extract items directly from a collection page."""
        html_content = self._fetch_page(url)
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        items = []
        
        # Find all item divs in the page
        item_divs = soup.find_all('div', class_=lambda c: c and any(x in c for x in ['ds-artifact-item', 'item']))
        
        if not item_divs:
            # Try other common patterns in DSpace sites
            item_divs = soup.find_all(['div', 'li'], class_=lambda c: c and any(x in c for x in ['artifact-description', 'ds-item']))
        
        logger.info(f"Found {len(item_divs)} potential item divs on page {url}")
        
        # Process each item div
        for item_div in item_divs:
            try:
                # Extract the title and link
                title_elem = item_div.find(['h4', 'h3', 'h2', 'a'], class_=lambda c: c and 'title' in c)
                if not title_elem:
                    title_elem = item_div.find(['h4', 'h3', 'h2'])
                
                if not title_elem:
                    continue
                    
                title_link = title_elem if title_elem.name == 'a' else title_elem.find('a')
                if not title_link or not title_link.get('href'):
                    continue
                    
                # Extract handle from link
                href = title_link.get('href')
                handle_match = re.search(r'/handle/([0-9/]+)', href)
                if not handle_match:
                    continue
                    
                handle = handle_match.group(1)
                
                # Skip if we've already seen this handle (deduplication)
                if handle in self.seen_handles:
                    continue
                    
                # Mark as seen
                self.seen_handles.add(handle)
                
                # Process the full item by parsing its metadata
                item_url = urljoin(self.base_url, href)
                item_content = self._fetch_page(item_url)
                if item_content:
                    item_soup = BeautifulSoup(item_content, 'html.parser')
                    processed_item = self._parse_item_page(item_soup, item_url, handle)
                    if processed_item:
                        items.append(processed_item)
            except Exception as e:
                logger.error(f"Error processing item div: {e}")
        
        return items

    def _extract_items_from_browse(self, url: str) -> List[Dict]:
        """Extract items from a browse page (e.g., by date, author, etc.)."""
        html_content = self._fetch_page(url)
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        items = []
        
        # Find all links to items
        item_links = soup.find_all('a', href=re.compile(r'/handle/123456789/\d+$'))
        logger.info(f"Found {len(item_links)} potential item links on browse page {url}")
        
        # Process each item link
        for link in item_links:
            try:
                href = link.get('href')
                if not href:
                    continue
                    
                # Extract handle from link
                handle_match = re.search(r'/handle/([0-9/]+)', href)
                if not handle_match:
                    continue
                    
                handle = handle_match.group(1)
                
                # Skip if we've already seen this handle (deduplication)
                if handle in self.seen_handles:
                    continue
                    
                # Mark as seen
                self.seen_handles.add(handle)
                
                # Process the full item by parsing its metadata
                item_url = urljoin(self.base_url, href)
                item_content = self._fetch_page(item_url)
                if item_content:
                    item_soup = BeautifulSoup(item_content, 'html.parser')
                    processed_item = self._parse_item_page(item_soup, item_url, handle)
                    if processed_item:
                        items.append(processed_item)
            except Exception as e:
                logger.error(f"Error processing item link: {e}")
        
        # Check for pagination in browse pages
        pagination_div = soup.find('div', class_='pagination')
        if pagination_div:
            next_link = pagination_div.find('a', text=re.compile(r'next|›|»', re.IGNORECASE))
            if next_link and next_link.get('href'):
                next_url = urljoin(self.base_url, next_link.get('href'))
                logger.info(f"Found next page in browse: {next_url}")
                # Recursively process next page
                next_items = self._extract_items_from_browse(next_url)
                items.extend(next_items)
        
        return items

    def _extract_title_element(self, element: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract title element with robust selector handling."""
        if not isinstance(element, BeautifulSoup) and not hasattr(element, 'find'):
            return None
            
        # Get selectors from config
        tag_types, class_names = self.config.html_selectors['title']
        
        # Try primary method
        title_elem = element.find(tag_types, class_=class_names)
        if title_elem:
            return title_elem
            
        # Try individual selectors
        for tag in tag_types:
            for class_name in class_names:
                title_elem = element.find(tag, class_=class_name)
                if title_elem:
                    return title_elem
        
        # Try finding any link with title class
        title_elem = element.find('a', class_='item-title')
        if title_elem:
            return title_elem
            
        # Last resort: find any heading tag
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5']:
            title_elem = element.find(tag)
            if title_elem:
                return title_elem
                
        return None

    def _extract_text_content(self, element: BeautifulSoup) -> str:
        """Safely extract text content from an element."""
        if not element:
            return ""
            
        if hasattr(element, 'get_text'):
            return element.get_text().strip()
        elif hasattr(element, 'text'):
            return element.text.strip()
        else:
            return str(element).strip()

    def _extract_url_and_handle(self, element: BeautifulSoup, title_elem: BeautifulSoup) -> Tuple[Optional[str], Optional[str], str]:
        """Extract URL and handle using multiple strategies."""
        url = None
        handle = None
        content_type = 'other'  # Default content type
        
        # Strategy 1: Try to find the link in the title element
        link = None
        if hasattr(title_elem, 'find'):
            link = title_elem.find('a')
        if not link and hasattr(title_elem, 'name') and title_elem.name == 'a':
            link = title_elem
            
        # Get URL and handle from link
        if link and hasattr(link, 'get'):
            href = link.get('href', '')
            if href:
                url = urljoin(self.base_url, href)
                handle_match = re.search(r'handle/([0-9/]+)', url)
                if handle_match:
                    handle = handle_match.group(1)
                    # Set content type based on handle path
                    if '123456789/1' in handle:
                        content_type = 'publications'
                    elif '123456789/2' in handle:
                        content_type = 'documents'
                    elif '123456789/3' in handle:
                        content_type = 'reports'
                    elif '123456789/4' in handle:
                        content_type = 'multimedia'
        
        # Strategy 2: If no URL found, try to find it elsewhere in the element
        if not url:
            # Try finding any link with href attribute
            url_elem = element.find('a', href=True)
            if url_elem:
                href = url_elem.get('href', '')
                if href:
                    url = urljoin(self.base_url, href)
                    handle_match = re.search(r'handle/([0-9/]+)', url)
                    if handle_match:
                        handle = handle_match.group(1)
                        # Set content type based on handle path
                        if '123456789/1' in handle:
                            content_type = 'publications'
                        elif '123456789/2' in handle:
                            content_type = 'documents'
                        elif '123456789/3' in handle:
                            content_type = 'reports'
                        elif '123456789/4' in handle:
                            content_type = 'multimedia'
        
        # Strategy 3: Look for any link with "handle" in the URL
        if not url:
            handle_links = element.find_all('a', href=lambda h: h and 'handle' in h)
            if handle_links:
                href = handle_links[0].get('href', '')
                if href:
                    url = urljoin(self.base_url, href)
                    handle_match = re.search(r'handle/([0-9/]+)', url)
                    if handle_match:
                        handle = handle_match.group(1)
                        # Determine content type
                        if '123456789/1' in handle:
                            content_type = 'publications'
                        elif '123456789/2' in handle:
                            content_type = 'documents'
                        elif '123456789/3' in handle:
                            content_type = 'reports'
                        elif '123456789/4' in handle:
                            content_type = 'multimedia'
        
        return url, handle, content_type

    def _extract_metadata(self, element: BeautifulSoup) -> Dict:
        """Extract metadata from publication element with improved error handling and normalization."""
        logger.debug("Extracting metadata fields...")
        metadata = {
            'authors': [],
            'keywords': [],
            'type': 'other',
            'date': None,
            'citation': None,
            'language': 'en',
            'abstract': ''
        }
        
        try:
            if not isinstance(element, BeautifulSoup) and not hasattr(element, 'find'):
                return metadata
                
            # Get metadata div selectors
            tag_types, class_names = self.config.html_selectors['metadata']
            
            # Try to find metadata div
            meta_div = None
            for tag in tag_types:
                for class_name in class_names:
                    meta_div = element.find(tag, class_=class_name)
                    if meta_div:
                        break
                if meta_div:
                    break
            
            if not meta_div:
                # Try more generic approach
                meta_div = element.find('div', class_=lambda c: c and any(x in c for x in ['metadata', 'info']))
                
            if not meta_div:
                return metadata
            
            # Extract authors with normalization
            author_elems = meta_div.find_all('span', class_=['author', 'creator'])
            authors_raw = [
                self._extract_text_content(author)
                for author in author_elems
                if author
            ]
            metadata['authors'] = [self._normalize_author_name(author) for author in authors_raw if author]
            
            # Extract date
            date_elem = meta_div.find('span', class_=['date', 'issued'])
            if date_elem:
                date_str = self._extract_text_content(date_elem)
                metadata['date'] = self._parse_date(date_str)
            
            # Extract type
            type_elem = meta_div.find('span', class_=['type', 'resourcetype'])
            if type_elem:
                type_str = self._extract_text_content(type_elem)
                metadata['type'] = self._normalize_publication_type(type_str)
            
            # Extract DOI if available (for potential future use)
            doi_elem = meta_div.find('span', class_='doi')
            if doi_elem:
                doi_text = self._extract_text_content(doi_elem)
                doi_match = re.search(r'10\.\d{4,}/\S+', doi_text)
                if doi_match:
                    metadata['doi'] = doi_match.group(0)
            
            # Extract keywords with normalization
            keyword_elems = meta_div.find_all('span', class_=['subject', 'keyword'])
            keywords_raw = [
                self._extract_text_content(kw)
                for kw in keyword_elems
                if kw
            ]
            metadata['keywords'] = [self._normalize_keyword(kw) for kw in keywords_raw if kw]
            
            # Extract abstract
            abstract_elem = meta_div.find('span', class_=['abstract', 'description'])
            if abstract_elem:
                metadata['abstract'] = safe_str(self._extract_text_content(abstract_elem))
            
            # Extract citation if available
            citation_elem = meta_div.find('span', class_=['citation', 'reference'])
            if citation_elem:
                metadata['citation'] = safe_str(self._extract_text_content(citation_elem))
            
            # Extract language if available
            lang_elem = meta_div.find('span', class_='language')
            if lang_elem:
                metadata['language'] = self._normalize_language(self._extract_text_content(lang_elem))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return metadata

    def _normalize_author_name(self, author_name: str) -> str:
        """Normalize author names to a consistent format."""
        if not author_name:
            return ""
            
        # Trim whitespace
        author_name = author_name.strip()
        
        # Remove trailing commas
        author_name = re.sub(r',\s*,', '', author_name)
        
        # Fix capitalization (handling special cases like "McName")
        parts = re.split(r'[\s,;-]+', author_name)
        normalized_parts = []
        
        for part in parts:
            if not part:
                continue
                
            # Handle special cases like "McDowell"
            if part.lower().startswith("mc") and len(part) > 2:
                normalized_part = "Mc" + part[2].upper() + part[3:].lower()
            else:
                normalized_part = part.capitalize()
                
            normalized_parts.append(normalized_part)
        
        # Rejoin with spaces
        return " ".join(normalized_parts)

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keywords."""
        if not keyword:
            return ""
            
        # Trim and lowercase
        keyword = keyword.strip().lower()
        
        # Remove trailing punctuation
        keyword = re.sub(r'[;,.\s]+$', '', keyword)
        
        return keyword
    def _normalize_language(self, language: str) -> str:
        """Normalize language codes."""
        if not language:
            return "en"
            
        language = language.strip().lower()
        
        # Map full language names to ISO codes
        language_map = {
            "english": "en",
            "french": "fr",
            "spanish": "es",
            "german": "de",
            "chinese": "zh",
            "japanese": "ja",
            "russian": "ru",
            "arabic": "ar",
            "portuguese": "pt",
            "italian": "it",
            "korean": "ko",
            "dutch": "nl",
            "swedish": "sv",
            "finnish": "fi",
            "danish": "da",
            "norwegian": "no"
        }
        
        # If it's a full language name, convert to code
        if language in language_map:
            return language_map[language]
            
        # If it's already a 2-letter code, return it
        if len(language) == 2:
            return language
            
        # Default to English
        return "en"

    def _normalize_publication_type(self, type_str: str) -> str:
        """Normalize publication type strings."""
        if not type_str:
            return "other"
            
        type_str = type_str.lower().strip()
        
        type_mapping = {
            'article': 'journal_article',
            'journal article': 'journal_article',
            'research article': 'journal_article',
            'review': 'review_article',
            'book': 'book',
            'book chapter': 'book_chapter',
            'conference': 'conference_paper',
            'conference paper': 'conference_paper',
            'proceedings': 'conference_proceedings',
            'report': 'report',
            'technical report': 'technical_report',
            'working paper': 'working_paper',
            'thesis': 'thesis',
            'dissertation': 'dissertation',
            'policy brief': 'policy_brief',
            'policy report': 'policy_brief',
            'data': 'dataset',
            'dataset': 'dataset',
            'presentation': 'presentation',
            'video': 'multimedia',
            'audio': 'multimedia',
            'image': 'multimedia',
            'poster': 'poster',
            'lecture': 'lecture',
            'interview': 'interview',
            'case study': 'case_study',
            'white paper': 'white_paper',
            'guideline': 'guideline',
            'protocol': 'protocol',
            'standard': 'standard',
            'survey': 'survey',
            'questionnaire': 'survey',
            'factsheet': 'factsheet',
            'manual': 'manual'
        }
        
        return type_mapping.get(type_str, 'other')

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string into ISO format with expanded patterns."""
        if not date_str:
            return None
            
        try:
            date_str = date_str.strip()
            
            # Try common DSpace date formats
            formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y',
                '%b %d, %Y',
                '%d %b %Y',
                '%Y-%m',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%d-%m-%Y',
                '%m-%d-%Y',
                '%d.%m.%Y',
                '%m.%d.%Y',
                '%Y.%m.%d'
            ]
            
            for fmt in formats:
                try:
                    date = datetime.strptime(date_str, fmt)
                    return date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # Try to extract year if full date parsing fails
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return f"{year_match.group(0)}-01-01"
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {e}")
            return None

    def _generate_summary(self, title: str, abstract: str) -> str:
        """Generate a summary using the TextSummarizer with caching."""
        try:
            # Create a cache key based on title and abstract
            cache_key = f"summary_{hashlib.md5((title + abstract).encode()).hexdigest()}"
            
            # Try to get from cache
            cached_summary = self.cache.get(cache_key)
            if cached_summary:
                logger.debug("Retrieved summary from cache")
                return cached_summary
            
            # Clean and truncate inputs
            title = truncate_text(title, max_length=200)
            abstract = truncate_text(abstract, max_length=1000)
            
            # Generate summary
            try:
                summary = self.summarizer.summarize(title, abstract)
                summary = truncate_text(summary, max_length=500)
                
                # Cache the result
                self.cache.set(cache_key, summary, expire=self.config.cache_ttl)
                
                return summary
            except Exception as e:
                logger.error(f"Summary generation error: {e}")
                fallback = abstract if abstract else f"Publication about {title}"
                
                # Cache even the fallback to avoid repeated failures
                self.cache.set(cache_key, fallback, expire=self.config.cache_ttl)
                
                return fallback
                
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return title

    def batch_save_to_database(self, publications: List[Dict], db_connection_string: Optional[str] = None) -> int:
        """Save multiple publications to database in efficient batches."""
        if not publications:
            logger.warning("No publications to save")
            return 0
            
        # Use provided connection string or fall back to instance's
        conn_string = db_connection_string or self.db_connection_string
        if not conn_string:
            logger.error("No database connection string provided")
            return 0
            
        logger.info(f"Saving {len(publications)} publications to database in batches")
        
        batch_size = self.config.batch_size
        successful = 0
        conn = None
        cursor = None
        
        try:
            # Connect to database
            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()
            
            # Process in batches
            for i in range(0, len(publications), batch_size):
                batch = publications[i:i+batch_size]
                
                try:
                    # Begin transaction for this batch
                    cursor.execute("BEGIN;")
                    
                    for pub in batch:
                        # Check if publication already exists
                        cursor.execute(
                            "SELECT id FROM resources_resource WHERE title = %s OR doi = %s",
                            (pub['title'], pub.get('doi'))
                        )
                        exists = cursor.fetchone()
                        
                        if not exists:
                            # Prepare data
                            values = (
                                pub['title'],
                                pub.get('doi'),
                                pub.get('abstract'),
                                pub.get('summary'),
                                pub.get('authors', []),
                                pub.get('description', ''),
                                pub.get('type', 'other'),
                                pub.get('source', 'knowhub'),
                                pub.get('date_issue'),
                                pub.get('citation'),
                                pub.get('language', 'en'),
                                pub.get('identifiers', '{}'),
                                pub.get('collection', 'knowhub'),
                                pub.get('publishers', '{}'),
                                pub.get('subtitles', '{}')
                            )
                            
                            # Insert publication
                            cursor.execute("""
                                INSERT INTO resources_resource
                                (title, doi, abstract, summary, authors, description, type, source, 
                                 date_issue, citation, language, identifiers, collection, publishers, subtitles)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, values)
                            
                            pub_id = cursor.fetchone()[0]
                            
                            # Insert tags if they exist
                            if 'tags' in pub:
                                for tag in pub['tags']:
                                    tag_values = (
                                        tag['name'],
                                        tag['tag_type'],
                                        tag.get('additional_metadata', '{}'),
                                        pub_id
                                    )
                                    cursor.execute("""
                                        INSERT INTO resources_tag
                                        (name, tag_type, additional_metadata, resource_id)
                                        VALUES (%s, %s, %s, %s)
                                    """, tag_values)
                            
                            successful += 1
                    
                    # Commit the batch
                    conn.commit()
                    logger.info(f"Saved batch of {len(batch)} publications (total successful: {successful})")
                    
                except Exception as e:
                    # Rollback this batch on error
                    conn.rollback()
                    logger.error(f"Error saving batch: {e}")
            
            return successful
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return successful
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            logger.info(f"Database operation completed, saved {successful} publications")

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics.get_summary()

    def close(self):
        """Close resources and perform cleanup."""
        try:
            # Close thread pool
            self.executor.shutdown(wait=True)
            
            # Close cache
            self.cache.close()
            
            # Close session
            self.session.close()
            
            # Close summarizer if we own it
            if self._summarizer and hasattr(self._summarizer, 'close'):
                self._summarizer.close()
            
            # Clear collections
            self.seen_handles.clear()
            
            logger.info("KnowhubScraper resources cleaned up")
            logger.info("Final metrics: %s", self.get_metrics())
            
        except Exception as e:
            logger.error(f"Error closing KnowhubScraper: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method to ensure cleanup."""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        # Re-raise any exception that occurred
        return False  # Propagate exceptions

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with rate limiting and metrics."""
        # Wait according to rate limiting strategy
        self.rate_limiter.wait()
        
        try:
            logger.debug(f"Fetching page: {url}")
            
            # Check if it's in the cache
            cache_key = f"page_{hashlib.md5(url.encode()).hexdigest()}"
            cached_page = self.cache.get(cache_key)
            if cached_page:
                logger.debug(f"Retrieved page from cache: {url}")
                return cached_page
            
            # Make the request
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=self.config.request_timeout,
                verify=False
            )
            
            # Record metrics
            self.metrics.record_request(response.status_code == 200, url)
            
            # Update rate limiter
            if response.status_code == 200:
                self.rate_limiter.record_success()
            else:
                self.rate_limiter.record_failure()
                
            # Process response
            if response.status_code == 200:
                # Cache the content
                self.cache.set(cache_key, response.text, expire=self.config.cache_ttl)
                return response.text
            else:
                logger.error(f"Failed to fetch page: {url}, status: {response.status_code}")
                return None
                
        except Exception as e:
            self.metrics.record_request(False, url, error=e)
            self.rate_limiter.record_failure()
            logger.error(f"Error fetching page {url}: {e}", exc_info=True)
            return None

    @retry(
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout, requests.ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    def _make_request(self, url: str, method: str = 'get', **kwargs) -> requests.Response:
        """Make an HTTP request with automatic retries and rate limiting."""
        # Wait according to rate limiting strategy
        self.rate_limiter.wait()
        
        try:
            logger.debug(f"Making {method.upper()} request to: {url}")
            kwargs['headers'] = {**self.headers, **kwargs.get('headers', {})}
            kwargs['verify'] = False  # Disable SSL verification
            kwargs['timeout'] = kwargs.get('timeout', self.config.request_timeout)
            
            response = self.session.request(method, url, **kwargs)
            
            # Record success metrics
            self.metrics.record_request(response.status_code == 200, url)
            
            # Update rate limiter based on response
            if response.status_code == 200:
                self.rate_limiter.record_success()
            else:
                self.rate_limiter.record_failure()
            
            response.raise_for_status()
            
            logger.debug(f"Request successful: {response.status_code}")
            return response


        except Exception as e:
            self.metrics.record_request(False, url, error=e)
            self.rate_limiter.record_failure()
            logger.error(f"Error making {method.upper()} request to {url}: {e}", exc_info=True)
            raise

    def _parse_item_page(self, soup: BeautifulSoup, url: str, handle: str) -> Optional[Dict]:
        """Parse a full item page to extract complete metadata."""
        try:
            # Extract title
            title_elem = soup.find(['h1', 'h2'], class_=lambda c: c and 'item-title' in c)
            if not title_elem:
                title_elem = soup.find(['h1', 'h2'])
            
            if not title_elem:
                return None
                
            title = title_elem.get_text().strip()
            if not title:
                return None
            
            # Extract type based on handle path
            content_type = 'other'
            if '123456789/1' in handle:
                content_type = 'publications'
            elif '123456789/2' in handle:
                content_type = 'documents'
            elif '123456789/3' in handle:
                content_type = 'reports'
            elif '123456789/4' in handle:
                content_type = 'multimedia'
            
            # Extract metadata fields
            metadata = {}
            
            # Find metadata section
            metadata_section = soup.find('div', class_=lambda c: c and any(x in c for x in ['item-page-field-wrapper', 'item-metadata']))
            if metadata_section:
                # Extract key-value pairs
                labels = metadata_section.find_all(['dt', 'label', 'span'], class_=lambda c: c and 'label' in c)
                for label in labels:
                    key = label.get_text().strip().lower().replace(':', '').replace(' ', '_')
                    # Find the corresponding value
                    value_elem = label.find_next(['dd', 'span', 'div'], class_=lambda c: c and ('value' in c or 'field-value' in c))
                    if value_elem:
                        value = value_elem.get_text().strip()
                        metadata[key] = value
            
            # Extract specific fields
            abstract = metadata.get('abstract', '') or metadata.get('description', '')
            authors = []
            if 'author' in metadata:
                author_text = metadata['author']
                authors = [a.strip() for a in author_text.split(';')]
            elif 'authors' in metadata:
                author_text = metadata['authors']
                authors = [a.strip() for a in author_text.split(';')]
            
            # Generate a summary
            summary = self._generate_summary(title, abstract)
            
            # Extract keywords
            keywords = []
            if 'keywords' in metadata:
                keyword_text = metadata['keywords']
                keywords = [k.strip() for k in keyword_text.split(';')]
            elif 'subject' in metadata:
                keyword_text = metadata['subject']
                keywords = [k.strip() for k in keyword_text.split(';')]
            
            # Create item record
            publication = {
                'title': title,
                'doi': url,  # Store the URL in the doi field
                'abstract': abstract or f"Publication about {title}",
                'summary': summary,
                'authors': authors,
                'description': abstract or f"Publication about {title}",
                'expert_id': None,
                'type': content_type,
                'subtitles': json.dumps({}),
                'publishers': json.dumps({
                    'name': 'APHRC',
                    'url': self.base_url,
                    'type': 'repository'
                }),
                'collection': 'knowhub',
                'date_issue': metadata.get('date_issued', None) or metadata.get('date', None),
                'citation': metadata.get('citation', None),
                'language': metadata.get('language', 'en'),
                'identifiers': json.dumps({
                    'handle': handle,
                    'source_id': f"knowhub-{handle.replace('/', '-')}",
                    'keywords': keywords,
                    'content_type': content_type
                }),
                'source': 'knowhub',
                'tags': [
                    {
                        'name': author,
                        'tag_type': 'author',
                        'additional_metadata': json.dumps({
                            'source': 'knowhub',
                            'affiliation': 'APHRC'
                        })
                    }
                    for author in authors
                ] + [
                    {
                        'name': keyword,
                        'tag_type': 'domain',
                        'additional_metadata': json.dumps({
                            'source': 'knowhub',
                            'type': 'keyword'
                        })
                    }
                    for keyword in keywords
                ] + [{
                    'name': content_type,
                    'tag_type': 'content_type',
                    'additional_metadata': json.dumps({
                        'source': 'knowhub',
                        'original_type': metadata.get('type', 'other')
                    })
                }]
            }
            
            return publication
        except Exception as e:
            logger.error(f"Error parsing item page: {e}")
            return None

    # Updated scrape_and_save_all to use the collection-based approach
    def scrape_and_save_all(self, limit_per_endpoint: int = None) -> Dict:
        """Scrape all content types using the hierarchical collection approach and save to database."""
        results = {}
        total_saved = 0
        
        try:
            # Log that we're scraping ALL content without limits
            logger.info("Scraping and saving ALL content using hierarchical collection structure")
            
            # Use the hierarchical collection fetching approach
            all_content = self.fetch_all_collections()
            
            # Save each content type to the database
            for content_type, items in all_content.items():
                if items:
                    logger.info(f"Saving {len(items)} {content_type} items to database")
                    saved = self.batch_save_to_database(items)
                    results[content_type] = {'fetched': len(items), 'saved': saved}
                    total_saved += saved
                else:
                    results[content_type] = {'fetched': 0, 'saved': 0}
                    
            # Add metrics and totals
            results['metrics'] = self.get_metrics()
            results['total_saved'] = total_saved
            
            logger.info(f"Hierarchical collection scraping and saving completed. Total saved: {total_saved} items")
            return results
                
        except Exception as e:
            logger.error(f"Error in scrape_and_save_all: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    
if __name__ == "__main__":
        import argparse
        
        parser = argparse.ArgumentParser(description='Run the KnowhubScraper with hierarchical collection traversal')
        parser.add_argument('--save', action='store_true', help='Save to database')
        parser.add_argument('--clear-cache', action='store_true', help='Clear the cache before starting')
        args = parser.parse_args()
        
        print(f"Starting KnowhubScraper with hierarchical collection traversal to fetch ALL content")
        
        with KnowhubScraper() as scraper:
            if args.clear_cache:
                print("Clearing cache...")
                scraper.cache.clear()
                
            if args.save:
                results = scraper.scrape_and_save_all()
                print(f"Scraping completed. Total saved: {results.get('total_saved', 0)}")
            else:
                all_content = scraper.fetch_all_collections()
                total_items = sum(len(items) for items in all_content.values())
                print(f"Scraping completed. Total items: {total_items}")
            
            print("Metrics:", scraper.get_metrics())