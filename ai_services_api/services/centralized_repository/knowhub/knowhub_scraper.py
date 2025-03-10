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
    request_timeout: int = int(os.getenv('KNOWHUB_REQUEST_TIMEOUT', '10'))
    max_workers: int = int(os.getenv('KNOWHUB_MAX_WORKERS', '4'))
    max_retries: int = int(os.getenv('KNOWHUB_MAX_RETRIES', '3'))
    initial_rate_limit_delay: float = float(os.getenv('KNOWHUB_INITIAL_RATE_LIMIT', '1.0'))
    batch_size: int = int(os.getenv('KNOWHUB_BATCH_SIZE', '50'))
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

    def fetch_publications(self, limit: int = 10, auto_save: bool = False) -> List[Dict]:
        """Fetch publications from Knowhub with parallel processing and caching."""
        publications = []
        try:
            logger.info(f"Starting to fetch up to {limit} publications from Knowhub")
            
            # Generate cache key for this request
            cache_key = f"publications_{limit}_{datetime.utcnow().strftime('%Y%m%d')}"
            
            # Try to get from cache first
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved {len(cached_data)} publications from cache")
                if auto_save and cached_data:
                    saved_count = self.batch_save_to_database(cached_data)
                    logger.info(f"Auto-saved {saved_count} publications from cache to database")
                return cached_data
            
            # Access the main publications page
            html_content = self._fetch_page(self.publications_url)
            if not html_content:
                return publications
            
            # Parse HTML content and find publication items
            pub_items = self._extract_items(html_content)
            total_items = len(pub_items)
            logger.info(f"Found {total_items} publication items")
            
            if total_items == 0:
                logger.warning("No publication items found. HTML structure may have changed.")
                return publications
            
            # Process items in parallel using thread pool
            items_to_process = pub_items[:min(limit, total_items)]
            
            # Submit all tasks to thread pool
            future_to_item = {
                self.executor.submit(self._process_publication_item, item): item 
                for item in items_to_process
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    publication = future.result()
                    if publication:
                        publications.append(publication)
                except Exception as e:
                    logger.error(f"Error processing publication item: {e}", exc_info=True)
            
            # Store in cache for future use
            if publications:
                self.cache.set(cache_key, publications, expire=self.config.cache_ttl)
            
            # Auto-save if requested
            if auto_save and publications:
                saved_count = self.batch_save_to_database(publications)
                logger.info(f"Auto-saved {saved_count} publications to database")
            
            logger.info(f"Successfully fetched {len(publications)} publications")
            return publications
            
        except Exception as e:
            logger.error(f"Error in fetch_publications: {e}", exc_info=True)
            return publications

    def _process_publication_item(self, item: BeautifulSoup) -> Optional[Dict]:
        """Process a single publication item with timing and metrics."""
        start_time = time.time()
        try:
            publication = self._parse_publication(item)
            processing_time = time.time() - start_time
            
            if publication:
                self.metrics.record_item_processed(True, processing_time)
                return publication
            else:
                self.metrics.record_item_processed(False, processing_time)
                return None
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.record_item_processed(False, processing_time, error=e)
            logger.error(f"Error processing publication: {e}", exc_info=True)
            return None

    def fetch_additional_content(self, content_type: str, limit: int = 10, auto_save: bool = False) -> List[Dict]:
        """Fetch content from additional endpoints with caching."""
        if content_type not in self.endpoints:
            logger.error(f"Invalid content type: {content_type}")
            return []
            
        url = self.endpoints[content_type]
        logger.info(f"Fetching {content_type} from: {url}")
        contents = []

        try:
            # Generate cache key for this request
            cache_key = f"{content_type}_{limit}_{datetime.utcnow().strftime('%Y%m%d')}"
            
            # Try to get from cache first
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved {len(cached_data)} {content_type} items from cache")
                if auto_save and cached_data:
                    saved_count = self.batch_save_to_database(cached_data)
                    logger.info(f"Auto-saved {saved_count} {content_type} items from cache to database")
                return cached_data
            
            # Fetch and parse page
            html_content = self._fetch_page(url)
            if not html_content:
                return contents
                
            # Extract items
            items = self._extract_items(html_content)
            
            if not items:
                logger.warning(f"No {content_type} items found. HTML structure may have changed.")
                return contents
            
            # Process items in parallel
            items_to_process = items[:min(limit, len(items))]
            
            # Submit all tasks to thread pool
            future_to_item = {
                self.executor.submit(self._process_additional_content_item, item, content_type): item 
                for item in items_to_process
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    content = future.result()
                    if content:
                        contents.append(content)
                except Exception as e:
                    logger.error(f"Error processing {content_type} item: {e}", exc_info=True)
            
            # Store in cache for future use
            if contents:
                self.cache.set(cache_key, contents, expire=self.config.cache_ttl)
                
            # Auto-save if requested
            if auto_save and contents:
                saved_count = self.batch_save_to_database(contents)
                logger.info(f"Auto-saved {saved_count} {content_type} items to database")
            
            logger.info(f"Successfully fetched {len(contents)} {content_type} items")
            return contents
            
        except Exception as e:
            logger.error(f"Error fetching {content_type}: {e}", exc_info=True)
            return contents

    def _process_additional_content_item(self, item: BeautifulSoup, content_type: str) -> Optional[Dict]:
        """Process an additional content item with specialized handling."""
        start_time = time.time()
        try:
            content = self._parse_publication(item)
            
            if not content:
                self.metrics.record_item_processed(False, time.time() - start_time)
                return None

            # Add content type to metadata
            try:
                identifiers = json.loads(content['identifiers'])
                identifiers['content_type'] = content_type
                content['identifiers'] = json.dumps(identifiers)
            except Exception as e:
                logger.error(f"Error updating identifiers: {e}")
            
            self.metrics.record_item_processed(True, time.time() - start_time)
            return content
                
        except Exception as e:
            self.metrics.record_item_processed(False, time.time() - start_time, error=e)
            logger.error(f"Error processing {content_type} item: {e}", exc_info=True)
            return None

    def fetch_all_content(self, limit: int = 1500, auto_save: bool = False) -> Dict[str, List[Dict]]:
        """Fetch content from all endpoints including original publications."""
        all_content = {}
        
        logger.info(f"Fetching all content with limit {limit} per endpoint")
        
        # Generate cache key for this batch
        cache_key = f"all_content_{limit}_{datetime.utcnow().strftime('%Y%m%d')}"
        
        # Try to get from cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Retrieved all content from cache")
            if auto_save:
                total_saved = 0
                for content_type, items in cached_data.items():
                    saved = self.batch_save_to_database(items)
                    logger.info(f"Auto-saved {saved} {content_type} items from cache to database")
                    total_saved += saved
                logger.info(f"Total auto-saved from cache: {total_saved} items")
            return cached_data
        
        # Fetch from original publications endpoint
        logger.info("Fetching from original publications endpoint...")
        all_content['publications'] = self.fetch_publications(limit=limit, auto_save=auto_save)
        
        # Fetch from additional endpoints using parallel processing
        future_to_content_type = {
            self.executor.submit(self.fetch_additional_content, content_type, limit, auto_save): content_type
            for content_type in self.endpoints
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_content_type):
            content_type = future_to_content_type[future]
            try:
                content = future.result()
                all_content[content_type] = content
                logger.info(f"Fetched {len(content)} {content_type} items")
            except Exception as e:
                logger.error(f"Error fetching {content_type}: {e}", exc_info=True)
                all_content[content_type] = []
        
        # Save to cache
        self.cache.set(cache_key, all_content, expire=self.config.cache_ttl)
        
        return all_content

    def _extract_items(self, html_content: str) -> List[BeautifulSoup]:
        """Extract items from HTML with flexible selectors."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get selectors from config
            tag_types, class_names = self.config.html_selectors['item']
            
            # Try primary selectors
            items = soup.find_all(tag_types, class_=class_names)
            
            # If nothing found, try alternative selectors
            if not items:
                logger.warning("Primary item selectors failed, trying alternatives")
                
                # Try different combinations
                for tag in tag_types:
                    for class_name in class_names:
                        items = soup.find_all(tag, class_=class_name)
                        if items:
                            logger.info(f"Found items using alternative selector: {tag}, {class_name}")
                            return items
                
                # If still nothing, try more generic approach
                items = soup.find_all('div', class_=lambda c: c and any(x in c for x in ['item', 'artifact', 'article']))
                if items:
                    logger.info("Found items using generic selector")
            
            return items
        except Exception as e:
            logger.error(f"Error extracting items: {e}", exc_info=True)
            return []

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
            # Record failure metrics
            self.metrics.record_request(False, url, error=e)
            self.rate_limiter.record_failure()
            logger.error(f"Request error for {url}: {e}")
            raise

    def _parse_publication(self, element: BeautifulSoup) -> Optional[Dict]:
        """Parse a DSpace publication element with enhanced error handling and metadata normalization."""
        start_time = time.time()
        try:
            # Check if we've seen this element before (deduplication)
            element_hash = hashlib.md5(str(element).encode()).hexdigest()
            cache_key = f"parsed_pub_{element_hash}"
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Retrieved parsed publication from cache (hash: {element_hash})")
                return cached_result
            
            logger.debug("Extracting publication information...")
            
            # Extract title with robust handling
            title_elem = self._extract_title_element(element)
            if not title_elem:
                logger.warning("No title element found, cannot process publication")
                return None
            
            # Get title text
            title = self._extract_text_content(title_elem)
            if not title:
                logger.warning("Empty title found, skipping publication")
                return None
                
            title = safe_str(title)
            logger.debug(f"Found title: {title[:100]}...")
            
            # Extract URL and handle using multiple strategies
            url, handle, content_type = self._extract_url_and_handle(element, title_elem)
            
            if not url or not handle:
                logger.warning(f"No URL or handle found for publication: {title}")
                return None
            
            # Check for duplicates using handle
            if handle in self.seen_handles:
                logger.debug(f"Duplicate publication found (handle: {handle}), skipping")
                return None
            
            self.seen_handles.add(handle)
            
            # Extract metadata
            metadata = self._extract_metadata(element)
            
            # Generate summary
            abstract = metadata.get('abstract', '')
            summary = self._generate_summary(title, abstract)
            
            # Create publication record with normalized fields
            publication = {
                'title': title,
                'doi': url,  # Store the URL in the doi field
                'abstract': abstract or f"Publication about {title}",
                'summary': summary,
                'authors': metadata.get('authors', []),
                'description': abstract or f"Publication about {title}",
                'expert_id': None,
                'type': content_type,  # Use our handle-based content type
                'subtitles': json.dumps({}),
                'publishers': json.dumps({
                    'name': 'APHRC',
                    'url': self.base_url,
                    'type': 'repository'
                }),
                'collection': 'knowhub',
                'date_issue': metadata.get('date'),
                'citation': metadata.get('citation'),
                'language': self._normalize_language(metadata.get('language', 'en')),
                'identifiers': json.dumps({
                    'handle': handle,
                    'source_id': f"knowhub-{handle.replace('/', '-')}",
                    'keywords': metadata.get('keywords', []),
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
                    for author in metadata.get('authors', [])
                ] + [
                    {
                        'name': keyword,
                        'tag_type': 'domain',
                        'additional_metadata': json.dumps({
                            'source': 'knowhub',
                            'type': 'keyword'
                        })
                    }
                    for keyword in metadata.get('keywords', [])
                ] + [{
                    'name': content_type,
                    'tag_type': 'content_type',
                    'additional_metadata': json.dumps({
                        'source': 'knowhub',
                        'original_type': metadata.get('type', 'other')
                    })
                }]
            }
            
            # Cache the result
            self.cache.set(cache_key, publication, expire=self.config.cache_ttl)
            
            # Record success
            self.metrics.record_item_processed(True, time.time() - start_time)
            
            return publication
        
        except Exception as e:
            # Record failure
            self.metrics.record_item_processed(False, time.time() - start_time, error=e)
            logger.error(f"Error parsing publication element: {str(e)}", exc_info=True)
            return None

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
            

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keywords."""
        if not keyword:
            return ""
            
        # Trim and lowercase
        keyword = keyword.strip().lower()
        
        # Remove trailing punctuation
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

    def scrape_and_save_all(self, limit_per_endpoint: int = 100) -> Dict:
        """Scrape all content types and save to database."""
        results = {}
        total_saved = 0
        
        try:
            # Fetch and save publications
            pubs = self.fetch_publications(limit=limit_per_endpoint)
            saved_pubs = self.batch_save_to_database(pubs)
            results['publications'] = {'fetched': len(pubs), 'saved': saved_pubs}
            total_saved += saved_pubs
            
            # Fetch and save each content type
            for content_type in self.endpoints.keys():
                content = self.fetch_additional_content(content_type, limit=limit_per_endpoint)
                saved = self.batch_save_to_database(content)
                results[content_type] = {'fetched': len(content), 'saved': saved}
                total_saved += saved
                
            # Add metrics and totals
            results['metrics'] = self.get_metrics()
            results['total_saved'] = total_saved
            
            return results
            
        except Exception as e:
            logger.error(f"Error in scrape_and_save_all: {e}", exc_info=True)
            results['error'] = str(e)
            return results

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
        """Context manager exit."""
        self.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the KnowhubScraper')
    parser.add_argument('--limit', type=int, default=100, help='Limit per endpoint')
    parser.add_argument('--save', action='store_true', help='Save to database')
    args = parser.parse_args()
    
    print(f"Starting KnowhubScraper with limit={args.limit}, save={args.save}")
    
    with KnowhubScraper() as scraper:
        if args.save:
            results = scraper.scrape_and_save_all(args.limit)
            print(f"Scraping completed. Total saved: {results.get('total_saved', 0)}")
        else:
            all_content = scraper.fetch_all_content(args.limit)
            print(f"Scraping completed. Total items: {sum(len(items) for items in all_content.values())}")
        
        print("Metrics:", scraper.get_metrics())