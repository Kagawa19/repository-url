import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from datetime import datetime, timedelta
import re
import hashlib
import json
import time
import threading
import concurrent.futures
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import diskcache
import psycopg2
from dataclasses import dataclass, field, asdict
import tempfile
import io
import traceback
from urllib.parse import urlparse, urljoin

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException, 
    TimeoutException, 
    WebDriverException, 
    NoSuchElementException
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.text_processor import safe_str
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
    def __init__(self, initial_delay: float = 2.0, min_delay: float = 1.0, max_delay: float = 10.0, backoff_factor: float = 1.5):
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

# Browser pool for parallel processing
class BrowserPool:
    def __init__(self, max_size: int = 3, chrome_options=None):
        self.max_size = max_size
        self.chrome_options = chrome_options or self._default_chrome_options()
        self.browsers = []
        self.browsers_in_use = {}
        self._lock = threading.Lock()
        
    def _default_chrome_options(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--remote-debugging-port=9222')
        options.add_argument('--disable-setuid-sandbox')
        options.add_argument('--disable-extensions')
        return options
        
    def get_browser(self):
        with self._lock:
            # Check if we have available browsers
            for browser in self.browsers:
                if browser not in self.browsers_in_use:
                    self.browsers_in_use[browser] = True
                    return browser
                    
            # If pool is not at max size, create a new browser
            if len(self.browsers) < self.max_size:
                try:
                    service = Service()
                    browser = webdriver.Chrome(
                        service=service,
                        options=self.chrome_options
                    )
                    self.browsers.append(browser)
                    self.browsers_in_use[browser] = True
                    return browser
                except Exception as e:
                    logger.error(f"Failed to create new browser: {e}")
                    raise
                
            # If we reach here, all browsers are in use and we're at max capacity
            return None
            
    def release_browser(self, browser):
        with self._lock:
            if browser in self.browsers_in_use:
                del self.browsers_in_use[browser]
                
    def close_all(self):
        with self._lock:
            for browser in self.browsers:
                try:
                    browser.quit()
                except:
                    pass
            self.browsers = []
            self.browsers_in_use = {}

@dataclass
class ScraperConfig:
    """Configuration for the WebsiteScraper."""
    base_url: str = os.getenv('WEBSITE_BASE_URL', 'https://aphrc.org') # Updated base URL
    cache_dir: str = os.getenv('WEBSITE_CACHE_DIR', '/tmp/website_cache')
    cache_ttl: int = int(os.getenv('WEBSITE_CACHE_TTL', '86400'))  # 24 hours in seconds
    request_timeout: int = int(os.getenv('WEBSITE_REQUEST_TIMEOUT', '30'))
    browser_timeout: int = int(os.getenv('WEBSITE_BROWSER_TIMEOUT', '20'))
    max_workers: int = int(os.getenv('WEBSITE_MAX_WORKERS', '5')) # Increased workers
    max_browsers: int = int(os.getenv('WEBSITE_MAX_BROWSERS', '5')) # Increased browsers
    max_retries: int = int(os.getenv('WEBSITE_MAX_RETRIES', '3'))
    initial_rate_limit_delay: float = float(os.getenv('WEBSITE_INITIAL_RATE_LIMIT', '2.0'))
    batch_size: int = int(os.getenv('WEBSITE_BATCH_SIZE', '100')) # Increased batch size
    css_selectors: Dict[str, List[str]] = field(default_factory=lambda: {
        'links': [ # Generic link selectors
            'a[href]', 'a[class*="-link"]', 'a[class*="button"]', 'a[class*="nav"]'
        ],  
        'text': [ # Generic text selectors
            'p', 'article', 'div[class*="content"]', 'div[class*="text"]',
            'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', '.post-content'
        ],
        'load_more': [
            '.alm-load-more-btn', '.load-more', 'a.next', '.pagination a', 
            'button[class*="load"]', '.nav-links .next'
        ]
    })

    def to_dict(self) -> Dict:
        return asdict(self)

class WebsiteScraper:
    def __init__(self, summarizer: Optional[TextSummarizer] = None, config: Optional[ScraperConfig] = None):
        """Initialize WebsiteScraper with enhanced features."""
        # Initialize configuration
        self.config = config or ScraperConfig()
        
        # Base URL
        self.base_url = self.config.base_url
        
        # Initialize database connection
        self.db = DatabaseManager()
        
        # Initialize rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            initial_delay=self.config.initial_rate_limit_delay,
            min_delay=1.0,
            max_delay=10.0,
            backoff_factor=1.5
        )
        
        # Initialize HTTP session with retry capabilities
        self.session = self._create_session()
        
        # Set headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Initialize cache
        self.cache = diskcache.Cache(self.config.cache_dir)
        
        # Initialize summarizer (with lazy loading)
        self._summarizer = summarizer
        
        # Track seen URLs
        self.seen_urls = set()
        
        # Initialize metrics collector
        self.metrics = ScraperMetrics()
        
        # Initialize browser pool
        chrome_options = self._setup_chrome_options()
        self.browser_pool = BrowserPool(max_size=self.config.max_browsers, chrome_options=chrome_options)
        
        # Create thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Initialize single browser for compatibility with original code
        try:
            self.driver = self.browser_pool.get_browser()
            self.wait = WebDriverWait(self.driver, self.config.browser_timeout)
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise
            
        logger.info("WebsiteScraper initialized with configuration: %s", self.config.to_dict())


    def _setup_chrome_options(self) -> Options:
        """Set up Chrome options for headless browsing."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--remote-debugging-port=9222')
        chrome_options.add_argument('--disable-setuid-sandbox')
        chrome_options.add_argument('--disable-extensions')
        # Add user agent to avoid detection
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36')
        return chrome_options

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

    @retry(
        retry=retry_if_exception_type((
            NoSuchElementException, 
            StaleElementReferenceException,
            WebDriverException
        )),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3)
    )
    def extract_publication_details(self, card) -> Optional[Dict]:
        """Extract publication details from a card element with enhanced error handling."""
        start_time = time.time()
        try:
            # Extract URL first (most important for identifying the publication)
            url = self._extract_element_attribute(card, self.config.css_selectors['url'], "href")
            
            if not url:
                self.metrics.record_item_processed(False, time.time() - start_time)
                return None
                
            # Generate cache key for this URL
            cache_key = f"publication_details_{hashlib.md5(url.encode()).hexdigest()}"
            
            # Check cache first
            cached_details = self.cache.get(cache_key)
            if cached_details:
                logger.debug(f"Retrieved publication details from cache for URL: {url}")
                return cached_details
                
            # Extract title with multiple selectors
            title = self._extract_element_text(card, self.config.css_selectors['title'])
            title = title or 'Untitled Publication'
            
            # Extract content/description with multiple selectors
            content = self._extract_element_text(card, self.config.css_selectors['content'])
            content = content or ''
            
            # Try to extract year from URL or content
            year = self._extract_year(url, content)
            
            # Try to extract authors from content
            authors = self._extract_authors(content)
            
            # Extract keywords or domains from content
            domains = self._extract_keywords(content)
            
            # Create publication details dictionary
            publication_details = {
                'title': title,
                'doi': url,
                'content': content,
                'year': year,
                'authors': authors,
                'domains': domains
            }
            
            # Cache the result
            self.cache.set(cache_key, publication_details, expire=self.config.cache_ttl)
            
            self.metrics.record_item_processed(True, time.time() - start_time)
            return publication_details
            
        except Exception as e:
            self.metrics.record_item_processed(False, time.time() - start_time, error=e)
            logger.error(f"Error extracting publication details: {e}")
            return None

    def _extract_element_text(self, element, selectors) -> Optional[str]:
        """Extract text from element with multiple selectors."""
        for selector in selectors:
            try:
                element_found = element.find_element(By.CSS_SELECTOR, selector)
                text = element_found.text.strip()
                if text:
                    return text
            except:
                continue
        return None

    def _extract_element_attribute(self, element, selectors, attribute) -> Optional[str]:
        """Extract attribute from element with multiple selectors."""
        for selector in selectors:
            try:
                element_found = element.find_element(By.CSS_SELECTOR, selector)
                value = element_found.get_attribute(attribute)
                if value:
                    return value
            except:
                continue
        return None

    def _extract_year(self, url: str, content: str) -> Optional[int]:
        """Extract publication year from URL or content."""
        # Try URL first
        year_match = re.search(r'/(\d{4})/', url)
        if year_match:
            year = int(year_match.group(1))
            # Validate year is reasonable
            if 1990 <= year <= datetime.now().year:
                return year

        # Try content next
        year_match = re.search(r'\b(19|20)\d{2}\b', content)
        if year_match:
            year = int(year_match.group(0))
            # Validate year is reasonable
            if 1990 <= year <= datetime.now().year:
                return year

        # Try publication date pattern
        date_match = re.search(r'(published|released|date).*?\b(19|20)\d{2}\b', content.lower())
        if date_match:
            year_match = re.search(r'\b(19|20)\d{2}\b', date_match.group(0))
            if year_match:
                return int(year_match.group(0))

        return None

    def _extract_authors(self, content: str) -> List[str]:
        """Extract author names from content."""
        authors = []
        
        # Common author patterns
        author_patterns = [
            r'by\s+([\w\s]+)(?:,|\s+and|\s+&|\.|$)',
            r'authors?:?\s+([\w\s,&]+)(?:\.|$)',
            r'written by\s+([\w\s]+)(?:,|\s+and|\s+&|\.|$)'
        ]
        
        for pattern in author_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                author_text = match.group(1).strip()
                # Split multiple authors
                if ',' in author_text or ' and ' in author_text or ' & ' in author_text:
                    parts = re.split(r',\s*|\s+and\s+|\s+&\s+', author_text)
                    authors.extend([p.strip() for p in parts if p.strip()])
                else:
                    authors.append(author_text)
        
        # Normalize author names
        normalized_authors = []
        for author in authors:
            # Remove titles
            author = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof)\.?\s+', '', author)
            # Trim and add if not empty
            author = author.strip()
            if author and len(author.split()) >= 1:
                normalized_authors.append(author)
                
        # Remove duplicates while preserving order
        seen = set()
        return [a for a in normalized_authors if not (a in seen or seen.add(a))]

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords or domains from content."""
        keywords = []
        
        # Common keyword patterns
        keyword_patterns = [
            r'keywords?:?\s+([\w\s,]+)(?:\.|$)',
            r'tags?:?\s+([\w\s,]+)(?:\.|$)',
            r'topics?:?\s+([\w\s,]+)(?:\.|$)',
            r'categories?:?\s+([\w\s,]+)(?:\.|$)'
        ]
        
        for pattern in keyword_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                keyword_text = match.group(1).strip()
                parts = re.split(r',\s*|\s+and\s+|\s+&\s+', keyword_text)
                keywords.extend([p.strip().lower() for p in parts if p.strip()])
        
        # If no explicit keywords, extract potential domain words
        if not keywords:
            # Common domains in public health
            domains = [
                'health', 'education', 'poverty', 'nutrition', 'policy', 
                'development', 'children', 'women', 'gender', 'environment',
                'climate', 'disease', 'economic', 'social', 'community',
                'sustainable', 'malaria', 'HIV', 'AIDS', 'research',
                'urban', 'rural', 'agriculture', 'food', 'security'
            ]
            
            content_lower = content.lower()
            for domain in domains:
                if domain in content_lower:
                    keywords.append(domain)
        
        # Remove duplicates while preserving order
        seen = set()
        return [k for k in keywords if not (k in seen or seen.add(k))]

    def fetch_content(self, search_term: Optional[str] = None, max_pages: Optional[int] = None) -> List[Dict]:
        """
        Fetch publications from website with improved caching and parallelism.
        
        Args:
            search_term (str, optional): Search term to filter publications
            max_pages (int, optional): Maximum number of pages to scrape (None for unlimited)
        """
        # Check cache first for the full result set
        cache_key = f"website_content_{search_term or 'all'}_{datetime.utcnow().strftime('%Y%m%d')}"
        cached_content = self.cache.get(cache_key)
        if cached_content:
            logger.info(f"Retrieved {len(cached_content)} publications from cache")
            return cached_content
            
        publications = []
        visited = set()
        
        try:
            # Log the request
            logger.info(f"Accessing URL: {self.base_url}")
            
            # Apply rate limiting
            self.rate_limiter.wait()
            
            # Load the main page
            try:
                self.driver.get(self.base_url)
                self.rate_limiter.record_success()
                time.sleep(5)  # Initial wait for page load
            except Exception as e:
                self.rate_limiter.record_failure()
                logger.error(f"Error loading base URL: {e}")
                return publications

            # Process publication cards
            page_num = 1
            while max_pages is None or page_num <= max_pages:
                try:
                    logger.info(f"Processing page {page_num}")
                    
                    # Get all publication cards using our multi-selector approach
                    publication_cards = self._find_all_with_selectors(
                        self.driver, 
                        self.config.css_selectors['publication_cards'],
                        wait_time=5
                    )
                    
                    if not publication_cards:
                        logger.warning("No publication cards found on current page")
                        break
                        
                    logger.info(f"Found {len(publication_cards)} publication cards")
                    
                    # Process cards in parallel
                    futures = []
                    for card in publication_cards:
                        # Submit the task to thread pool
                        future = self.executor.submit(self._process_publication_card, card, visited)
                        futures.append(future)
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            publication = future.result()
                            if publication:
                                publications.append(publication)
                        except Exception as e:
                            logger.error(f"Error processing publication card: {e}")
                    
                    # Try to load more
                    if not self._click_load_more():
                        logger.info("No more publications to load")
                        break
                        
                    page_num += 1
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    break
                    
            # Cache the results
            if publications:
                self.cache.set(cache_key, publications, expire=self.config.cache_ttl)
                
            return publications
            
        except Exception as e:
            logger.error(f"Error in fetch_content: {e}")
            return publications
        finally:
            if not publications:
                logger.warning("No publications were found")

    def _find_all_with_selectors(self, driver, selectors, wait_time=5):
        """Find all elements using multiple selectors with fallbacks."""
        elements = []
        
        # Try each selector in the list
        for selector in selectors:
            try:
                wait = WebDriverWait(driver, wait_time)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                found_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if found_elements:
                    logger.debug(f"Found {len(found_elements)} elements using selector: {selector}")
                    elements.extend(found_elements)
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
                
        # Return unique elements (remove duplicates)
        unique_elements = []
        element_ids = set()
        
        for element in elements:
            try:
                element_id = element.id
                if element_id not in element_ids:
                    element_ids.add(element_id)
                    unique_elements.append(element)
            except:
                # If we can't get the element ID, just add it
                unique_elements.append(element)
                
        return unique_elements

    def _click_load_more(self) -> bool:
        """Click load more button with multiple selector support."""
        for selector in self.config.css_selectors['load_more']:
            try:
                # Try to find and click the load more button
                load_more = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                
                if load_more.is_displayed():
                    # Scroll to the button
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
                    time.sleep(1)
                    
                    # Click it
                    load_more.click()
                    time.sleep(3)  # Wait for new content to load
                    return True
            except:
                continue
                
        return False

    def _process_publication_card(self, card, visited: Set[str]) -> Optional[Dict]:
        """Process a single publication card with error handling."""
        start_time = time.time()
        browser = None
        
        try:
            # Extract basic details from the card
            details = self.extract_publication_details(card)
            if not details:
                return None
                
            url = details['doi']
            
            # Skip if already visited
            if url in visited or url in self.seen_urls:
                logger.debug(f"Skipping already processed URL: {url}")
                return None
                
            # Mark as visited
            visited.add(url)
            self.seen_urls.add(url)
            
            logger.info(f"Processing publication URL: {url}")
            
            # Handle PDFs differently
            if url.lower().endswith('.pdf'):
                # For PDFs, use the details extracted from the card
                publication = {
                    'title': details['title'],
                    'doi': url,
                    'authors': details.get('authors', []),
                    'domains': details.get('domains', []),
                    'type': 'publication',
                    'publication_year': details.get('year'),
                    'summary': details.get('content', '')[:1000] if details.get('content') else "PDF Publication",
                    'source': 'website'
                }
                
                logger.info(f"Processed PDF Publication: {details['title'][:100]}...")
                self.metrics.record_item_processed(True, time.time() - start_time)
                return publication
            
            # For webpages, get a dedicated browser
            browser = self.browser_pool.get_browser()
            if not browser:
                logger.warning("No browser available for processing webpage - falling back to basic details")
                # Return basic publication with what we already know
                publication = {
                    'title': details['title'],
                    'doi': url,
                    'authors': details.get('authors', []),
                    'domains': details.get('domains', []),
                    'type': 'publication',
                    'publication_year': details.get('year'),
                    'summary': details.get('content', '')[:1000] if details.get('content') else f"Publication about {details['title']}",
                    'source': 'website'
                }
                self.metrics.record_item_processed(True, time.time() - start_time)
                return publication
            
            # Check cache for webpage
            cache_key = f"webpage_{hashlib.md5(url.encode()).hexdigest()}"
            cached_publication = self.cache.get(cache_key)
            if cached_publication:
                logger.debug(f"Retrieved webpage publication from cache: {url}")
                self.browser_pool.release_browser(browser)
                return cached_publication
            
            # Navigate to the url
            try:
                logger.debug(f"Navigating to URL: {url}")
                browser.get(url)
                wait = WebDriverWait(browser, self.config.browser_timeout)
                
                # Wait for content to load
                content_selectors = self.config.css_selectors['content_page']
                content_text = ""
                
                for selector in content_selectors:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        content_elems = browser.find_elements(By.CSS_SELECTOR, selector)
                        content_text = "\n".join(elem.text.strip() for elem in content_elems if elem.text.strip())
                        if content_text:
                            break
                    except:
                        continue
                
                # Get more detailed information
                if not details['authors']:
                    details['authors'] = self._extract_authors(content_text)
                
                if not details['domains']:
                    details['domains'] = self._extract_keywords(content_text)
                
                if not details['year']:
                    details['year'] = self._extract_year(url, content_text)
                
                # Create publication object
                publication = {
                    'title': details['title'],
                    'doi': url,
                    'authors': details['authors'],
                    'domains': details['domains'],
                    'type': 'publication',
                    'publication_year': details['year'],
                    'summary': content_text[:1000] or details.get('content', '')[:1000] or f"Publication about {details['title']}",
                    'source': 'website'
                }
                
                # Cache the publication
                self.cache.set(cache_key, publication, expire=self.config.cache_ttl)
                
                logger.info(f"Processed Web Publication: {details['title'][:100]}...")
                self.metrics.record_item_processed(True, time.time() - start_time)
                return publication
                
            except Exception as e:
                logger.error(f"Error extracting webpage content: {e}")
                # Return basic publication with what we already know
                publication = {
                    'title': details['title'],
                    'doi': url,
                    'authors': details.get('authors', []),
                    'domains': details.get('domains', []),
                    'type': 'publication',
                    'publication_year': details.get('year'),
                    'summary': details.get('content', '')[:1000] if details.get('content') else f"Publication about {details['title']}",
                    'source': 'website'
                }
                self.metrics.record_item_processed(True, time.time() - start_time)
                return publication
                
        except Exception as e:
            self.metrics.record_item_processed(False, time.time() - start_time, error=e)
            logger.error(f"Error processing publication card: {e}")
            return None
            
        finally:
            # Always release the browser
            if browser:
                self.browser_pool.release_browser(browser)

    @retry(
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout, requests.ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    def _make_request(self, url: str) -> requests.Response:
        """Make HTTP request with error handling and retry."""
        # Apply rate limiting
        self.rate_limiter.wait()
        
        try:
            # Check cache first
            cache_key = f"http_request_{hashlib.md5(url.encode()).hexdigest()}"
            cached_response = self.cache.get(cache_key)
            
            if cached_response:
                logger.debug(f"Retrieved HTTP response from cache: {url}")
                self.metrics.record_request(True, url)
                self.rate_limiter.record_success()
                return cached_response
            
            # Make request
            logger.debug(f"Making HTTP request: {url}")
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=self.config.request_timeout
            )
            
            # Record metrics
            self.metrics.record_request(response.status_code == 200, url)
            
            # Update rate limiter
            if response.status_code == 200:
                self.rate_limiter.record_success()
            else:
                self.rate_limiter.record_failure()
            
            response.raise_for_status()
            
            # Cache successful response
            self.cache.set(cache_key, response, expire=self.config.cache_ttl)
            
            return response
            
        except Exception as e:
            self.metrics.record_request(False, url, error=e)
            self.rate_limiter.record_failure()
            logger.error(f"Request failed for {url}: {e}")
            raise

    def batch_save_to_database(self, publications: List[Dict]) -> int:
        """Save multiple publications to database in efficient batches."""
        if not publications:
            logger.warning("No publications to save")
            return 0
            
        logger.info(f"Saving {len(publications)} publications to database in batches")
        
        successful = 0
        batch_size = self.config.batch_size
        
        try:
            # Process in batches
            for i in range(0, len(publications), batch_size):
                batch = publications[i:i+batch_size]
                
                try:
                    batch_successful = 0
                    
                    # Save each publication in the batch
                    for pub in batch:
                        # Normalize data
                        title = pub.get('title', '')
                        doi = pub.get('doi', '')
                        
                        if not title or not doi:
                            continue
                        
                        # Check if exists by DOI or title
                        exists = self.db.exists_publication(title=title, doi=doi)
                        
                        if exists:
                            logger.debug(f"Publication already exists: {title}")
                            continue
                        
                        # Generate summary if not present
                        summary = pub.get('summary')
                        if not summary or len(summary) < 50:
                            abstract = pub.get('content', '') or f"Publication about {title}"
                            try:
                                summary = self.summarizer.summarize(title, abstract)
                            except Exception as e:
                                logger.error(f"Error generating summary: {e}")
                                summary = abstract[:500]
                        
                        # Add publication to database
                        success = self.db.add_publication(
                            title=title,
                            doi=doi,
                            authors=pub.get('authors', []),
                            domains=pub.get('domains', []),
                            publication_year=pub.get('publication_year'),
                            summary=summary,
                            source='website'
                        )
                        
                        if success:
                            batch_successful += 1
                            
                    # Update counter
                    successful += batch_successful
                    logger.info(f"Successfully saved {batch_successful} publications in batch (total: {successful})")
                    
                except Exception as e:
                    logger.error(f"Error saving publication batch: {e}")
                    continue
            
            return successful
            
        except Exception as e:
            logger.error(f"Error in batch save: {e}")
            return successful

    def get_metrics(self) -> Dict:
        """Get current metrics summary."""
        return self.metrics.get_summary()

    def close(self):
        """Clean up resources with improved error handling."""
        try:
            # Close thread pool
            logger.debug("Shutting down thread pool...")
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Close browser pool
            logger.debug("Closing browser pool...")
            if hasattr(self, 'browser_pool'):
                self.browser_pool.close_all()
            
            # Close single driver for backward compatibility
            logger.debug("Closing main driver...")
            if hasattr(self, 'driver'):
                try:
                    self.driver.quit()
                except:
                    pass
            
            # Close cache
            logger.debug("Closing cache...")
            if hasattr(self, 'cache'):
                self.cache.close()
            
            # Close session
            logger.debug("Closing HTTP session...")
            if hasattr(self, 'session'):
                self.session.close()
            
            # Close summarizer if we own it
            logger.debug("Closing summarizer...")
            if self._summarizer and hasattr(self._summarizer, 'close'):
                self._summarizer.close()
            
            # Clear collections
            self.seen_urls.clear()
            
            logger.info("WebsiteScraper resources cleaned up")
            logger.info("Final metrics: %s", self.get_metrics())
            
        except Exception as e:
            logger.error(f"Error closing WebsiteScraper: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """Context manager exit."""
        self.close()