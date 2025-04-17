import logging
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from retrying import retry
import time
import hashlib
from datetime import datetime
import os
import re
from ..utils.text_cleaner import TextCleaner
from ...database.database_setup import insert_webpage, insert_expert

logger = logging.getLogger(__name__)

class WebsiteScraper:
    """
    Handles web scraping operations using Selenium and BeautifulSoup.
    Supports JavaScript rendering and dynamic content loading.
    """
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.base_url = os.getenv('WEBSITE_URL')
        if not self.base_url:
            raise ValueError("WEBSITE_URL environment variable not set")
            
        self.base_domain = urlparse(self.base_url).netloc
        self.visited_urls: Set[str] = set()
        self.pdf_links: Set[str] = set()
        self.text_cleaner = TextCleaner()
        
        self.selenium_timeout = int(os.getenv('SELENIUM_TIMEOUT', '30'))
        self.scroll_pause = float(os.getenv('SCROLL_PAUSE_TIME', '1.0'))
        self.max_depth = int(os.getenv('MAX_DEPTH', '3'))
        
        self.setup_selenium()
        logger.info(f"WebsiteScraper initialized with base_url: {self.base_url}")

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def setup_selenium(self):
        try:
            options = Options()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            chrome_driver_path = os.getenv('CHROMEDRIVER_PATH')
            service = Service(chrome_driver_path) if chrome_driver_path else Service()
            
            remote_url = os.getenv('SELENIUM_REMOTE_URL')
            if remote_url:
                logger.info(f"Using remote WebDriver at {remote_url}")
                self.driver = webdriver.Remote(
                    command_executor=remote_url,
                    options=options
                )
            else:
                self.driver = webdriver.Chrome(service=service, options=options)
            
            self.driver.set_page_load_timeout(self.selenium_timeout)
            logger.info("Selenium WebDriver setup complete")
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {str(e)}")
            raise

    def is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc.endswith(self.base_domain),
                len(url) < 2048
            ])
        except Exception:
            return False

    def is_expert_url(self, url: str) -> bool:
        """Check if URL is an expert profile"""
        return bool(re.match(r'^https?://aphrc\.org/person/.*$', url))

    def scroll_page(self):
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(self.scroll_pause)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
        except Exception as e:
            logger.error(f"Error scrolling page: {str(e)}")

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Set[str]]:
        nav_links = set()
        pdf_links = set()
        try:
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)
                if self.is_valid_url(absolute_url):
                    if absolute_url.lower().endswith('.pdf'):
                        pdf_links.add(absolute_url)
                    else:
                        nav_links.add(absolute_url)
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
        return {
            'nav_links': nav_links,
            'pdf_links': pdf_links
        }

    def extract_expert_name(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract expert name from page title or content"""
        try:
            title = soup.title.string if soup.title else ''
            h1 = soup.find('h1')
            name = h1.get_text(strip=True) if h1 else title.split('|')[0].strip()
            return name if name else None
        except Exception as e:
            logger.error(f"Error extracting expert name: {str(e)}")
            return None

    def get_page_content(self, url: str) -> Optional[Dict]:
        try:
            logger.info(f"Fetching content from: {url}")
            self.driver.get(url)
            WebDriverWait(self.driver, self.selenium_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self.scroll_page()
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article'])
            text_content = ' '.join([elem.get_text(strip=True) for elem in text_elements])
            cleaned_content = self.text_cleaner.clean_text(text_content)
            
            if not cleaned_content.strip():
                logger.warning(f"No content found at: {url}")
                return None
                
            links = self.extract_links(soup, url)
            self.pdf_links.update(links['pdf_links'])
            
            metadata = {
                'title': soup.title.string if soup.title else '',
                'meta_description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else '',
                'last_modified': self.driver.execute_script("return document.lastModified;"),
                'word_count': len(cleaned_content.split())
            }
            
            content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()
            logger.debug(f"Scraped {url}: {len(cleaned_content)} chars, hash: {content_hash}")
            
            result = {
                'url': url,
                'content': cleaned_content,
                'title': metadata['title'],
                'nav_links': list(links['nav_links']),
                'pdf_links': list(links['pdf_links']),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'content_hash': content_hash
            }
            
            # Store in the appropriate table
            if self.is_expert_url(url):
                name = self.extract_expert_name(soup)
                expert_id = insert_expert(
                    url=url,
                    name=name,
                    content=cleaned_content,
                    metadata=metadata,
                    last_modified=metadata['last_modified']
                )
                result['expert_id'] = expert_id
                result['content_type'] = 'expert'
            else:
                webpage_id = insert_webpage(
                    url=url,
                    title=metadata['title'],
                    content=cleaned_content,
                    metadata=metadata,
                    last_modified=metadata['last_modified']
                )
                result['webpage_id'] = webpage_id
                result['content_type'] = 'webpage'
            
            return result
            
        except TimeoutException:
            logger.error(f"Timeout while loading: {url}")
            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    def scrape_site(self) -> List[Dict]:
        pages_data = []
        urls_to_visit = [(self.base_url, 0)]  # (url, depth)
        try:
            while urls_to_visit:
                url, depth = urls_to_visit.pop(0)
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                logger.info(f"Scraping: {url} (Depth: {depth})")
                page_data = self.get_page_content(url)
                if page_data:
                    pages_data.append(page_data)
                    self.visited_urls.add(url)
                    if depth < self.max_depth:
                        new_urls = [
                            (url, depth + 1) 
                            for url in page_data['nav_links'] 
                            if url not in self.visited_urls
                        ]
                        urls_to_visit.extend(new_urls)
                time.sleep(1)
            logger.info(f"Scraping complete. Processed {len(pages_data)} pages")
            return pages_data
        except Exception as e:
            logger.error(f"Error during site scraping: {str(e)}")
            return pages_data
        finally:
            self.close()

    def close(self):
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()