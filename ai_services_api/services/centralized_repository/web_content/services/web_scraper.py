"""
Website scraper for fetching and processing web content.

WARNING: Do not import ContentPipeline in this module to avoid circular imports.
Pass pipeline instances as method arguments if needed.
"""
import logging
import time
from typing import Dict, List, Optional
import os
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

class WebsiteScraper:
    """Handles web content scraping with Selenium and BeautifulSoup"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 60):
        self.max_retries = max_retries
        self.timeout = timeout
        self.website_url = os.getenv('WEBSITE_URL')
        self.driver = None
        self.session = requests.Session()
        self.setup_driver()

    def setup_driver(self):
        """Initialize Selenium WebDriver"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Selenium WebDriver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise

    def get_page_content(self, url: str) -> Optional[Dict]:
        """Fetch content for a single URL with Selenium and fallback to requests"""
        # Try Selenium first
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempting to scrape {url} (Attempt {attempt + 1})")
                self.driver.get(url)
                
                # Wait for main content to load (adjust selector as needed)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )
                
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Extract content
                title = soup.find('title').text if soup.find('title') else 'No Title'
                content = ' '.join([p.text for p in soup.find_all('p')])
                
                if not content.strip():
                    logger.warning(f"No content extracted from {url} with Selenium")
                    raise ValueError("Empty content")
                
                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'content_type': 'webpage',
                    'metadata': {}
                }
            except (TimeoutException, WebDriverException, ValueError) as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # Delay between retries
                continue
        
        # Fallback to requests
        logger.info(f"Falling back to requests for {url}")
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title').text if soup.find('title') else 'No Title'
            content = ' '.join([p.text for p in soup.find_all('p')])
            
            if not content.strip():
                logger.warning(f"No content extracted from {url} with requests")
                return None
                
            return {
                'url': url,
                'title': title,
                'content': content,
                'content_type': 'webpage',
                'metadata': {}
            }
        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url} with requests: {str(e)}")
            return None
        
        logger.error(f"Failed to scrape {url} after {self.max_retries} attempts")
        return None

    def scrape(self) -> List[Dict]:
        """Scrape content from the website"""
        if not self.website_url:
            logger.error("WEBSITE_URL not set")
            return []

        results = []
        try:
            # Start with the base URL
            response = self.session.get(self.website_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all links
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(self.website_url, href)
                if full_url.startswith(self.website_url):
                    links.add(full_url)
            
            # Scrape each link
            for url in links:
                page_data = self.get_page_content(url)
                if page_data:
                    results.append(page_data)
                else:
                    logger.warning(f"Skipping {url} due to scraping failure")
            
            return results
        except requests.RequestException as e:
            logger.error(f"Error during initial scraping: {str(e)}")
            return results
        except Exception as e:
            logger.error(f"Unexpected error during scraping: {str(e)}", exc_info=True)
            return results

    def cleanup(self):
        """Clean up WebDriver and requests resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver cleaned up")ppro
            if self.session:
                self.session.close()
                logger.info("Requests session cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")