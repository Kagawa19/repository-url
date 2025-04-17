"""
Website scraper for fetching and processing web content.

WARNING: Do not import ContentPipeline in this module to avoid circular imports.
Pass pipeline instances as method arguments if needed.
"""
import logging
from typing import Dict, List, Optional
import os
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

class WebsiteScraper:
    """Handles web content scraping with Selenium and BeautifulSoup"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout
        self.website_url = os.getenv('WEBSITE_URL')
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Initialize Selenium WebDriver"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Selenium WebDriver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise

    def get_page_content(self, url: str) -> Optional[Dict]:
        """Fetch content for a single URL"""
        for attempt in range(self.max_retries):
            try:
                self.driver.get(url)
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Extract content (simplified example)
                title = soup.find('title').text if soup.find('title') else 'No Title'
                content = ' '.join([p.text for p in soup.find_all('p')])
                
                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'content_type': 'webpage',
                    'metadata': {}
                }
            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to scrape {url} after {self.max_retries} attempts")
                    return None
        return None

    def scrape(self) -> List[Dict]:
        """Scrape content from the website"""
        if not self.website_url:
            logger.error("WEBSITE_URL not set")
            return []

        results = []
        try:
            # Start with the base URL
            response = requests.get(self.website_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all links (simplified example)
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
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return results

    def cleanup(self):
        """Clean up WebDriver resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver cleaned up")
        except Exception as e:
            logger.error(f"Error during WebDriver cleanup: {str(e)}")