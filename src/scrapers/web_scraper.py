# src/scrapers/web_scraper.py
import logging
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ..utils.text_cleaner import TextCleaner
import time
import hashlib
import json
from datetime import datetime
from ..config.settings import (
    WEBSITE_URL, MAX_PAGES, MAX_DEPTH, SELENIUM_TIMEOUT, 
    SCROLL_PAUSE_TIME
)

class WebsiteScraper:
    def __init__(self):
        self.setup_logging()
        self.setup_selenium()
        self.visited_urls: Set[str] = set()
        self.pdf_links: Set[str] = set()
        self.base_domain = urlparse(WEBSITE_URL).netloc

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def setup_selenium(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(SELENIUM_TIMEOUT)

    def scroll_page(self):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def get_page_content(self, url: str) -> Dict:
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self.scroll_page()
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract and clean text content
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article'])
            text_content = ' '.join([elem.get_text(strip=True) for elem in text_elements])
            cleaned_content = TextCleaner.clean_text(text_content)
            
            # Rest of the method remains the same...
            return {
                'url': url,
                'content': cleaned_content,
                'pdf_links': list(pdf_links),
                'nav_links': list(nav_links),
                'title': soup.title.string if soup.title else '',
                'timestamp': datetime.now().isoformat(),
                'hash': hashlib.sha256(cleaned_content.encode()).hexdigest()
            }

        def scrape_site(self) -> List[Dict]:
            pages_data = []
            urls_to_visit = [(WEBSITE_URL, 0)]  # (url, depth)
            
            while urls_to_visit and len(self.visited_urls) < MAX_PAGES:
                url, depth = urls_to_visit.pop(0)
                
                if url in self.visited_urls or depth > MAX_DEPTH:
                    continue
                    
                self.logger.info(f"Scraping: {url} (Depth: {depth})")
                page_data = self.get_page_content(url)
                
                if page_data:
                    pages_data.append(page_data)
                    self.visited_urls.add(url)
                    
                    if depth < MAX_DEPTH:
                        new_urls = [
                            (url, depth + 1) 
                            for url in page_data['nav_links'] 
                            if url not in self.visited_urls
                        ]
                        urls_to_visit.extend(new_urls)
            
            return pages_data

    def save_results(self, data: List[Dict], filename: str = 'scraped_data.json'):
        with open(filename, 'w') as f:
            json.dump({
                'pages': data,
                'pdf_links': list(self.pdf_links),
                'total_pages': len(data),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    def close(self):
        self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()