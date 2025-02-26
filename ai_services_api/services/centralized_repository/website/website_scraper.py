import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re
import hashlib
import json
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException

from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager

from ai_services_api.services.centralized_repository.text_processor import safe_str

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s: %(message)s', 
   datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WebsiteScraper:
   def __init__(self, summarizer: Optional[TextSummarizer] = None):
       """Initialize WebsiteScraper."""
       self.base_url = "https://aphrc.org/publications/"
       self.db = DatabaseManager()  # Add this line
       self.headers = {
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
       }
       self.summarizer = summarizer or TextSummarizer()
       self.seen_urls = set()
       
       # Enhanced Chrome options for container environment
       chrome_options = Options()
       chrome_options.add_argument('--headless')
       chrome_options.add_argument('--no-sandbox')
       chrome_options.add_argument('--disable-dev-shm-usage')
       chrome_options.add_argument('--disable-gpu')
       chrome_options.add_argument('--window-size=1920,1080')
       chrome_options.add_argument('--remote-debugging-port=9222')
       chrome_options.add_argument('--disable-setuid-sandbox')
       chrome_options.add_argument('--disable-extensions')
       
       try:
           service = Service()
           self.driver = webdriver.Chrome(
               service=service,
               options=chrome_options
           )
           self.wait = WebDriverWait(self.driver, 20)  # Increased timeout
           logger.info("Chrome WebDriver initialized successfully")
       except Exception as e:
           logger.error(f"Failed to initialize Chrome WebDriver: {e}")
           raise

   def extract_publication_details(self, card) -> Optional[Dict]:
       """Extract publication details from a card element."""
       try:
           # Try different methods to get the title
           title = None
           for selector in ["h3", "h2", ".title", ".entry-title"]:
               try:
                   title_elem = card.find_element(By.CSS_SELECTOR, selector)
                   title = title_elem.text.strip()
                   if title:
                       break
               except:
                   continue

           # Try to get content/description
           content = ""
           for selector in [".excerpt", ".description", ".content", "p"]:
               try:
                   content_elem = card.find_element(By.CSS_SELECTOR, selector)
                   content = content_elem.text.strip()
                   if content:
                       break
               except:
                   continue

           # Try to get URL
           url = None
           try:
               # First try direct link on title
               url_elem = card.find_element(By.CSS_SELECTOR, "h3 a, h2 a, .title a")
               url = url_elem.get_attribute("href")
           except:
               try:
                   # Then try any link in the card
                   url_elem = card.find_element(By.CSS_SELECTOR, "a[href*='publication'], a[href*='wp-content']")
                   url = url_elem.get_attribute("href")
               except:
                   return None

           # Try to extract year from URL or content
           year = None
           year_match = re.search(r'/(\d{4})/', url)
           if year_match:
               year = int(year_match.group(1))
           else:
               year_match = re.search(r'\b(20\d{2})\b', content)
               if year_match:
                   year = int(year_match.group(1))

           return {
               'title': title or 'Untitled Publication',
               'doi': url,
               'content': content,
               'year': year
           }
       except Exception as e:
           logger.error(f"Error extracting publication details: {e}")
           return None

   def fetch_content(self, limit: int = 10) -> List[Dict]:
       """Fetch publications from APHRC website."""
       publications = []
       visited = set()

       try:
           logger.info(f"Accessing URL: {self.base_url}")
           self.driver.get(self.base_url)
           sleep(5)

           while len(publications) < limit:
               try:
                   # Wait for publication cards with multiple selectors
                   publication_cards = self.wait.until(
                       EC.presence_of_all_elements_located(
                           (By.CSS_SELECTOR, ".highlighted-publication, .alm-item, .publication-card, div[class*='publication']")
                       )
                   )
                   
                   logger.info(f"Found {len(publication_cards)} publication cards")
                   
                   for card in publication_cards:
                       if len(publications) >= limit:
                           break
                           
                       try:
                           pub_details = self.extract_publication_details(card)
                           if not pub_details or pub_details['doi'] in visited:
                               continue

                           url = pub_details['doi']
                           logger.info(f"Processing publication URL: {url}")

                           if url.lower().endswith('.pdf'):
                               # For PDFs, use the details extracted from the card
                               publication = {
                                   'title': pub_details['title'],
                                   'doi': url,
                                   'authors': [],
                                   'domains': [],
                                   'type': 'publication',
                                   'publication_year': pub_details['year'],
                                   'summary': pub_details['content'][:1000] if pub_details['content'] else "PDF Publication",
                                   'source': 'website'
                               }
                               publications.append(publication)
                               logger.info(f"Extracted PDF Publication: {pub_details['title'][:100]}...")
                           
                           else:
                               # For web pages, navigate and extract content
                               self.driver.execute_script(f"window.open('{url}');")
                               self.driver.switch_to.window(self.driver.window_handles[-1])
                               sleep(2)

                               try:
                                   content_elems = self.wait.until(
                                       EC.presence_of_all_elements_located(
                                           (By.CSS_SELECTOR, "div.entry-content, article.content-area, div.publication-content, .post-content")
                                       )
                                   )
                                   content_text = "\n".join(elem.text.strip() for elem in content_elems if elem.text.strip())

                                   if content_text:
                                       publication = {
                                           'title': pub_details['title'],
                                           'doi': url,
                                           'authors': [],
                                           'domains': [],
                                           'type': 'publication',
                                           'publication_year': pub_details['year'],
                                           'summary': content_text[:1000],
                                           'source': 'website'
                                       }
                                       publications.append(publication)
                                       logger.info(f"Extracted Web Publication: {pub_details['title'][:100]}...")

                               except Exception as e:
                                   logger.error(f"Error extracting webpage content: {e}")

                               finally:
                                   self.driver.close()
                                   self.driver.switch_to.window(self.driver.window_handles[0])

                           visited.add(url)

                       except Exception as e:
                           logger.error(f"Error processing publication card: {e}")
                           if len(self.driver.window_handles) > 1:
                               self.driver.close()
                               self.driver.switch_to.window(self.driver.window_handles[0])
                           continue

                   # Try to load more
                   try:
                       load_more = self.wait.until(
                           EC.element_to_be_clickable((By.CLASS_NAME, "alm-load-more-btn"))
                       )
                       if load_more.is_displayed():
                           load_more.click()
                           sleep(3)
                       else:
                           break
                   except:
                       logger.info("No more publications to load")
                       break

               except Exception as e:
                   logger.error(f"Error in main loop: {e}")
                   break

       except Exception as e:
           logger.error(f"Error in fetch_content: {e}")
       finally:
           if not publications:
               logger.warning("No publications were found")
           
       return publications

   def _make_request(self, url: str) -> requests.Response:
       """Make HTTP request with error handling."""
       try:
           response = requests.get(url, headers=self.headers, timeout=30)
           response.raise_for_status()
           return response
       except Exception as e:
           logger.error(f"Request failed for {url}: {e}")
           raise

   def close(self):
       """Clean up resources."""
       try:
           if hasattr(self, 'driver'):
               self.driver.quit()
           if hasattr(self.summarizer, 'close'):
               self.summarizer.close()
           self.seen_urls.clear()
           logger.info("WebsiteScraper resources cleaned up")
       except Exception as e:
           logger.error(f"Error in cleanup: {e}")

   def __enter__(self):
       """Context manager entry."""
       return self

   def __exit__(self, exc_type, exc_val, traceback):
       """Context manager exit."""
       self.close()