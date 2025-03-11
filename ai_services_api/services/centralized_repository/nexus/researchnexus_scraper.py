import os
import re
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.text_processor import safe_str

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ResearchNexusScraper:
    def __init__(self, summarizer: Optional[TextSummarizer] = None):
        """Initialize scraper with database integration."""
        self.summarizer = summarizer or TextSummarizer()
        self.driver = None
        self._initialize_chrome()
        logger.info("ResearchNexusScraper initialized")

    def _initialize_chrome(self):
        """Initialize Chrome with headless mode."""
        try:
            options = Options()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(30)
            logger.info("Chrome initialized successfully")
        except Exception as e:
            logger.error(f"Chrome initialization failed: {e}")
            raise

    def fetch_content(self, search_term: str = "aphrc", max_pages: Optional[int] = None) -> List[Dict]:
        """
        Fetch publications data in format compatible with resources_resource table.
        
        Args:
            search_term (str): Keyword to search for
            max_pages (int, optional): Maximum number of result pages to scrape (None for unlimited)
        """
        publications = []
        try:
            # Set up download directory for exports
            download_dir = os.path.join(os.getcwd(), 'temp_downloads')
            os.makedirs(download_dir, exist_ok=True)
            
            # Configure Chrome download behavior
            self.driver.execute_cdp_cmd('Page.setDownloadBehavior', {
                'behavior': 'allow',
                'downloadPath': download_dir
            })
            
            # Tracking for pagination
            current_page = 1
            
            while max_pages is None or current_page <= max_pages:
                # Navigate to search page with larger result limit
                url = f"https://research-nexus.net/research/?kwd={search_term}&page={current_page}&limit=100"
                self.driver.get(url)
                
                # Step 1: Wait for page load and results
                outer_container = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.ID, "search-outer-container"))
                )
                
                WebDriverWait(self.driver, 20).until_not(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#search-outer-container.d-none"))
                )
                
                # Step 2: Wait for actual results to appear
                results = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "ar-search-result"))
                )
                
                if not results:
                    logger.warning(f"No search results found on page {current_page}")
                    break
                    
                # Step 3: Deal with potential overlays
                self.driver.execute_script("""
                    // Remove any overlay elements
                    document.querySelectorAll('.elementor-location-header').forEach(e => e.remove());
                    document.querySelectorAll('.elementor-element').forEach(e => {
                        if(e.style.position === 'fixed') e.remove();
                    });
                """)
                
                # Step 4: Find and prepare export button
                try:
                    export_button = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "export-csv"))
                    )
                    
                    # Scroll to top first
                    self.driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(1)
                    
                    # Make sure button is visible and clickable
                    self.driver.execute_script("""
                        arguments[0].style.position = 'relative';
                        arguments[0].style.zIndex = '99999';
                    """, export_button)
                    
                    # Try clicking using JavaScript
                    self.driver.execute_script("arguments[0].click();", export_button)
                    
                except Exception as e:
                    logger.error(f"Failed to click export button: {e}")
                    # Alternative approach - try direct actions
                    try:
                        actions = webdriver.ActionChains(self.driver)
                        actions.move_to_element(export_button).click().perform()
                    except Exception as e2:
                        logger.error(f"ActionChains click failed: {e2}")
                        raise
                
                # Step 5: Wait for and process download
                timeout = time.time() + 30
                csv_path = None
                
                while time.time() < timeout:
                    csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
                    if csv_files:
                        csv_path = os.path.join(download_dir, csv_files[0])
                        break
                    time.sleep(1)
                    
                if not csv_path:
                    raise Exception("Download timeout - no CSV file found")
                
                # Ensure file is completely downloaded
                time.sleep(2)
                    
                # Process CSV
                df = pd.read_csv(csv_path)

                def get_doi_from_paper_page(paper_url):
                    """Helper function to extract DOI from individual paper pages"""
                    try:
                        self.driver.get(paper_url)
                        # Wait for DOI element and extract it
                        doi_element = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-doi]"))
                        )
                        return doi_element.get_attribute("data-doi")
                    except Exception as e:
                        logger.error(f"Error getting DOI from {paper_url}: {e}")
                        return None
                
                for _, row in df.iterrows():
                    try:
                        title = safe_str(row['Title'])
                        
                        # Fixed ID processing
                        paper_id = str(int(row['ID'])) if pd.notna(row.get('ID')) else None
                        
                        # Construct and visit paper URL to get DOI
                        if paper_id:
                            paper_url = f"https://research-nexus.net/paper/{paper_id}/"
                            doi = get_doi_from_paper_page(paper_url)
                        else:
                            doi = None
                            logger.warning(f"No paper ID found for: {title}")
                        
                        authors = []
                        if pd.notna(row.get('Authors')):
                            author_count = row['Authors']
                            if isinstance(author_count, (int, float)):
                                authors = [f"Author {i+1}" for i in range(int(author_count))]
                        
                        domains = []
                        if pd.notna(row.get('Countries')):
                            locations = str(row['Countries'])
                            domains = [loc.strip() for loc in locations.split(',') if loc.strip()]
                        
                        abstract = safe_str(row.get('Excerpt', f"Research about {title}"))
                        
                        try:
                            summary = self.summarizer.summarize(title, abstract)
                        except Exception as e:
                            logger.error(f"Summary generation failed: {e}")
                            summary = abstract[:500] if abstract else f"Research about {title}"
                        
                        publication = {
                            'title': title,
                            'doi': doi,
                            'summary': summary,
                            'source': 'researchnexus',
                            'type': 'publication',
                            'authors': authors,
                            'domains': domains,
                            'citations': int(row.get('Citations', 0)),
                            'scrape_date': datetime.now().isoformat()
                        }
                        
                        publications.append(publication)
                        logger.info(f"Processed publication: {title}")
                        
                    except Exception as e:
                        logger.error(f"Error processing publication row: {e}")
                        continue
                    
                # Cleanup downloaded CSV
                try:
                    os.remove(csv_path)
                except Exception as e:
                    logger.error(f"Error removing CSV file: {e}")
                
                # Check for next page
                try:
                    next_page_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination .next")
                    if next_page_button and next_page_button.is_enabled():
                        current_page += 1
                    else:
                        break
                except NoSuchElementException:
                    # No more pages
                    break
        
        except Exception as e:
            logger.error(f"Error in fetch_content: {e}")
            raise
            
        finally:
            if len(publications) == 0:
                logger.warning("No publications were fetched")
                
            # Clean up download directory
            try:
                if os.path.exists(download_dir):
                    os.rmdir(download_dir)
            except Exception as e:
                logger.error(f"Error cleaning up download directory: {e}")
                
        return publications

            def get_doi_from_paper_page(paper_url):
                """Helper function to extract DOI from individual paper pages"""
                try:
                    self.driver.get(paper_url)
                    # Wait for DOI element and extract it
                    doi_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-doi]"))
                    )
                    return doi_element.get_attribute("data-doi")
                except Exception as e:
                    logger.error(f"Error getting DOI from {paper_url}: {e}")
                    return None
            
            for _, row in df.iterrows():
                if len(publications) >= limit:
                    break
                    
                try:
                    title = safe_str(row['Title'])
                    
                    # Fixed ID processing
                    paper_id = str(int(row['ID'])) if pd.notna(row.get('ID')) else None
                    
                    # Construct and visit paper URL to get DOI
                    if paper_id:
                        paper_url = f"https://research-nexus.net/paper/{paper_id}/"
                        doi = get_doi_from_paper_page(paper_url)
                    else:
                        doi = None
                        logger.warning(f"No paper ID found for: {title}")
                    
                    authors = []
                    if pd.notna(row.get('Authors')):
                        author_count = row['Authors']
                        if isinstance(author_count, (int, float)):
                            authors = [f"Author {i+1}" for i in range(int(author_count))]
                    
                    domains = []
                    if pd.notna(row.get('Countries')):
                        locations = str(row['Countries'])
                        domains = [loc.strip() for loc in locations.split(',') if loc.strip()]
                    
                    abstract = safe_str(row.get('Excerpt', f"Research about {title}"))
                    
                    try:
                        summary = self.summarizer.summarize(title, abstract)
                    except Exception as e:
                        logger.error(f"Summary generation failed: {e}")
                        summary = abstract[:500] if abstract else f"Research about {title}"
                    
                    publication = {
                        'title': title,
                        'doi': doi,
                        'summary': summary,
                        'source': 'researchnexus',
                        'type': 'publication',
                        'authors': authors,
                        'domains': domains,
                        'citations': int(row.get('Citations', 0)),
                        'scrape_date': datetime.now().isoformat()
                    }
                    
                    publications.append(publication)
                    logger.info(f"Processed publication: {title}")
                    
                except Exception as e:
                    logger.error(f"Error processing publication row: {e}")
                    continue
                    
            # Cleanup
            try:
                os.remove(csv_path)
                os.rmdir(download_dir)
            except Exception as e:
                logger.error(f"Error cleaning up files: {e}")
                
        except Exception as e:
            logger.error(f"Error in fetch_content: {e}")
            raise
            
        finally:
            if len(publications) == 0:
                logger.warning("No publications were fetched")
                
        return publications

    def _generate_doi(self, title: str) -> str:
        """Generate a consistent DOI from title."""
        hash_object = hashlib.sha256(title.encode())
        return f"10.5555/researchnexus-{hash_object.hexdigest()[:16]}"

    def close(self):
        """Clean up resources."""
        try:
            if self.driver:
                self.driver.quit()
            logger.info("ResearchNexusScraper resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing resources: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()