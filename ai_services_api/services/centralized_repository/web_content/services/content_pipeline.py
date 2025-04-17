
import logging
import os
import requests
from typing import Dict, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ContentPipeline:
    """Manages web content scraping and processing pipeline."""
    
    def __init__(self, batch_size: int = 50):
        """Initialize pipeline with HTTP session and configuration."""
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; ContentPipeline/1.0)'})
        # Example starting URLs (replace with actual sitemap or config)
        self.start_urls = ['https://aphrc.org', 'https://aphrc.org/projects/', 'https://aphrc.org/publications/']
        logger.info(f"ContentPipeline initialized with batch_size={batch_size}")

    def process_webpage(self, page_data: Dict) -> Dict:
        """Process webpage data and assign content_type."""
        content_type = 'webpage'
        url = page_data.get('url', '').lower()
        title = page_data.get('title', '').lower()
        if any(keyword in url or keyword in title for keyword in ['publication', 'report', 'research', 'project']):
            content_type = 'publication'
        page_data['content_type'] = content_type
        return page_data

    def scrape_page(self, url: str) -> Dict:
        """Scrape a single webpage and return its data."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.string if soup.title else 'Untitled'
            # Extract main content (adjust selector as needed)
            content = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
            
            return {
                'url': url,
                'title': title,
                'content': content[:5000],  # Truncate for database
                'content_type': 'webpage'  # Will be updated by process_webpage
            }
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {}

    def run(self) -> Dict:
        """
        Run the scraping pipeline and return processed webpage data.
        Returns:
            {
                'processed_pages': int,
                'processing_details': {
                    'webpage_results': [
                        {'batch': [{'url': str, 'title': str, 'content': str, 'content_type': str}, ...]},
                        ...
                    ]
                }
            }
        """
        results = {
            'processed_pages': 0,
            'processing_details': {'webpage_results': []}
        }
        
        try:
            logger.info("Starting content pipeline...")
            webpage_results = []
            current_batch = []
            processed_pages = 0
            
            # Scrape starting URLs and discover links
            for url in self.start_urls:
                page_data = self.scrape_page(url)
                if not page_data:
                    continue
                
                # Process webpage (assign content_type)
                page_data = self.process_webpage(page_data)
                current_batch.append(page_data)
                processed_pages += 1
                
                # Discover additional links on the page
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                links = set(
                    urljoin(url, a.get('href'))
                    for a in soup.find_all('a', href=True)
                    if urljoin(url, a.get('href')).startswith('https://aphrc.org')
                )
                
                # Scrape up to 10 additional links per page
                for link in list(links)[:10]:
                    page_data = self.scrape_page(link)
                    if not page_data:
                        continue
                    page_data = self.process_webpage(page_data)
                    current_batch.append(page_data)
                    processed_pages += 1
                    
                    # Create a new batch if batch_size is reached
                    if len(current_batch) >= self.batch_size:
                        webpage_results.append({'batch': current_batch})
                        current_batch = []
                
                # Add any remaining items to a batch
                if current_batch:
                    webpage_results.append({'batch': current_batch})
                    current_batch = []
            
            results['processed_pages'] = processed_pages
            results['processing_details']['webpage_results'] = webpage_results
            logger.info(f"Content pipeline completed: processed {processed_pages} pages")
            
        except Exception as e:
            logger.error(f"Error running content pipeline: {str(e)}")
        
        return results

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.session:
                self.session.close()
                logger.info("Requests session cleaned up")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")
