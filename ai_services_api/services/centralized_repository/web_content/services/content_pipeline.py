
import logging
import os
import requests
from typing import Dict, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ContentPipeline:
    """Manages web content scraping and processing pipeline with summarization."""
    
    def __init__(self, batch_size: int = 50):
        """Initialize pipeline with HTTP session, summarizer, and configuration."""
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; ContentPipeline/1.0)'})
        self.summarizer = TextSummarizer()
        self.start_urls = ['https://aphrc.org']
        self.visited_urls = set()
        logger.info(f"ContentPipeline initialized with batch_size={batch_size}")

    def process_webpage(self, page_data: Dict) -> Dict:
        """Process webpage data, summarize, and assign content_type."""
        url = page_data.get('url', '').lower()
        title = page_data.get('title', '').lower()
        content = page_data.get('content', '')

        # Assign content_type
        if '/person/' in url:
            content_type = 'expert'
        else:
            content_type = 'webpage'
            if any(keyword in url or keyword in title for keyword in ['publication', 'report', 'research', 'project']):
                content_type = 'publication'

        # Extract additional data for experts
        affiliation = None
        contact_email = None
        if content_type == 'expert':
            soup = BeautifulSoup(page_data.get('raw_html', ''), 'html.parser')
            # Extract affiliation (e.g., from a specific class or tag)
            aff_elem = soup.find(class_='affiliation') or soup.find('div', string=re.compile('affiliation|organization', re.I))
            affiliation = aff_elem.get_text(strip=True) if aff_elem else 'APHRC'
            # Extract email (simplified, assumes email is in text or href)
            email_elem = soup.find('a', href=re.compile(r'^mailto:'))
            contact_email = email_elem.get('href').replace('mailto:', '') if email_elem else None

        # Summarize content
        try:
            summary, gemini_content_type = self.summarizer.summarize(title, content)
            if summary and not summary.startswith("Failed") and not summary.startswith("Cannot"):
                page_data['content'] = summary
                if gemini_content_type and content_type != 'expert':
                    content_type = gemini_content_type
            else:
                logger.warning(f"Failed to summarize {url}: {summary}")
        except Exception as e:
            logger.error(f"Error summarizing {url}: {str(e)}")

        page_data['content_type'] = content_type
        if content_type == 'expert':
            page_data['affiliation'] = affiliation
            page_data['contact_email'] = contact_email
        return page_data

    def scrape_page(self, url: str) -> Dict:
        """Scrape a single webpage and return its data."""
        if url in self.visited_urls:
            return {}
        self.visited_urls.add(url)

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.string if soup.title else 'Untitled'
            content = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
            
            return {
                'url': url,
                'title': title,
                'content': content[:5000],
                'content_type': 'webpage',
                'raw_html': str(soup)  # Store raw HTML for expert parsing
            }
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {}

    def run(self) -> Dict:
        """
        Run the scraping pipeline with summarization.
        Returns:
            {
                'processed_pages': int,
                'processing_details': {
                    'webpage_results': [
                        {'batch': [{'url': str, 'title': str, 'content': str, 'content_type': str, 'affiliation': str, 'contact_email': str}, ...]},
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
            
            sitemap_url = 'https://aphrc.org/sitemap.xml'
            try:
                response = self.session.get(sitemap_url, timeout=10)
                response.raise_for_status()
                sitemap = BeautifulSoup(response.text, 'xml')
                urls = [loc.text for loc in sitemap.find_all('loc') if loc.text.startswith('https://aphrc.org')]
            except Exception as e:
                logger.warning(f"Failed to fetch sitemap: {str(e)}. Falling back to start_urls.")
                urls = self.start_urls

            for url in urls[:100]:
                page_data = self.scrape_page(url)
                if not page_data:
                    continue
                
                page_data = self.process_webpage(page_data)
                if not page_data.get('url'):
                    continue
                
                current_batch.append(page_data)
                processed_pages += 1
                
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                links = set(
                    urljoin(url, a.get('href'))
                    for a in soup.find_all('a', href=True)
                    if urljoin(url, a.get('href')).startswith('https://aphrc.org')
                )
                
                for link in list(links)[:5]:
                    page_data = self.scrape_page(link)
                    if not page_data:
                        continue
                    page_data = self.process_webpage(page_data)
                    if not page_data.get('url'):
                        continue
                    current_batch.append(page_data)
                    processed_pages += 1
                    
                    if len(current_batch) >= self.batch_size:
                        webpage_results.append({'batch': current_batch})
                        current_batch = []
                
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
