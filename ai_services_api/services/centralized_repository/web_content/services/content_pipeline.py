
import logging
import os
import requests
from typing import Dict, List
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urljoin
from text_summarizer import TextSummarizer
import re
import warnings

# Suppress XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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
        self.visited_urls = set()  # Reset per run
        logger.info(f"ContentPipeline initialized with batch_size={batch_size}")

    def process_webpage(self, page_data: Dict) -> Dict:
        """Process webpage data, summarize, and assign content_type."""
        url = page_data.get('url', '').lower()
        title = page_data.get('title', '').lower()
        content = page_data.get('content', '')

        if not url:
            logger.warning(f"Skipping page with no URL: {page_data}")
            return {}

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
            aff_elem = soup.find(class_='affiliation') or soup.find('div', string=re.compile('affiliation|organization', re.I))
            affiliation = aff_elem.get_text(strip=True) if aff_elem else 'APHRC'
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
        logger.debug(f"Processed page: url={url}, content_type={content_type}, title={title[:50]}...")
        return page_data

    def scrape_page(self, url: str) -> Dict:
        """Scrape a single webpage and return its data."""
        if url in self.visited_urls:
            logger.debug(f"Skipping already visited URL: {url}")
            return {}
        self.visited_urls.add(url)

        try:
            logger.debug(f"Scraping URL: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Use XML parser for sitemap URLs
            parser = 'xml' if url.endswith('.xml') else 'html.parser'
            soup = BeautifulSoup(response.text, parser)
            
            title = soup.title.string if soup.title else 'Untitled'
            content = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
            
            # Extract navigation text
            nav_text = ''
            nav_elements = (soup.find_all('nav') or 
                           soup.find_all('ul', class_=re.compile('menu|nav|navigation|main-menu|primary-menu', re.I)) or
                           soup.find_all(class_=re.compile('menu|nav|navigation', re.I)))
            if nav_elements:
                nav_links = []
                for nav in nav_elements:
                    links = nav.find_all('a')
                    nav_links.extend(link.get_text(strip=True) for link in links if link.get_text(strip=True))
                nav_text = ' | '.join(set(nav_links))[:1000]
                logger.debug(f"Extracted navigation text for {url}: {nav_text[:100]}...")
            else:
                logger.warning(f"No navigation elements found for {url}")
            
            page_data = {
                'url': url,
                'title': title,
                'content': content[:5000],
                'content_type': 'webpage',
                'raw_html': str(soup),
                'navigation_text': nav_text
            }
            logger.debug(f"Scraped page: url={url}, title={title[:50]}..., content_length={len(content)}, nav_text_length={len(nav_text)}")
            return page_data
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {}

    def run(self) -> Dict:
        """
        Run the scraping pipeline with summarization.
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
            self.visited_urls.clear()  # Reset visited URLs
            
            sitemap_url = 'https://aphrc.org/sitemap.xml'
            urls = []
            try:
                logger.info(f"Fetching sitemap: {sitemap_url}")
                response = self.session.get(sitemap_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'xml')
                urls = [loc.text for loc in soup.find_all('loc') if loc.text.startswith('https://aphrc.org')]
                logger.info(f"Fetched {len(urls)} URLs from sitemap")
            except Exception as e:
                logger.warning(f"Failed to fetch sitemap: {str(e)}. Falling back to start_urls.")
                urls = self.start_urls
                logger.info(f"Using start_urls: {urls}")

            for url in urls[:100]:
                logger.debug(f"Processing URL: {url}")
                page_data = self.scrape_page(url)
                if not page_data:
                    logger.debug(f"No data scraped for {url}")
                    continue
                
                page_data = self.process_webpage(page_data)
                if not page_data.get('url'):
                    logger.debug(f"No valid data after processing for {url}")
                    continue
                
                current_batch.append(page_data)
                processed_pages += 1
                
                try:
                    response = self.session.get(url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = set(
                        urljoin(url, a.get('href'))
                        for a in soup.find_all('a', href=True)
                        if urljoin(url, a.get('href')).startswith('https://aphrc.org')
                    )
                    logger.debug(f"Found {len(links)} links on {url}")
                    
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
                
                except Exception as e:
                    logger.error(f"Error fetching links from {url}: {str(e)}")
                
                if current_batch:
                    webpage_results.append({'batch': current_batch})
                    current_batch = []
            
            results['processed_pages'] = processed_pages
            results['processing_details']['webpage_results'] = webpage_results
            logger.info(f"Content pipeline completed: processed {processed_pages} pages")
            
        except Exception as e:
            logger.error(f"Error running content pipeline: {str(e)}", exc_info=True)
        
        return results

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.session:
                self.session.close()
                logger.info("Requests session cleaned up")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")
