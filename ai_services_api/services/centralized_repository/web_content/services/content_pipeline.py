import logging
import os
import requests
from typing import Dict, List
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urljoin
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper, ScraperConfig
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor
import re
import warnings

import psutil
import gc
import hashlib
from urllib.parse import urljoin
from typing import Dict, List, Set, Optional
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor
import requests
from ai_services_api.services.centralized_repository.ai_summarizer import TextSummarizer
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)




class ContentPipeline:
    def __init__(
        self,
        start_urls: List[str],
        scraper: WebsiteScraper,
        pdf_processor: PDFProcessor,
        db: DatabaseManager,
        batch_size: int = 100
    ):
        self.start_urls = start_urls
        self.scraper = scraper
        self.pdf_processor = pdf_processor
        self.db = db
        self.batch_size = batch_size
        self.visited_urls: Set[str] = set()
        # Add this line to initialize the summarizer
        self.summarizer = TextSummarizer()
        logger.info("ContentPipeline initialized with %d start URLs", len(start_urls))



    def process_webpage(self, page_data: Dict) -> Dict:
        """Process webpage data, summarize, and assign content_type."""
        url = page_data.get('url', page_data.get('doi', '')).lower()
        title = page_data.get('title', '').lower()
        content = page_data.get('summary', page_data.get('content', ''))

        if not url:
            logger.warning(f"Skipping page with no URL: {page_data}")
            return {}

        # Assign content_type
        content_type = 'webpage'
        if url.endswith('.pdf'):
            content_type = 'pdf'
        elif any(keyword in url for keyword in ['/person/', '/staff/', '/team/', '/researcher/']):
            content_type = 'expert'
        elif any(keyword in url or keyword in title for keyword in ['publication', 'report', 'research', 'project', 'article']):
            content_type = 'publication'

        # Heuristic-based expert detection using content
        if content_type == 'webpage' and any(
            keyword in content.lower() for keyword in ['researcher profile', 'staff profile', 'team member', 'expert bio']
        ):
            content_type = 'expert'

        # Extract additional data for experts
        affiliation = None
        contact_email = None
        if content_type == 'expert':
            soup = BeautifulSoup(page_data.get('raw_html', ''), 'html.parser')
            aff_elem = soup.find(class_=re.compile('affiliation|organization|institute', re.I)) or \
                    soup.find('div', string=re.compile('affiliation|organization|institute', re.I))
            affiliation = aff_elem.get_text(strip=True) if aff_elem else 'APHRC'
            email_elem = soup.find('a', href=re.compile(r'^mailto:'))
            contact_email = email_elem.get('href').replace('mailto:', '') if email_elem else None

        # Summarize content for non-PDFs
        if content_type != 'pdf':
            try:
                summary, gemini_content_type = self.summarizer.summarize(title, content)
                if summary and not summary.startswith("Failed") and not summary.startswith("Cannot"):
                    page_data['summary'] = summary
                    if gemini_content_type and content_type != 'expert':
                        content_type = gemini_content_type
                else:
                    logger.warning(f"Failed to summarize {url}: {summary}")
            except Exception as e:
                logger.error(f"Error summarizing {url}: {str(e)}")

        page_data['type'] = content_type
        if content_type == 'expert':
            page_data['affiliation'] = affiliation
            page_data['contact_email'] = contact_email

        # Add validation before returning expert data
        if page_data['type'] == 'expert' and not self.validate_expert(page_data):
            logger.warning(f"Skipping invalid expert data: {page_data.get('url')}")
            return None

        logger.debug(f"Processed page: url={url}, type={content_type}, title={title[:50]}...")
        return page_data

    async def run(self, exclude_publications: bool = False) -> Dict:
        """
        Run the scraping pipeline, saving experts, webpages, and optionally publications every 10 pages.
        Stores navigation text in webpages.navigation_text.
        """
        results = {
            'processed_pages': 0,
            'updated_pages': 0,
            'processed_publications': 0,
            'processed_experts': 0,
            'processed_resources': 0,
            'processing_details': {'webpage_results': []}
        }
        
        try:
            logger.info("Starting content pipeline...")
            webpage_results = []
            current_batch = []
            processed_urls: Set[str] = set()
            to_visit_urls: Set[str] = set(self.start_urls)
            max_pages = 500
            max_depth = 3
            depth_map = {url: 0 for url in to_visit_urls}
            save_batch_size = 10
            
            # Log initial memory usage
            process = psutil.Process()
            logger.debug(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            # Fetch existing URLs
            try:
                query = """
                    SELECT url FROM experts WHERE url IS NOT NULL
                    UNION
                    SELECT url FROM webpages WHERE url IS NOT NULL
                """
                if not exclude_publications:
                    query = "SELECT doi FROM publications WHERE doi IS NOT NULL UNION " + query
                existing_urls = self.db.execute(query)
                processed_urls.update(row[0] for row in existing_urls)
                logger.info(f"Loaded {len(processed_urls)} existing URLs from database")
            except Exception as e:
                logger.error(f"Error loading existing URLs: {str(e)}", exc_info=True)
                processed_urls = set()
            
            # Fetch initial content using WebsiteScraper
            try:
                publications = self.scraper.fetch_content(max_pages=20)
                logger.info(f"Fetched {len(publications)} items from WebsiteScraper")
                current_batch.extend([p for p in publications if p.get('url') not in processed_urls])
                if not exclude_publications:
                    results['processed_publications'] += len([p for p in publications if p.get('type') == 'publication'])
                    results['processed_resources'] += len([p for p in publications if p.get('type') == 'pdf'])
                results['processed_experts'] += len([p for p in publications if p.get('type') == 'expert'])
            except Exception as e:
                logger.error(f"Error fetching publications from WebsiteScraper: {str(e)}", exc_info=True)
            
            # Save initial batch if >= 10 items
            if len(current_batch) >= save_batch_size:
                try:
                    self._save_batch(current_batch, results, exclude_publications)
                    webpage_results.append({'batch': current_batch})
                    results['updated_pages'] += len(current_batch)
                    processed_urls.update(item.get('url') or item.get('doi', '') for item in current_batch)
                    current_batch = []
                    gc.collect()
                    logger.debug(f"Memory usage after saving batch: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                except Exception as e:
                    logger.error(f"Error saving initial batch: {str(e)}", exc_info=True)
            
            # Recursively crawl additional pages
            while to_visit_urls and results['processed_pages'] < max_pages:
                url = to_visit_urls.pop()
                current_depth = depth_map.get(url, 0)
                if url in processed_urls or url in self.visited_urls:
                    logger.debug(f"Skipping already processed URL: {url}")
                    continue
                if current_depth >= max_depth:
                    logger.debug(f"Skipping URL {url} at depth {current_depth} (max_depth={max_depth})")
                    continue
                
                try:
                    page_data = self.scrape_page(url)
                    if not page_data or not page_data.get('url'):
                        logger.debug(f"No valid data scraped for URL: {url}")
                        continue
                    
                    # Extract navigation text
                    try:
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        nav_elements = (
                            soup.select('.nav-menu, .main-nav, .menu, .navbar, nav') +
                            soup.select('.breadcrumb, .breadcrumbs, .nav-path')
                        )
                        nav_texts = [elem.get_text(strip=True) for elem in nav_elements if elem.get_text(strip=True)]
                        page_data['navigation_text'] = ' | '.join(nav_texts)[:1000] if nav_texts else ''
                    except Exception as e:
                        logger.error(f"Error extracting navigation text for {url}: {str(e)}")
                        page_data['navigation_text'] = ''
                    
                    page_data = self.process_webpage(page_data)
                    if not page_data.get('url'):
                        logger.debug(f"No valid processed data for URL: {url}")
                        continue
                    
                    current_batch.append(page_data)
                    self.visited_urls.add(url)
                    results['processed_pages'] += 1
                    
                    # Extract internal links
                    try:
                        links = set(
                            urljoin(url, a.get('href'))
                            for a in soup.find_all('a', href=True)
                            if urljoin(url, a.get('href')).startswith('https://aphrc.org')
                            and not any(ext in a.get('href').lower() for ext in ['.xml', '.jpg', '.png', '.css', '.js'])
                        )
                        logger.debug(f"Found {len(links)} internal links on {url}")
                        for link in links:
                            if link not in processed_urls and link not in self.visited_urls:
                                to_visit_urls.add(link)
                                depth_map[link] = current_depth + 1
                    except Exception as e:
                        logger.error(f"Error fetching links from {url}: {str(e)}")
                    
                    # Save batch every 10 pages
                    if len(current_batch) >= save_batch_size:
                        try:
                            self._save_batch(current_batch, results, exclude_publications)
                            webpage_results.append({'batch': current_batch})
                            results['updated_pages'] += len(current_batch)
                            processed_urls.update(item.get('url') or item.get('doi', '') for item in current_batch)
                            current_batch = []
                            gc.collect()
                            logger.debug(f"Memory usage after saving batch: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                        except Exception as e:
                            logger.error(f"Error saving batch: {str(e)}", exc_info=True)
                    
                    # Log progress
                    if results['processed_pages'] % 50 == 0:
                        logger.info(f"Processed {results['processed_pages']} pages, {len(to_visit_urls)} URLs remaining")
                        logger.debug(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
                    continue
            
            # Save remaining batch
            if current_batch:
                try:
                    self._save_batch(current_batch, results, exclude_publications)
                    webpage_results.append({'batch': current_batch})
                    results['updated_pages'] += len(current_batch)
                    processed_urls.update(item.get('url') or item.get('doi', '') for item in current_batch)
                except Exception as e:
                    logger.error(f"Error saving final batch: {str(e)}", exc_info=True)
            
            # Process PDFs if not excluding publications
            if not exclude_publications:
                pdf_urls = [item['url'] for item in current_batch if item.get('type') == 'pdf' and item['url'] not in processed_urls]
                if pdf_urls:
                    try:
                        pdf_results = self.pdf_processor.process_pdfs(pdf_urls)
                        self._save_batch(pdf_results, results, exclude_publications)
                        webpage_results.append({'batch': pdf_results})
                        results['processed_resources'] += len(pdf_results)
                        results['updated_pages'] += len(pdf_results)
                        logger.info(f"Processed {len(pdf_results)} PDFs")
                    except Exception as e:
                        logger.error(f"Error processing PDFs: {str(e)}", exc_info=True)
            
            results['processing_details']['webpage_results'] = webpage_results
            logger.info(f"Content pipeline completed: processed {results['processed_pages']} pages, "
                       f"{results['processed_publications']} publications, "
                       f"{results['processed_experts']} experts, "
                       f"{results['processed_resources']} PDFs")
            
            # Log final memory usage and scraper metrics
            logger.debug(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            logger.info(f"WebsiteScraper metrics: {self.scraper.get_metrics()}")
            
        except Exception as e:
            logger.error(f"Critical error in content pipeline: {str(e)}", exc_info=True)
        
        return results

    def _save_batch(self, batch: List[Dict], results: Dict, exclude_publications: bool) -> None:
        """Save a batch of items to experts, webpages, and optionally publications."""
        for item in batch:
            try:
                content_type = item.get('type')
                url = item.get('url') or item.get('doi')
                title = item.get('title')
                if not url:
                    logger.warning(f"Skipping item with missing url: {item}")
                    continue
                
                # Save to webpages (content and navigation_text)
                try:
                    existing = self.db.execute("SELECT id FROM webpages WHERE url = %s", (url,))
                    if not existing:
                        content_hash = hashlib.md5(item.get('content', '').encode('utf-8')).hexdigest()
                        self.db.execute("""
                            INSERT INTO webpages (url, content, navigation_text, content_hash, last_updated)
                            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """, (
                            url,
                            item.get('content', '')[:10000],
                            item.get('navigation_text', '')[:1000],
                            content_hash
                        ))
                        logger.debug(f"Saved webpage with navigation text for: {url}")
                except Exception as e:
                    logger.error(f"Error saving webpage for {url}: {str(e)}")
                
                # Save to experts or publications
                if content_type == 'expert':
                    if not title:
                        logger.warning(f"Skipping expert with missing title: {item}")
                        continue
                    existing = self.db.execute("SELECT id FROM experts WHERE url = %s", (url,))
                    if not existing:
                        self.db.execute("""
                            INSERT INTO experts (name, affiliation, contact_email, url, is_active)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            title or 'Unknown Expert',
                            item.get('affiliation', 'APHRC'),
                            item.get('contact_email'),
                            url,
                            True
                        ))
                        results['processed_experts'] += 1
                        logger.debug(f"Saved expert to experts table: {url}")
                
                elif content_type in ['publication', 'pdf'] and not exclude_publications:
                    if not title:
                        logger.warning(f"Skipping publication/pdf with missing title: {item}")
                        continue
                    existing = self.db.execute("SELECT id FROM publications WHERE doi = %s", (url,))
                    if not existing:
                        self.db.execute("""
                            INSERT INTO publications (title, summary, source, type, authors, domains, publication_year, doi, field, subfield)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            title or 'Untitled',
                            item.get('summary', item.get('content', ''))[:1000],
                            item.get('source', 'website'),
                            content_type,
                            item.get('authors', []),
                            item.get('domains', []),
                            item.get('publication_year'),
                            url,
                            item.get('field'),
                            item.get('subfield')
                        ))
                        if content_type == 'publication':
                            results['processed_publications'] += 1
                        else:
                            results['processed_resources'] += 1
                        logger.debug(f"Saved {content_type} to publications table: {url}")
                
            except Exception as e:
                logger.error(f"Error saving item {url or 'unknown'} to database: {str(e)}", exc_info=True)
    
    def scrape_page(self, url: str) -> Dict:
        """Scrape a single webpage using requests and return its data."""
        if url in self.visited_urls:
            logger.debug(f"Skipping already visited URL: {url}")
            return {}
        self.visited_urls.add(url)

        try:
            logger.debug(f"Scraping URL: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Skip non-HTML pages for navigation and content extraction
            if any(ext in url.lower() for ext in ['.xml', '.json']):
                logger.debug(f"Skipping content extraction for non-HTML URL: {url}")
                return {'url': url, 'title': 'Non-HTML', 'content': '', 'type': 'webpage', 'raw_html': ''}

            parser = 'html.parser'
            soup = BeautifulSoup(response.text, parser)
            
            title = soup.title.string if soup.title else 'Untitled'
            content = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
            
            nav_text = ''
            nav_elements = (
                soup.find_all('nav') or 
                soup.find_all('ul', class_=re.compile('menu|nav|navigation|main-menu|primary-menu|footer-menu', re.I)) or
                soup.find_all(class_=re.compile('menu|nav|navigation|site-nav|header-nav', re.I))
            )
            if nav_elements:
                nav_links = []
                for nav in nav_elements:
                    links = nav.find_all('a')
                    nav_links.extend(link.get_text(strip=True) for link in links if link.get_text(strip=True))
                nav_text = ' | '.join(set(nav_links))[:1000]
                logger.debug(f"Extracted navigation text for {url}: {nav_text[:100]}...")
            else:
                logger.debug(f"No navigation elements found for {url}")
            
            page_data = {
                'url': url,
                'title': title,
                'content': content[:5000],
                'type': 'webpage',
                'raw_html': str(soup),
                'navigation_text': nav_text
            }
            logger.debug(f"Scraped page: url={url}, title={title[:50]}..., content_length={len(content)}, nav_text_length={len(nav_text)}")
            return page_data
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {}

    


    def cleanup(self):
        """Clean up resources."""
        try:
            if self.scraper:
                self.scraper.close()
                logger.info("WebsiteScraper cleaned up")
            if self.pdf_processor:
                self.pdf_processor.cleanup()
                logger.info("PDFProcessor cleaned up")
            if self.summarizer and hasattr(self.summarizer, 'close'):
                self.summarizer.close()
                logger.info("TextSummarizer cleaned up")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")