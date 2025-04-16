import logging
from typing import List, Dict, Set, Optional
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from urllib.parse import urlparse
from ai_services_api.services.centralized_repository.web_content.services.redis_handler import ContentRedisHandler
from ai_services_api.services.centralized_repository.web_content.utils.text_cleaner import TextCleaner
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class ContentPipeline:
    """
    Pipeline for processing web content including webpage scraping,
    PDF processing, and content preparation.
    """
    
    def __init__(self, max_workers: int = 4, webdriver_retries: int = 3):
        self.max_workers = max_workers
        self.webdriver_retries = webdriver_retries
        self.visited_urls: Set[str] = set()
        self.pdf_links: Set[str] = set()
        self.website_url = os.getenv('WEBSITE_URL')
        self.setup_components()

    def setup_components(self):
        """Initialize all required components"""
        try:
            if not self.website_url:
                raise ValueError("WEBSITE_URL environment variable not set")
            self.web_scraper = WebsiteScraper()
            self.pdf_processor = PDFProcessor()
            self.text_cleaner = TextCleaner()
            logger.info(f"Pipeline initialized with WEBSITE_URL: {self.website_url}")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise

    def validate_url(self, url: str) -> bool:
        """Validate URL format and allowed domains"""
        try:
            result = urlparse(url)
            base_domain = urlparse(self.website_url).netloc
            return all([
                result.scheme in ['http', 'https'],
                result.netloc.endswith(base_domain),
                len(url) < 2048
            ])
        except Exception:
            return False

    def process_webpage(self, page_data: Dict) -> Optional[Dict]:
        """Process a single webpage"""
        try:
            if not self.validate_url(page_data['url']):
                logger.error(f"Invalid URL format: {page_data['url']}")
                return None

            cleaned_content = self.text_cleaner.clean_text(page_data['content'])
            if not cleaned_content.strip():
                logger.warning(f"Empty content for URL: {page_data['url']}")
                return None

            content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()
            metadata = {
                'url': page_data['url'],
                'title': page_data.get('title', ''),
                'nav_links': page_data.get('nav_links', []),
                'pdf_links': page_data.get('pdf_links', []),
                'last_modified': page_data.get('last_modified'),
                'scrape_timestamp': datetime.now().isoformat()
            }

            if page_data.get('pdf_links'):
                self.pdf_links.update(page_data['pdf_links'])

            logger.debug(f"Processed webpage {page_data['url']}: hash {content_hash}")
            return {
                'url': page_data['url'],
                'title': page_data.get('title', ''),
                'content': cleaned_content,
                'content_hash': content_hash,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'content_type': 'webpage'
            }

        except Exception as e:
            logger.error(f"Error processing webpage {page_data.get('url', 'unknown')}: {str(e)}")
            return None

    def process_pdf(self, pdf_data: Dict) -> Optional[Dict]:
        """Process a single PDF document"""
        try:
            if not self.validate_url(pdf_data['url']):
                logger.error(f"Invalid PDF URL: {pdf_data['url']}")
                return None

            cleaned_chunks = []
            for chunk in pdf_data['chunks']:
                cleaned_chunk = self.text_cleaner.clean_pdf_text(chunk)
                if cleaned_chunk.strip():
                    cleaned_chunks.append(cleaned_chunk)

            if not cleaned_chunks:
                logger.warning(f"No valid content in PDF: {pdf_data['url']}")
                return None

            full_content = ' '.join(cleaned_chunks)
            content_hash = hashlib.md5(full_content.encode()).hexdigest()

            metadata = {
                'url': pdf_data['url'],
                'file_path': pdf_data.get('file_path', ''),
                'total_pages': pdf_data.get('total_pages', 0),
                'total_chunks': len(cleaned_chunks),
                'file_size': pdf_data.get('file_size', 0),
                'scrape_timestamp': datetime.now().isoformat()
            }

            logger.debug(f"Processed PDF {pdf_data['url']}: {len(cleaned_chunks)} chunks, hash {content_hash}")
            return {
                'url': pdf_data['url'],
                'chunks': cleaned_chunks,
                'content_hash': content_hash,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_data.get('url', 'unknown')}: {str(e)}")
            return None

    def run(self) -> Dict:
        """Run the complete content processing pipeline"""
        try:
            results = {
                'webpage_results': [],
                'pdf_results': [],
                'status': 'initialized',
                'timestamp': datetime.now().isoformat()
            }

            pages_data = self.web_scraper.scrape_site()
            logger.info(f"Scraped {len(pages_data)} pages")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_page = {
                    executor.submit(self.process_webpage, page): page 
                    for page in pages_data
                }
                for future in as_completed(future_to_page):
                    page = future_to_page[future]
                    try:
                        result = future.result()
                        if result:
                            results['webpage_results'].append(result)
                    except Exception as e:
                        logger.error(f"Failed to process page {page.get('url', 'unknown')}: {str(e)}")

            pdf_data = self.pdf_processor.process_pdfs(list(self.pdf_links))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {
                    executor.submit(self.process_pdf, pdf): pdf 
                    for pdf in pdf_data
                }
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        result = future.result()
                        if result:
                            results['pdf_results'].append(result)
                    except Exception as e:
                        logger.error(f"Failed to process PDF {pdf.get('url', 'unknown')}: {str(e)}")

            results.update({
                'total_webpages': len(results['webpage_results']),
                'total_pdf_chunks': sum(len(pdf['chunks']) for pdf in results['pdf_results']),
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            })

            if not results['webpage_results'] and not results['pdf_results']:
                logger.warning("No content processed. Check WEBSITE_URL and WebDriver.")
            
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.web_scraper.close()
            self.pdf_processor.cleanup()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()