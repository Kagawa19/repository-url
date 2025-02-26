# services/web_content/content_pipeline.py

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
from ai_services_api.services.centralized_repository.web_content.embeddings.model_handler import EmbeddingModel
from ai_services_api.services.centralized_repository.database_setup import get_db_cursor




logger = logging.getLogger(__name__)

class ContentPipeline:
    """
    Pipeline for processing web content including webpage scraping,
    PDF processing, and content preparation.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.visited_urls: Set[str] = set()
        self.pdf_links: Set[str] = set()
        self.setup_components()

    def setup_components(self):
        """Initialize all required components"""
        try:
            self.web_scraper = WebsiteScraper()
            self.pdf_processor = PDFProcessor()
            self.embedding_model = EmbeddingModel()
            self.text_cleaner = TextCleaner()
            logger.info("Successfully initialized pipeline components")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise

    def validate_url(self, url: str) -> bool:
        """Validate URL format and allowed domains"""
        try:
            result = urlparse(url)
            base_domain = os.getenv('WEBSITE_URL')
            if not base_domain:
                raise ValueError("WEBSITE_URL environment variable not set")
            base_domain = urlparse(base_domain).netloc
            return all([
                result.scheme in ['http', 'https'],
                result.netloc.endswith(base_domain),
                len(url) < 2048  # Standard URL length limit
            ])
        except Exception:
            return False

    def process_webpage(self, page_data: Dict) -> Optional[Dict]:
        """Process a single webpage"""
        try:
            if not self.validate_url(page_data['url']):
                logger.error(f"Invalid URL format: {page_data['url']}")
                return None

            # Clean and process content
            cleaned_content = self.text_cleaner.clean_text(page_data['content'])
            
            if not cleaned_content.strip():
                logger.warning(f"Empty content for URL: {page_data['url']}")
                return None

            # Create hash for content
            content_hash = hashlib.sha256(cleaned_content.encode()).hexdigest()

            # Generate metadata
            metadata = {
                'url': page_data['url'],
                'title': page_data.get('title', ''),
                'nav_links': page_data.get('nav_links', []),
                'pdf_links': page_data.get('pdf_links', []),
                'last_modified': page_data.get('last_modified'),
                'scrape_timestamp': datetime.now().isoformat()
            }

            # Update PDF links set
            if page_data.get('pdf_links'):
                self.pdf_links.update(page_data['pdf_links'])

            return {
                'url': page_data['url'],
                'title': page_data.get('title', ''),
                'content': cleaned_content,
                'content_hash': content_hash,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
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

            # Clean and process content
            cleaned_chunks = []
            for chunk in pdf_data['chunks']:
                cleaned_chunk = self.text_cleaner.clean_pdf_text(chunk)
                if cleaned_chunk.strip():  # Only keep non-empty chunks
                    cleaned_chunks.append(cleaned_chunk)

            if not cleaned_chunks:
                logger.warning(f"No valid content in PDF: {pdf_data['url']}")
                return None

            # Create hash for entire PDF content
            full_content = ' '.join(cleaned_chunks)
            content_hash = hashlib.sha256(full_content.encode()).hexdigest()

            # Generate metadata
            metadata = {
                'url': pdf_data['url'],
                'file_path': pdf_data.get('file_path', ''),
                'total_pages': pdf_data.get('total_pages', 0),
                'total_chunks': len(cleaned_chunks),
                'file_size': pdf_data.get('file_size', 0),
                'scrape_timestamp': datetime.now().isoformat()
            }

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

            # Scrape website pages
            pages_data = self.web_scraper.scrape_site()
            logger.info(f"Scraped {len(pages_data)} pages")

            # Process webpages in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit webpage processing tasks
                future_to_page = {
                    executor.submit(self.process_webpage, page): page 
                    for page in pages_data
                }
                
                # Collect webpage results
                for future in as_completed(future_to_page):
                    page = future_to_page[future]
                    try:
                        result = future.result()
                        if result:
                            results['webpage_results'].append(result)
                    except Exception as e:
                        logger.error(f"Failed to process page {page.get('url', 'unknown')}: {str(e)}")

            # Process PDFs in parallel
            pdf_data = self.pdf_processor.process_pdfs(list(self.pdf_links))
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit PDF processing tasks
                future_to_pdf = {
                    executor.submit(self.process_pdf, pdf): pdf 
                    for pdf in pdf_data
                }
                
                # Collect PDF results
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        result = future.result()
                        if result:
                            results['pdf_results'].append(result)
                    except Exception as e:
                        logger.error(f"Failed to process PDF {pdf.get('url', 'unknown')}: {str(e)}")

            # Update results
            results.update({
                'total_webpages': len(results['webpage_results']),
                'total_pdf_chunks': sum(len(pdf['chunks']) for pdf in results['pdf_results']),
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            })

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
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()