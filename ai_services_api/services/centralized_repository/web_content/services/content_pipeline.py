
import logging
from typing import Dict, List
from .web_scraper import WebsiteScraper
from .pdf_processor import PDFProcessor
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ContentPipeline:
    """Manages content processing pipeline for web and PDF content"""
    
    def __init__(self, max_workers: int = 4, webdriver_retries: int = 3):
        self.max_workers = max_workers
        self.web_scraper = WebsiteScraper(max_retries=webdriver_retries)
        self.pdf_processor = PDFProcessor()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_webpage(self, page_data: Dict) -> Dict:
        """Process a single webpage and prepare for storage"""
        try:
            content_type = page_data.get('content_type', 'webpage')
            if 'publication' in page_data.get('url', '').lower() or 'report' in page_data.get('title', '').lower():
                content_type = 'publication'
            
            processed_data = {
                'url': page_data['url'],
                'title': page_data.get('title', 'Untitled'),
                'content': page_data.get('content', ''),
                'content_type': content_type,
                'metadata': page_data.get('metadata', {})
            }
            logger.debug(f"Processed webpage: {page_data['url']}")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing webpage {page_data.get('url', 'unknown')}: {str(e)}")
            return None

    def run(self) -> Dict[str, List]:
        """Run the content processing pipeline"""
        results = {
            'webpage_results': [],
            'pdf_results': []
        }
        try:
            logger.info("Starting content pipeline...")
            webpages = self.web_scraper.scrape()
            
            with self.executor:
                webpage_results = list(self.executor.map(self.process_webpage, webpages))
                results['webpage_results'] = [r for r in webpage_results if r is not None]
                
                pdf_urls = [page['url'] for page in webpages if page['url'].endswith('.pdf')]
                if pdf_urls:
                    pdf_results = list(self.executor.map(self.pdf_processor.process_pdf, pdf_urls))
                    results['pdf_results'] = [r for r in pdf_results if r is not None]
            
            logger.info(f"Pipeline completed: {len(results['webpage_results'])} webpages, {len(results['pdf_results'])} PDFs")
            return results
        except Exception as e:
            logger.error(f"Error in content pipeline: {str(e)}", exc_info=True)
            return results
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up pipeline resources"""
        try:
            self.web_scraper.cleanup()
            self.pdf_processor.cleanup()
            self.executor.shutdown(wait=True)
            logger.info("Pipeline resources cleaned up")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")
