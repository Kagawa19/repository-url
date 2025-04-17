"""
Content pipeline for coordinating web scraping and PDF processing.

WARNING: Do not import WebContentProcessor in this module to avoid circular imports.
"""
import logging
from typing import Dict, List
from .web_scraper import WebsiteScraper
from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class ContentPipeline:
    """Coordinates web scraping and PDF processing"""
    
    def __init__(self, max_workers: int = 4, webdriver_retries: int = 3):
        self.max_workers = max_workers
        self.web_scraper = WebsiteScraper(max_retries=webdriver_retries)
        self.pdf_processor = PDFProcessor()
    
    def run(self) -> Dict:
        """Run the content pipeline"""
        logger.info("Running content pipeline...")
        try:
            webpage_results = self.web_scraper.scrape()
            pdf_results = self.pdf_processor.process_pdfs([item['url'] for item in webpage_results if item['content_type'] == 'webpage'])
            return {
                'webpage_results': webpage_results,
                'pdf_results': pdf_results
            }
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            return {'webpage_results': [], 'pdf_results': []}
    
    def process_webpage(self, page_data: Dict) -> Dict:
        """Process webpage data (e.g., extract metadata, classify content)"""
        try:
            # Simplified processing (add actual logic as needed)
            content_type = page_data.get('content_type', 'webpage')
            if 'publication' in page_data.get('title', '').lower():
                content_type = 'publication'
            return {
                'url': page_data['url'],
                'content': page_data['content'],
                'content_type': content_type,
                'content_id': None  # Set by WebContentProcessor after DB insert
            }
        except Exception as e:
            logger.error(f"Error processing webpage {page_data.get('url', 'unknown')}: {str(e)}")
            return page_data
    
    def cleanup(self):
        """Clean up pipeline resources"""
        try:
            self.web_scraper.cleanup()
            self.pdf_processor.cleanup()
            logger.info("Pipeline resources cleaned up")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")