
import logging
import asyncio
import os
from typing import Dict, List
from datetime import datetime
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WebContentProcessor:
    """Processes web content by scraping and storing webpage data without embeddings."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 50, processing_checkpoint_hours: int = 24):
        """Initialize processor with database and pipeline."""
        self.db = DatabaseManager()
        self.content_pipeline = ContentPipeline()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.processing_checkpoint_hours = processing_checkpoint_hours
        self._validate_config()
        logger.info(f"WebContentProcessor initialized with max_workers={max_workers}, "
                   f"batch_size={batch_size}, checkpoint_hours={processing_checkpoint_hours}")

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.processing_checkpoint_hours < 1:
            raise ValueError("processing_checkpoint_hours must be positive")

    async def insert_webpage(self, item: Dict) -> int:
        """Insert a webpage into the webpages table and return its ID."""
        try:
            url = item.get('url')
            title = item.get('title', 'Untitled')
            content = item.get('content', '')
            content_type = item.get('content_type', 'webpage')
            
            if not url:
                logger.warning(f"Skipping webpage with missing URL: {item}")
                return 0
            
            # Check if webpage exists
            result = self.db.execute(
                "SELECT id FROM webpages WHERE url = %s",
                (url,)
            )
            if result:
                logger.debug(f"Webpage {url} already exists with ID {result[0][0]}")
                return result[0][0]
            
            # Insert new webpage
            result = self.db.execute(
                """
                INSERT INTO webpages (url, title, content, content_type, last_updated)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (url, title, content, content_type, datetime.utcnow())
            )
            webpage_id = result[0][0]
            logger.info(f"Inserted webpage {url} with ID {webpage_id}")
            return webpage_id
        except Exception as e:
            logger.error(f"Error inserting webpage {url or 'unknown'}: {str(e)}")
            return 0

    async def process_content(self) -> Dict:
        """Process web content and store scraped data in the webpages table."""
        try:
            logger.info("Starting web content processing...")
            
            # Run the content pipeline (synchronous call, no await)
            results = self.content_pipeline.run()
            processed_pages = results.get('processed_pages', 0)
            webpage_results = results.get('processing_details', {}).get('webpage_results', [])
            
            if not webpage_results:
                logger.warning("No webpage results received from pipeline")
                return results
            
            # Process each batch and insert webpages
            updated_pages = 0
            for batch in webpage_results:
                batch_items = batch.get('batch', [])
                for item in batch_items:
                    webpage_id = await self.insert_webpage(item)
                    if webpage_id:
                        updated_pages += 1
            
            logger.info(f"Processed {processed_pages} pages, inserted/updated {updated_pages} webpages")
            
            # Update results with actual insertions
            results['updated_pages'] = updated_pages
            results['processed_publications'] = sum(
                1 for batch in webpage_results
                for item in batch.get('batch', [])
                if item.get('content_type') == 'publication'
            )
            
            return results
        except Exception as e:
            logger.error(f"Error processing web content: {str(e)}", exc_info=True)
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.content_pipeline:
                self.content_pipeline.cleanup()
                logger.info("Content pipeline cleaned up")
            if self.db:
                self.db.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
