import logging
import asyncio
from typing import Dict
from datetime import datetime
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WebContentProcessor:
    """Processes web content by scraping and storing webpage data without embeddings."""
    
    def __init__(self, max_workers: int = 5, batch_size: int = 100, processing_checkpoint_hours: int = 24):
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
            url = item.get('url') or item.get('doi')
            title = item.get('title', 'Untitled')
            content = item.get('content', '') or item.get('summary', '')
            content_type = item.get('content_type', 'webpage') or item.get('type', 'webpage')
            navigation_text = item.get('navigation_text', '')
            affiliation = item.get('affiliation')
            contact_email = item.get('contact_email')
            authors = item.get('authors', [])
            domains = item.get('domains', [])
            publication_year = item.get('publication_year')
            
            if not url:
                logger.warning(f"Skipping webpage with missing URL: {item}")
                return 0
            
            if content_type == 'expert':
                logger.debug(f"Inserting expert profile: {url}")
                result = self.db.execute(
                    """
                    INSERT INTO experts (url, name, affiliation, contact_email)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                        name = EXCLUDED.name,
                        affiliation = EXCLUDED.affiliation,
                        contact_email = EXCLUDED.contact_email
                    RETURNING id
                    """,
                    (url, title, affiliation, contact_email)
                )
                expert_id = result[0][0]
                logger.info(f"Inserted/updated expert {url} with ID {expert_id}")
                return expert_id
            
            if content_type == 'publication':
                exists = self.db.exists_publication(title=title, doi=url)
                if exists:
                    logger.debug(f"Publication already exists: {title}")
                    return exists
                success = self.db.add_publication(
                    title=title,
                    doi=url,
                    authors=authors,
                    domains=domains,
                    publication_year=publication_year,
                    summary=content,
                    source='website'
                )
                if success:
                    result = self.db.execute("SELECT id FROM publications WHERE doi = %s", (url,))
                    publication_id = result[0][0]
                    logger.info(f"Inserted publication {url} with ID {publication_id}")
                    return publication_id
                return 0
            
            # Generate content_hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest() if content else ''
            
            logger.debug(f"Attempting to insert webpage: url={url}, title={title[:50]}..., content_type={content_type}, content_length={len(content)}, navigation_text_length={len(navigation_text)}, content_hash={content_hash}")
            
            result = self.db.execute(
                "SELECT id FROM webpages WHERE url = %s",
                (url,)
            )
            if result:
                logger.debug(f"Webpage {url} already exists with ID {result[0][0]}")
                return result[0][0]
            
            result = self.db.execute(
                """
                INSERT INTO webpages (url, title, content, content_type, navigation_text, last_updated, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (url, title, content, content_type, navigation_text, datetime.utcnow(), content_hash)
            )
            webpage_id = result[0][0]
            logger.info(f"Inserted webpage {url} with ID {webpage_id}")
            return webpage_id
        except Exception as e:
            logger.error(f"Error inserting webpage {url or 'unknown'}: {str(e)}", exc_info=True)
            return 0

    async def process_content(self) -> Dict:
        """
        Process web content and store scraped data in the webpages table.
        """
        try:
            logger.info("Starting web content processing...")
            
            results = self.content_pipeline.run()
            
            if not isinstance(results, dict):
                logger.error("ContentPipeline.run() did not return a dictionary")
                raise ValueError("Invalid pipeline results format")
            
            processed_pages = results.get('processed_pages', 0)
            webpage_results = results.get('processing_details', {}).get('webpage_results', [])
            
            if not webpage_results:
                logger.warning("No webpage results received from pipeline")
                return {
                    'processed_pages': processed_pages,
                    'updated_pages': 0,
                    'processed_publications': 0,
                    'processed_experts': 0,
                    'processed_resources': 0,
                    'processing_details': {'webpage_results': []}
                }
            
            updated_pages = 0
            for batch in webpage_results:
                batch_items = batch.get('batch', [])
                if not isinstance(batch_items, list):
                    logger.warning(f"Invalid batch format: {batch}")
                    continue
                for item in batch_items:
                    if not isinstance(item, dict):
                        logger.warning(f"Invalid item format: {item}")
                        continue
                    logger.debug(f"Processing item: {item.get('url', item.get('doi', 'unknown'))}")
                    item_id = await self.insert_webpage(item)
                    if item_id:
                        updated_pages += 1
            
            processed_publications = sum(
                1 for batch in webpage_results
                for item in batch.get('batch', [])
                if isinstance(item, dict) and item.get('content_type', item.get('type', '')) == 'publication'
            )
            processed_experts = sum(
                1 for batch in webpage_results
                for item in batch.get('batch', [])
                if isinstance(item, dict) and item.get('content_type', item.get('type', '')) == 'expert'
            )
            processed_resources = sum(
                1 for batch in webpage_results
                for item in batch.get('batch', [])
                if isinstance(item, dict) and item.get('content_type', item.get('type', '')) == 'pdf'
            )
            
            logger.info(f"Processed {processed_pages} pages, inserted/updated {updated_pages} items, "
                       f"found {processed_publications} publications, {processed_experts} experts, {processed_resources} resources")
            
            results['updated_pages'] = updated_pages
            results['processed_publications'] = processed_publications
            results['processed_experts'] = processed_experts
            results['processed_resources'] = processed_resources
            
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