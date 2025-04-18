import logging
import asyncio
from typing import Dict
from datetime import datetime
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
import hashlib
from ai_services_api.services.centralized_repository.database_manager import DatabaseManager
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor
from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class WebContentProcessor:
    def __init__(self, db: DatabaseManager, config: dict = None):
        self.db = db
        self.config = config or {}
        self.skip_publications = self.config.get('skip_publications', False)
        
        # Initialize ContentPipeline with required arguments
        start_urls = [os.getenv('WEBSITE_BASE_URL', 'https://aphrc.org')]
        self.content_pipeline = ContentPipeline(
            start_urls=start_urls,
            scraper=WebsiteScraper(max_browsers=2),  # Reduced for OOM prevention
            pdf_processor=PDFProcessor(),
            db=self.db,
            batch_size=self.config.get('batch_size', 100)
        )
        logger.info("WebContentProcessor initialized with start URLs: %s", start_urls)

    async def process_content(self) -> Dict:
        """Process web content using the content pipeline."""
        try:
            logger.info("Starting web content processing...")
            if self.skip_publications:
                logger.warning("Skipping publication processing due to --skip-publications flag")
                # Optionally process only experts and webpages
                results = await self.content_pipeline.run(exclude_publications=True)
            else:
                results = await self.content_pipeline.run()
            
            if not isinstance(results, dict):
                logger.error("ContentPipeline.run() did not return a dictionary")
                raise ValueError("Invalid pipeline results format")
            
            required_keys = ['processed_pages', 'processing_details']
            if not all(key in results for key in required_keys):
                logger.error(f"Pipeline results missing required keys: {results}")
                raise ValueError("Invalid pipeline results format")
            
            if 'webpage_results' not in results['processing_details']:
                logger.error("Pipeline results missing webpage_results")
                raise ValueError("Invalid pipeline results format")
            
            logger.info(f"Processed {results['processed_pages']} pages, "
                       f"inserted/updated {results['updated_pages']} items, "
                       f"found {results['processed_publications']} publications, "
                       f"{results['processed_experts']} experts, "
                       f"{results['processed_resources']} resources")
            
            return results
        except Exception as e:
            logger.error(f"Error processing web content: {str(e)}", exc_info=True)
            raise
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