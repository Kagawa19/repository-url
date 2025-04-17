from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor
from ai_services_api.services.centralized_repository.web_content.embeddings.model_handler import EmbeddingModel
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio
import os
import hashlib
import numpy as np
import json
from ...database.database_setup import insert_embedding, update_scrape_state, get_scrape_state, check_content_changes

logger = logging.getLogger(__name__)

class WebContentProcessor:
    """Optimized web content processor with PostgreSQL storage"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 batch_size: int = 50,
                 processing_checkpoint_hours: int = 24):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.website_url = os.getenv('WEBSITE_URL')
        self.processing_checkpoint_hours = processing_checkpoint_hours
        
        self.pipeline = None
        self.embedding_model = None
        
        self.setup_components()
        self.scrape_state = self.load_scrape_state()

    def load_scrape_state(self):
        """Load scrape state from database"""
        try:
            state = get_scrape_state()
            logger.info("Loaded scrape state from database")
            return state
        except Exception as e:
            logger.error(f"Error loading scrape state: {str(e)}")
            return {
                'last_run': None,
                'processed_urls': [],
                'failed_urls': [],
                'timestamp': None,
                'content_hashes': {}
            }

    def save_scrape_state(self):
        """Save scrape state to database"""
        try:
            update_scrape_state(
                last_run=self.scrape_state['last_run'],
                processed_urls=self.scrape_state['processed_urls'],
                failed_urls=self.scrape_state['failed_urls'],
                content_hashes=self.scrape_state['content_hashes']
            )
            logger.info("Saved scrape state to database")
        except Exception as e:
            logger.error(f"Error saving scrape state: {str(e)}")

    def setup_components(self):
        """Initialize components"""
        try:
            self.pipeline = ContentPipeline(
                max_workers=self.max_workers,
                webdriver_retries=3
            )
            self.embedding_model = EmbeddingModel()
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def process_batch(self, items: List[Dict]) -> Dict[str, int]:
        """Synchronous wrapper for async batch processing"""
        return asyncio.run(self._async_process_batch(items))

    async def _async_process_batch(self, items: List[Dict]) -> Dict[str, int]:
        """Actual async batch processing method"""
        results = {'processed': 0, 'updated': 0}
        try:
            changed_items = await self.batch_check_content_changes(items)
            results['processed'] = len(items)
            if changed_items:
                item_embeddings = await self.batch_create_embeddings(changed_items)
                keys = await self.batch_store_embeddings(item_embeddings)
                results['updated'] = len(keys)
                self.scrape_state['processed_urls'].extend([item['url'] for item in changed_items])
                self.save_scrape_state()  # Save state after successful batch
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            self.scrape_state['failed_urls'].extend([item['url'] for item in items])
            self.save_scrape_state()  # Save state even on failure
        return results

    async def batch_check_content_changes(self, items: List[Dict]) -> List[Dict]:
        """Check for content changes using database and scrape state"""
        changed_items = []
        try:
            for item in items:
                url = item['url']
                new_hash = hashlib.md5(item['content'].encode()).hexdigest()
                stored_hash = self.scrape_state['content_hashes'].get(url)
                last_modified = self.scrape_state.get('timestamp')
                
                if last_modified:
                    try:
                        last_modified_dt = datetime.fromisoformat(last_modified)
                    except (ValueError, TypeError):
                        last_modified_dt = None
                
                if (stored_hash is None or 
                    stored_hash != new_hash or 
                    (last_modified_dt and 
                     (datetime.now() - last_modified_dt) > timedelta(hours=self.processing_checkpoint_hours))):
                    changed_items.append(item)
                    self.scrape_state['content_hashes'][url] = new_hash
        except Exception as e:
            logger.error(f"Error checking content changes: {str(e)}", exc_info=True)
        return changed_items

    async def batch_create_embeddings(self, items: List[Dict]) -> List[tuple]:
        """Optimize embedding creation with additional error handling"""
        try:
            batch_size = min(32, len(items))
            embeddings = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                contents = [item['content'] for item in batch]
                try:
                    batch_embeddings = self.embedding_model.create_embeddings_batch(contents)
                    embeddings.extend(zip(batch, batch_embeddings))
                except Exception as batch_error:
                    logger.warning(f"Error in embedding batch {i//batch_size}: {batch_error}")
            return embeddings
        except Exception as e:
            logger.error(f"Error in batch embedding creation: {str(e)}", exc_info=True)
            return []

    async def batch_store_embeddings(self, item_embeddings: List[tuple]) -> List[str]:
        """Store embeddings in database"""
        keys = []
        for item, embedding in item_embeddings:
            try:
                content_type = item['content_type']
                content_id = item.get('content_id')
                if not content_id:
                    logger.error(f"No content_id for {item.get('url', 'unknown')}")
                    continue
                
                content_hash = hashlib.md5(item['content'].encode()).hexdigest()
                embedding_id = insert_embedding(
                    content_type=content_type,
                    content_id=content_id,
                    embedding=embedding.tolist(),
                    content_hash=content_hash
                )
                keys.append(str(embedding_id))
            except Exception as e:
                logger.error(f"Error storing embedding for {item.get('url', 'unknown')}: {str(e)}", exc_info=True)
                self.scrape_state['failed_urls'].append(item.get('url', 'unknown'))
        return keys

    async def process_content(self) -> Dict:
        """Enhanced processing method with database storage, diagnostics, and retry logic"""
        try:
            logger.info("\n" + "="*50)
            logger.info("Starting Resumable Web Content Processing...")
            logger.info(f"Website URL: {self.website_url}")
            logger.info("="*50)

            results = {
                'processed_pages': 0,
                'updated_pages': 0,
                'processed_chunks': 0,
                'updated_chunks': 0,
                'processed_publications': 0,
                'updated_publications': 0,
                'retry_attempts': 0,
                'retry_successes': 0,
                'timestamp': datetime.now().isoformat(),
                'processing_details': {
                    'webpage_results': [],
                    'pdf_results': [],
                    'publication_results': [],
                    'retry_results': []
                }
            }

            if not self.website_url:
                logger.error("No WEBSITE_URL provided in environment variables")
                self.scrape_state['last_run'] = results['timestamp']
                self.scrape_state['timestamp'] = results['timestamp']
                self.save_scrape_state()
                return results

            # Retry failed URLs from previous run
            failed_urls = self.scrape_state['failed_urls'][:]
            if failed_urls:
                logger.info(f"Retrying {len(failed_urls)} failed URLs")
                self.scrape_state['failed_urls'] = []
                self.save_scrape_state()
                retry_pipeline = ContentPipeline(max_workers=self.max_workers)
                retry_successes = 0
                for url in failed_urls:
                    try:
                        page_data = retry_pipeline.web_scraper.get_page_content(url)
                        if page_data:
                            processed = retry_pipeline.process_webpage(page_data)
                            if processed:
                                batch_results = await self.process_batch([processed])
                                retry_successes += batch_results['processed']
                                results['processed_pages'] += batch_results['processed']
                                results['updated_pages'] += batch_results['updated']
                                if processed['content_type'] == 'publication':
                                    results['processed_publications'] += batch_results['processed']
                                    results['updated_publications'] += batch_results['updated']
                        else:
                            self.scrape_state['failed_urls'].append(url)
                    except Exception as e:
                        logger.error(f"Retry failed for {url}: {str(e)}", exc_info=True)
                        self.scrape_state['failed_urls'].append(url)
                results['retry_attempts'] = len(failed_urls)
                results['retry_successes'] = retry_successes
                results['processing_details']['retry_results'].append({
                    'attempted_urls': len(failed_urls),
                    'successful_urls': retry_successes
                })
                retry_pipeline.cleanup()
                self.save_scrape_state()

            pipeline_results = self.pipeline.run()
            logger.info(f"Pipeline results: {len(pipeline_results['webpage_results'])} webpages, {len(pipeline_results['pdf_results'])} PDFs")

            if not pipeline_results['webpage_results'] and not pipeline_results['pdf_results']:
                logger.warning("No content retrieved from pipeline. Check WEBSITE_URL and WebDriver configuration.")

            # Process webpages and publications
            webpage_items = [item for item in pipeline_results['webpage_results'] if item['content_type'] in ['webpage', 'expert', 'publication']]
            for i in range(0, len(webpage_items), self.batch_size):
                batch = webpage_items[i:i + self.batch_size]
                logger.debug(f"Processing webpage batch {i//self.batch_size + 1}: {len(batch)} items")
                batch_results = await self.process_batch(batch)
                results['processed_pages'] += batch_results['processed']
                results['updated_pages'] += batch_results['updated']
                if any(item['content_type'] == 'publication' for item in batch):
                    results['processed_publications'] += batch_results['processed']
                    results['updated_publications'] += batch_results['updated']
                results['processing_details']['webpage_results'].append({
                    'batch_start_index': i,
                    'batch_size': len(batch),
                    'processed': batch_results['processed'],
                    'updated': batch_results['updated']
                })
                self.save_scrape_state()

            # Process PDF chunks
            pdf_chunks = []
            for pdf in pipeline_results['pdf_results']:
                for chunk_index, chunk in enumerate(pdf['chunks']):
                    pdf_chunks.append({
                        'url': f"{pdf['url']}#chunk{chunk_index}",
                        'content': chunk,
                        'content_type': 'pdf_chunk',
                        'content_id': pdf['chunk_ids'][chunk_index]
                    })

            for i in range(0, len(pdf_chunks), self.batch_size):
                batch = pdf_chunks[i:i + self.batch_size]
                logger.debug(f"Processing PDF chunk batch {i//self.batch_size + 1}: {len(batch)} items")
                batch_results = await self.process_batch(batch)
                results['processed_chunks'] += batch_results['processed']
                results['updated_chunks'] += batch_results['updated']
                results['processing_details']['pdf_results'].append({
                    'batch_start_index': i,
                    'batch_size': len(batch),
                    'processed': batch_results['processed'],
                    'updated': batch_results['updated']
                })
                self.save_scrape_state()

            # Process publication PDFs
            publication_pdfs = [pdf for pdf in pipeline_results['pdf_results'] if pdf.get('content_type') == 'publication']
            for i in range(0, len(publication_pdfs), self.batch_size):
                batch = publication_pdfs[i:i + self.batch_size]
                logger.debug(f"Processing publication PDF batch {i//self.batch_size + 1}: {len(batch)} items")
                batch_results = await self.process_batch([{
                    'url': pdf['url'],
                    'content': ' '.join(pdf['chunks']),
                    'content_type': 'publication',
                    'content_id': pdf['publication_id']
                } for pdf in batch])
                results['processed_publications'] += batch_results['processed']
                results['updated_publications'] += batch_results['updated']
                results['processing_details']['publication_results'].append({
                    'batch_start_index': i,
                    'batch_size': len(batch),
                    'processed': batch_results['processed'],
                    'updated': batch_results['updated']
                })
                self.save_scrape_state()

            logger.info(f"""Resumable Web Content Processing Results:
                - Processed Pages: {results['processed_pages']}
                - Updated Pages: {results['updated_pages']}
                - Processed PDF Chunks: {results['processed_chunks']}
                - Updated PDF Chunks: {results['updated_chunks']}
                - Processed Publications: {results['processed_publications']}
                - Updated Publications: {results['updated_publications']}
                - Retry Attempts: {results['retry_attempts']}
                - Retry Successes: {results['retry_successes']}
                - Timestamp: {results['timestamp']}
            """)

            self.scrape_state['last_run'] = results['timestamp']
            self.scrape_state['timestamp'] = results['timestamp']
            self.scrape_state['processed_urls'] = list(set(self.scrape_state['processed_urls']))
            self.save_scrape_state()
            return results

        except Exception as e:
            logger.error(f"Error in content processing: {str(e)}", exc_info=True)
            self.scrape_state['timestamp'] = datetime.now().isoformat()
            self.scrape_state['failed_urls'] = list(set(self.scrape_state['failed_urls']))
            self.save_scrape_state()
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup with database state management"""
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.cleanup()
            if hasattr(self, 'embedding_model'):
                self.embedding_model.cleanup()
            logger.info("Resources cleaned up successfully")
            self.save_scrape_state()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            self.save_scrape_state()

    def close(self):
        self.cleanup()