from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
from ai_services_api.services.centralized_repository.web_content.services.redis_handler import ContentRedisHandler
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
import redis

logger = logging.getLogger(__name__)

class WebContentProcessor:
    """Optimized web content processor with Redis storage and state saving"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 batch_size: int = 50,
                 processing_checkpoint_hours: int = 24,
                 state_file: str = "scrape_state.json"):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.website_url = os.getenv('WEBSITE_URL')
        self.processing_checkpoint_hours = processing_checkpoint_hours
        self.state_file = state_file
        
        self.redis_pool = None
        self.pipeline = None
        self.redis_handler = None
        self.embedding_model = None
        
        self.setup_components()
        self.load_scrape_state()

    def load_scrape_state(self):
        """Load scrape state from JSON file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.scrape_state = json.load(f)
                logger.info(f"Loaded scrape state from {self.state_file}")
            else:
                self.scrape_state = {
                    'last_run': None,
                    'processed_urls': [],
                    'failed_urls': [],
                    'timestamp': None,
                    'content_hashes': {}  # Track content hashes for change detection
                }
                logger.info("Initialized new scrape state")
        except Exception as e:
            logger.error(f"Error loading scrape state: {str(e)}")
            self.scrape_state = {
                'last_run': None,
                'processed_urls': [],
                'failed_urls': [],
                'timestamp': None,
                'content_hashes': {}
            }

    def save_scrape_state(self):
        """Save scrape state to JSON file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.scrape_state, f, indent=2)
            logger.info(f"Saved scrape state to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving scrape state: {str(e)}")

    def setup_components(self):
        """Initialize components with Redis connection pooling"""
        try:
            self.pipeline = ContentPipeline(
                max_workers=self.max_workers,
                webdriver_retries=3
            )
            self.redis_handler = ContentRedisHandler()
            self.embedding_model = EmbeddingModel()
            
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_handler.redis_url,
                max_connections=self.max_workers,
                db=0
            )
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
                if keys:
                    await self.store_url_mappings(
                        [item['url'] for item in changed_items],
                        keys
                    )
                    results['updated'] = len(keys)
                    self.scrape_state['processed_urls'].extend([item['url'] for item in changed_items])
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
        return results

    async def batch_check_content_changes(self, items: List[Dict]) -> List[Dict]:
        """Check for content changes using Redis and scrape state"""
        changed_items = []
        async with redis.Redis(connection_pool=self.redis_pool) as redis_client:
            try:
                for item in items:
                    url = item['url']
                    new_hash = hashlib.md5(item['content'].encode()).hexdigest()
                    stored_hash = self.scrape_state['content_hashes'].get(url)
                    last_modified = self.scrape_state.get('timestamp')
                    
                    # Check Redis for last modified timestamp
                    redis_key = f"url_map:{hashlib.md5(url.encode()).hexdigest()}"
                    stored_data = await redis_client.get(redis_key)
                    if stored_data:
                        stored_data = json.loads(stored_data)
                        last_modified = stored_data.get('updated_at', last_modified)
                    
                    # Convert last_modified to datetime if it exists
                    last_modified_dt = None
                    if last_modified:
                        try:
                            last_modified_dt = datetime.fromisoformat(last_modified)
                        except (ValueError, TypeError):
                            last_modified_dt = None
                    
                    # Determine if content has changed
                    if (stored_hash is None or 
                        stored_hash != new_hash or 
                        (last_modified_dt and 
                         (datetime.now() - last_modified_dt) > timedelta(hours=self.processing_checkpoint_hours))):
                        changed_items.append(item)
                        self.scrape_state['content_hashes'][url] = new_hash
            except Exception as e:
                logger.error(f"Error checking content changes: {str(e)}")
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
            logger.error(f"Error in batch embedding creation: {str(e)}")
            return []

    async def batch_store_embeddings(self, item_embeddings: List[tuple]) -> List[str]:
        """Store embeddings in Redis"""
        keys = []
        async with redis.Redis(connection_pool=self.redis_pool) as redis_client:
            for item, embedding in item_embeddings:
                try:
                    key = f"emb:{hashlib.md5((item['url'] + '_' + datetime.now().isoformat()).encode()).hexdigest()}"
                    data = {
                        'embedding': embedding.tolist(),
                        'url': item['url'],
                        'content_hash': hashlib.md5(item['content'].encode()).hexdigest(),
                        'stored_at': datetime.now().isoformat(),
                        'source_metadata': {
                            'url': item.get('url', 'unknown'),
                            'content_type': item.get('content_type', 'unknown')
                        }
                    }
                    await redis_client.set(key, json.dumps(data), ex=timedelta(days=30))
                    keys.append(key)
                except Exception as e:
                    logger.error(f"Error storing embedding for {item.get('url', 'unknown')}: {str(e)}")
                    self.scrape_state['failed_urls'].append(item.get('url', 'unknown'))
        return keys

    async def store_url_mappings(self, urls: List[str], keys: List[str]):
        """Store URL to embedding key mappings in Redis"""
        async with redis.Redis(connection_pool=self.redis_pool) as redis_client:
            try:
                for url, key in zip(urls, keys):
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    redis_key = f"url_map:{url_hash}"
                    data = {
                        'url': url,
                        'embedding_key': key,
                        'updated_at': datetime.now().isoformat()
                    }
                    await redis_client.set(redis_key, json.dumps(data), ex=timedelta(days=30))
            except Exception as e:
                logger.error(f"Error storing URL mappings: {str(e)}")
                self.scrape_state['failed_urls'].extend(urls)

    async def process_content(self) -> Dict:
        """Enhanced processing method with Redis storage and diagnostics"""
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
                'timestamp': datetime.now().isoformat(),
                'processing_details': {
                    'webpage_results': [],
                    'pdf_results': []
                }
            }

            if not self.website_url:
                logger.error("No WEBSITE_URL provided in environment variables")
                self.scrape_state['last_run'] = results['timestamp']
                self.scrape_state['timestamp'] = results['timestamp']
                self.save_scrape_state()
                return results

            pipeline_results = self.pipeline.run()
            logger.info(f"Pipeline results: {len(pipeline_results['webpage_results'])} webpages, {len(pipeline_results['pdf_results'])} PDFs")

            if not pipeline_results['webpage_results'] and not pipeline_results['pdf_results']:
                logger.warning("No content retrieved from pipeline. Check WEBSITE_URL and WebDriver configuration.")

            for i in range(0, len(pipeline_results['webpage_results']), self.batch_size):
                batch = pipeline_results['webpage_results'][i:i + self.batch_size]
                logger.debug(f"Processing webpage batch {i//self.batch_size + 1}: {len(batch)} items")
                batch_results = await self.process_batch(batch)
                results['processed_pages'] += batch_results['processed']
                results['updated_pages'] += batch_results['updated']
                results['processing_details']['webpage_results'].append({
                    'batch_start_index': i,
                    'batch_size': len(batch),
                    'processed': batch_results['processed'],
                    'updated': batch_results['updated']
                })

            pdf_chunks = []
            for pdf in pipeline_results['pdf_results']:
                for chunk_index, chunk in enumerate(pdf['chunks']):
                    pdf_chunks.append({
                        'url': f"{pdf['url']}#chunk{chunk_index}",
                        'content': chunk,
                        'content_type': 'pdf_chunk'
                    })

            for i in range(0, len(pdf_chunks), self.batch_size):
                batch = pdf_chunks[i:i + self.batch_size]
                logger.debug(f"Processing PDF chunk batch {i//batch_size + 1}: {len(batch)} items")
                batch_results = await self.process_batch(batch)
                results['processed_chunks'] += batch_results['processed']
                results['updated_chunks'] += batch_results['updated']
                results['processing_details']['pdf_results'].append({
                    'batch_start_index': i,
                    'batch_size': len(batch),
                    'processed': batch_results['processed'],
                    'updated': batch_results['updated']
                })

            logger.info(f"""Resumable Web Content Processing Results:
                - Processed Pages: {results['processed_pages']}
                - Updated Pages: {results['updated_pages']}
                - Processed PDF Chunks: {results['processed_chunks']}
                - Updated PDF Chunks: {results['updated_chunks']}
                - Timestamp: {results['timestamp']}
            """)

            self.scrape_state['last_run'] = results['timestamp']
            self.scrape_state['timestamp'] = results['timestamp']
            self.scrape_state['processed_urls'] = list(set(self.scrape_state['processed_urls']))
            self.save_scrape_state()
            return results

        except Exception as e:
            logger.error(f"Error in content processing: {str(e)}")
            self.scrape_state['timestamp'] = datetime.now().isoformat()
            self.save_scrape_state()
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup with Redis and state management"""
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.cleanup()
            if hasattr(self, 'redis_handler'):
                self.redis_handler.close()
            if hasattr(self, 'embedding_model'):
                self.embedding_model.cleanup()
            logger.info("Resources cleaned up successfully")
            self.save_scrape_state()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def close(self):
        """Cleanup method compatible with various usage patterns"""
        self.cleanup()