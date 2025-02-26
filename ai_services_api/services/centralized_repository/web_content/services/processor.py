

from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
from ai_services_api.services.centralized_repository.web_content.services.redis_handler import ContentRedisHandler
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor
from ai_services_api.services.centralized_repository.web_content.embeddings.model_handler import EmbeddingModel
from ai_services_api.services.centralized_repository.web_content.database.database_setup import ContentTracker, get_db_cursor, DatabaseInitializer
import logging
from typing import Dict, Optional, List, Set
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib
from functools import partial
from itertools import islice
import numpy as np
import json
import psycopg2.pool

# Update Redis imports
import redis

# Import the DATABASE_CONFIG 
from ..config.settings import DATABASE_CONFIG

from ai_services_api.services.centralized_repository.web_content.services.content_pipeline import ContentPipeline
from ai_services_api.services.centralized_repository.web_content.services.redis_handler import ContentRedisHandler
from ai_services_api.services.centralized_repository.web_content.services.web_scraper import WebsiteScraper
from ai_services_api.services.centralized_repository.web_content.services.pdf_processor import PDFProcessor
from ai_services_api.services.centralized_repository.web_content.embeddings.model_handler import EmbeddingModel
from ai_services_api.services.centralized_repository.web_content.database.database_setup import ContentTracker, get_db_cursor, DatabaseInitializer

logger = logging.getLogger(__name__)

class WebContentProcessor:
    """Optimized web content processor with batch processing and parallelization"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 batch_size: int = 50,
                 processing_checkpoint_hours: int = 24):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.website_url = os.getenv('WEBSITE_URL')
        
        # New parameter for controlling content re-processing
        self.processing_checkpoint_hours = processing_checkpoint_hours
        
        # Initialize placeholders for pools and components
        self.db_pool = None
        self.redis_pool = None
        self.pipeline = None
        self.redis_handler = None
        self.embedding_model = None
        self.content_tracker = None
        
        # Initialize database schema
        self._initialize_database_schema()
        
        # Setup components
        self.setup_components()

    def _initialize_database_schema(self):
        """Ensure necessary database schema exists"""
        try:
            db_initializer = DatabaseInitializer()
            db_initializer.initialize_schema()
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {str(e)}")
            raise

    def batch_update_references(self, urls: List[str], keys: List[str]):
        """Synchronous method to update embedding references"""
        try:
            conn = self.db_pool.getconn()
            cur = conn.cursor()
            
            # Batch update
            cur.executemany("""
                INSERT INTO content_embeddings (url, embedding_key)
                VALUES (%s, %s)
                ON CONFLICT (url) DO UPDATE
                SET embedding_key = EXCLUDED.embedding_key,
                    updated_at = CURRENT_TIMESTAMP
            """, list(zip(urls, keys)))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating references: {str(e)}")
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                self.db_pool.putconn(conn)

    def setup_components(self):
        """Initialize components with connection pooling"""
        try:
            # Initialize main components
            self.pipeline = ContentPipeline(max_workers=self.max_workers)
            self.redis_handler = ContentRedisHandler()
            self.embedding_model = EmbeddingModel()
            self.content_tracker = ContentTracker()
            
            # Create connection pools
            self.setup_connection_pools()
            
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def setup_connection_pools(self):
        """Setup database and Redis connection pools"""
        try:
            # PostgreSQL connection pool
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self.max_workers,
                **DATABASE_CONFIG
            )
            
            # Redis connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_handler.redis_url,
                max_connections=self.max_workers,
                db=0  # Explicitly use db 0
            )
            
            logger.info("Connection pools initialized")
        except Exception as e:
            logger.error(f"Failed to setup connection pools: {str(e)}")
            raise

    def process_batch(self, items: List[Dict]) -> Dict[str, int]:
        """Synchronous wrapper for async batch processing"""
        import asyncio
        return asyncio.run(self._async_process_batch(items))

    async def _async_process_batch(self, items: List[Dict]) -> Dict[str, int]:
        """Actual async batch processing method"""
        results = {'processed': 0, 'updated': 0}
        
        try:
            # Check which items have changed
            changed_items = await self.batch_check_content_changes(items)
            results['processed'] = len(items)
            
            if changed_items:
                # Create embeddings for changed items
                item_embeddings = await self.batch_create_embeddings(changed_items)
                
                # Store embeddings
                keys = await self.batch_store_embeddings(item_embeddings)
                
                # Update references
                if keys:
                    await self.batch_update_references(
                        [item['url'] for item in changed_items],
                        keys
                    )
                    results['updated'] = len(keys)
                    
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            
        return results

    async def batch_check_content_changes(self, items: List[Dict]) -> List[Dict]:
        """Enhanced content change detection with checkpoint mechanism"""
        changed_items = []
        
        try:
            # Get connection from pool
            conn = self.db_pool.getconn()
            cur = conn.cursor()
            
            # Prepare batch query
            urls_and_hashes = [(
                item['url'],
                hashlib.md5(item['content'].encode()).hexdigest()
            ) for item in items]
            
            # Check existing hashes with checkpoint
            cur.execute("""
                SELECT url, content_hash, last_modified 
                FROM content_hashes 
                WHERE url = ANY(%s)
            """, ([url for url, _ in urls_and_hashes],))
            
            existing_records = {
                url: (content_hash, last_modified) 
                for url, content_hash, last_modified in cur.fetchall()
            }
            
            # Compare and identify changes
            for item, (url, new_hash) in zip(items, urls_and_hashes):
                # Check if item exists and hasn't been processed recently
                if (url not in existing_records or 
                    existing_records[url][0] != new_hash or 
                    (datetime.now() - existing_records[url][1]) > timedelta(hours=self.processing_checkpoint_hours)):
                    
                    changed_items.append(item)
                    
                    # Update hash in database
                    cur.execute("""
                        INSERT INTO content_hashes (url, content_hash)
                        VALUES (%s, %s)
                        ON CONFLICT (url) DO UPDATE
                        SET content_hash = EXCLUDED.content_hash,
                            last_modified = CURRENT_TIMESTAMP
                    """, (url, new_hash))
            
            conn.commit()
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                self.db_pool.putconn(conn)
                
        return changed_items

    async def batch_create_embeddings(self, items: List[Dict]) -> List[tuple]:
        """Optimize embedding creation with additional error handling"""
        try:
            # Process in small batches to avoid memory issues
            batch_size = min(32, len(items))
            embeddings = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                contents = [item['content'] for item in batch]
                
                try:
                    batch_embeddings = self.embedding_model.create_embeddings_batch(contents)
                    embeddings.extend(zip(batch, batch_embeddings))
                except Exception as batch_error:
                    # Log individual batch error but continue processing
                    logger.warning(f"Error in embedding batch {i//batch_size}: {batch_error}")
                    # Optionally, you could add partial processing logic here
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding creation: {str(e)}")
            return []

    async def batch_store_embeddings(self, item_embeddings: List[tuple]) -> List[str]:
        """Enhanced embedding storage with more robust error handling"""
        keys = []
        async with redis.Redis(connection_pool=self.redis_pool) as redis_client:
            for item, embedding in item_embeddings:
                try:
                    # Generate more robust key
                    key = f"emb:{hashlib.md5((item['url'] + '_' + datetime.now().isoformat()).encode()).hexdigest()}"
                    
                    # Prepare enriched data
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
                    
                    # Store in Redis with expiration (optional, can be adjusted)
                    await redis_client.set(key, json.dumps(data), ex=timedelta(days=30))
                    keys.append(key)
                    
                except Exception as e:
                    logger.error(f"Error storing embedding for {item.get('url', 'unknown')}: {str(e)}")
                    
        return keys

    


    async def process_content(self) -> Dict:
        """Enhanced processing method with more detailed tracking"""
        try:
            logger.info("\n" + "="*50)
            logger.info("Starting Resumable Web Content Processing...")
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
            
            # Run content pipeline
            pipeline_results = self.pipeline.run()
            
            # Process webpages in batches
            for i in range(0, len(pipeline_results['webpage_results']), self.batch_size):
                batch = pipeline_results['webpage_results'][i:i + self.batch_size]
                batch_results = await self.process_batch(batch)
                results['processed_pages'] += batch_results['processed']
                results['updated_pages'] += batch_results['updated']
                
                # Track batch details
                results['processing_details']['webpage_results'].append({
                    'batch_start_index': i,
                    'batch_size': len(batch),
                    'processed': batch_results['processed'],
                    'updated': batch_results['updated']
                })

            # Process PDF chunks in batches
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
                batch_results = await self.process_batch(batch)
                results['processed_chunks'] += batch_results['processed']
                results['updated_chunks'] += batch_results['updated']
                
                # Track PDF chunk batch details
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

            return results
                
        except Exception as e:
            logger.error(f"Error in content processing: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup with more comprehensive resource management"""
        try:
            # Close pipeline resources
            if hasattr(self, 'pipeline'):
                self.pipeline.cleanup()
            
            # Close Redis handler
            if hasattr(self, 'redis_handler'):
                self.redis_handler.close()
            
            # Close database connection pool
            if hasattr(self, 'db_pool'):
                self.db_pool.closeall()
            
            # Optional: Add any additional cleanup for other resources
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    
    def close(self):
        """Cleanup method compatible with various usage patterns"""
        self.cleanup()



