# redis_handler.py
import redis
import json
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ContentRedisHandler:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.from_url(self.redis_url, db=0)  # Explicitly use db 0
        
    def store_webpage_embedding(self, 
                              url: str, 
                              embedding: np.ndarray, 
                              metadata: Dict[str, Any]) -> bool:
        """Store webpage embedding and metadata in Redis db 0"""
        try:
            key = f"webpage:{self._generate_key(url)}"
            
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Prepare data for storage
            data = {
                'embedding': embedding_list,
                'metadata': {
                    'url': url,
                    'title': metadata.get('title', ''),
                    'content_hash': metadata.get('content_hash', ''),
                    'type': 'webpage',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Store in Redis
            self.redis_client.set(key, json.dumps(data))
            logger.info(f"Stored webpage embedding for URL: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing webpage embedding: {e}")
            return False
            
    def store_pdf_chunk_embedding(self, 
                                url: str,
                                chunk_index: int,
                                embedding: np.ndarray,
                                metadata: Dict[str, Any]) -> bool:
        """Store PDF chunk embedding and metadata in Redis db 0"""
        try:
            key = f"pdf:{self._generate_key(url)}:chunk:{chunk_index}"
            
            # Convert embedding to list
            embedding_list = embedding.tolist()
            
            # Prepare data
            data = {
                'embedding': embedding_list,
                'metadata': {
                    'url': url,
                    'chunk_index': chunk_index,
                    'total_chunks': metadata.get('total_chunks', 1),
                    'content_hash': metadata.get('content_hash', ''),
                    'type': 'pdf',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Store in Redis
            self.redis_client.set(key, json.dumps(data))
            logger.info(f"Stored PDF chunk embedding for URL: {url}, chunk: {chunk_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing PDF chunk embedding: {e}")
            return False
    
    def _generate_key(self, url: str) -> str:
        """Generate a consistent key from URL"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
        
    def get_embedding(self, key: str) -> Optional[Dict]:
        """Retrieve embedding and metadata by key"""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
            return None
            
    def clear_embeddings(self) -> bool:
        """Clear all embeddings from Redis db 0"""
        try:
            # Only clear keys matching our patterns
            for pattern in ["webpage:*", "pdf:*"]:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            logger.info("Cleared all content embeddings from Redis")
            return True
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
            return False
            
    def get_all_embeddings(self) -> Dict[str, Dict]:
        """Retrieve all embeddings and their metadata"""
        try:
            results = {}
            for pattern in ["webpage:*", "pdf:*"]:
                keys = self.redis_client.keys(pattern)
                for key in keys:
                    data = self.get_embedding(key)
                    if data:
                        results[key] = data
            return results
        except Exception as e:
            logger.error(f"Error retrieving all embeddings: {e}")
            return {}
            
    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")