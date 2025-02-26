# services/web_content/redis_handler.py

import redis
import json
import numpy as np
from typing import Dict, Any, Optional
import logging
import hashlib
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ContentRedisHandler:
    """Handles storing and retrieving embeddings in Redis"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.from_url(self.redis_url, db=0)
        
    def store_embedding(self, 
                       url: str, 
                       embedding: np.ndarray,
                       last_modified: Optional[str] = None) -> str:
        """
        Store embedding in Redis and return key.
        
        Args:
            url: Content URL
            embedding: Embedding vector
            last_modified: Last modified timestamp
            
        Returns:
            str: Redis key for the embedding
        """
        try:
            # Generate key from URL
            key = f"emb:{hashlib.md5(url.encode()).hexdigest()}"
            
            # Prepare data
            data = {
                'embedding': embedding.tolist(),
                'url': url,
                'stored_at': datetime.now().isoformat(),
                'last_modified': last_modified
            }
            
            # Store in Redis
            self.redis_client.set(key, json.dumps(data))
            logger.info(f"Stored embedding for URL: {url}")
            return key
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise

    def get_embedding(self, key: str) -> Optional[Dict]:
        """Retrieve embedding by key"""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding: {str(e)}")
            return None
            
    def delete_embedding(self, key: str) -> bool:
        """Delete embedding by key"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}")
            return False

    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")