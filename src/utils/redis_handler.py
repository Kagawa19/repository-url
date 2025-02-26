# src/utils/redis_handler.py
import redis
import numpy as np
import json
from typing import Dict, List, Optional
from ..config.settings import REDIS_HOST, REDIS_PORT, REDIS_EMBEDDINGS_DB

class RedisHandler:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_EMBEDDINGS_DB,
            decode_responses=True
        )
        self.redis_binary = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_EMBEDDINGS_DB,
            decode_responses=False
        )

    def store_embedding(self, key: str, embedding: np.ndarray, metadata: Dict):
        """Store embedding vector and metadata"""
        pipeline = self.redis_binary.pipeline()
        pipeline.hset(
            f"emb:{key}",
            mapping={
                'vector': embedding.tobytes(),
                'metadata': json.dumps(metadata)
            }
        )
        pipeline.execute()

    def store_text(self, key: str, text: str, metadata: Dict):
        """Store original text and metadata"""
        self.redis_client.hset(
            f"text:{key}",
            mapping={
                'content': text,
                'metadata': json.dumps(metadata)
            }
        )

    def get_embedding(self, key: str) -> Optional[Dict]:
        """Retrieve embedding and metadata"""
        data = self.redis_binary.hgetall(f"emb:{key}")
        if not data:
            return None
            
        vector = np.frombuffer(data[b'vector'])
        metadata = json.loads(data[b'metadata'])
        return {'vector': vector, 'metadata': metadata}

    def get_text(self, key: str) -> Optional[Dict]:
        """Retrieve text and metadata"""
        data = self.redis_client.hgetall(f"text:{key}")
        if not data:
            return None
        return {
            'content': data['content'],
            'metadata': json.loads(data['metadata'])
        }