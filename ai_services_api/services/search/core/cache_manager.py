import redis
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host='localhost',  # Change as needed
                port=6379,        # Change as needed
                db=0,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            # Fallback to in-memory cache
            self.cache = {}
            logger.info("Using in-memory cache fallback")
            self.using_redis = False
        else:
            self.using_redis = True

    def get(self, key: str) -> Optional[Any]:
        try:
            if self.using_redis:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                return self.cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, expire: int = 300):
        try:
            if self.using_redis:
                self.redis_client.setex(
                    key,
                    expire,
                    json.dumps(value)
                )
            else:
                self.cache[key] = value
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def delete(self, key: str):
        try:
            if self.using_redis:
                self.redis_client.delete(key)
            else:
                self.cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    def clear(self):
        try:
            if self.using_redis:
                self.redis_client.flushdb()
            else:
                self.cache.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")