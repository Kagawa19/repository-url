from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional, Dict
from pydantic import BaseModel
import json
from datetime import datetime
import logging
import asyncio
from slowapi import Limiter
from slowapi.util import get_remote_address
from redis.asyncio import Redis
import time
from ai_services_api.services.chatbot.utils.redis_connection import DynamicRedisPool


logger = logging.getLogger(__name__)
redis_pool = DynamicRedisPool()

class DynamicRateLimiter:
    def __init__(self):
        # Increase base and max limits
        self.base_limit = 20  # Increased from 5 to 20 requests per minute
        self.max_limit = 50   # Increased from 20 to 50 requests per minute
        self.window_size = 60  # Window size in seconds
        self._redis_client = None

    async def get_user_limit(self, user_id: str) -> int:
        """Dynamically calculate user's rate limit based on usage patterns."""
        try:
            redis_client = await redis_pool.get_redis()
            
            # Get user's historical usage
            usage_key = f"usage_history:{user_id}"
            usage_pattern = await redis_client.get(usage_key)
            
            if not usage_pattern:
                return self.base_limit
            
            # Parse historical usage
            usage_data = float(usage_pattern)
            
            # More generous limit adjustments
            if usage_data < 0.3:  # Low usage
                return self.base_limit
            elif usage_data < 0.7:  # Medium usage
                return int(self.base_limit * 2)  # Double the base limit
            else:  # High usage
                return self.max_limit  # Give maximum limit
                
        except Exception as e:
            logger.error(f"Error calculating rate limit: {e}")
            return self.base_limit

    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded their rate limit."""
        try:
            redis_client = await redis_pool.get_redis()
            limit_key = await self.get_limit_key(user_id)
            
            # Get current request count
            current_count = int(await redis_client.get(limit_key) or 0)
            
            # Get user's current limit
            user_limit = await self.get_user_limit(user_id)
            
            if current_count >= user_limit:
                return False
            
            # Use pipeline for atomic operations
            pipe = redis_client.pipeline()
            pipe.incr(limit_key)
            pipe.expire(limit_key, self.window_size)
            await pipe.execute()
            
            # Update usage pattern in background
            asyncio.create_task(self.update_usage_pattern(user_id, current_count + 1))
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow request on error to prevent blocking users

    async def get_limit_key(self, user_id: str) -> str:
        """Generate a rate limit key based on user and time window."""
        current_window = int(time.time() / self.window_size)
        return f"ratelimit:{user_id}:{current_window}"

    async def get_window_remaining(self, user_id: str) -> int:
        """Get remaining time in current window."""
        current_time = time.time()
        current_window = int(current_time / self.window_size)
        next_window = (current_window + 1) * self.window_size
        return int(next_window - current_time)

    
    async def update_usage_pattern(self, user_id: str, request_count: int):
        """Update user's usage pattern."""
        try:
            redis_client = await redis_pool.get_redis()
            usage_key = f"usage_history:{user_id}"
            
            # Calculate usage ratio
            current_limit = await self.get_user_limit(user_id)
            usage_ratio = min(request_count / current_limit, 1.0)
            
            # Update exponential moving average
            old_pattern = float(await redis_client.get(usage_key) or 0)
            new_pattern = 0.7 * old_pattern + 0.3 * usage_ratio
            
            # Store updated pattern
            await redis_client.setex(usage_key, 86400, str(new_pattern))  # 24 hour expiry
            
        except Exception as e:
            logger.error(f"Error updating usage pattern: {e}")
