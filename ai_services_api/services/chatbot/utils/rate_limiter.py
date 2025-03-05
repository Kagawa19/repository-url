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
        self.base_limit = 200  # Increased from 5 to 20 requests per minute
        self.max_limit = 400   # Increased from 20 to 50 requests per minute
        self.window_size = 300  # Window size in seconds
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
        """
        Check if user has exceeded their rate limit, with awareness of Google API quotas.
        
        Args:
            user_id (str): The user ID to check rate limits for
            
        Returns:
            bool: True if the request is allowed, False if rate limited
        """
        try:
            redis_client = await redis_pool.get_redis()
            
            # First check global circuit breaker for API quota issues
            global_circuit_breaker = "global:api_circuit_breaker"
            if await redis_client.get(global_circuit_breaker):
                remaining = await redis_client.ttl(global_circuit_breaker)
                logger.warning(f"Global circuit breaker active. Blocking request for user {user_id}. Resets in {remaining}s")
                return False
                
            # Check global API quota counter
            quota_key = "global:gemini_api_quota"
            current_minute = int(time.time() / 60)
            quota_window_key = f"{quota_key}:{current_minute}"
            
            # Get current API quota usage
            pipe = redis_client.pipeline()
            pipe.incr(quota_window_key)
            pipe.expire(quota_window_key, 120)  # Keep for 2 minutes (current + next minute)
            results = await pipe.execute()
            
            current_api_usage = results[0]
            
            # Check if we're approaching Google's API quota limit
            # Assume Google limit is ~60 requests per minute per project
            if current_api_usage > 45:  # 75% of theoretical limit
                # If we're getting close to the quota, reduce allowed traffic
                quota_reduction = min(0.9, (current_api_usage - 45) / 15)  # Scale from 0 to 0.9 as usage increases
                logger.warning(f"Approaching API quota limit: {current_api_usage}/60 requests. Reducing traffic by {quota_reduction*100:.0f}%")
                
                # Apply probabilistic rate limiting based on quota usage
                if random.random() < quota_reduction:
                    logger.warning(f"Probabilistic rate limiting applied to user {user_id} due to high API usage")
                    return False
            
            # If we've exceeded the safe limit, activate circuit breaker
            if current_api_usage >= 55:  # 92% of limit
                logger.critical(f"API quota nearly exhausted: {current_api_usage}/60. Activating circuit breaker")
                # Set global circuit breaker for 45 seconds to allow quota to reset
                await redis_client.setex(global_circuit_breaker, 45, "1")
                return False
            
            # Now check user-specific limit
            limit_key = await self.get_limit_key(user_id)
            
            # Get current request count
            current_count = int(await redis_client.get(limit_key) or 0)
            
            # Get user's current limit
            user_limit = await self.get_user_limit(user_id)
            
            # Reduce user limit if we're approaching API quota
            if current_api_usage > 30:  # 50% of limit
                # Apply graduated reductions as API usage increases
                if current_api_usage > 40:
                    user_limit = int(user_limit * 0.5)  # 50% reduction
                elif current_api_usage > 35:
                    user_limit = int(user_limit * 0.7)  # 30% reduction
                else:
                    user_limit = int(user_limit * 0.85)  # 15% reduction
                    
                logger.info(f"Reduced user limit to {user_limit} due to API quota usage ({current_api_usage}/60)")
            
            if current_count >= user_limit:
                logger.warning(f"Rate limit exceeded for user {user_id}. Count: {current_count}, Limit: {user_limit}")
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
            # Allow request on error to prevent blocking users
            return True

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
