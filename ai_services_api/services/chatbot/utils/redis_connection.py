import redis
from redis.asyncio import Redis
import asyncio
import psutil
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DynamicRedisPool:
    def __init__(self):
        self.min_connections = 5
        self.max_connections = 50
        self.current_connections = self.min_connections
        self.last_adjustment = datetime.now()
        self.adjustment_interval = timedelta(seconds=30)
        self._redis_client: Optional[Redis] = None
        self._lock = asyncio.Lock()

    async def get_redis(self) -> Redis:
        """Get Redis client with dynamically adjusted connection pool."""
        async with self._lock:
            if self._redis_client is None or not await self._redis_client.ping():
                self._redis_client = await self._create_redis_client()
            
            # Check if it's time to adjust the pool
            if datetime.now() - self.last_adjustment > self.adjustment_interval:
                await self._adjust_pool_size()
            
            return self._redis_client

    async def _create_redis_client(self) -> Redis:
        """Create new Redis client with current connection settings."""
        try:
            return Redis(
                host='redis',
                port=6379,
                db=3,
                decode_responses=True,
                max_connections=self.current_connections,
                health_check_interval=30,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True
            )
        except Exception as e:
            logger.error(f"Error creating Redis client: {e}")
            raise

    async def _adjust_pool_size(self):
        """Dynamically adjust connection pool size based on system metrics."""
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Get Redis metrics
            info = await self._redis_client.info()
            connected_clients = info.get('connected_clients', 0)
            
            # Calculate load score (0-100)
            load_score = (cpu_percent + memory_percent) / 2
            
            # Calculate optimal connections based on load
            if load_score < 30:  # Low load
                target_connections = max(
                    self.min_connections,
                    int(self.current_connections * 0.8)  # Reduce by 20%
                )
            elif load_score > 70:  # High load
                target_connections = min(
                    self.max_connections,
                    int(self.current_connections * 1.2)  # Increase by 20%
                )
            else:  # Medium load - fine tune based on current usage
                usage_ratio = connected_clients / self.current_connections
                if usage_ratio > 0.8:  # High usage
                    target_connections = min(
                        self.max_connections,
                        self.current_connections + 5
                    )
                elif usage_ratio < 0.4:  # Low usage
                    target_connections = max(
                        self.min_connections,
                        self.current_connections - 5
                    )
                else:
                    target_connections = self.current_connections

            # Apply changes if significant difference
            if abs(target_connections - self.current_connections) >= 5:
                logger.info(
                    f"Adjusting Redis pool: {self.current_connections} -> {target_connections} "
                    f"(Load: {load_score:.1f}%, Usage: {connected_clients}/{self.current_connections})"
                )
                
                # Create new client with updated pool size
                self.current_connections = target_connections
                new_client = await self._create_redis_client()
                
                # Gracefully switch clients
                old_client = self._redis_client
                self._redis_client = new_client
                
                # Close old client after delay to allow ongoing operations to complete
                asyncio.create_task(self._graceful_shutdown(old_client))
            
            self.last_adjustment = datetime.now()
            
        except Exception as e:
            logger.error(f"Error adjusting pool size: {e}")

    async def _graceful_shutdown(self, client: Redis, delay: int = 10):
        """Gracefully shutdown old Redis client."""
        try:
            await asyncio.sleep(delay)  # Wait for ongoing operations
            await client.close()
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")

    async def close(self):
        """Close Redis connection pool."""
        if self._redis_client:
            await self._redis_client.close()

# Create singleton instance
redis_pool = DynamicRedisPool()

# Use in FastAPI dependency
async def get_redis() -> Redis:
    """FastAPI dependency for getting Redis client."""
    return await redis_pool.get_redis()