import logging
from typing import List, Dict, Optional, Union
from redis.asyncio import Redis, ConnectionPool
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class RedisConnectionManager:
    """Centralized Redis connection manager to handle all async Redis connections."""
    
    _instance = None
    _pools = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RedisConnectionManager()
        return cls._instance
    
    async def get_connection(self, db: int = 0) -> Redis:
        """Get an async Redis connection from the appropriate connection pool."""
        if db not in self._pools:
            self._pools[db] = ConnectionPool(
                host='redis',
                port=6379,
                db=db,
                decode_responses=True,
                max_connections=10,  # Limit concurrent connections
                health_check_interval=30  # Check connection health periodically
            )
        
        return Redis(connection_pool=self._pools[db])
    
    async def close_all(self):
        """Close all connection pools."""
        for pool in self._pools.values():
            await pool.disconnect()

# Factory function to create MLPredictor instances asynchronously
async def create_ml_predictor(suggestions_db: int = 0):
    """
    Factory function to create and initialize an MLPredictor with async Redis.
    
    Args:
        suggestions_db: Redis database number for suggestions
        
    Returns:
        An initialized MLPredictor instance
    """
    predictor = MLPredictor()
    await predictor.initialize(suggestions_db)
    return predictor

class MLPredictor:
    """Prediction service for search suggestions."""
    
    def __init__(self):
        """
        Create an uninitialized predictor.
        Use the create_ml_predictor factory function instead of calling this directly.
        """
        # Constants
        self.SUGGESTION_SCORE_BOOST = 1.0
        self.MAX_SUGGESTIONS = 10
        self.MIN_CHARS = 2
        
        # Key prefixes for better organization
        self.USER_KEY_PREFIX = "suggestions:user:"
        self.GLOBAL_KEY = "suggestions:global"
        
        # Redis client will be set in initialize()
        self.redis_client = None
        self.redis_manager = None
    
    async def initialize(self, suggestions_db: int = 0):
        """
        Initialize the predictor with Redis connection.
        
        Args:
            suggestions_db: Redis database number for suggestions
        """
        # Use centralized connection manager for async Redis
        from ai_services_api.services.search.ml.ml_predictor import RedisConnectionManager
        
        self.redis_manager = RedisConnectionManager.get_instance()
        self.redis_client = await self.redis_manager.get_connection(suggestions_db)
        
        return self
    async def predict(self, partial_query: str, user_id: str, limit: int = 10) -> List[str]:
        """Get search suggestions for partial query using async Redis."""
        try:
            # If query is empty, return trending searches instead of empty list
            if not partial_query:
                return await self._get_trending_suggestions(limit)
                
            partial_query = partial_query.lower().strip()
            
            # For 1 character, provide broad category suggestions
            if len(partial_query) < self.MIN_CHARS:
                return await self._get_category_suggestions(partial_query, limit)
            
            # Try user-specific suggestions first
            user_key = f"{self.USER_KEY_PREFIX}{user_id}"
            suggestions = await self._get_suggestions(user_key, partial_query, limit)
            
            # If not enough user suggestions, add global ones
            if len(suggestions) < limit:
                global_suggestions = await self._get_suggestions(
                    self.GLOBAL_KEY, 
                    partial_query, 
                    limit - len(suggestions)
                )
                suggestions.extend(
                    [s for s in global_suggestions if s not in suggestions]
                )
            
            # No fallback suggestions as requested
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}", exc_info=True)
            # Return empty list with warning log instead of silent failure
            logger.warning(f"Returning empty suggestions list due to error")
            return []

    async def _get_trending_suggestions(self, limit: int) -> List[str]:
        """Get trending/popular searches when query is empty."""
        try:
            # Get highest scoring items from global suggestions
            trending = await self.redis_client.zrevrange(
                self.GLOBAL_KEY,
                0,
                limit - 1,
                withscores=True
            )
            return [item[0] for item in trending]
        except Exception as e:
            logger.error(f"Error getting trending suggestions: {e}", exc_info=True)
            return []

    async def _get_category_suggestions(self, char: str, limit: int) -> List[str]:
        """Get broad category suggestions for single character inputs."""
        try:
            # Get suggestions that start with the character, sorted by score
            matches = await self.redis_client.zrangebylex(
                self.GLOBAL_KEY,
                f"[{char}",
                f"[{char}\xff",
                0,
                limit * 2  # Get more than needed to sort by popularity
            )
            
            if matches:
                # Sort by popularity and return top results
                scored_matches = [
                    (m, await self.redis_client.zscore(self.GLOBAL_KEY, m) or 0) 
                    for m in matches
                ]
                scored_matches.sort(key=lambda x: x[1], reverse=True)
                return [m[0] for m in scored_matches[:limit]]
            
            return []
        except Exception as e:
            logger.error(f"Error getting category suggestions: {e}", exc_info=True)
            return []

    async def _get_suggestions(self, key: str, prefix: str, limit: int) -> List[str]:
        """Get suggestions from a specific sorted set with improved error handling."""
        try:
            # Check if key exists before attempting operations
            if not await self.redis_client.exists(key):
                return []
                
            # Use ZRANGEBYLEX for prefix matching
            matches = await self.redis_client.zrangebylex(
                key,
                f"[{prefix}",
                f"[{prefix}\xff",
                0,
                limit
            )
            
            # Sort by score (popularity)
            if matches:
                scored_matches = [
                    (m, await self.redis_client.zscore(key, m) or 0) 
                    for m in matches
                ]
                scored_matches.sort(key=lambda x: x[1], reverse=True)
                return [m[0] for m in scored_matches]
                
            return []
            
        except Exception as e:
            logger.error(f"Error in _get_suggestions for key {key}: {e}", exc_info=True)
            return []

    async def update(self, query: str, user_id: str = None):
        """Record a successful search query and invalidate related caches."""
        try:
            if not query or not user_id:
                return

            query = query.lower().strip()
            timestamp = datetime.now().timestamp()

            # Use pipeline for batched operations
            pipeline = self.redis_client.pipeline()
            
            # Update user-specific suggestions
            user_key = f"{self.USER_KEY_PREFIX}{user_id}"
            pipeline.zadd(user_key, {query: timestamp})
            
            # Update global suggestions
            pipeline.zincrby(self.GLOBAL_KEY, self.SUGGESTION_SCORE_BOOST, query)
            
            # Maintain size limits
            pipeline.zremrangebyrank(user_key, 0, -self.MAX_SUGGESTIONS-1)
            pipeline.zremrangebyrank(self.GLOBAL_KEY, 0, -self.MAX_SUGGESTIONS-1)
            
            # Execute all commands
            await pipeline.execute()
            
            # Invalidate related caches
            await self._invalidate_suggestion_caches(query, user_id)
            
        except Exception as e:
            logger.error(f"Error updating suggestions: {e}", exc_info=True)

    async def _invalidate_suggestion_caches(self, query: str, user_id: str):
        """Invalidate caches related to a query when data changes."""
        try:
            # Get the cache DB connection
            cache_redis = await self.redis_manager.get_connection(3)  # Assuming DB 3 is for caching
            
            # Find keys to invalidate
            # This pattern will match any cache keys that might contain this query
            pattern = f"*{query[:3]}*"
            
            # User-specific cache invalidation
            user_pattern = f"*{user_id}*{pattern}"
            keys_to_delete = await cache_redis.keys(user_pattern)
            
            # Also invalidate prediction cache for this user
            prediction_pattern = f"query_prediction:{user_id}:*"
            keys_to_delete.extend(await cache_redis.keys(prediction_pattern))
            
            # Delete all matched keys
            if keys_to_delete:
                await cache_redis.delete(*keys_to_delete)
                logger.info(f"Invalidated {len(keys_to_delete)} cache keys")
                
        except Exception as e:
            logger.error(f"Error invalidating caches: {e}", exc_info=True)

    async def train(self, historical_queries: List[str], user_id: str = "default"):
        """Train with historical search queries."""
        try:
            if not historical_queries:
                return

            pipeline = self.redis_client.pipeline()
            timestamp = datetime.now().timestamp()

            # Process each query
            for query in historical_queries:
                query = query.lower().strip()
                if not query:
                    continue

                # Add to user suggestions
                if user_id != "default":
                    user_key = f"{self.USER_KEY_PREFIX}{user_id}"
                    pipeline.zadd(user_key, {query: timestamp})

                # Add to global suggestions
                pipeline.zadd(self.GLOBAL_KEY, {query: 1.0})

            await pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error training suggestions: {e}", exc_info=True)

    async def clear_user_suggestions(self, user_id: str):
        """Clear suggestions for a specific user."""
        try:
            user_key = f"{self.USER_KEY_PREFIX}{user_id}"
            await self.redis_client.delete(user_key)
            
            # Also invalidate related caches
            try:
                cache_redis = await self.redis_manager.get_connection(3)
                pattern = f"query_prediction:{user_id}:*"
                keys = await cache_redis.keys(pattern)
                if keys:
                    await cache_redis.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing user caches: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error clearing user suggestions: {e}", exc_info=True)

    async def close(self):
        """Close Redis connection."""
        try:
            # No need to close individual connections as they're managed by the pool
            pass
        except:
            pass