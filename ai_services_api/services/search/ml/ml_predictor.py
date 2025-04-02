import logging
from typing import List, Dict, Optional
from redis import Redis
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        # Initialize Redis connection (DB 5 for suggestions)
        self.redis_client = Redis(
            host='redis',
            port=6379,
            db=5,
            decode_responses=True
        )
        
        # Constants
        self.SUGGESTION_SCORE_BOOST = 1.0
        self.MAX_SUGGESTIONS = 10
        self.MIN_CHARS = 2

    # Add to the MLPredictor class in ml_predictor.py
    def predict(self, partial_query: str, user_id: str, limit: int = 5) -> List[str]:
        """Get search suggestions for partial query"""
        try:
            if not partial_query or len(partial_query) < self.MIN_CHARS:
                return []

            partial_query = partial_query.lower().strip()
            
            # Try user-specific suggestions first
            user_key = f"suggestions:user:{user_id}"
            suggestions = self._get_suggestions(user_key, partial_query, limit)
            
            # If not enough user suggestions, add global ones
            if len(suggestions) < limit:
                global_key = "suggestions:global"
                global_suggestions = self._get_suggestions(
                    global_key, 
                    partial_query, 
                    limit - len(suggestions)
                )
                suggestions.extend(
                    [s for s in global_suggestions if s not in suggestions]
                )
                
            # NEW CODE: Always add fallback suggestions if we don't have enough
            if len(suggestions) < limit:
                # Generate some basic contextual suggestions 
                fallback_suggestions = [
                    f"{partial_query} research",
                    f"{partial_query} expert",
                    f"{partial_query} specialist",
                    f"top {partial_query} researcher",
                    f"{partial_query} study"
                ]
                # Only add suggestions that start with our partial query
                fallback_suggestions = [s for s in fallback_suggestions 
                                    if s.lower().startswith(partial_query.lower())]
                
                # Add unique fallbacks
                for suggestion in fallback_suggestions:
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
                        if len(suggestions) >= limit:
                            break

            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []

    def _get_suggestions(self, key: str, prefix: str, limit: int) -> List[str]:
        """Get suggestions from a specific sorted set"""
        try:
            # Use ZRANGEBYLEX for prefix matching
            matches = self.redis_client.zrangebylex(
                key,
                f"[{prefix}",
                f"[{prefix}\xff",
                0,
                limit
            )
            
            # Sort by score (popularity)
            if matches:
                scored_matches = [
                    (m, self.redis_client.zscore(key, m) or 0) 
                    for m in matches
                ]
                scored_matches.sort(key=lambda x: x[1], reverse=True)
                return [m[0] for m in scored_matches]
                
            return []
            
        except Exception as e:
            logger.error(f"Error in _get_suggestions: {e}")
            return []

    def update(self, query: str, user_id: str = None):
        """Record a successful search query"""
        try:
            if not query or not user_id:
                return

            query = query.lower().strip()
            timestamp = datetime.now().timestamp()

            pipeline = self.redis_client.pipeline()
            
            # Update user-specific suggestions
            user_key = f"suggestions:user:{user_id}"
            pipeline.zadd(user_key, {query: timestamp})
            
            # Update global suggestions
            global_key = "suggestions:global"
            pipeline.zincrby(global_key, self.SUGGESTION_SCORE_BOOST, query)
            
            # Maintain size limits
            pipeline.zremrangebyrank(user_key, 0, -self.MAX_SUGGESTIONS-1)
            pipeline.zremrangebyrank(global_key, 0, -self.MAX_SUGGESTIONS-1)
            
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error updating suggestions: {e}")

    def train(self, historical_queries: List[str], user_id: str = "default"):
        """Train with historical search queries"""
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
                    user_key = f"suggestions:user:{user_id}"
                    pipeline.zadd(user_key, {query: timestamp})

                # Add to global suggestions
                pipeline.zadd("suggestions:global", {query: 1.0})

            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error training suggestions: {e}")

    def clear_user_suggestions(self, user_id: str):
        """Clear suggestions for a specific user"""
        try:
            user_key = f"suggestions:user:{user_id}"
            self.redis_client.delete(user_key)
        except Exception as e:
            logger.error(f"Error clearing user suggestions: {e}")

    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
        except:
            pass