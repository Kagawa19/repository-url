import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# For Gemini API
import google.generativeai as genai

# For database operations
from ai_services_api.services.message.core.database import get_db_connection

# For circuit breaker pattern
from ai_services_api.services.search.utils.circuit_breaker import CircuitBreaker
class SearchCategorizer:
    """
    Categorizes search queries using Gemini API with caching and fallbacks.
    
    This class is designed to work with the existing search system and
    provides automatic categorization with minimal performance impact.
    """
    
    def __init__(self, redis_client=None, gemini_api_key=None):
        """
        Initialize the SearchCategorizer.
        
        Args:
            redis_client: Optional Redis client for caching
            gemini_api_key: Optional Gemini API key (will try to get from env if not provided)
        """
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis_client
        
        # Initialize Gemini API
        self.gemini_model = None
        try:
            import google.generativeai as genai
            
            # Get API key from provided value or environment
            api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                self.logger.info("Successfully initialized Gemini API for categorization")
            else:
                self.logger.warning("No Gemini API key found for categorization")
        except ImportError:
            self.logger.warning("google.generativeai module not available")
        except Exception as e:
            self.logger.error(f"Error initializing Gemini API: {e}")
        
        # Set up cache with TTL
        self.cache_ttl = 86400 * 7  # 7 days
        self.local_cache = {}
        self.local_cache_timestamps = {}
        
        # Define predefined categories
        self.predefined_categories = {
            "person": ["researcher", "professor", "scientist", "expert", "director", "lead", "fellow"],
            "theme": ["research", "study", "field", "area", "domain", "topic", "discipline"],
            "designation": ["position", "role", "job", "title", "appointment", "post", "rank"],
            "publication": ["paper", "article", "journal", "publication", "review", "report", "study"],
            "dataset": ["data", "dataset", "statistics", "metrics", "indicators", "variables", "numbers"],
            "method": ["method", "approach", "technique", "procedure", "protocol", "process", "workflow"],
            "general": []  # Fallback category
        }
        
        # Set up categorization queue for background processing
        self.categorization_queue = asyncio.Queue()
        self.is_worker_running = False
        
        # Circuit breaker for Gemini API
        self.circuit_breaker = CircuitBreaker(
            name="gemini-categorization",
            failure_threshold=5,
            recovery_timeout=300,  # 5 minutes
            retry_timeout=1800     # 30 minutes
        )
    
    async def start_background_worker(self):
        """Start the background worker for asynchronous categorization."""
        if not self.is_worker_running:
            self.is_worker_running = True
            asyncio.create_task(self._background_categorization_worker())
            self.logger.info("Started background categorization worker")
    
    async def categorize_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """
        Categorize a search query with minimal performance impact.
        
        This method returns immediately with best-effort categorization
        and schedules full categorization in the background if needed.
        
        Args:
            query: The search query to categorize
            user_id: Optional user ID for personalization
            
        Returns:
            Dictionary with category information
        """
        if not query:
            return {"category": "general", "confidence": 1.0}
        
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Try to get from cache first
        cache_key = f"category:{normalized_query}"
        category_info = await self._check_cache(cache_key)
        
        if category_info:
            return category_info
        
        # If not in cache, do quick rule-based categorization
        quick_category = self._rule_based_categorization(normalized_query)
        
        # Queue for full categorization if Gemini is available
        if self.gemini_model:
            # Only add to queue if not already processed
            try:
                self.categorization_queue.put_nowait({
                    "query": normalized_query,
                    "user_id": user_id,
                    "cache_key": cache_key
                })
                
                # Ensure worker is running
                if not self.is_worker_running:
                    await self.start_background_worker()
            except asyncio.QueueFull:
                self.logger.warning("Categorization queue is full, skipping full categorization")
        
        return quick_category
    
    async def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if categorization is cached.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            Cached category information or None
        """
        # Check local cache first
        if cache_key in self.local_cache:
            # Check if expired
            timestamp = self.local_cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp <= self.cache_ttl:
                return self.local_cache[cache_key]
            else:
                # Expired, remove from local cache
                del self.local_cache[cache_key]
                del self.local_cache_timestamps[cache_key]
        
        # Then check Redis if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    try:
                        category_info = json.loads(cached_data)
                        
                        # Also update local cache
                        self.local_cache[cache_key] = category_info
                        self.local_cache_timestamps[cache_key] = time.time()
                        
                        return category_info
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                self.logger.warning(f"Error checking Redis cache: {e}")
        
        return None
    
    async def _update_cache(self, cache_key: str, category_info: Dict[str, Any]):
        """
        Update cache with category information.
        
        Args:
            cache_key: Cache key to update
            category_info: Category information to cache
        """
        # Update local cache
        self.local_cache[cache_key] = category_info
        self.local_cache_timestamps[cache_key] = time.time()
        
        # Update Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(category_info)
                )
            except Exception as e:
                self.logger.warning(f"Error updating Redis cache: {e}")
    
    def _rule_based_categorization(self, query: str) -> Dict[str, Any]:
        """
        Perform quick rule-based categorization without API calls.
        
        Args:
            query: The search query to categorize
            
        Returns:
            Dictionary with best-guess category information
        """
        query = query.lower()
        
        # Check for explicit category markers
        if ':' in query:
            prefix = query.split(':', 1)[0].strip()
            if prefix in self.predefined_categories:
                return {
                    "category": prefix,
                    "confidence": 0.9,
                    "method": "rule_explicit"
                }
        
        # Check each category's keywords
        best_category = None
        best_score = 0.0
        
        for category, keywords in self.predefined_categories.items():
            # Skip empty keyword lists
            if not keywords:
                continue
                
            # Count matching keywords
            matches = sum(1 for keyword in keywords if keyword in query)
            
            # Calculate match score (0-1)
            if matches > 0:
                # More keyword matches = higher score
                score = min(0.8, 0.3 + (matches * 0.1))
                
                if score > best_score:
                    best_score = score
                    best_category = category
        
        # Check for person name pattern if no strong match yet
        if (not best_category or best_score < 0.5) and all(word.istitle() for word in query.split()) and len(query.split()) <= 3:
            return {
                "category": "person",
                "confidence": 0.7,
                "method": "rule_pattern"
            }
        
        # If we found a category with good confidence
        if best_category and best_score >= 0.3:
            return {
                "category": best_category,
                "confidence": best_score,
                "method": "rule_keyword"
            }
        
        # Fallback to general category
        return {
            "category": "general",
            "confidence": 0.5,
            "method": "rule_fallback"
        }
    
    async def _background_categorization_worker(self):
        """Background worker for processing categorization queue."""
        self.logger.info("Background categorization worker started")
        
        try:
            while True:
                try:
                    # Get item from queue with timeout
                    item = await asyncio.wait_for(self.categorization_queue.get(), timeout=60)
                    
                    query = item.get("query")
                    user_id = item.get("user_id")
                    cache_key = item.get("cache_key")
                    
                    # Skip if already in cache
                    if await self._check_cache(cache_key):
                        self.categorization_queue.task_done()
                        continue
                    
                    # Process with Gemini
                    try:
                        category_info = await self._categorize_with_gemini(query)
                        
                        # Update cache with results
                        if category_info:
                            await self._update_cache(cache_key, category_info)
                            
                            # Also record in database if needed
                            try:
                                await self._record_category_in_db(query, category_info, user_id)
                            except Exception as db_error:
                                self.logger.warning(f"Error recording category in database: {db_error}")
                    except Exception as e:
                        self.logger.error(f"Error in background categorization: {e}")
                    
                    # Mark task as done
                    self.categorization_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Check if we should keep running (no items for 60 seconds)
                    if self.categorization_queue.empty():
                        self.logger.info("Background categorization worker stopping due to inactivity")
                        self.is_worker_running = False
                        break
                except Exception as e:
                    self.logger.error(f"Error in categorization worker: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
        except asyncio.CancelledError:
            self.logger.info("Background categorization worker cancelled")
            self.is_worker_running = False
    
    async def _categorize_with_gemini(self, query: str) -> Dict[str, Any]:
        """
        Categorize query using Gemini API.
        
        Args:
            query: The search query to categorize
            
        Returns:
            Dictionary with category information from Gemini
        """
        if not self.gemini_model:
            return None
        
        try:
            # Use circuit breaker to manage API calls
            result = await self.circuit_breaker.execute(
                self._execute_gemini_categorization,
                query
            )
            
            if result:
                return result
        except Exception as e:
            self.logger.error(f"Error in Gemini categorization: {e}")
        
        # Fall back to rule-based categorization
        return self._rule_based_categorization(query)
    
    async def _execute_gemini_categorization(self, query: str) -> Dict[str, Any]:
        """
        Execute the actual Gemini API call for categorization.
        
        Args:
            query: The search query to categorize
            
        Returns:
            Processed category information from Gemini
        """
        # Create prompt for categorization
        categories = list(self.predefined_categories.keys())
        
        prompt = f"""Categorize the following search query into exactly one of these categories:
{', '.join(categories)}

Search query: "{query}"

Response format: JSON with the following fields:
- category: The most appropriate category from the list
- confidence: Confidence score between 0-1
- reasoning: Brief explanation of why this category was chosen

JSON response:"""
        
        # Call Gemini API with timeout
        response = await asyncio.wait_for(
            self.gemini_model.generate_content_async(prompt),
            timeout=3.0  # 3 second timeout
        )
        
        if not response or not hasattr(response, 'text'):
            raise ValueError("Invalid response from Gemini API")
        
        # Extract and parse JSON response
        try:
            # Find JSON in response
            response_text = response.text.strip()
            
            # Check if response is wrapped in code blocks
            if "```json" in response_text and "```" in response_text:
                json_str = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response_text.startswith("{") and "}" in response_text:
                # Find the closing brace of the JSON object
                json_str = response_text
            else:
                raise ValueError("Response does not contain valid JSON")
            
            # Parse JSON
            category_data = json.loads(json_str)
            
            # Validate and clean up
            if "category" not in category_data or category_data["category"] not in categories:
                raise ValueError("Invalid category in response")
            
            if "confidence" not in category_data or not isinstance(category_data["confidence"], (int, float)):
                category_data["confidence"] = 0.7  # Default confidence
            
            # Add method to indicate source
            category_data["method"] = "gemini"
            
            return category_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing Gemini response: {e}")
            self.logger.debug(f"Response text: {response.text}")
            raise ValueError("Failed to parse Gemini response")
        except Exception as e:
            self.logger.error(f"Error processing Gemini response: {e}")
            raise
    
    async def _record_category_in_db(self, query: str, category_info: Dict[str, Any], user_id: str = None):
        """
        Record query categorization in database.
        
        Args:
            query: The search query
            category_info: Category information
            user_id: Optional user ID
        """
        try:
            # Import database connection function
            from ai_services_api.services.message.core.database import get_db_connection
            
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    # Check if search_categories table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'search_categories'
                        );
                    """)
                    
                    if not cur.fetchone()[0]:
                        # Create search_categories table if it doesn't exist
                        cur.execute("""
                            CREATE TABLE search_categories (
                                id SERIAL PRIMARY KEY,
                                query TEXT NOT NULL,
                                category TEXT NOT NULL,
                                confidence FLOAT,
                                method TEXT,
                                user_id TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                        """)
                        conn.commit()
                    
                    # Insert category information
                    cur.execute("""
                        INSERT INTO search_categories 
                            (query, category, confidence, method, user_id)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        query,
                        category_info.get("category"),
                        category_info.get("confidence"),
                        category_info.get("method"),
                        user_id
                    ))
                    
                    conn.commit()
                    
            except Exception as db_error:
                if conn:
                    conn.rollback()
                self.logger.error(f"Database error while recording category: {db_error}")
                raise
            finally:
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error recording category in database: {e}")
            raise