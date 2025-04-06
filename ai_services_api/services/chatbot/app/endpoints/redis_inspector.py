import time
import redis
import json
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RedisPublicationInspector:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        try:
            # Create Redis connection with explicit host and port
            self.redis_client = redis.Redis(
                host='localhost',  # or '127.0.0.1'
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Verify connection with a retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.redis_client.ping()
                    logger.info("Successfully connected to Redis")
                    break
                except Exception as ping_error:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Redis connection attempt {attempt + 1} failed: {ping_error}")
                    time.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    def list_publication_keys(self, pattern='meta:resource:*', limit=50):
        """
        List publication keys in Redis
        
        Args:
            pattern (str): Pattern to match keys
            limit (int): Maximum number of keys to return
        """
        try:
            # Use scan to find keys matching the pattern
            cursor = 0
            keys = []
            while cursor != 0 or len(keys) == 0:
                cursor, batch = self.redis_client.scan(
                    cursor=cursor, 
                    match=pattern, 
                    count=limit
                )
                keys.extend(batch)
                
                if cursor == 0:
                    break
            
            logger.info(f"Found {len(keys)} publication keys")
            return keys
        except Exception as e:
            logger.error(f"Error listing publication keys: {e}")
            return []

    def inspect_publication(self, key):
        """
        Inspect details of a specific publication
        
        Args:
            key (str): Redis key for the publication
        
        Returns:
            dict: Publication metadata
        """
        try:
            # Retrieve all hash fields for the publication
            metadata = self.redis_client.hgetall(key)
            
            # Special handling for JSON-encoded fields
            for field in ['authors', 'domains']:
                if field in metadata:
                    try:
                        metadata[field] = json.loads(metadata[field])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse {field} for {key}")
            
            return metadata
        except Exception as e:
            logger.error(f"Error inspecting publication {key}: {e}")
            return {}

    def summarize_publications(self):
        """
        Provide a summary of publications in Redis
        """
        try:
            publication_keys = self.list_publication_keys()
            
            summary = {
                'total_publications': len(publication_keys),
                'publication_years': {},
                'dois_count': 0,
                'fields': {}
            }
            
            for key in publication_keys:
                pub = self.inspect_publication(key)
                
                # Count publications by year
                year = pub.get('publication_year', 'Unknown')
                summary['publication_years'][year] = summary['publication_years'].get(year, 0) + 1
                
                # Count DOIs
                if pub.get('doi'):
                    summary['dois_count'] += 1
                
                # Count publications by field
                field = pub.get('field', 'Uncategorized')
                summary['fields'][field] = summary['fields'].get(field, 0) + 1
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing publications: {e}")
            return {}

    def search_publications(self, query):
        """
        Search publications by query
        
        Args:
            query (str): Search term
        
        Returns:
            list: Matching publications
        """
        try:
            publication_keys = self.list_publication_keys()
            matches = []
            
            for key in publication_keys:
                pub = self.inspect_publication(key)
                
                # Case-insensitive search across multiple fields
                search_text = ' '.join([
                    str(pub.get('title', '')).lower(),
                    str(pub.get('field', '')).lower(),
                    str(pub.get('summary_snippet', '')).lower(),
                    ' '.join([str(a).lower() for a in pub.get('authors', [])]),
                    str(pub.get('doi', '')).lower()
                ])
                
                if query.lower() in search_text:
                    matches.append(pub)
            
            return matches
        except Exception as e:
            logger.error(f"Error searching publications: {e}")
            return []

def main():
    inspector = RedisPublicationInspector()
    
    print("\n=== Publication Keys ===")
    keys = inspector.list_publication_keys()
    for key in keys[:10]:  # Show first 10 keys
        print(key)
    
    print("\n=== Publication Summary ===")
    summary = inspector.summarize_publications()
    print(json.dumps(summary, indent=2))
    
    # Example search
    print("\n=== Sample Search: 'health' ===")
    health_pubs = inspector.search_publications('health')
    for pub in health_pubs[:5]:  # Show first 5 matches
        print(f"Title: {pub.get('title', 'N/A')}")
        print(f"DOI: {pub.get('doi', 'N/A')}\n")

if __name__ == '__main__':
    main()