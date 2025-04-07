import json
import numpy as np
import redis
import logging
import os
from typing import Dict, Any, Optional, List

class RedisDataInspector:
    def __init__(self, redis_url: str = None):
        """
        Initialize Redis connection for data inspection.
        
        :param redis_url: Optional Redis URL. Defaults to Docker service URL if not provided.
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Determine Redis URL
        # Priority: 
        # 1. Explicitly provided URL
        # 2. Environment variable
        # 3. Docker service URL
        # 4. Localhost fallback
        if redis_url:
            final_redis_url = redis_url
        else:
            final_redis_url = os.getenv(
                'REDIS_URL', 
                'redis://localhost:6379'  # Localhost fallback
            )
        
        try:
            # Create both text and binary Redis connections
            self.redis_text = redis.StrictRedis.from_url(
                final_redis_url, 
                decode_responses=True,
                db=0
            )
            self.redis_binary = redis.StrictRedis.from_url(
                final_redis_url, 
                decode_responses=False,
                db=0
            )
            
            # Verify connection
            self.redis_text.ping()
            self.logger.info(f"Redis connections established successfully to {final_redis_url}")
        except Exception as e:
            self.logger.error(f"Failed to establish Redis connection to {final_redis_url}: {e}")
            raise

    def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List all keys matching a given pattern.
        
        :param pattern: Redis key pattern to match. Defaults to all keys.
        :return: List of matching keys
        """
        try:
            # Use SCAN to safely list keys
            cursor = 0
            keys = []
            while True:
                cursor, batch = self.redis_text.scan(cursor, match=pattern, count=1000)
                keys.extend(batch)
                if cursor == 0:
                    break
            return keys
        except Exception as e:
            self.logger.error(f"Error listing keys: {e}")
            return []

    def count_keys(self, pattern: str = "*") -> int:
        """
        Count keys matching a given pattern.
        
        :param pattern: Redis key pattern to match. Defaults to all keys.
        :return: Number of matching keys
        """
        try:
            return len(self.list_keys(pattern))
        except Exception as e:
            self.logger.error(f"Error counting keys: {e}")
            return 0

    def get_expert_details(self, expert_id: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive details for a specific expert.
        
        :param expert_id: ID of the expert to retrieve
        :return: Dictionary of expert details
        """
        try:
            # Construct base key
            base_key = f"expert:{expert_id}"
            
            # Retrieve metadata
            metadata = self.redis_text.hgetall(f"meta:{base_key}")
            
            # Parse JSON fields
            if metadata.get('expertise'):
                try:
                    metadata['expertise'] = json.loads(metadata['expertise'])
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse expertise for expert {expert_id}")
            
            # Retrieve text content
            text_content = self.redis_text.get(f"text:{base_key}")
            
            # Retrieve embedding
            embedding_bytes = self.redis_binary.get(f"emb:{base_key}")
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32) if embedding_bytes else None
            
            return {
                'metadata': metadata,
                'text_content': text_content,
                'embedding': {
                    'shape': embedding.shape if embedding is not None else None,
                    'first_few_values': embedding[:5].tolist() if embedding is not None else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error retrieving expert details for {expert_id}: {e}")
            return {}

    def get_resource_details(self, resource_id: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive details for a specific resource.
        
        :param resource_id: ID of the resource to retrieve
        :return: Dictionary of resource details
        """
        try:
            # Construct base key
            base_key = f"resource:{resource_id}"
            
            # Retrieve metadata
            metadata = self.redis_text.hgetall(f"meta:{base_key}")
            
            # Parse JSON fields
            json_fields = ['domains', 'topics', 'subtitles', 'publishers', 'identifiers', 'authors']
            for field in json_fields:
                if metadata.get(field):
                    try:
                        metadata[field] = json.loads(metadata[field])
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not parse {field} for resource {resource_id}")
            
            # Retrieve text content
            text_content = self.redis_text.get(f"text:{base_key}")
            
            # Retrieve embedding
            embedding_bytes = self.redis_binary.get(f"emb:{base_key}")
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32) if embedding_bytes else None
            
            return {
                'metadata': metadata,
                'text_content': text_content,
                'embedding': {
                    'shape': embedding.shape if embedding is not None else None,
                    'first_few_values': embedding[:5].tolist() if embedding is not None else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error retrieving resource details for {resource_id}: {e}")
            return {}

    def get_summary_stats(self) -> Dict[str, int]:
        """
        Get summary statistics of stored data.
        
        :return: Dictionary with counts of experts, resources, and total keys
        """
        try:
            return {
                'total_keys': self.count_keys(),
                'expert_keys': self.count_keys('expert:*'),
                'expert_metadata_keys': self.count_keys('meta:expert:*'),
                'expert_text_keys': self.count_keys('text:expert:*'),
                'expert_embedding_keys': self.count_keys('emb:expert:*'),
                'resource_keys': self.count_keys('resource:*'),
                'resource_metadata_keys': self.count_keys('meta:resource:*'),
                'resource_text_keys': self.count_keys('text:resource:*'),
                'resource_embedding_keys': self.count_keys('emb:resource:*')
            }
        except Exception as e:
            self.logger.error(f"Error getting summary statistics: {e}")
            return {}

    def clear_data(self, pattern: str = "*") -> int:
        """
        Clear data matching a specific pattern.
        
        :param pattern: Pattern of keys to delete. Defaults to all keys.
        :return: Number of keys deleted
        """
        try:
            keys = self.list_keys(pattern)
            if keys:
                deleted_count = self.redis_text.delete(*keys)
                self.logger.info(f"Deleted {deleted_count} keys matching pattern {pattern}")
                return deleted_count
            return 0
        except Exception as e:
            self.logger.error(f"Error clearing data: {e}")
            return 0

def main():
    """
    Main function to demonstrate Redis data inspection.
    """
    # Create an inspector instance
    inspector = RedisDataInspector()
    
    print("=== Redis Data Summary Statistics ===")
    # Get and print summary statistics
    stats = inspector.get_summary_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Sample Expert Keys ===")
    # List first 10 expert keys
    expert_keys = inspector.list_keys('meta:expert:*')[:10]
    for key in expert_keys:
        print(key)
    
    # Inspect sample expert if exists
    if expert_keys:
        sample_expert_id = expert_keys[0].split(':')[-1]
        print(f"\n=== Sample Expert Details (ID: {sample_expert_id}) ===")
        expert_details = inspector.get_expert_details(sample_expert_id)
        print(json.dumps(expert_details, indent=2))
    
    print("\n=== Sample Resource Keys ===")
    # List first 10 resource keys
    resource_keys = inspector.list_keys('meta:resource:*')[:10]
    for key in resource_keys:
        print(key)
    
    # Inspect sample resource if exists
    if resource_keys:
        sample_resource_id = resource_keys[0].split(':')[-1]
        print(f"\n=== Sample Resource Details (ID: {sample_resource_id}) ===")
        resource_details = inspector.get_resource_details(sample_resource_id)
        print(json.dumps(resource_details, indent=2))

def clear_redis_data():
    """
    Function to offer clearing Redis data with user confirmation.
    """
    inspector = RedisDataInspector()
    
    print("WARNING: This will delete ALL data in Redis!")
    confirm = input("Are you sure you want to clear ALL Redis data? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        print("Clearing all Redis data...")
        deleted_count = inspector.clear_data()
        print(f"Deleted {deleted_count} keys.")
    else:
        print("Data clearing cancelled.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Redis Data Inspector")
    parser.add_argument('--clear', action='store_true', help='Clear all Redis data')
    
    args = parser.parse_args()
    
    if args.clear:
        clear_redis_data()
    else:
        main()