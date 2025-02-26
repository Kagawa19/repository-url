import redis
from typing import List
import click
import logging
from datetime import datetime
import os

class RedisCleanup:
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        log_file: str = None
    ):
        # Set up logging
        self.setup_logging(log_file)
        
        # Create Redis connections
        self.redis_text = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        self.redis_binary = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False
        )
        
        self.logger.info(f"Initialized Redis connections to {redis_host}:{redis_port} DB:{redis_db}")

    def setup_logging(self, log_file: str = None) -> None:
        """Set up logging configuration"""
        self.logger = logging.getLogger('RedisCleanup')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Default log file name with timestamp if none provided
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'logs/redis_cleanup_{timestamp}.log'
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def list_all_keys(self) -> List[str]:
        """List all keys in the Redis database"""
        self.logger.info("Retrieving all keys from database...")
        keys = [key.decode('utf-8') if isinstance(key, bytes) else key 
                for key in self.redis_text.keys('*')]
        self.logger.info(f"Found {len(keys)} keys in database")
        return keys

    def delete_keys_by_pattern(self, pattern: str) -> int:
        """Delete all keys matching a specific pattern"""
        self.logger.info(f"Attempting to delete keys matching pattern: {pattern}")
        keys = self.redis_text.keys(pattern)
        
        if keys:
            self.logger.info(f"Found {len(keys)} keys matching pattern")
            deleted = self.redis_text.delete(*keys)
            self.logger.info(f"Successfully deleted {deleted} keys")
            return deleted
        else:
            self.logger.info("No keys found matching pattern")
            return 0

    def clean_database(self, confirm: bool = True) -> None:
        """Clean the entire database, optionally asking for confirmation"""
        self.logger.info("Starting database cleanup process")
        
        keys = self.list_all_keys()
        if not keys:
            self.logger.info("Database is already empty")
            return

        self.logger.info(f"Found {len(keys)} keys in database")
        
        if confirm:
            print("\nCurrent keys in database:")
            for key in keys:
                print(f"- {key}")
            
            response = input("\nAre you sure you want to delete all keys? (yes/no): ")
            if response.lower() != 'yes':
                self.logger.info("Database cleanup cancelled by user")
                print("Operation cancelled.")
                return

        try:
            self.redis_text.flushdb()
            self.redis_binary.flushdb()
            self.logger.info("Successfully cleaned database")
            print("Database cleaned successfully.")
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {str(e)}")
            raise

    def show_database_stats(self) -> None:
        """Show statistics about the current database"""
        self.logger.info("Generating database statistics")
        keys = self.list_all_keys()
        
        # Group keys by prefix
        key_groups = {}
        for key in keys:
            prefix = key.split(':')[0] if ':' in key else 'other'
            key_groups.setdefault(prefix, []).append(key)

        stats_message = "\nDatabase Statistics:\n" + "-" * 50
        stats_message += f"\nTotal number of keys: {len(keys)}"
        stats_message += "\n\nKeys by prefix:"
        
        for prefix, prefix_keys in key_groups.items():
            stats_message += f"\n{prefix}: {len(prefix_keys)} keys"
            # Log some example keys for each prefix
            examples = prefix_keys[:3]
            if examples:
                stats_message += f"\n  Examples: {', '.join(examples)}"
        
        print(stats_message)
        self.logger.info("Database statistics generated")
        self.logger.debug(stats_message)

@click.command()
@click.option('--host', default='localhost', help='Redis host')
@click.option('--port', default=6379, help='Redis port')
@click.option('--db', default=0, help='Redis database number')
@click.option('--clean', is_flag=True, help='Clean the entire database')
@click.option('--force', is_flag=True, help='Skip confirmation when cleaning')
@click.option('--stats', is_flag=True, help='Show database statistics')
@click.option('--pattern', help='Delete keys matching pattern')
@click.option('--log-file', help='Custom log file path')
def main(host: str, port: int, db: int, clean: bool, force: bool, stats: bool, pattern: str, log_file: str):
    """Redis Database Cleanup Utility"""
    cleanup = RedisCleanup(host, port, db, log_file)

    try:
        if stats:
            cleanup.show_database_stats()

        if pattern:
            count = cleanup.delete_keys_by_pattern(pattern)
            cleanup.logger.info(f"Deleted {count} keys matching pattern: {pattern}")
            print(f"Deleted {count} keys matching pattern: {pattern}")

        if clean:
            cleanup.clean_database(confirm=not force)
            
    except Exception as e:
        cleanup.logger.error(f"Error during cleanup operation: {str(e)}")
        raise

if __name__ == "__main__":
    main()