import os
import asyncpg
import logging
from urllib.parse import urlparse
from typing import Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncio
from functools import wraps

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    _pool: Optional[asyncpg.Pool] = None
    _config: Optional[Dict] = None

    @classmethod
    async def initialize(cls, min_size: int = 5, max_size: int = 20):
        """Initialize the database pool."""
        if cls._pool is None:
            try:
                params = cls.get_connection_params()
                cls._pool = await asyncpg.create_pool(
                    host=params['host'],
                    port=params['port'],
                    database=params['dbname'],
                    user=params['user'],
                    password=params['password'],
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=60,
                    server_settings={'search_path': 'public'}
                )
                logger.info(f"Initialized connection pool for database: {params['dbname']} at {params['host']}")
            except Exception as e:
                logger.error(f"Failed to initialize connection pool: {e}", exc_info=True)
                raise

    @classmethod
    async def close(cls):
        """Close the database pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("Database pool closed")

    @staticmethod
    def get_connection_params() -> Dict[str, str]:
        """Get database connection parameters with enhanced error checking."""
        try:
            database_url = os.getenv('DATABASE_URL')
            
            if database_url:
                parsed_url = urlparse(database_url)
                if not all([parsed_url.hostname, parsed_url.username, parsed_url.password]):
                    raise ValueError("Invalid DATABASE_URL format")
                    
                return {
                    'host': parsed_url.hostname,
                    'port': parsed_url.port or 5432,
                    'dbname': parsed_url.path[1:],
                    'user': parsed_url.username,
                    'password': parsed_url.password
                }
            
            in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
            params = {
                'host': os.getenv('POSTGRES_HOST', '167.86.85.127'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
            }
            
            # Validate parameters
            if not all(params.values()):
                missing = [k for k, v in params.items() if not v]
                raise ValueError(f"Missing required database parameters: {', '.join(missing)}")
                
            return params
            
        except Exception as e:
            logger.error(f"Error getting connection parameters: {e}", exc_info=True)
            raise

    @classmethod
    @asynccontextmanager
    async def get_connection(cls, dbname: Optional[str] = None) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Async context manager for getting a database connection from the pool.
        
        Usage:
            async with DatabaseConnector.get_connection() as conn:
                result = await conn.fetch("SELECT * FROM your_table")
        """
        if not cls._pool:
            await cls.initialize()
            
        try:
            async with cls._pool.acquire() as connection:
                if dbname:
                    await connection.execute(f'SET search_path TO {dbname}, public')
                yield connection
        except asyncpg.PostgresError as e:
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}", exc_info=True)
            raise

    @staticmethod
    def retry_on_failure(max_retries: int = 3, retry_delay: float = 0.1):
        """
        Decorator for retrying database operations on failure.
        
        Usage:
            @DatabaseConnector.retry_on_failure(max_retries=3)
            async def your_db_function():
                async with DatabaseConnector.get_connection() as conn:
                    # Your database operations
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except (asyncpg.DeadlockDetectedError, asyncpg.ConnectionDoesNotExistError) as e:
                        last_error = f"Attempt {attempt + 1} failed: {str(e)}"
                        logger.warning(last_error)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    except Exception as e:
                        logger.error(f"Unrecoverable error: {e}", exc_info=True)
                        raise
                
                raise asyncpg.PostgresError(f"Operation failed after {max_retries} attempts. Last error: {last_error}")
            return wrapper
        return decorator

    @classmethod
    async def test_connection(cls) -> bool:
        """Test database connectivity."""
        try:
            async with cls.get_connection() as conn:
                await conn.execute('SELECT 1')
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}", exc_info=True)
            return False

# Example usage:
@DatabaseConnector.retry_on_failure(max_retries=3)
async def example_query():
    async with DatabaseConnector.get_connection() as conn:
        return await conn.fetch("SELECT * FROM your_table")

# For FastAPI integration
async def startup_db_handler():
    """Initialize database connection pool on startup."""
    await DatabaseConnector.initialize()

async def shutdown_db_handler():
    """Close database connections on shutdown."""
    await DatabaseConnector.close()