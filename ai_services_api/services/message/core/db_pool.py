import logging
import os
from functools import lru_cache
import psycopg2
import psycopg2.pool
import asyncio
from typing import Tuple, Optional, Any

# Configure logger
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_connection_pool(min_conn=5, max_conn=500):
    """Create or return a connection pool singleton with better connection management."""
    try:
        # Import inside function to avoid circular imports
        from ai_services_api.services.message.core.database import get_db_connection
        
        # Get a sample connection to extract connection parameters
        sample_conn = get_db_connection()
        conn_params = sample_conn.get_dsn_parameters()
        sample_conn.close()
        
        # Get database parameters from environment or connection params
        db_params = {
            'user': os.getenv('DB_USER', conn_params.get('postgres')),
            'password': os.getenv('DB_PASSWORD', conn_params.get('p0stgres')),
            'host': os.getenv('DB_HOST', conn_params.get('postgres')),
            'port': os.getenv('DB_PORT', conn_params.get('5342')),
            'database': os.getenv('DB_NAME', conn_params.get('aphrc'))
        }
        
        logger.info(f"Creating connection pool for database: {db_params['database']} at {db_params['host']}")
        
        # Create the connection pool with the same parameters
        pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, 
            max_conn,
            **db_params
        )
        
        # Test the pool by getting and immediately returning a connection
        test_conn = pool.getconn()
        pool.putconn(test_conn)
        
        logger.info(f"Successfully created connection pool with {min_conn}-{max_conn} connections")
        return pool
    except Exception as e:
        logger.error(f"Error creating connection pool: {str(e)}", exc_info=True)
        # Fall back to original connection method if pool creation fails
        return None

def get_pooled_connection() -> Tuple[Optional[Any], Optional[Any], bool]:
    """
    Get a connection from the pool with proper error handling.
    
    Returns:
        Tuple containing:
        - database connection
        - pool reference (or None if direct connection)
        - boolean flag indicating if pool was used
    """
    pool = get_connection_pool()
    conn = None
    
    if pool:
        try:
            conn = pool.getconn()
            logger.debug("Successfully obtained connection from pool")
            return conn, pool, True  # Connection, pool, using_pool flag
        except Exception as e:
            logger.error(f"Error getting connection from pool: {str(e)}")
    
    # Fallback to direct connection
    try:
        from ai_services_api.services.message.core.database import get_db_connection
        conn = get_db_connection()
        logger.info("Falling back to direct connection")
        logger.info(f"Successfully connected to database: {conn.get_dsn_parameters().get('dbname')} at {conn.get_dsn_parameters().get('host')}")
        return conn, None, False  # Connection, no pool, not using pool
    except Exception as e:
        logger.error(f"Error getting direct database connection: {str(e)}")
        raise

def return_connection(conn, pool, using_pool):
    """
    Safely return a connection to the pool or close it if not using pool.
    """
    if conn:
        try:
            if using_pool and pool:
                try:
                    pool.putconn(conn)
                    logger.debug("Returned connection to pool")
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {str(e)}")
                    # Don't try to close it if returning to pool failed
            else:
                try:
                    conn.close()
                    logger.debug("Closed direct database connection")
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")
        except Exception as e:
            logger.error(f"Error in return_connection: {str(e)}")

class DatabaseConnection:
    """Context manager for safely handling database connections."""
    
    def __init__(self):
        self.conn = None
        self.pool = None
        self.using_pool = False
    
    def __enter__(self):
        self.conn, self.pool, self.using_pool = get_pooled_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return_connection(self.conn, self.pool, self.using_pool)
        self.conn = None

def log_pool_status():
    """Log current status of the connection pool."""
    pool = get_connection_pool()
    if pool and hasattr(pool, '_used') and hasattr(pool, '_unused'):
        logger.info(f"Pool status: {len(pool._used)} used, {len(pool._unused)} unused")
    else:
        logger.info("Pool status unavailable - pool may not be initialized")

# Example usage with the context manager
async def example_db_operation():
    """Example of how to use the connection context manager."""
    try:
        with DatabaseConnection() as conn:
            # Use the connection here
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            cur.close()
            return result
    except Exception as e:
        logger.error(f"Database operation failed: {str(e)}")
        return None

# For functions that need to pass connections to background tasks
async def safe_background_task(func, *args, **kwargs):
    """
    Safely run a function as a background task with its own database connection.
    
    Args:
        func: The async function to run as a background task
        *args, **kwargs: Arguments to pass to the function
    """
    conn = None
    pool = None
    using_pool = False
    
    try:
        # Get connection for this background task
        conn, pool, using_pool = get_pooled_connection()
        
        # Create a new set of args with the connection as the first arg
        new_args = (conn,) + args
        
        # Create the task
        task = asyncio.create_task(func(*new_args, **kwargs))
        
        # No need to wait for it, but we need to ensure the connection gets closed
        # So we'll add a done callback
        def close_conn(_):
            nonlocal conn, pool, using_pool
            if conn:
                return_connection(conn, pool, using_pool)
        
        task.add_done_callback(close_conn)
        
        return task
    except Exception as e:
        # If task creation fails, ensure connection is closed
        if conn:
            return_connection(conn, pool, using_pool)
        logger.error(f"Failed to create background task: {str(e)}")
        raise

# Example of an async task that requires a connection
async def _record_data_async(conn, *args, **kwargs):
    """Example of a background task that requires a database connection."""
    try:
        # Use the connection
        cur = conn.cursor()
        # ... do work with connection
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error in background task: {str(e)}")
    # Note: Don't close the connection here, it will be closed by the done callback

# Example of how to spawn a background task
async def example_spawn_background_task():
    """Example of how to spawn a background task with database access."""
    try:
        # This will create a task, give it its own connection, and ensure the connection is closed
        await safe_background_task(_record_data_async, "arg1", "arg2", keyword_arg="value")
    except Exception as e:
        logger.error(f"Failed to spawn background task: {str(e)}")