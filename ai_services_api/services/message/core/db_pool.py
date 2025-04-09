import logging
import os
from functools import lru_cache
import psycopg2
import psycopg2.pool
import asyncio
import traceback
from typing import Tuple, Optional, Any
import time
import uuid

# Configure logger
logger = logging.getLogger(__name__)

# Track connection states for debugging
connection_tracker = {}

@lru_cache(maxsize=1)
def get_connection_pool(min_conn=200, max_conn=500):
    """Create or return a connection pool singleton with better connection management."""
    try:
        logger.info("Creating connection pool with min_conn=%d, max_conn=%d", min_conn, max_conn)
        
        # Import inside function to avoid circular imports
        from ai_services_api.services.message.core.database import get_db_connection
        
        logger.debug("Successfully imported get_db_connection")
        
        # Get a sample connection to extract connection parameters
        logger.debug("Getting sample connection to extract parameters")
        sample_conn = get_db_connection()
        conn_params = sample_conn.get_dsn_parameters()
        logger.info(f"Got sample connection parameters: host={conn_params.get('host')}, dbname={conn_params.get('dbname')}")
        sample_conn.close()
        logger.debug("Closed sample connection")
        
        # Get database parameters from environment or connection params
        db_params = {
            'user': os.getenv('DB_USER', conn_params.get('user')),
            'password': os.getenv('DB_PASSWORD', conn_params.get('password')),
            'host': os.getenv('DB_HOST', conn_params.get('host')),
            'port': os.getenv('DB_PORT', conn_params.get('port')),
            'database': os.getenv('DB_NAME', conn_params.get('dbname'))
        }
        
        logger.info(f"Creating connection pool for database: {db_params['database']} at {db_params['host']}:{db_params['port']}")
        
        # Create the connection pool with the same parameters
        pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, 
            max_conn,
            **db_params
        )
        
        # Test the pool by getting and immediately returning a connection
        logger.debug("Testing pool with a test connection")
        test_conn = pool.getconn()
        logger.debug("Got test connection from pool")
        pool.putconn(test_conn)
        logger.debug("Successfully returned test connection to pool")
        
        logger.info(f"Successfully created connection pool with {min_conn}-{max_conn} connections")
        return pool
    except Exception as e:
        logger.error(f"Error creating connection pool: {str(e)}")
        logger.error(f"Connection pool creation stacktrace: {traceback.format_exc()}")
        # Fall back to original connection method if pool creation fails
        return None

def get_pooled_connection() -> Tuple[Any, Optional[Any], bool, str]:
    """
    Get a connection from the pool with proper error handling.
    
    Returns:
        Tuple containing:
        - database connection
        - pool reference (or None if direct connection)
        - boolean flag indicating if pool was used
        - connection ID for tracking
    """
    pool = get_connection_pool()
    conn = None
    conn_id = str(uuid.uuid4())[:8]  # Generate a unique ID for this connection
    
    logger.debug(f"Getting pooled connection (id: {conn_id})")
    
    if pool:
        try:
            # Log pool status before getting connection
            if hasattr(pool, '_used') and hasattr(pool, '_unused'):
                logger.info(f"Pool status before getconn: {len(pool._used)} used, {len(pool._unused)} unused")
            
            start_time = time.time()
            conn = pool.getconn()
            elapsed = time.time() - start_time
            
            # Log pool status after getting connection
            if hasattr(pool, '_used') and hasattr(pool, '_unused'):
                logger.info(f"Pool status after getconn: {len(pool._used)} used, {len(pool._unused)} unused")
            
            logger.info(f"Successfully obtained connection {conn_id} from pool (took {elapsed:.3f}s)")
            
            # Track this connection
            connection_tracker[conn_id] = {
                "created_at": time.time(),
                "from_pool": True,
                "stack": traceback.format_stack(),
                "conn_object_id": id(conn)
            }
            
            return conn, pool, True, conn_id  # Add connection ID to return tuple
        except Exception as e:
            logger.error(f"Error getting connection from pool: {str(e)}")
            logger.error(f"Get connection stacktrace: {traceback.format_exc()}")
    
    # Fallback to direct connection
    try:
        logger.info(f"Falling back to direct connection (id: {conn_id})")
        from ai_services_api.services.message.core.database import get_db_connection
        
        start_time = time.time()
        conn = get_db_connection()
        elapsed = time.time() - start_time
        
        logger.info(f"Successfully connected to database (id: {conn_id}) in {elapsed:.3f}s: {conn.get_dsn_parameters().get('dbname')} at {conn.get_dsn_parameters().get('host')}")
        
        # Track this connection
        connection_tracker[conn_id] = {
            "created_at": time.time(),
            "from_pool": False,
            "stack": traceback.format_stack(),
            "conn_object_id": id(conn)
        }
        
        return conn, None, False, conn_id  # Connection, no pool, not using pool, conn_id
    except Exception as e:
        logger.error(f"Error getting direct database connection: {str(e)}")
        logger.error(f"Direct connection stacktrace: {traceback.format_exc()}")
        raise

def return_connection(conn, pool, using_pool, conn_id=None):
    """
    Safely return a connection to the pool or close it if not using pool.
    
    Args:
        conn: Database connection to return/close
        pool: Connection pool (or None for direct connections)
        using_pool: Whether the connection came from the pool
        conn_id: Connection tracking ID (optional)
    """
    if conn_id and conn_id in connection_tracker:
        logger.info(f"Returning connection {conn_id} (created {time.time() - connection_tracker[conn_id]['created_at']:.1f}s ago)")
    else:
        logger.info(f"Returning untracked connection (likely manually created)")
    
    if conn:
        try:
            if using_pool and pool:
                try:
                    # Log pool status before returning connection
                    if hasattr(pool, '_used') and hasattr(pool, '_unused'):
                        logger.info(f"Pool status before putconn: {len(pool._used)} used, {len(pool._unused)} unused")
                    
                    # Verify connection state
                    conn_status = "Unknown"
                    try:
                        cur = conn.cursor()
                        cur.execute("SELECT 1")
                        cur.fetchone()
                        cur.close()
                        conn_status = "Good"
                    except Exception as status_e:
                        conn_status = f"Bad: {str(status_e)}"
                    
                    logger.info(f"Connection {conn_id} status before return: {conn_status}")
                    
                    # Try to return to pool
                    pool.putconn(conn)
                    logger.info(f"Successfully returned connection {conn_id} to pool")
                    
                    # Log pool status after returning connection
                    if hasattr(pool, '_used') and hasattr(pool, '_unused'):
                        logger.info(f"Pool status after putconn: {len(pool._used)} used, {len(pool._unused)} unused")
                    
                except Exception as e:
                    logger.error(f"Error returning connection {conn_id} to pool: {str(e)}")
                    logger.error(f"Return connection stacktrace: {traceback.format_exc()}")
                    
                    # Try to close the connection if return to pool failed
                    try:
                        conn.close()
                        logger.info(f"Closed connection {conn_id} after failed pool return")
                    except Exception as close_e:
                        logger.error(f"Error closing connection {conn_id} after failed pool return: {str(close_e)}")
            else:
                try:
                    conn.close()
                    logger.info(f"Closed direct database connection {conn_id}")
                except Exception as e:
                    logger.error(f"Error closing connection {conn_id}: {str(e)}")
                    logger.error(f"Close connection stacktrace: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error in return_connection for {conn_id}: {str(e)}")
            logger.error(f"General return connection error stacktrace: {traceback.format_exc()}")
        finally:
            # Remove from tracking
            if conn_id and conn_id in connection_tracker:
                del connection_tracker[conn_id]

class DatabaseConnection:
    """Context manager for safely handling database connections."""
    
    def __init__(self):
        self.conn = None
        self.pool = None
        self.using_pool = False
        self.conn_id = None
    
    def __enter__(self):
        self.conn, self.pool, self.using_pool, self.conn_id = get_pooled_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return_connection(self.conn, self.pool, self.using_pool, self.conn_id)
        self.conn = None

def log_pool_status():
    """Log current status of the connection pool."""
    pool = get_connection_pool()
    if pool and hasattr(pool, '_used') and hasattr(pool, '_unused'):
        logger.info(f"Pool status: {len(pool._used)} used, {len(pool._unused)} unused")
        
        # Print details about tracked connections
        logger.info(f"Currently tracking {len(connection_tracker)} connections")
        for conn_id, data in connection_tracker.items():
            age = time.time() - data['created_at']
            logger.info(f"Connection {conn_id}: age={age:.1f}s, from_pool={data['from_pool']}")
    else:
        logger.info("Pool status unavailable - pool may not be initialized")

async def safe_background_task(func, *args, **kwargs):
    """
    Safely run a function as a background task with its own database connection.
    
    Args:
        func: The async function to run as a background task
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        asyncio.Task object representing the background task
    """
    conn = None
    pool = None
    using_pool = False
    conn_id = None
    
    try:
        # Get connection for this background task
        conn, pool, using_pool, conn_id = get_pooled_connection()
        logger.info(f"Created connection {conn_id} for background task {func.__name__}")
        
        # Create a new set of args with the connection as the first arg
        new_args = (conn,) + args
        
        # Create the task
        task = asyncio.create_task(func(*new_args, **kwargs))
        
        # Add done callback to close connection
        def close_conn(_):
            nonlocal conn, pool, using_pool, conn_id
            if conn:
                logger.info(f"Background task {func.__name__} completed, returning connection {conn_id}")
                return_connection(conn, pool, using_pool, conn_id)
        
        task.add_done_callback(close_conn)
        
        return task
    except Exception as e:
        # If task creation fails, ensure connection is closed
        if conn:
            logger.error(f"Background task {func.__name__} creation failed, returning connection {conn_id}")
            return_connection(conn, pool, using_pool, conn_id)
        logger.error(f"Failed to create background task: {str(e)}")
        logger.error(f"Background task creation stacktrace: {traceback.format_exc()}")
        raise

# Example usage
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
        logger.error(f"Database operation stacktrace: {traceback.format_exc()}")
        return None