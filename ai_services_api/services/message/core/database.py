import os
import psycopg2
import psycopg2.pool
from urllib.parse import urlparse
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global connection pool (singleton)
_DB_POOL = None
_pool_lock = threading.Lock()

def get_connection_params():
    """Get database connection parameters from environment variables."""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        parsed_url = urlparse(database_url)
        return {
            'host': parsed_url.hostname,
            'port': parsed_url.port,
            'dbname': parsed_url.path[1:],  # Remove leading '/'
            'user': parsed_url.username,
            'password': parsed_url.password
        }
    else:
        in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
        }

def get_db_pool(min_conn=5, max_conn=20, dbname=None):
    """Get or create database connection pool (thread-safe singleton)."""
    global _DB_POOL
    
    # Use double-checked locking pattern for thread safety
    if _DB_POOL is None:
        with _pool_lock:
            if _DB_POOL is None:
                params = get_connection_params()
                if dbname:
                    params['dbname'] = dbname
                
                try:
                    logger.info(f"Creating database connection pool with {min_conn}-{max_conn} connections")
                    _DB_POOL = psycopg2.pool.ThreadedConnectionPool(
                        min_conn,
                        max_conn,
                        **params
                    )
                    logger.info(f"Connection pool created successfully for {params['dbname']} at {params['host']}")
                except Exception as e:
                    logger.error(f"Error creating connection pool: {e}")
                    logger.error(f"Connection params: {params}")
                    raise
    
    return _DB_POOL

def get_db_connection(dbname=None):
    """Get a database connection from the pool or create a new one if pool fails."""
    pool = get_db_pool(dbname=dbname)
    
    if pool:
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                # Explicitly set the schema
                cur.execute('SET search_path TO public')
            
            logger.info(f"Successfully got connection from pool")
            
            # Return connection with custom close method to return to pool
            orig_close = conn.close
            def pooled_close():
                try:
                    pool.putconn(conn)
                    logger.debug("Connection returned to pool")
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    orig_close()
            
            # Replace the close method
            conn.close = pooled_close
            return conn
            
        except Exception as pool_error:
            logger.error(f"Error getting connection from pool: {pool_error}")
            logger.info("Falling back to direct connection")
    
    # Fallback to direct connection if pooling fails
    params = get_connection_params()
    if dbname:
        params['dbname'] = dbname
        
    try:
        conn = psycopg2.connect(**params)
        with conn.cursor() as cur:
            # Explicitly set the schema
            cur.execute('SET search_path TO public')
        logger.info(f"Successfully connected to database: {params['dbname']} at {params['host']}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to the database: {e}")
        logger.error(f"Connection params: {params}")
        raise

def close_connection_pool():
    """Close all connections in the pool."""
    global _DB_POOL
    if _DB_POOL is not None:
        logger.info("Closing database connection pool")
        _DB_POOL.closeall()
        _DB_POOL = None

if __name__ == "__main__":
    try:
        # Test a pooled connection
        conn = get_db_connection()
        print("Successfully connected to the database")
        
        # Test that returning to pool works
        conn.close()
        
        # Test closing the pool
        close_connection_pool()
    except Exception as e:
        print(f"Failed to connect: {e}")