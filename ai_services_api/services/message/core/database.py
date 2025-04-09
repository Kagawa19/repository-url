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
            'host': os.getenv('POSTGRES_HOST', 'postgres'),  # Default to 'postgres' in Docker
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
        }

def get_db_pool(min_conn=10, max_conn=50, dbname=None):
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

# This class tracks if a connection comes from a pool
class PooledConnection:
    """Wrapper for a database connection that returns it to the pool on close."""
    
    def __init__(self, connection, pool=None, from_pool=False):
        self.connection = connection
        self.pool = pool
        self.from_pool = from_pool
        self.closed = False
    
    def close(self):
        """Return connection to the pool or close it directly."""
        if self.closed:
            return
            
        if self.from_pool and self.pool:
            try:
                self.pool.putconn(self.connection)
                logger.debug("Connection returned to pool")
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")
                self.connection.close()
        else:
            self.connection.close()
            
        self.closed = True
    
    def __getattr__(self, name):
        """Delegate all other attribute access to the underlying connection."""
        return getattr(self.connection, name)

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
            
            # Return wrapped connection
            return PooledConnection(conn, pool, True)
            
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
        
        # Return wrapped connection
        return PooledConnection(conn)
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

# Context manager for easier connection handling
class DatabaseConnection:
    """Context manager for database connections."""
    
    def __init__(self, dbname=None):
        self.dbname = dbname
        self.conn = None
        
    def __enter__(self):
        self.conn = get_db_connection(self.dbname)
        return self.conn
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

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