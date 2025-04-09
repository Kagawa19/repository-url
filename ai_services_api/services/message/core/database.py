import os
import psycopg2
from urllib.parse import urlparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
            'password': parsed_url.password,
            'connect_timeout': 10  # Timeout for connection attempt
        }
    else:
        in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres'),
            'connect_timeout': 10  # Timeout for connection attempt
        }

def get_db_connection(dbname=None):
    """Get a direct database connection (no pooling)."""
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

def close_connection(conn):
    """Close the database connection."""
    if conn:
        conn.close()
        logger.info("Connection closed.")

if __name__ == "__main__":
    try:
        # Test a direct connection
        conn = get_db_connection()
        print("Successfully connected to the database")
        
        # Close the connection after use
        close_connection(conn)
    except Exception as e:
        print(f"Failed to connect: {e}")
