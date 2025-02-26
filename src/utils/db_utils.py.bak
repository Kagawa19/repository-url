import os
import psycopg2
from urllib.parse import urlparse
import logging
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    @staticmethod
    def get_connection_params() -> Dict[str, str]:
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            parsed_url = urlparse(database_url)
            return {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'dbname': parsed_url.path[1:],
                'user': parsed_url.username,
                'password': parsed_url.password
            }
        
        in_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        return {
            'host': os.getenv('POSTGRES_HOST', '167.86.85.127'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'dbname': os.getenv('POSTGRES_DB', 'aphrc'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'p0stgres')
        }

    @staticmethod
    def get_connection(dbname: Optional[str] = None):
        params = DatabaseConnector.get_connection_params()
        if dbname:
            params['dbname'] = dbname
            
        try:
            conn = psycopg2.connect(**params)
            with conn.cursor() as cur:
                cur.execute('SET search_path TO public')
            logger.info(f"Connected to database: {params['dbname']} at {params['host']}")
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection error: {e}")
            logger.error(f"Connection params: {params}")
            raise