import os
import psycopg2
import logging

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Simple database connection utility
    """
    @classmethod
    def get_connection(cls):
        """
        Create and return a database connection
        
        Returns database connection using environment variables
        """
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'postgres'),
                database=os.getenv('DB_NAME', 'aphrc'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'p0stgres')
            )
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    @classmethod
    def get_connection_params(cls):
        """
        Retrieve connection parameters from environment
        """
        return {
            'host': os.getenv('DB_HOST', 'postgres'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'aphrc'),
            'password': os.getenv('DB_PASSWORD', 'p0stgres')
        }