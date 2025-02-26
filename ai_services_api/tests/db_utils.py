import psycopg2
import logging

logger = logging.getLogger(__name__)

class DatabaseConnector:
    def __init__(self):
        self.connection = None

    def get_connection(self):
        if self.connection is None:
            params = {
                "host": "postgres",  # Update the hostname to "postgres"
                "port": 5432,
                "dbname": "aphrc",
                "user": "postgres",
                "password": "p0stgres"
            }
            try:
                self.connection = psycopg2.connect(**params)
                logger.info("Connected to the database")
            except psycopg2.OperationalError as e:
                logger.error(f"Database connection error: {e}")
                logger.error(f"Connection params: {params}")
                raise e
        return self.connection

    def close_connection(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from the database")
