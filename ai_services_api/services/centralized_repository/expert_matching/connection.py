import os
import psycopg2
from psycopg2 import sql

def get_db_connection():
    """Establish a connection to the database using connection parameters."""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        return psycopg2.connect(database_url)
    
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME', 'your_database_name')
    user = os.getenv('DB_USER', 'your_username')
    password = os.getenv('DB_PASSWORD', 'your_password')
    
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )