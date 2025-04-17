"""
Database setup and utility functions for PostgreSQL.

WARNING: Ensure all database operations are atomic and avoid importing application logic to prevent circular dependencies.
"""
import logging
import os
from typing import Dict, List, Any
from contextlib import contextmanager
from psycopg2.extras import Json, DictCursor
import psycopg2
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")

# Initialize connection pool
pool = SimpleConnectionPool(minconn=1, maxconn=20, dsn=DATABASE_URL)

@contextmanager
def get_db_cursor(cursor_factory=None):
    """Context manager for database cursor"""
    conn = pool.getconn()
    try:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        yield cursor, conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        pool.putconn(conn)

def insert_embedding(content_type: str, content_id: int, embedding: List[float], content_hash: str) -> int:
    """Insert an embedding into the database"""
    try:
        with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
            cursor.execute("""
                INSERT INTO embeddings (content_type, content_id, embedding, content_hash, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                RETURNING id
            """, (content_type, content_id, Json(embedding), content_hash))
            return cursor.fetchone()['id']
    except Exception as e:
        logger.error(f"Error inserting embedding: {str(e)}")
        raise

def get_scrape_state() -> Dict:
    """Retrieve the latest scrape state"""
    try:
        with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
            cursor.execute("""
                SELECT last_run, processed_urls, failed_urls, content_hashes, timestamp
                FROM scrape_state
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return {
                    'last_run': row['last_run'].isoformat() if row['last_run'] else None,
                    'processed_urls': row['processed_urls'] or [],
                    'failed_urls': row['failed_urls'] or [],
                    'content_hashes': row['content_hashes'] or {},
                    'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None
                }
            return {
                'last_run': None,
                'processed_urls': [],
                'failed_urls': [],
                'content_hashes': {},
                'timestamp': None
            }
    except Exception as e:
        logger.error(f"Error retrieving scrape state: {str(e)}")
        raise

def update_scrape_state(last_run: str, processed_urls: List[str], failed_urls: List[str], content_hashes: Dict) -> None:
    """Update the scrape state"""
    try:
        with get_db_cursor() as (cursor, conn):
            cursor.execute("""
                INSERT INTO scrape_state (last_run, processed_urls, failed_urls, content_hashes, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
            """, (
                last_run,
                processed_urls,
                failed_urls,
                Json(content_hashes)
            ))
    except Exception as e:
        logger.error(f"Error updating scrape state: {str(e)}")
        raise

def check_content_changes(items: List[Dict]) -> List[Dict]:
    """Check for content changes (simplified for compatibility)"""
    return items  # Implement actual logic as needed