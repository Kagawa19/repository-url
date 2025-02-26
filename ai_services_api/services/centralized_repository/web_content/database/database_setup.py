# database/database_setup.py

import logging
from typing import Optional, Dict, List
import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import redis

from ..config.settings import DATABASE_CONFIG

logger = logging.getLogger(__name__)

@contextmanager
def get_db_cursor(cursor_factory=None):
    """Database cursor context manager"""
    conn = None
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor(cursor_factory=cursor_factory)
        yield cursor, conn
    finally:
        if conn:
            conn.close()

class DatabaseInitializer:
    """Minimal database initialization for content hashing and tracking"""
    
    def initialize_schema(self):
        """Initialize minimal database schema"""
        try:
            with get_db_cursor() as (cur, conn):
                # Table to track content hashes
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS content_hashes (
                        url TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL,
                        last_checked TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Only store embedding reference in database
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS content_embeddings (
                        url TEXT PRIMARY KEY,
                        embedding_key TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Schema initialization complete")
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            raise

class ContentTracker:
    """Tracks content changes and embedding references"""
    
    def has_content_changed(self, url: str, new_hash: str) -> bool:
        """Check if content has changed based on hash"""
        try:
            with get_db_cursor() as (cur, conn):
                cur.execute("""
                    SELECT content_hash 
                    FROM content_hashes 
                    WHERE url = %s
                """, (url,))
                
                result = cur.fetchone()
                if result is None:
                    # New URL, insert hash
                    cur.execute("""
                        INSERT INTO content_hashes (url, content_hash)
                        VALUES (%s, %s)
                    """, (url, new_hash))
                    conn.commit()
                    return True
                
                old_hash = result[0]
                if old_hash != new_hash:
                    # Content changed, update hash
                    cur.execute("""
                        UPDATE content_hashes 
                        SET content_hash = %s,
                            last_modified = CURRENT_TIMESTAMP
                        WHERE url = %s
                    """, (new_hash, url))
                    conn.commit()
                    return True
                
                # Update last checked timestamp
                cur.execute("""
                    UPDATE content_hashes 
                    SET last_checked = CURRENT_TIMESTAMP
                    WHERE url = %s
                """, (url,))
                conn.commit()
                return False
                
        except Exception as e:
            logger.error(f"Error checking content change: {str(e)}")
            return True  # Assume changed on error to force update

    def update_embedding_reference(self, url: str, embedding_key: str):
        """Store or update embedding reference"""
        try:
            with get_db_cursor() as (cur, conn):
                cur.execute("""
                    INSERT INTO content_embeddings (url, embedding_key)
                    VALUES (%s, %s)
                    ON CONFLICT (url) DO UPDATE
                    SET embedding_key = EXCLUDED.embedding_key,
                        updated_at = CURRENT_TIMESTAMP
                """, (url, embedding_key))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating embedding reference: {str(e)}")
            raise

    def get_embedding_key(self, url: str) -> Optional[str]:
        """Get embedding key for URL"""
        try:
            with get_db_cursor() as (cur, conn):
                cur.execute("""
                    SELECT embedding_key
                    FROM content_embeddings
                    WHERE url = %s
                """, (url,))
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting embedding key: {str(e)}")
            return None

    def get_urls_to_check(self, older_than_hours: int = 24) -> List[str]:
        """Get URLs that haven't been checked recently"""
        try:
            with get_db_cursor() as (cur, conn):
                cur.execute("""
                    SELECT url
                    FROM content_hashes
                    WHERE last_checked < NOW() - INTERVAL '%s hours'
                    ORDER BY last_checked ASC
                """, (older_than_hours,))
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting URLs to check: {str(e)}")
            return []