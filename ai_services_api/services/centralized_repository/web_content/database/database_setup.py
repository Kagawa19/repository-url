import logging
from typing import Optional, Dict, List, Any
import psycopg2
from psycopg2.extras import DictCursor, Json
from contextlib import contextmanager
from datetime import datetime
import hashlib
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
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def insert_webpage(url: str, title: str, content: str, metadata: Dict, last_modified: str) -> Optional[int]:
    """Insert or update a webpage record"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO webpages (url, title, content, content_hash, metadata, timestamp, last_modified)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE
            SET title = EXCLUDED.title,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                timestamp = EXCLUDED.timestamp,
                last_modified = EXCLUDED.last_modified
            RETURNING id
        """, (url, title, content, content_hash, Json(metadata), datetime.now(), last_modified))
        return cursor.fetchone()['id']

def insert_expert(url: str, name: str, content: str, metadata: Dict, last_modified: str) -> Optional[int]:
    """Insert or update an expert profile"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO experts (url, name, content, content_hash, metadata, timestamp, last_modified)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE
            SET name = EXCLUDED.name,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                timestamp = EXCLUDED.timestamp,
                last_modified = EXCLUDED.last_modified
            RETURNING id
        """, (url, name, content, content_hash, Json(metadata), datetime.now(), last_modified))
        return cursor.fetchone()['id']

def insert_pdf(url: str, file_path: str, metadata: Dict, content_hash: str) -> Optional[int]:
    """Insert or update a PDF record"""
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO pdfs (url, file_path, content_hash, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE
            SET file_path = EXCLUDED.file_path,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                timestamp = EXCLUDED.timestamp
            RETURNING id
        """, (url, file_path, content_hash, Json(metadata), datetime.now()))
        return cursor.fetchone()['id']

def insert_pdf_chunk(pdf_id: int, chunk_index: int, content: str) -> Optional[int]:
    """Insert a PDF chunk"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO pdf_chunks (pdf_id, chunk_index, content, content_hash, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (pdf_id, chunk_index, content, content_hash, datetime.now()))
        return cursor.fetchone()['id']

def insert_publication(url: str, title: str, authors: List[str], abstract: str, publication_date: Optional[str], content_type: str, content_id: int, content: str, metadata: Dict) -> Optional[int]:
    """Insert or update a publication record"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO publications (url, title, authors, abstract, publication_date, content_type, content_id, content_hash, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE
            SET title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                abstract = EXCLUDED.abstract,
                publication_date = EXCLUDED.publication_date,
                content_type = EXCLUDED.content_type,
                content_id = EXCLUDED.content_id,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                timestamp = EXCLUDED.timestamp
            RETURNING id
        """, (url, title, authors, abstract, publication_date, content_type, content_id, content_hash, Json(metadata), datetime.now()))
        return cursor.fetchone()['id']

def insert_embedding(content_type: str, content_id: int, embedding: List[float], content_hash: str) -> Optional[int]:
    """Insert an embedding"""
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO embeddings (content_type, content_id, embedding, content_hash, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (content_type, content_id, Json(embedding), content_hash, datetime.now()))
        return cursor.fetchone()['id']

def update_scrape_state(last_run: str, processed_urls: List[str], failed_urls: List[str], content_hashes: Dict) -> None:
    """Update the scrape state"""
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("""
            INSERT INTO scrape_state (last_run, timestamp, processed_urls, failed_urls, content_hashes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET last_run = EXCLUDED.last_run,
                timestamp = EXCLUDED.timestamp,
                processed_urls = EXCLUDED.processed_urls,
                failed_urls = EXCLUDED.failed_urls,
                content_hashes = EXCLUDED.content_hashes
        """, (last_run, datetime.now(), processed_urls, failed_urls, Json(content_hashes)))

def get_scrape_state() -> Dict:
    """Retrieve the scrape state"""
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("SELECT * FROM scrape_state ORDER BY timestamp DESC LIMIT 1")
        state = cursor.fetchone()
        if state:
            return {
                'last_run': state['last_run'],
                'timestamp': state['timestamp'],
                'processed_urls': state['processed_urls'],
                'failed_urls': state['failed_urls'],
                'content_hashes': state['content_hashes']
            }
        return {
            'last_run': None,
            'timestamp': None,
            'processed_urls': [],
            'failed_urls': [],
            'content_hashes': {}
        }

def check_content_changes(url: str, new_hash: str) -> bool:
    """Check if content has changed based on stored hash"""
    with get_db_cursor(cursor_factory=DictCursor) as (cursor, conn):
        cursor.execute("SELECT content_hashes->>%s AS hash FROM scrape_state ORDER BY timestamp DESC LIMIT 1", (url,))
        stored_hash = cursor.fetchone()['hash'] if cursor.fetchone() else None
        return stored_hash is None or stored_hash != new_hash