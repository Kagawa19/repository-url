# File: ai_services_api/services/search/core/database.py
import asyncpg
import os
import logging

logger = logging.getLogger(__name__)

# Database connection pool
_pool: asyncpg.pool.Pool = None

# Read credentials from environment
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "p0stgres"),
    "database": os.getenv("POSTGRES_DB", "aphrc"),
    "min_size": 5,
    "max_size": 15,
    "command_timeout": 30
}

async def create_pool():
    """Initialize the PostgreSQL connection pool"""
    global _pool
    try:
        if not _pool:
            logger.info("Creating database connection pool")
            _pool = await asyncpg.create_pool(**DB_CONFIG)
            logger.info(f"Connected to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        return _pool
    except Exception as e:
        logger.error(f"Failed to create database pool: {str(e)}")
        raise

async def close_pool():
    """Close all connections in the pool"""
    global _pool
    if _pool:
        logger.info("Closing database connection pool")
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")

async def test_connection():
    """Validate database connectivity"""
    conn = await get_connection()
    try:
        # Simple test query
        result = await conn.fetchval("SELECT 1")
        logger.info("Database connection test successful")
        return result == 1
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False
    finally:
        await release_connection(conn)

async def get_connection() -> asyncpg.Connection:
    """Acquire a connection from the pool"""
    if not _pool:
        await create_pool()
    return await _pool.acquire()

async def release_connection(conn: asyncpg.Connection):
    """Release a connection back to the pool"""
    if _pool and conn:
        await _pool.release(conn)