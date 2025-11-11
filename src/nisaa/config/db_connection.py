import os
import logging
import threading
from typing import Optional
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASS = os.getenv("DATABASE_PASS")
DB_NAME = os.getenv("DB_NAME")

CONN_STRING = (
    f"dbname='{DB_NAME}' "
    f"user='{DATABASE_USER}' "
    f"password='{DATABASE_PASS}' "
    f"host='{DATABASE_HOST}' "
    f"port='{DATABASE_PORT}'"
)

DB_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DB_NAME}"

if not all([DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASS, DB_NAME]):
    raise ValueError("One or more database environment variables are not set!")

_pg_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> pool.ThreadedConnectionPool:
    """
    Get or create the PostgreSQL connection pool (thread-safe).
    
    CRITICAL FIX:
    - Uses ThreadedConnectionPool instead of SimpleConnectionPool
    - Double-checked locking pattern for thread safety
    - Pool creation is synchronized
    
    Returns:
        ThreadedConnectionPool: The connection pool instance
    """
    global _pg_pool
    
    if _pg_pool is not None:
        return _pg_pool
    
    with _pool_lock:
        if _pg_pool is None:
            _pg_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=DB_URI
            )
            logger.info(" PostgreSQL ThreadedConnectionPool created (thread-safe)")
    
    return _pg_pool


def get_connection():
    """
    Get a direct PostgreSQL connection (non-pooled).
    Useful for simple operations that don't need pooling.
    """
    try:
        return psycopg2.connect(dsn=CONN_STRING)
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        raise ConnectionError(f"Unable to connect to database: {e}")


@contextmanager
def get_pooled_connection():
    """
    Context manager for getting a connection from the pool (thread-safe).
    Automatically commits on success, rolls back on error, and returns connection to pool.

    """
    pool_instance = get_pool()
    conn = None
    try:
        conn = pool_instance.getconn()
        yield conn
        conn.commit()  # ✅ Auto-commit on success
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        if conn:
            pool_instance.putconn(conn)


@contextmanager
def get_dict_cursor(use_pool: bool = True):

    if use_pool:
        with get_pooled_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                yield cur
    else:
        conn = None
        try:
            conn = get_connection()
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                yield cur
                conn.commit()
        finally:
            if conn:
                conn.close()


def close_pool():
    """
    Close all connections in the pool (thread-safe).
    Should be called during application shutdown.
    """
    global _pg_pool
    
    with _pool_lock:
        if _pg_pool:
            _pg_pool.closeall()
            _pg_pool = None
            logger.info("PostgreSQL connection pool closed")