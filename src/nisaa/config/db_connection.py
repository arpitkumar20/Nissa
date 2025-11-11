import os
import logging
from typing import Optional
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, extras
# from psycopg2.pool import ThreadedConnectionPool
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
    raise ValueError("One or more database environment variables are not set")

# Optimized connection pool with better settings
_pg_pool: Optional[pool.ThreadedConnectionPool] = None


def get_pool() -> pool.ThreadedConnectionPool:
    """
    Get or create optimized PostgreSQL connection pool
    """
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = pool.ThreadedConnectionPool(
            minconn=2,  # Increased from 1
            maxconn=30,  # Increased from 20
            dsn=DB_URI,
            # Connection optimization
            options="-c statement_timeout=30000",  # 30s timeout
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
        logger.info("PostgreSQL connection pool created")
    return _pg_pool


def get_connection():
    """
    Get a direct PostgreSQL connection with optimizations
    """
    try:
        conn = psycopg2.connect(
            dsn=CONN_STRING,
            options="-c statement_timeout=30000",
            keepalives=1,
            keepalives_idle=30,
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        raise ConnectionError(f"Unable to connect to database: {e}")


@contextmanager
def get_pooled_connection():
    """
    Context manager for pooled connections with auto-commit
    """
    pool_instance = get_pool()
    conn = None
    try:
        conn = pool_instance.getconn()
        conn.autocommit = False
        yield conn
        conn.commit()
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
    """
    Context manager for RealDictCursor with auto-commit
    """
    if use_pool:
        with get_pooled_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                yield cur
    else:
        conn = None
        try:
            conn = get_connection()
            conn.autocommit = False
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                yield cur
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()


def close_pool():
    """
    Close all connections in the pool
    """
    global _pg_pool
    if _pg_pool:
        _pg_pool.closeall()
        _pg_pool = None
        logger.info("PostgreSQL connection pool closed")

