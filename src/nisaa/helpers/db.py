import os
from psycopg2 import pool
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASS = os.getenv("DATABASE_PASS")
DB_NAME = os.getenv("DB_NAME")

DB_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DB_NAME}"

if not DB_URI:
    raise ValueError("DB_URI environment variable is not set!")

# Connection pool
_pg_pool: Optional[pool.SimpleConnectionPool] = None


def get_pool():
    """Get or create the connection pool"""
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = pool.SimpleConnectionPool(minconn=1, maxconn=20, dsn=DB_URI)
        logger.info("PostgreSQL connection pool created")
    return _pg_pool


def save_message(thread_id: str, role: str, content: str):
    """
    Save a single message to PostgreSQL

    Args:
        thread_id: User's phone number (thread identifier)
        role: 'user' or 'assistant'
        content: Message content
    """
    conn = None
    try:
        pool = get_pool()
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (thread_id, role, content, created_at) 
                VALUES (%s, %s, %s, NOW())
                """,
                (thread_id, role, content),
            )
        conn.commit()
        logger.info(f"Saved {role} message for thread {thread_id}")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to save message: {e}")
        raise e
    finally:
        if conn:
            pool.putconn(conn)


def initialize_db():
    """Initialize database tables if they don't exist"""
    conn = None
    try:
        pool = get_pool()
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    thread_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_thread_id 
                ON messages(thread_id);
                
                CREATE INDEX IF NOT EXISTS idx_thread_created 
                ON messages(thread_id, created_at);
            """
            )
        conn.commit()
        logger.info("Database tables initialized")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to initialize database: {e}")
        raise e
    finally:
        if conn:
            pool.putconn(conn)
