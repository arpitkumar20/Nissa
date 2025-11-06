import os
from psycopg2 import pool
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASS = os.getenv("DATABASE_PASS")
DB_NAME = os.getenv("DB_NAME")

DB_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DB_NAME}"

if not DB_URI:
    raise ValueError("DB_URI environment variable is not set!")

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
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id VARCHAR(50) PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    skipped_files INTEGER DEFAULT 0,
                    failed_files INTEGER DEFAULT 0,
                    total_vectors INTEGER DEFAULT 0,
                    error_message TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_job_company 
                ON ingestion_jobs(company_name);
                
                CREATE INDEX IF NOT EXISTS idx_job_status 
                ON ingestion_jobs(status);
                
                CREATE INDEX IF NOT EXISTS idx_job_created 
                ON ingestion_jobs(created_at DESC);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id SERIAL PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash VARCHAR(64) NOT NULL,
                    file_size BIGINT,
                    file_type VARCHAR(50),
                    vector_count INTEGER DEFAULT 0,
                    job_id VARCHAR(50),
                    processed_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(company_name, file_hash)
                );
                
                CREATE INDEX IF NOT EXISTS idx_processed_company_path 
                ON processed_files(company_name, file_path);
                
                CREATE INDEX IF NOT EXISTS idx_processed_hash 
                ON processed_files(file_hash);
                
                CREATE INDEX IF NOT EXISTS idx_processed_company_job 
                ON processed_files(company_name, job_id);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_websites (
                    id SERIAL PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    website_url TEXT NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    page_count INTEGER DEFAULT 0,
                    vector_count INTEGER DEFAULT 0,
                    job_id VARCHAR(50),
                    processed_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(company_name, content_hash)
                );
                
                CREATE INDEX IF NOT EXISTS idx_processed_website_company_url 
                ON processed_websites(company_name, website_url);
                
                CREATE INDEX IF NOT EXISTS idx_processed_website_hash 
                ON processed_websites(content_hash);
                
                CREATE INDEX IF NOT EXISTS idx_processed_website_job 
                ON processed_websites(company_name, job_id);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_databases (
                    id SERIAL PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    db_uri TEXT NOT NULL,
                    db_hash VARCHAR(64) NOT NULL,
                    db_type VARCHAR(50),
                    db_name VARCHAR(255),
                    vector_count INTEGER DEFAULT 0,
                    job_id VARCHAR(50),
                    processed_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(company_name, db_hash)
                );
                
                CREATE INDEX IF NOT EXISTS idx_processed_db_company_uri 
                ON processed_databases(company_name, db_uri);
                
                CREATE INDEX IF NOT EXISTS idx_processed_db_hash 
                ON processed_databases(db_hash);
                
                CREATE INDEX IF NOT EXISTS idx_processed_db_job 
                ON processed_databases(company_name, job_id);
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_zoho_reports (
                    id SERIAL PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    report_name VARCHAR(255) NOT NULL,
                    app_name VARCHAR(255),
                    content_hash VARCHAR(64) NOT NULL,
                    record_count INTEGER DEFAULT 0,
                    vector_count INTEGER DEFAULT 0,
                    job_id VARCHAR(50),
                    processed_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(company_name, content_hash)
                );
                
                CREATE INDEX IF NOT EXISTS idx_processed_zoho_company_report 
                ON processed_zoho_reports(company_name, report_name);
                
                CREATE INDEX IF NOT EXISTS idx_processed_zoho_hash 
                ON processed_zoho_reports(content_hash);
                
                CREATE INDEX IF NOT EXISTS idx_processed_zoho_job 
                ON processed_zoho_reports(company_name, job_id);
            """)
            
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