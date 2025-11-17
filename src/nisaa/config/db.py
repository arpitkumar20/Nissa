import logging
from src.nisaa.config.db_connection import get_pooled_connection, get_pool

logger = logging.getLogger(__name__)

def initialize_db():
    """Initialize database tables if they don't exist"""
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                # Ingestion jobs table
                cur.execute(
                    """
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
                """
                )

                # Processed files table
                cur.execute(
                    """
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
                """
                )

                # Processed websites table
                cur.execute(
                    """
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
                """
                )

                # Processed databases table (LEGACY - URI level)
                cur.execute(
                    """
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
                """
                )

                # NEW: Processed database tables (TABLE-LEVEL TRACKING)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processed_db_tables (
                        id SERIAL PRIMARY KEY,
                        company_name VARCHAR(255) NOT NULL,
                        db_uri TEXT NOT NULL,
                        db_hash VARCHAR(64) NOT NULL,
                        table_name VARCHAR(255) NOT NULL,
                        table_hash VARCHAR(64) NOT NULL,
                        row_count INTEGER DEFAULT 0,
                        vector_count INTEGER DEFAULT 0,
                        job_id VARCHAR(50) NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        processed_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(company_name, db_hash, table_name, table_hash)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_db_tables_company 
                    ON processed_db_tables(company_name);
                    
                    CREATE INDEX IF NOT EXISTS idx_db_tables_db_hash 
                    ON processed_db_tables(db_hash);
                    
                    CREATE INDEX IF NOT EXISTS idx_db_tables_table_name 
                    ON processed_db_tables(table_name);
                    
                    CREATE INDEX IF NOT EXISTS idx_db_tables_job_id 
                    ON processed_db_tables(job_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_db_tables_dedup 
                    ON processed_db_tables(company_name, db_hash, table_name, table_hash);
                """
                )

                # Processed Zoho reports table
                cur.execute(
                    """
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
                """
                )

                # Leads table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS leads (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        wa_id VARCHAR(100),
                        first_name TEXT,
                        full_name TEXT,
                        phone VARCHAR(20) NOT NULL UNIQUE,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_leads_phone ON leads(phone);
                    CREATE INDEX IF NOT EXISTS idx_leads_is_active ON leads(is_active);
                """)

            conn.commit()

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
