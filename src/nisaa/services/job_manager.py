import uuid
import json
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from psycopg2 import pool
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """Manages ingestion job lifecycle and status tracking"""
    
    def __init__(self, db_pool: pool.SimpleConnectionPool):
        self.pool = db_pool
    
    def create_job(
        self,
        company_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new ingestion job"""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        conn = None
        
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ingestion_jobs 
                    (job_id, company_name, status, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """, (
                    job_id,
                    company_name,
                    JobStatus.PENDING,
                    json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f"Created job {job_id} for company {company_name}")
            return job_id
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create job: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        **kwargs
    ):
        """Update job status and statistics"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                update_fields = ["status = %s", "updated_at = NOW()"]
                params = [status]
                
                if status == JobStatus.RUNNING:
                    update_fields.append("started_at = NOW()")
                
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    update_fields.append("completed_at = NOW()")
                
                if error_message:
                    update_fields.append("error_message = %s")
                    params.append(error_message)
                
                for key, value in kwargs.items():
                    if key in ['total_files', 'processed_files', 'skipped_files', 
                               'failed_files', 'total_vectors']:
                        update_fields.append(f"{key} = %s")
                        params.append(value)
                
                params.append(job_id)
                
                query = f"""
                    UPDATE ingestion_jobs 
                    SET {', '.join(update_fields)}
                    WHERE job_id = %s
                """
                
                cur.execute(query, params)
            
            conn.commit()
            logger.info(f"Updated job {job_id} status to {status}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update job {job_id}: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        job_id, company_name, status, 
                        total_files, processed_files, skipped_files, failed_files,
                        total_vectors, error_message, metadata,
                        created_at, started_at, completed_at, updated_at
                    FROM ingestion_jobs
                    WHERE job_id = %s
                """, (job_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                job_data = {
                    "job_id": row[0],
                    "company_name": row[1],
                    "status": row[2],
                    "total_files": row[3] or 0,
                    "processed_files": row[4] or 0,
                    "skipped_files": row[5] or 0,
                    "failed_files": row[6] or 0,
                    "total_vectors": row[7] or 0,
                    "error_message": row[8],
                    "metadata": row[9],
                    "created_at": row[10].isoformat() if row[10] else None,
                    "started_at": row[11].isoformat() if row[11] else None,
                    "completed_at": row[12].isoformat() if row[12] else None,
                    "updated_at": row[13].isoformat() if row[13] else None
                }
                
                if job_data["total_files"] > 0:
                    job_data["progress_percentage"] = round(
                        (job_data["processed_files"] / job_data["total_files"]) * 100, 2
                    )
                else:
                    job_data["progress_percentage"] = 0
                
                if job_data["started_at"] and job_data["completed_at"]:
                    start = datetime.fromisoformat(job_data["started_at"])
                    end = datetime.fromisoformat(job_data["completed_at"])
                    job_data["duration_seconds"] = (end - start).total_seconds()
                
                return job_data
                
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def is_file_processed(
        self,
        company_name: str,
        file_path: str,
        file_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Check if file has already been processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        file_path, file_hash, file_size, file_type,
                        vector_count, job_id, processed_at
                    FROM processed_files
                    WHERE company_name = %s AND file_hash = %s
                    ORDER BY processed_at DESC
                    LIMIT 1
                """, (company_name, file_hash))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return {
                    "file_path": row[0],
                    "file_hash": row[1],
                    "file_size": row[2],
                    "file_type": row[3],
                    "vector_count": row[4],
                    "job_id": row[5],
                    "processed_at": row[6].isoformat() if row[6] else None
                }
                
        except Exception as e:
            logger.error(f"Failed to check file: {e}")
            return None
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def mark_file_processed(
        self,
        company_name: str,
        file_path: str,
        file_hash: str,
        file_size: int,
        file_type: str,
        vector_count: int,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark a file as processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed_files
                    (company_name, file_path, file_hash, file_size, file_type,
                     vector_count, job_id, processed_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (company_name, file_hash) 
                    DO UPDATE SET
                        file_path = EXCLUDED.file_path,
                        vector_count = EXCLUDED.vector_count,
                        job_id = EXCLUDED.job_id,
                        processed_at = NOW(),
                        metadata = EXCLUDED.metadata
                """, (
                    company_name, file_path, file_hash, file_size, file_type,
                    vector_count, job_id, json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f"Marked file as processed: {file_path}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to mark file processed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def is_website_processed(
        self,
        company_name: str,
        website_url: str,
        content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Check if website has already been processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        website_url, content_hash, page_count,
                        vector_count, job_id, processed_at
                    FROM processed_websites
                    WHERE company_name = %s AND content_hash = %s
                    ORDER BY processed_at DESC
                    LIMIT 1
                """, (company_name, content_hash))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return {
                    "website_url": row[0],
                    "content_hash": row[1],
                    "page_count": row[2],
                    "vector_count": row[3],
                    "job_id": row[4],
                    "processed_at": row[5].isoformat() if row[5] else None
                }
                
        except Exception as e:
            logger.error(f"Failed to check website: {e}")
            return None
        finally:
            if conn:
                self.pool.putconn(conn)

    def mark_website_processed(
        self,
        company_name: str,
        website_url: str,
        content_hash: str,
        page_count: int,
        vector_count: int,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark a website as processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed_websites
                    (company_name, website_url, content_hash, page_count,
                    vector_count, job_id, processed_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (company_name, content_hash) 
                    DO UPDATE SET
                        website_url = EXCLUDED.website_url,
                        page_count = EXCLUDED.page_count,
                        vector_count = EXCLUDED.vector_count,
                        job_id = EXCLUDED.job_id,
                        processed_at = NOW(),
                        metadata = EXCLUDED.metadata
                """, (
                    company_name, website_url, content_hash, page_count,
                    vector_count, job_id, json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f"Marked website as processed: {website_url}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to mark website processed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def is_database_processed(
        self,
        company_name: str,
        db_uri: str,
        db_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Check if database has already been processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        db_uri, db_hash, db_type, db_name,
                        vector_count, job_id, processed_at
                    FROM processed_databases
                    WHERE company_name = %s AND db_hash = %s
                    ORDER BY processed_at DESC
                    LIMIT 1
                """, (company_name, db_hash))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return {
                    "db_uri": row[0],
                    "db_hash": row[1],
                    "db_type": row[2],
                    "db_name": row[3],
                    "vector_count": row[4],
                    "job_id": row[5],
                    "processed_at": row[6].isoformat() if row[6] else None
                }
                
        except Exception as e:
            logger.error(f"Failed to check database: {e}")
            return None
        finally:
            if conn:
                self.pool.putconn(conn)

    def mark_database_processed(
        self,
        company_name: str,
        db_uri: str,
        db_hash: str,
        db_type: str,
        db_name: str,
        vector_count: int,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark a database as processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed_databases
                    (company_name, db_uri, db_hash, db_type, db_name,
                     vector_count, job_id, processed_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (company_name, db_hash) 
                    DO UPDATE SET
                        db_uri = EXCLUDED.db_uri,
                        vector_count = EXCLUDED.vector_count,
                        job_id = EXCLUDED.job_id,
                        processed_at = NOW(),
                        metadata = EXCLUDED.metadata
                """, (
                    company_name, db_uri, db_hash, db_type, db_name,
                    vector_count, job_id, json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f"Marked database as processed: {db_name}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to mark database processed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def is_zoho_report_processed(
        self,
        company_name: str,
        report_name: str,
        content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if Zoho report has already been processed
        
        Uses content hash to detect if data has changed
        """
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        report_name, app_name, content_hash, record_count,
                        vector_count, job_id, processed_at
                    FROM processed_zoho_reports
                    WHERE company_name = %s AND content_hash = %s
                    ORDER BY processed_at DESC
                    LIMIT 1
                """, (company_name, content_hash))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return {
                    "report_name": row[0],
                    "app_name": row[1],
                    "content_hash": row[2],
                    "record_count": row[3],
                    "vector_count": row[4],
                    "job_id": row[5],
                    "processed_at": row[6].isoformat() if row[6] else None
                }
                
        except Exception as e:
            logger.error(f"Failed to check Zoho report: {e}")
            return None
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def mark_zoho_report_processed(
        self,
        company_name: str,
        report_name: str,
        content_hash: str,
        record_count: int,
        vector_count: int,
        job_id: str,
        app_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark a Zoho report as processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed_zoho_reports
                    (company_name, report_name, app_name, content_hash, record_count,
                     vector_count, job_id, processed_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (company_name, content_hash) 
                    DO UPDATE SET
                        report_name = EXCLUDED.report_name,
                        app_name = EXCLUDED.app_name,
                        record_count = EXCLUDED.record_count,
                        vector_count = EXCLUDED.vector_count,
                        job_id = EXCLUDED.job_id,
                        processed_at = NOW(),
                        metadata = EXCLUDED.metadata
                """, (
                    company_name, report_name, app_name, content_hash, record_count,
                    vector_count, job_id, json.dumps(metadata or {})
                ))
            
            conn.commit()
            logger.info(f"Marked Zoho report as processed: {report_name}")

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to mark Zoho report processed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
                
    def is_db_table_processed(
        self,
        company_name: str,
        db_hash: str,
        table_name: str,
        table_hash: str
    ) -> Optional[Dict]:
        """Check if a specific table has been processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT job_id, vector_count, processed_at
                    FROM processed_db_tables
                    WHERE company_name = %s 
                    AND db_hash = %s
                    AND table_name = %s
                    AND table_hash = %s
                    ORDER BY processed_at DESC
                    LIMIT 1
                """, (company_name, db_hash, table_name, table_hash))
                
                row = cur.fetchone()
                if row:
                    return {
                        "job_id": row[0],
                        "vector_count": row[1],
                        "processed_at": row[2]
                    }
                return None
        finally:
            if conn:
                self.pool.putconn(conn)

    def mark_db_table_processed(
        self,
        company_name: str,
        db_uri: str,
        db_hash: str,
        table_name: str,
        table_hash: str,
        row_count: int,
        vector_count: int,
        job_id: str,
        metadata: Dict = None
    ):
        """Mark a specific table as processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed_db_tables 
                    (company_name, db_uri, db_hash, table_name, table_hash, 
                    row_count, vector_count, job_id, metadata, processed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (company_name, db_hash, table_name, table_hash)
                    DO UPDATE SET
                        row_count = EXCLUDED.row_count,
                        vector_count = EXCLUDED.vector_count,
                        job_id = EXCLUDED.job_id,
                        processed_at = NOW(),
                        metadata = EXCLUDED.metadata
                """, (
                    company_name, db_uri, db_hash, table_name, table_hash,
                    row_count, vector_count, job_id, json.dumps(metadata or {})
                ))
                conn.commit()
                logger.info(f"Marked table {table_name} as processed")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to mark table processed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)