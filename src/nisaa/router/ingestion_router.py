import ast
import os
import asyncio
import json
import base64
import re
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from nisaa.services.file_deduplication import FileDeduplicator, DBDeduplicator, ZohoDeduplicator
from nisaa.services.job_manager import JobManager, JobStatus
from nisaa.utils.zoho_data_downloader import ZohoCreatorExporter
from nisaa.utils.s3_downloader import download_all_files_from_s3
from src.nisaa.config.logger import logger
from src.nisaa.config.db import get_pool
from src.nisaa.controllers.ingestion_pipeline import DataIngestionPipeline

router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])

# Decode base64 encoded Zoho credentials from the request payload
def decode_zoho_credentials(base64_str: str) -> Dict[str, str]:
    """Decode base64 encoded Zoho credentials"""
    try:
        decoded_bytes = base64.b64decode(base64_str)
        decoded_str = decoded_bytes.decode("utf-8")
        try:
            credentials = json.loads(decoded_str)
        except json.JSONDecodeError:
            credentials = ast.literal_eval(decoded_str)

        logger.info("Zoho credentials decoded successfully")
        return credentials
    except Exception as e:
        logger.error(f"Failed to decode credentials: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid credentials: {str(e)}")

# Save company namespace to JSON file
def save_name(namespace: str, folder_path: str = "web_info", filename: str = "web_info.json"):
    """Save namespace to JSON file"""
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data["namespace"] = namespace

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Namespace '{namespace}' saved to {file_path} successfully!")
    return namespace

# Sanitize company name to prevent path traversal attacks
def sanitize_company_name(company_name: str) -> str:
    """
    Sanitize company name to prevent path traversal attacks

    Returns:
        Safe company name or raises HTTPException
    """
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required")

    # Only allow alphanumeric, underscore, hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", company_name):
        raise HTTPException(
            status_code=400,
            detail="Company name can only contain letters, numbers, hyphens, and underscores",
        )

    # Prevent overly long names
    if len(company_name) > 100:
        raise HTTPException(
            status_code=400, detail="Company name too long (max 100 chars)"
        )

    # Additional check: resolve path and ensure it's within base_directory
    base_dir = Path("data").resolve()
    company_dir = (base_dir / company_name).resolve()

    if not str(company_dir).startswith(str(base_dir)):
        raise HTTPException(status_code=400, detail="Invalid company name")

    return company_name

# Run Knowledgebase ingestion pipeline for a company
async def run_ingestion_pipeline(
    job_id: str,
    company_name: str,
    company_directory: str,
    s3_file_keys_list: List[str],
    zoho_cred_encrypted: Optional[str],
    zoho_region: str,
    db_uri_list: List[str],
    website_url_list: List[str],
):
    """Background task to run ingestion pipeline"""
    job_manager = JobManager(get_pool())

    try:
        job_manager.update_job_status(job_id, JobStatus.RUNNING)
        logger.info(f"[{job_id}] Starting ingestion for {company_name}")

        zoho_files = []
        all_file_paths = []

        if zoho_cred_encrypted and zoho_cred_encrypted.strip():
            logger.info(f"[{job_id}] PHASE 1: ZOHO DATA EXTRACTION")

            try:
                decoded_creds = decode_zoho_credentials(zoho_cred_encrypted)
                zoho_client_id = decoded_creds.get("zoho_client_id")
                zoho_client_secret = decoded_creds.get("zoho_client_secret")
                zoho_refresh_token = decoded_creds.get("zoho_refresh_token")

                if not all([zoho_client_id, zoho_client_secret, zoho_refresh_token]):
                    raise ValueError("Incomplete Zoho credentials")

                zoho_exporter = ZohoCreatorExporter(
                    client_id=zoho_client_id,
                    client_secret=zoho_client_secret,
                    refresh_token=zoho_refresh_token,
                    output_dir=company_directory,
                    zoho_region=zoho_region,
                )

                zoho_stats = zoho_exporter.export_all_data()
                zoho_files = zoho_stats["json_file_paths"]

                logger.info(f"[{job_id}] Zoho: {len(zoho_files)} files exported")

            except Exception as zoho_error:
                logger.error(f"[{job_id}] Zoho extraction failed: {zoho_error}")
                raise

        if len(s3_file_keys_list) > 0:
            logger.info(f"[{job_id}] PHASE 2: S3 FILE DOWNLOAD")

            downloaded_s3_files = await asyncio.to_thread(
                download_all_files_from_s3,
                file_list=s3_file_keys_list,
                company_name=company_name,
            )

            all_file_paths.extend(downloaded_s3_files)
            logger.info(
                f"[{job_id}] ✓ Downloaded {len(downloaded_s3_files)} files from S3"
            )

        zoho_info = {"new": [], "skipped": []}
        if zoho_files:
            logger.info(f"[{job_id}] PHASE 3: ZOHO FILE DEDUPLICATION")

            new_zoho_reports, skipped_zoho_reports = (
                ZohoDeduplicator.filter_new_zoho_reports(
                    zoho_json_files=zoho_files,
                    job_manager=job_manager,
                    company_name=company_name,
                )
            )

            zoho_info = {"new": new_zoho_reports, "skipped": skipped_zoho_reports}

            new_zoho_file_paths = [report["file_path"] for report in new_zoho_reports]
            all_file_paths.extend(new_zoho_file_paths)
            logger.info(
                f"[{job_id}] Zoho: {len(new_zoho_reports)} new, "
                f"{len(skipped_zoho_reports)} skipped"
            )

        logger.info(f"[{job_id}] PHASE 4: REGULAR FILE DEDUPLICATION")

        new_files, skipped_files = FileDeduplicator.filter_new_files(
            file_paths=all_file_paths,
            job_manager=job_manager,
            company_name=company_name,
        )

        # NEW: Add database table-level deduplication
        db_info = {"new": [], "skipped": []}
        new_db_uri_list = []

        if db_uri_list:
            logger.info(f"[{job_id}] PHASE 4.5: DATABASE TABLE-LEVEL DEDUPLICATION")

            all_new_tables = []
            all_skipped_tables = []

            for db_uri in db_uri_list:
                try:
                    # Get available tables from database
                    from nisaa.utils.sql_database import get_database_tables

                    available_tables = get_database_tables(db_uri)

                    # Filter new vs already-processed tables
                    new_tables, skipped_tables = DBDeduplicator.filter_new_tables(
                        db_uri=db_uri,
                        available_tables=available_tables,
                        job_manager=job_manager,
                        company_name=company_name,
                    )

                    all_new_tables.extend(new_tables)
                    all_skipped_tables.extend(skipped_tables)

                    # If any new tables exist for this DB, add it to processing list
                    if new_tables:
                        new_db_uri_list.append(
                            {
                                "db_uri": db_uri,
                                "new_tables": [t["table_name"] for t in new_tables],
                            }
                        )

                except Exception as e:
                    logger.error(f"[{job_id}] Failed to process DB {db_uri}: {e}")

            db_info = {"new": all_new_tables, "skipped": all_skipped_tables}

            logger.info(
                f"[{job_id}] Tables: {len(all_new_tables)} new, "
                f"{len(all_skipped_tables)} skipped"
            )

        total_items = (
            len(new_files)
            + len(skipped_files)
            + len(zoho_info["skipped"])
            + len(db_info["skipped"])
        )

        job_manager.update_job_status(
            job_id,
            JobStatus.RUNNING,
            total_files=total_items,
            skipped_files=len(skipped_files)
            + len(zoho_info["skipped"])
            + len(db_info["skipped"]),
        )

        logger.info(
            f"[{job_id}] Summary: {len(new_files)} new files, "
            f"{len(skipped_files)} skipped files, "
            f"{len(zoho_info['skipped'])} skipped Zoho, "
            f"{len(db_info['new'])} new tables, "
            f"{len(db_info['skipped'])} skipped tables"
        )

        if len(new_files) == 0 and not new_db_uri_list and not website_url_list:
            logger.info(f"[{job_id}] No new content to process")
            job_manager.update_job_status(
                job_id, JobStatus.COMPLETED, processed_files=0, total_vectors=0
            )
            return

        logger.info(f"[{job_id}] PHASE 5: DATA INGESTION PIPELINE")

        new_file_paths = [file_info["file_path"] for file_info in new_files]

        logger.info(
            f"[{job_id}] Processing {len(new_file_paths)} new files and "
            f"{len(db_info['new'])} new tables"
        )

        pipeline = DataIngestionPipeline(
            company_namespace=company_name,
            directory_path=company_directory if len(new_files) > 0 else None,
            file_paths=new_file_paths if len(new_files) > 0 else None,
            db_uris=new_db_uri_list,
            website_urls=website_url_list,
            proxies=None,
        )

        stats = await asyncio.to_thread(pipeline.run, job_manager, company_name)

        logger.info(f"[{job_id}] PHASE 6: MARKING FILES AS PROCESSED")

        vectors_per_item = stats["vectors_upserted"] // max(
            len(new_files) + len(db_info["new"]), 1
        )

        for file_info in new_files:
            try:
                job_manager.mark_file_processed(
                    company_name=company_name,
                    file_path=file_info["file_path"],
                    file_hash=file_info["file_hash"],
                    file_size=file_info["file_size"],
                    file_type=file_info["file_type"],
                    vector_count=vectors_per_item,
                    job_id=job_id,
                    metadata={"source": "ingestion_pipeline"},
                )
            except Exception as e:
                logger.error(
                    f"[{job_id}] Failed to mark file {file_info['file_path']}: {e}"
                )

        if zoho_info["new"]:
            logger.info(f"[{job_id}] PHASE 7: MARKING ZOHO REPORTS AS PROCESSED")

            for zoho_report in zoho_info["new"]:
                try:
                    job_manager.mark_zoho_report_processed(
                        company_name=company_name,
                        report_name=zoho_report["report_name"],
                        content_hash=zoho_report["content_hash"],
                        record_count=zoho_report["record_count"],
                        vector_count=vectors_per_item * zoho_report["record_count"],
                        job_id=job_id,
                        app_name=zoho_report.get("app_name", "Unknown"),
                        metadata={"source": "zoho_export"},
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to mark Zoho report: {e}")

        if hasattr(pipeline, "website_info") and pipeline.website_info:
            logger.info(f"[{job_id}] PHASE 8: MARKING WEBSITES AS PROCESSED")

            for website_info in pipeline.website_info.get("new", []):
                try:
                    job_manager.mark_website_processed(
                        company_name=company_name,
                        website_url=website_info["url"],
                        content_hash=website_info["content_hash"],
                        page_count=website_info["page_count"],
                        vector_count=vectors_per_item * website_info["page_count"],
                        job_id=job_id,
                        metadata={"source": "website_scraping"},
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to mark website: {e}")

        # NEW: Mark processed tables
        if db_info["new"]:
            logger.info(f"[{job_id}] PHASE 9: MARKING DATABASE TABLES AS PROCESSED")

            for table_info in db_info["new"]:
                try:
                    job_manager.mark_db_table_processed(
                        company_name=company_name,
                        db_uri=table_info["db_uri"],
                        db_hash=table_info["db_hash"],
                        table_name=table_info["table_name"],
                        table_hash=table_info["table_hash"],
                        row_count=table_info["row_count"],
                        vector_count=vectors_per_item,
                        job_id=job_id,
                        metadata={
                            "source": "database_ingestion",
                            "db_type": table_info["db_type"],
                            "db_name": table_info.get("db_name", "unknown"),
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"[{job_id}] Failed to mark table {table_info['table_name']}: {e}"
                    )

        save_name(namespace=company_name)

        job_manager.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            processed_files=len(new_files) + len(db_info["new"]),
            total_vectors=stats["vectors_upserted"],
        )

        logger.info(
            f"[{job_id}] COMPLETED: {len(new_files)} files, "
            f"{len(db_info['new'])} tables, "
            f"{stats['vectors_upserted']} vectors"
        )

    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        logger.error(f"[{job_id}] : {error_msg}", exc_info=True)

        job_manager.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)

# API Endpoints for Data Ingestion
@router.post("/ingest")
async def create_ingestion_job(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Create ingestion job and return immediately with job_id
    """
    data = await request.json()

    company_name = data.get("company_name")
    db_uris = data.get("db_uris", [])
    website_urls = data.get("website_urls", [])
    s3_file_keys = data.get("s3_file_keys", [])
    zoho_cred_encrypted = data.get("zoho_cred_encrypted")
    zoho_region = data.get("zoho_region", "IN")

    if not company_name:
        raise HTTPException(status_code=400, detail="'company_name' is required")

    company_name = sanitize_company_name(company_name)

    base_directory = "data"
    company_directory = os.path.join(base_directory, company_name)
    os.makedirs(company_directory, exist_ok=True)

    s3_file_keys_list = []
    if s3_file_keys:
        if isinstance(s3_file_keys, list):
            s3_file_keys_list = [str(key).strip() for key in s3_file_keys if key]
        elif isinstance(s3_file_keys, str):
            try:
                parsed = json.loads(s3_file_keys)
                if isinstance(parsed, list):
                    s3_file_keys_list = [str(key).strip() for key in parsed if key]
            except json.JSONDecodeError:
                pass

    db_uri_list = []
    if db_uris:
        if isinstance(db_uris, list):
            db_uri_list = [str(uri).strip() for uri in db_uris if uri]
        elif isinstance(db_uris, str):
            try:
                parsed = json.loads(db_uris)
                if isinstance(parsed, list):
                    db_uri_list = [str(uri).strip() for uri in parsed if uri]
                else:
                    db_uri_list = [db_uris.strip()]
            except json.JSONDecodeError:
                db_uri_list = [uri.strip() for uri in db_uris.split(",") if uri.strip()]

    website_url_list = []
    if website_urls:
        if isinstance(website_urls, list):
            website_url_list = [str(url).strip() for url in website_urls if url]
        elif isinstance(website_urls, str):
            try:
                parsed = json.loads(website_urls)
                if isinstance(parsed, list):
                    website_url_list = [str(url).strip() for url in parsed if url]
                else:
                    website_url_list = [website_urls.strip()]
            except json.JSONDecodeError:
                website_url_list = [url.strip() for url in website_urls.split(",") if url.strip()]

    job_manager = JobManager(get_pool())
    job_id = job_manager.create_job(
        company_name=company_name,
        metadata={
            "s3_files": len(s3_file_keys_list),
            "databases": len(db_uri_list),
            "websites": len(website_url_list),
            "has_zoho": bool(zoho_cred_encrypted)
        }
    )

    background_tasks.add_task(
        run_ingestion_pipeline,
        job_id=job_id,
        company_name=company_name,
        company_directory=company_directory,
        s3_file_keys_list=s3_file_keys_list,
        zoho_cred_encrypted=zoho_cred_encrypted,
        zoho_region=zoho_region,
        db_uri_list=db_uri_list,
        website_url_list=website_url_list
    )

    logger.info(f"Created job {job_id} for {company_name}")

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "job_id": job_id,
            "company_name": company_name,
            "message": f"Ingestion job created. Check status at /data-ingestion/jobs/{job_id}/status",
            "check_status_url": f"/data-ingestion/jobs/{job_id}/status"
        }
    )

# Get status of an ingestion job
@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of an ingestion job"""
    job_manager = JobManager(get_pool())
    job_data = job_manager.get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JSONResponse(status_code=200, content=job_data)

# Get all processed files for a company
@router.get("/companies/{company_name}/files")
async def get_company_files(company_name: str):
    """Get all processed files for a company"""
    pool = get_pool()
    conn = None
    
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT file_path, file_hash, file_type, file_size, 
                       vector_count, job_id, processed_at
                FROM processed_files
                WHERE company_name = %s
                ORDER BY processed_at DESC
            """, (company_name,))
            
            rows = cur.fetchall()
            files = [
                {
                    "file_path": row[0],
                    "file_hash": row[1],
                    "file_type": row[2],
                    "file_size": row[3],
                    "vector_count": row[4],
                    "job_id": row[5],
                    "processed_at": row[6].isoformat() if row[6] else None
                }
                for row in rows
            ]
            
            return JSONResponse(
                status_code=200,
                content={
                    "company_name": company_name,
                    "total_files": len(files),
                    "files": files
                }
            )
    finally:
        if conn:
            pool.putconn(conn)

# Get all processed websites for a company
@router.get("/companies/{company_name}/websites")
async def get_company_websites(company_name: str):
    """Get all processed websites for a company"""
    pool = get_pool()
    conn = None
    
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT website_url, content_hash, page_count, 
                       vector_count, job_id, processed_at
                FROM processed_websites
                WHERE company_name = %s
                ORDER BY processed_at DESC
            """, (company_name,))
            
            rows = cur.fetchall()
            websites = [
                {
                    "website_url": row[0],
                    "content_hash": row[1],
                    "page_count": row[2],
                    "vector_count": row[3],
                    "job_id": row[4],
                    "processed_at": row[5].isoformat() if row[5] else None
                }
                for row in rows
            ]
            
            return JSONResponse(
                status_code=200,
                content={
                    "company_name": company_name,
                    "total_websites": len(websites),
                    "websites": websites
                }
            )
    finally:
        if conn:
            pool.putconn(conn)

    # 

# Get all processed databases for a company
@router.get("/companies/{company_name}/databases")
async def get_company_databases(company_name: str):
    """Get all processed databases for a company"""
    pool = get_pool()
    conn = None
    
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT db_uri, db_hash, db_type, db_name,
                       vector_count, job_id, processed_at
                FROM processed_databases
                WHERE company_name = %s
                ORDER BY processed_at DESC
            """, (company_name,))
            
            rows = cur.fetchall()
            databases = [
                {
                    "db_uri": row[0],
                    "db_hash": row[1],
                    "db_type": row[2],
                    "db_name": row[3],
                    "vector_count": row[4],
                    "job_id": row[5],
                    "processed_at": row[6].isoformat() if row[6] else None
                }
                for row in rows
            ]
            
            return JSONResponse(
                status_code=200,
                content={
                    "company_name": company_name,
                    "total_databases": len(databases),
                    "databases": databases
                }
            )
    finally:
        if conn:
            pool.putconn(conn)

# Get all processed Zoho reports for a company
@router.get("/companies/{company_name}/zoho-reports")
async def get_company_zoho_reports(company_name: str):
    """Get all processed Zoho reports for a company"""
    pool = get_pool()
    conn = None
    
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT report_name, app_name, content_hash, record_count,
                       vector_count, job_id, processed_at
                FROM processed_zoho_reports
                WHERE company_name = %s
                ORDER BY processed_at DESC
            """, (company_name,))
            
            rows = cur.fetchall()
            reports = [
                {
                    "report_name": row[0],
                    "app_name": row[1],
                    "content_hash": row[2],
                    "record_count": row[3],
                    "vector_count": row[4],
                    "job_id": row[5],
                    "processed_at": row[6].isoformat() if row[6] else None
                }
                for row in rows
            ]
            
            return JSONResponse(
                status_code=200,
                content={
                    "company_name": company_name,
                    "total_reports": len(reports),
                    "zoho_reports": reports
                }
            )
    finally:
        if conn:
            pool.putconn(conn)
