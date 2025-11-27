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
from nisaa.services.checkpoint_manager import CheckpointManager


router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])


def should_resume_job(job_manager, company_name: str) -> Optional[str]:
    """
    Check if there's a resumable job for this company
    Returns job_id if found, None otherwise
    """
    existing_job = job_manager.get_latest_interruptible_job(company_name)
    
    if existing_job:
        job_id = existing_job['job_id']
        
        from nisaa.services.checkpoint_manager import CheckpointManager
        checkpoint_manager = CheckpointManager(job_manager.pool)
        checkpoints = checkpoint_manager.get_all_checkpoints(job_id)
        
        if checkpoints:
            logger.info(
                f"Found resumable job {job_id} with {len(checkpoints)} checkpoint(s): "
                f"{list(checkpoints.keys())}"
            )
            return job_id
        else:
            logger.info(f"Job {job_id} has no checkpoints, will create new job")
            return None
    
    return None

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

    return namespace

def sanitize_company_name(company_name: str) -> str:
    """
    Sanitize company name to prevent path traversal attacks

    Returns:
        Safe company name or raises HTTPException
    """
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required")

    if not re.match(r"^[a-zA-Z0-9_-]+$", company_name):
        raise HTTPException(
            status_code=400,
            detail="Company name can only contain letters, numbers, hyphens, and underscores",
        )

    if len(company_name) > 100:
        raise HTTPException(
            status_code=400, detail="Company name too long (max 100 chars)"
        )

    base_dir = Path("data").resolve()
    company_dir = (base_dir / company_name).resolve()

    if not str(company_dir).startswith(str(base_dir)):
        raise HTTPException(status_code=400, detail="Invalid company name")

    return company_name

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
    """FIXED: Background task with proper file tracking on resume"""
    job_manager = JobManager(get_pool())
    db_pool = get_pool()
    checkpoint_manager = None
    pipeline = None

    try:
        checkpoint_manager = CheckpointManager(db_pool)
        checkpoint_manager.company_name = company_name
        
        # In run_ingestion_pipeline function, around line 140:

        if checkpoint_manager and job_id:
            checkpoints = checkpoint_manager.get_all_checkpoints(job_id)
            is_resuming = bool(checkpoints)
            
            # CRITICAL: Check what phase we're resuming FROM
            if 'upserting_vectors' in checkpoints or 'upserting_json_vectors' in checkpoints:
                logger.info(f"[{job_id}] Resuming from UPSERT checkpoint - skipping to vector storage")
                # Skip all preprocessing, jump to upsert
            elif checkpoints:
                logger.info(
                    f"[{job_id}] Resuming from checkpoint "
                    f"(found {len(checkpoints)} saved phases: {list(checkpoints.keys())})"
                )
        else:
            logger.info(f"[{job_id}] Starting new ingestion for {company_name}")
        
        job_manager.update_job_status(job_id, JobStatus.RUNNING)

        # Initialize variables
        zoho_files = []
        all_file_paths = []
        zoho_info = {"new": [], "skipped": []}
        db_info = {"new": [], "skipped": []}
        new_db_uri_list = []
        new_files = []
        skipped_files = []
        
        # =================================================================
        # CRITICAL FIX: Always check what files exist, even on resume
        # =================================================================
        
        # Check if we need to download (only skip if resuming from late stages)
        skip_downloads = is_resuming and 'embedding_documents' in checkpoints
        
        if skip_downloads:
            logger.info(f"[{job_id}] ✓ Skipping S3/Zoho downloads (resuming from embedding)")
            
            # CRITICAL: Still need to identify which files were being processed!
            # Check if files already exist in company directory
            if os.path.exists(company_directory):
                existing_files = []
                for root, dirs, files in os.walk(company_directory):
                    for file in files:
                        if file.lower().endswith(('.pdf', '.txt', '.docx', '.xlsx', '.xls', '.csv', '.json')):
                            existing_files.append(os.path.join(root, file))
                
                if existing_files:
                    logger.info(f"[{job_id}] Found {len(existing_files)} existing files from previous run")
                    all_file_paths.extend(existing_files)
                else:
                    logger.warning(f"[{job_id}] No files found in {company_directory}, this may cause issues")
        else:
            # Phase 1: Zoho Data Extraction
            if zoho_cred_encrypted and zoho_cred_encrypted.strip():
                logger.info(f"[{job_id}] Phase 1: Zoho data extraction")
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

            # Phase 2: S3 File Download
            if len(s3_file_keys_list) > 0:
                logger.info(f"[{job_id}] Phase 2: S3 file download")
                downloaded_s3_files = await asyncio.to_thread(
                    download_all_files_from_s3,
                    file_list=s3_file_keys_list,
                    company_name=company_name,
                )
                all_file_paths.extend(downloaded_s3_files)
                logger.info(f"[{job_id}] Downloaded {len(downloaded_s3_files)} files from S3")

        # =================================================================
        # ALWAYS do deduplication (fast, needed for tracking)
        # =================================================================
        
        # Phase 3: Zoho File Deduplication
        if zoho_files:
            logger.info(f"[{job_id}] Phase 3: Zoho file deduplication")
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

        # Phase 4: Regular File Deduplication
        if all_file_paths:
            logger.info(f"[{job_id}] Phase 4: Regular file deduplication")
            new_files, skipped_files = FileDeduplicator.filter_new_files(
                file_paths=all_file_paths,
                job_manager=job_manager,
                company_name=company_name,
            )
            logger.info(f"[{job_id}] Files: {len(new_files)} new, {len(skipped_files)} skipped")

        # Phase 4.5: Database Table-Level Deduplication
        if db_uri_list:
            logger.info(f"[{job_id}] Phase 4.5: Database table-level deduplication")
            all_new_tables = []
            all_skipped_tables = []

            for db_uri in db_uri_list:
                try:
                    from nisaa.utils.sql_database import get_database_tables
                    available_tables = get_database_tables(db_uri)

                    new_tables, skipped_tables = DBDeduplicator.filter_new_tables(
                        db_uri=db_uri,
                        available_tables=available_tables,
                        job_manager=job_manager,
                        company_name=company_name,
                    )

                    all_new_tables.extend(new_tables)
                    all_skipped_tables.extend(skipped_tables)

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

        # Update job status with file counts
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

        # Check if there's anything to process
        if len(new_files) == 0 and not new_db_uri_list and not website_url_list:
            if not is_resuming:
                logger.info(f"[{job_id}] No new content to process")
                job_manager.update_job_status(
                    job_id, JobStatus.COMPLETED, processed_files=0, total_vectors=0
                )
                
                try:
                    import shutil
                    if os.path.exists(company_directory):
                        shutil.rmtree(company_directory)
                        logger.info(f"[{job_id}] Cleaned up data directory: {company_directory}")
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to cleanup data directory: {e}")
                
                return

        # =================================================================
        # Phase 5: Data Ingestion Pipeline
        # Pipeline will handle document loading internally
        # =================================================================
        logger.info(f"[{job_id}] Phase 5: Data ingestion pipeline")

        new_file_paths = [file_info["file_path"] for file_info in new_files]

        if not is_resuming:
            logger.info(
                f"[{job_id}] Processing {len(new_file_paths)} new files and "
                f"{len(db_info.get('new', []))} new tables"
            )
        else:
            logger.info(f"[{job_id}] Resuming pipeline from checkpoint")

        # CRITICAL: Pass file_paths even on resume so pipeline knows what to process
        pipeline = DataIngestionPipeline(
            company_namespace=company_name,
            directory_path=company_directory if (len(new_files) > 0 or is_resuming) else None,
            file_paths=new_file_paths if len(new_files) > 0 else (all_file_paths if is_resuming else None),  # FIX THIS LINE
            db_uris=new_db_uri_list,
            website_urls=website_url_list,
            proxies=None,
            db_pool=db_pool,
            job_id=job_id
        )

        try:
            stats = await asyncio.to_thread(pipeline.run, job_manager, company_name)
        except asyncio.CancelledError:
            logger.warning(f"[{job_id}] Pipeline cancelled - checkpoints saved")
            
            if pipeline:
                pipeline.cancellation_event.set()
                logger.info(f"[{job_id}] Cancellation signal sent to pipeline thread")
            
            await asyncio.sleep(5)

            job_manager.update_job_status(
                job_id, 
                JobStatus.PENDING, 
                error_message="Interrupted by shutdown. Checkpoints saved. Rerun to resume."
            )
            raise

        # If pipeline returned but was interrupted internally, preserve checkpoint and don't mark completed
        if isinstance(stats, dict) and stats.get('interrupted'):
            logger.info(f"[{job_id}] Pipeline returned interrupted state; preserving checkpoints for resume")
            job_manager.update_job_status(
                job_id,
                JobStatus.PENDING,
                error_message="Interrupted during run. Checkpoints saved. Rerun to resume."
            )
            return

        # Phase 6-9: Mark processed items (only if not resuming)
        if not is_resuming:
            vectors_per_item = stats["vectors_upserted"] // max(
                len(new_files) + len(db_info.get("new", [])), 1
            )

            # Phase 6: Mark Files as Processed
            if new_files:
                logger.info(f"[{job_id}] Phase 6: Marking files as processed")
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
                        logger.error(f"[{job_id}] Failed to mark file: {e}")

            # Phase 7: Mark Zoho Reports as Processed
            if zoho_info.get("new"):
                logger.info(f"[{job_id}] Phase 7: Marking Zoho reports as processed")
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

            # Phase 8: Mark Websites as Processed
            if hasattr(pipeline, "website_info") and pipeline.website_info:
                website_new = pipeline.website_info.get("new", [])
                if website_new:
                    logger.info(f"[{job_id}] Phase 8: Marking websites as processed")
                    for website_info in website_new:
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

            # Phase 9: Mark Database Tables as Processed
            if db_info.get("new"):
                logger.info(f"[{job_id}] Phase 9: Marking database tables as processed")
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
                        logger.error(f"[{job_id}] Failed to mark table: {e}")
        else:
            logger.info(f"[{job_id}] ✓ Skipping marking phase - resumed job")

        # Save namespace
        save_name(namespace=company_name)

        # Update job status to completed
        job_manager.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            processed_files=len(new_files) + len(db_info.get("new", [])),
            total_vectors=stats["vectors_upserted"],
        )

        logger.info(
            f"[{job_id}] ✓ Completed: {len(new_files)} files, "
            f"{len(db_info.get('new', []))} tables, "
            f"{stats['vectors_upserted']} vectors"
        )

        # Cleanup
        try:
            import shutil
            if os.path.exists(company_directory):
                shutil.rmtree(company_directory)
                logger.info(f"[{job_id}] Cleaned up data directory: {company_directory}")
        except Exception as e:
            logger.error(f"[{job_id}] Failed to cleanup data directory: {e}")

    except asyncio.CancelledError:
        logger.info(f"[{job_id}] Task cancelled gracefully")
        
        if checkpoint_manager:
            checkpoints = checkpoint_manager.get_all_checkpoints(job_id)
            if checkpoints:
                logger.info(
                    f"[{job_id}] Saved checkpoints: {list(checkpoints.keys())}. "
                    f"Rerun ingestion to resume."
                )
        raise

    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        logger.error(f"[{job_id}]: {error_msg}", exc_info=True)

        if checkpoint_manager:
            checkpoints = checkpoint_manager.get_all_checkpoints(job_id)
            
            if checkpoints:
                logger.info(
                    f"[{job_id}] Saved checkpoints: {list(checkpoints.keys())}. "
                    f"Rerun to resume."
                )

        job_manager.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)

@router.post("/ingest")
async def create_ingestion_job(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    FIXED: Create or resume ingestion job with proper checkpoint detection
    """
    data = await request.json()

    # Extract request parameters
    company_name = data.get("company_name")
    db_uris = data.get("db_uris", [])
    website_urls = data.get("website_urls", [])
    s3_file_keys = data.get("s3_file_keys", [])
    zoho_cred_encrypted = data.get("zoho_cred_encrypted")
    zoho_region = data.get("zoho_region", "IN")
    force_new_job = data.get("force_new_job", False)

    # Validate company name
    if not company_name:
        raise HTTPException(status_code=400, detail="'company_name' is required")

    company_name = sanitize_company_name(company_name)

    # Setup data directory
    base_directory = "data"
    company_directory = os.path.join(base_directory, company_name)
    os.makedirs(company_directory, exist_ok=True)

    # Parse S3 file keys
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

    # Parse database URIs
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

    # Parse website URLs
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

    # Initialize job manager
    job_manager = JobManager(get_pool())
    
    job_id = None
    resumed = False
    checkpoints = []
    
    # FIXED: Proper resume logic - check for resumable jobs
    if not force_new_job:
        resumable_job = job_manager.get_latest_interruptible_job(company_name)
        if resumable_job:
            job_id = resumable_job['job_id']
            resumed = True
            checkpoints = resumable_job.get('checkpoints', [])
            logger.info(
                f"✓ Resuming job {job_id} for {company_name} "
                f"with checkpoints: {checkpoints}"
            )
        else:
            logger.info(f"No resumable job found for {company_name}")

    # Create new job if not resuming
    if not job_id:
        logger.info(f"Creating new job for {company_name}")
        
        # Check if we should clear previous completed job's data
        latest_completed = job_manager.get_latest_completed_job(company_name)
        if latest_completed:
            error_msg = latest_completed.get('error_message') or ''
            # Only clear if previous job didn't fail
            if not error_msg.startswith('Ingestion failed'):
                cleared_count = job_manager.clear_processed_tables_for_company(company_name)
                if cleared_count > 0:
                    logger.info(
                        f"Cleared {cleared_count} previous processing records for fresh restart"
                    )
        
        # Create new job
        job_id = job_manager.create_job(
            company_name=company_name,
            metadata={
                "s3_files": len(s3_file_keys_list),
                "databases": len(db_uri_list),
                "websites": len(website_url_list),
                "has_zoho": bool(zoho_cred_encrypted)
            }
        )
        logger.info(f"Created new job {job_id} for {company_name}")

    # Create background task
    task = asyncio.create_task(
        run_ingestion_pipeline(
            job_id=job_id,
            company_name=company_name,
            company_directory=company_directory,
            s3_file_keys_list=s3_file_keys_list,
            zoho_cred_encrypted=zoho_cred_encrypted,
            zoho_region=zoho_region,
            db_uri_list=db_uri_list,
            website_url_list=website_url_list
        )
    )
    
    # Register task for graceful shutdown
    try:
        from main import background_tasks_running
        background_tasks_running.add(task)
        task.add_done_callback(lambda t: background_tasks_running.discard(t))
    except ImportError:
        logger.warning("Could not import background_tasks_running from main")

    # Prepare response
    response_data = {
        "status": "accepted",
        "job_id": job_id,
        "company_name": company_name,
        "resumed": resumed,
        "check_status_url": f"/data-ingestion/jobs/{job_id}/status",
        "checkpoints_url": f"/data-ingestion/jobs/{job_id}/checkpoints"
    }
    
    if resumed:
        response_data["message"] = (
            f"Ingestion job resumed from checkpoint. "
            f"Phases saved: {', '.join(checkpoints)}. "
            f"Check status at /data-ingestion/jobs/{job_id}/status"
        )
    else:
        response_data["message"] = (
            f"Ingestion job created. "
            f"Check status at /data-ingestion/jobs/{job_id}/status"
        )

    return JSONResponse(
        status_code=202,
        content=response_data
    )


@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of an ingestion job"""
    job_manager = JobManager(get_pool())
    job_data = job_manager.get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JSONResponse(status_code=200, content=job_data)

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


@router.get("/jobs/{job_id}/checkpoints")
async def get_job_checkpoints(job_id: str):
    """Get checkpoint status for a job"""
    from nisaa.services.checkpoint_manager import CheckpointManager
    
    checkpoint_manager = CheckpointManager(get_pool())
    checkpoints = checkpoint_manager.get_all_checkpoints(job_id)
    
    if not checkpoints:
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "has_checkpoints": False,
                "message": "No checkpoints found (job completed or not started)"
            }
        )
    
    progress_info = {}
    for phase, data in checkpoints.items():
        if 'last_batch_index' in data and 'total_batches' in data:
            progress_pct = (data['last_batch_index'] + 1) / data['total_batches'] * 100
            progress_info[phase] = {
                "last_batch": data['last_batch_index'] + 1,
                "total_batches": data['total_batches'],
                "progress_percentage": round(progress_pct, 2),
                "last_updated": data.get('last_updated'),
                "error": data.get('error')
            }
    
    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "has_checkpoints": True,
            "can_resume": True,
            "checkpoints": progress_info,
            "message": "Job can be resumed by rerunning with same job_id"
        }
    )


@router.delete("/jobs/{job_id}/checkpoints")
async def clear_job_checkpoints(job_id: str):
    """Manually clear checkpoints for a job"""
    from nisaa.services.checkpoint_manager import CheckpointManager, ProcessedItemTracker
    
    try:
        checkpoint_manager = CheckpointManager(get_pool())
        item_tracker = ProcessedItemTracker(get_pool())
        
        checkpoint_manager.clear_checkpoint(job_id)
        item_tracker.clear_items(job_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "job_id": job_id,
                "message": "All checkpoints and tracked items cleared"
            }
        )
    except Exception as e:
        logger.error(f"Failed to clear checkpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))