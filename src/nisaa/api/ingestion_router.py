import os
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Form, File, UploadFile, HTTPException

from nisaa.helpers.logger import logger
from nisaa.controllers.ingestion_pipeline import DataIngestionPipeline

router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])


@router.post("/run")
async def run_data_ingestion(
    company_name: str = Form(None),
    db_uris: Optional[List[str]] = Form(None),
    website_urls: Optional[List[str]] = Form(None),
    file_documents: Optional[List[UploadFile]] = File(None),
):
    """
    Run data ingestion from files, databases, and websites.
    """
    company_directory = None  # ✅ initialize upfront

    try:
        # ✅ Basic validation
        if not company_name:
            raise HTTPException(status_code=400, detail="'company_name' is required.")
        if not db_uris and not website_urls and not file_documents:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'db_uris', 'website_urls', or 'file_documents' must be provided.",
            )

        # ✅ Define directories
        base_directory = "data"
        company_directory = os.path.join(base_directory, company_name)

        os.makedirs(company_directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {company_directory}")

        saved_files = []

        # ✅ Save uploaded files inside company folder
        if file_documents:
            for file in file_documents:
                file_path = os.path.join(company_directory, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                saved_files.append(file_path)
                logger.info(f"Saved uploaded file for {company_name}: {file_path}")

        logger.info(f"Starting data ingestion for company: {company_name}")

        # ✅ Initialize and run pipeline
        pipeline = DataIngestionPipeline(
            directory_path=company_directory,
            company_namespace=company_name,
            db_uris=db_uris or [],
            website_urls=website_urls or [],
            proxies=None,
        )

        all_documents = pipeline.run()
        total_docs = len(all_documents)
        stats = getattr(pipeline, "stats", {})

        logger.info(
            f"Data ingestion completed: {total_docs} documents processed for company: {company_name}"
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "company_name": company_name,
                "company_directory": company_directory,
                "uploaded_files": saved_files,
                "total_processed_documents": total_docs,
                "source_stats": {
                    "files": stats.get("file_documents", 0),
                    "databases": stats.get("database_documents", 0),
                    "websites": stats.get("website_documents", 0),
                },
                "message": f"Data ingestion completed successfully for {company_name} ✅",
            },
        )

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        # ✅ Safe fallback logging — won’t break if company_directory is None
        logger.error(
            f"Data ingestion failed for {company_name or 'Unknown Company'}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {e}")
