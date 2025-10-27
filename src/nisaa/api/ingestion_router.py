import os
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Form, HTTPException

from nisaa.helpers.logger import logger
from nisaa.controllers.ingestion_pipeline import DataIngestionPipeline

router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])

@router.post("/run")
async def run_data_ingestion(
    company_name: str = Form(None),
    db_uris: Optional[List[str]] = Form(None),
    website_urls: Optional[List[str]] = Form(None),
    directory_path: Optional[List[str]] = Form(None)
):
    """
    Run data ingestion from files, databases, and websites.

    Example usage in Postman (form-data):
    ---------------------------------------
    Key: company_name   | Value: AcmeCorp
    Key: db_uris        | Value: postgresql://user:pass@localhost:5432/mydb
    Key: db_uris        | Value: mysql://root:root@localhost:3306/testdb   ← (multiple allowed)
    Key: website_urls   | Value: https://example.com
    Key: website_urls   | Value: https://another-site.com
    """
    try:
        if not company_name:
            raise HTTPException(status_code=400, detail="'company_name' is required.")
        if not db_uris and not website_urls:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'db_uris' or 'website_urls' must be provided.",
            )

        # directory_path = "data"
        # if not os.path.exists(directory_path):
        #     os.makedirs(directory_path, exist_ok=True)
        #     logger.info(f"Created missing directory: {directory_path}")

        logger.info(f"Starting data ingestion for company: {company_name}")

        pipeline = DataIngestionPipeline(
            directory_path=directory_path,
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
                "total_processed_documents": total_docs,
                "source_stats": {
                    "files": stats.get("file_documents", 0),
                    "databases": stats.get("database_documents", 0),
                    "websites": stats.get("website_documents", 0),
                },
                "message": "Data ingestion completed successfully ✅",
            },
        )

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Data ingestion failed for {company_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {e}")