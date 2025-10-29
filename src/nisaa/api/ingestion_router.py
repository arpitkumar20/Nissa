import os
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from openai import BaseModel

from src.nisaa.helpers.logger import logger
from src.nisaa.controllers.ingestion_pipeline import DataIngestionPipeline
router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str
    company_name: str
    company_directory: str
    uploaded_files: List[str]
    statistics: dict
    message: str


@router.post("/ingest", response_model=IngestionResponse)
async def run_data_ingestion(
    company_name: str = Form(None),
    db_uris: Optional[str] = Form(None),
    website_urls: Optional[str] = Form(None),
    file_documents: Optional[List[UploadFile]] = File(None),
):
    """
    Run comprehensive data ingestion pipeline
    
    This endpoint:
    1. Creates a company-specific directory
    2. Saves uploaded files
    3. Loads data from files, databases, and websites
    4. Processes and cleans text (except database data)
    5. Handles JSON files separately with entity extraction
    6. Chunks and embeds all content
    7. Stores vectors in Pinecone with company namespace
    
    Args:
        company_name: Company identifier (used as Pinecone namespace)
        db_uris: Database connection strings (comma-separated)
        website_urls: Website URLs to scrape (comma-separated)
        file_documents: Files to upload and process
        
    Returns:
        Detailed statistics and status of the ingestion process
    """
    company_directory = None
    
    try:
        # Validation
        if not company_name:
            raise HTTPException(status_code=400, detail="'company_name' is required")
        
        if not db_uris and not website_urls and not file_documents:
            raise HTTPException(
                status_code=400,
                detail="At least one data source (db_uris, website_urls, or file_documents) must be provided"
            )
        
        # Parse inputs
        db_uri_list = [uri.strip() for uri in db_uris.split(",")] if db_uris else []
        website_url_list = [url.strip() for url in website_urls.split(",")] if website_urls else []
        
        # Setup directory
        base_directory = "data"
        company_directory = os.path.join(base_directory, company_name)
        os.makedirs(company_directory, exist_ok=True)
        
        logger.info(f"📁 Company directory: {company_directory}")
        
        # Save uploaded files
        saved_files = []
        if file_documents:
            logger.info(f"💾 Saving {len(file_documents)} uploaded files...")
            for file in file_documents:
                file_path = os.path.join(company_directory, file.filename)
                
                # Read and save file
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                
                saved_files.append(file_path)
                logger.info(f"   ✓ Saved: {file.filename}")
        
        # Initialize and run pipeline
        logger.info(f"🚀 Starting data ingestion for: {company_name}")
        
        pipeline = DataIngestionPipeline(
            company_namespace=company_name,
            directory_path=company_directory if (file_documents or True) else None,
            db_uris=db_uri_list,
            website_urls=website_url_list,
            proxies=None,
        )
        
        # Run the complete pipeline
        stats = pipeline.run()
        
        # Prepare response
        response_data = {
            "status": "success",
            "company_name": company_name,
            "company_directory": company_directory,
            "uploaded_files": [os.path.basename(f) for f in saved_files],
            "statistics": {
                "total_documents_loaded": stats["total_documents"],
                "source_breakdown": {
                    "files": stats["file_documents"],
                    "databases": stats["database_documents"],
                    "websites": stats["website_documents"],
                    "json_files": stats["json_documents"],
                },
                "processing": {
                    "successfully_processed": stats["processed_documents"],
                    "failed_or_skipped": stats["failed_documents"],
                    "total_chunks": stats["total_chunks"],
                    "json_chunks": stats["json_chunks"],
                },
                "embeddings": {
                    "total_generated": stats["total_embeddings"] + stats["json_chunks"],
                    "vectors_stored": stats["vectors_upserted"],
                },
                "performance": {
                    "processing_time_seconds": round(stats["processing_time"], 2),
                    "documents_per_second": round(
                        stats["total_documents"] / stats["processing_time"]
                        if stats["processing_time"] > 0 else 0,
                        2
                    ),
                },
            },
            "message": f"✅ Knowledge base for '{company_name}' created successfully with {stats['vectors_upserted']} vectors",
        }
        
        logger.info(f"✅ Ingestion completed for {company_name}")
        
        return JSONResponse(status_code=200, content=response_data)
        
    except HTTPException as http_err:
        raise http_err
        
    except Exception as e:
        error_msg = f"Data ingestion failed for {company_name or 'Unknown Company'}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
