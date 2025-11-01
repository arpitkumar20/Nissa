import os
import requests
import json
import time
import base64
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Request
from pydantic import BaseModel

# IMPORTANT: Use the fixed exporter
from src.nisaa.services.zoho_data_downloader import ZohoCreatorExporter
from nisaa.services.s3_downloader import download_all_files_from_s3
from src.nisaa.helpers.logger import logger
from src.nisaa.controllers.ingestion_pipeline import DataIngestionPipeline

router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""

    status: str
    company_name: str
    company_directory: str
    uploaded_files: List[str]
    zoho_files: List[str]
    statistics: dict
    message: str


def decode_zoho_credentials(base64_str: str) -> Dict[str, str]:
    """Decode base64 encoded Zoho credentials"""
    try:
        decoded_bytes = base64.b64decode(base64_str)
        decoded_str = decoded_bytes.decode("utf-8")
        credentials = json.loads(decoded_str)
        logger.info("✅ Zoho credentials decoded successfully")
        return credentials
    except Exception as e:
        logger.error(f"❌ Failed to decode credentials: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid credentials: {str(e)}")

def save_name(namespace: str, folder_path: str = "web_info", filename: str = "web_info.json"):
        """
        Save a single 'namespace' to a JSON file.
        If the file exists, overwrite the namespace value (only one key is stored).
        Always ensures the folder exists before saving.
        """
        # Ensure folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Full file path
        file_path = os.path.join(folder_path, filename)

        # Load old data if file exists (not strictly necessary since we overwrite)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Update namespace (only one key)
        data["namespace"] = namespace

        # Write updated data back to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Namespace '{namespace}' saved to {file_path} successfully!")
        return namespace

@router.post("/ingest", response_model=IngestionResponse)
async def run_data_ingestion(request: Request):
    """
    PRODUCTION: Data ingestion pipeline with Zoho export

    Payload:
    {
        "company_name": "mycompany",
        "zoho_cred_encrypted": "base64_encoded_json",
        "zoho_region": "IN",
        "s3_file_keys": ["file1.pdf", "file2.docx"],
        "db_uris": "mongodb://...",
        "website_urls": "https://example.com"
    }
    """
    data = await request.json()

    company_name = data.get("company_name", None)
    db_uris = data.get("db_uris", None)
    website_urls = data.get("website_urls", None)
    s3_file_keys = data.get("s3_file_keys", [])
    zoho_cred_encrypted = data.get("zoho_cred_encrypted", None)
    zoho_region = data.get("zoho_region", "IN")

    company_directory = None
    zoho_files = []
    base_directory = "data"
    company_directory = os.path.join(base_directory, company_name)
    os.makedirs(company_directory, exist_ok=True)

    try:
        # ============= VALIDATION =============
        if not company_name:
            raise HTTPException(status_code=400, detail="'company_name' is required")

        print("\n" + "=" * 70)
        print(f"🚀 DATA INGESTION PIPELINE: {company_name.upper()}")
        print("=" * 70)

        # ============= PARSE S3 FILE KEYS =============
        s3_file_keys_list = []
        if s3_file_keys:
            if isinstance(s3_file_keys, list):
                s3_file_keys_list = [
                    str(key).strip() for key in s3_file_keys if key and str(key).strip()
                ]
            elif isinstance(s3_file_keys, str):
                try:
                    parsed = json.loads(s3_file_keys)
                    if isinstance(parsed, list):
                        s3_file_keys_list = [
                            str(key).strip()
                            for key in parsed
                            if key and str(key).strip()
                        ]
                    else:
                        raise HTTPException(
                            status_code=400, detail="s3_file_keys must be a JSON array"
                        )
                except json.JSONDecodeError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON for s3_file_keys: {str(e)}",
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="s3_file_keys must be a JSON array or string",
                )

        # ============= PHASE 1: ZOHO DATA EXTRACTION =============
        if zoho_cred_encrypted and zoho_cred_encrypted.strip():
            print("\n📥 PHASE 1: ZOHO DATA EXTRACTION")
            print("-" * 70)

            try:
                # Decode credentials
                decoded_creds = decode_zoho_credentials(zoho_cred_encrypted)
                zoho_client_id = decoded_creds.get("zoho_client_id")
                zoho_client_secret = decoded_creds.get("zoho_client_secret")
                zoho_refresh_token = decoded_creds.get("zoho_refresh_token")
                zoho_owner_name = decoded_creds.get("zoho_owner_name")

                if not all(
                    [
                        zoho_client_id,
                        zoho_client_secret,
                        zoho_refresh_token,
                        zoho_owner_name,
                    ]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Incomplete Zoho credentials. Required: zoho_client_id, zoho_client_secret, zoho_refresh_token, zoho_owner_name",
                    )

                # Create exporter (uses fixed version with pagination)
                zoho_exporter = ZohoCreatorExporter(
                    client_id=zoho_client_id,
                    client_secret=zoho_client_secret,
                    refresh_token=zoho_refresh_token,
                    owner_name=zoho_owner_name,
                    output_dir=company_directory,
                    zoho_region=zoho_region or "IN",
                )

                # Export all data
                zoho_stats = zoho_exporter.export_all_data()
                zoho_files = [
                    os.path.basename(f) for f in zoho_stats["json_file_paths"]
                ]

                print(
                    f"\n✅ Zoho extraction complete: {len(zoho_files)} file(s), {zoho_stats['total_records']:,} records"
                )

            except HTTPException:
                raise
            except Exception as zoho_error:
                print(f"\n❌ Zoho extraction failed: {zoho_error}")
                logger.error(f"Zoho error: {zoho_error}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Zoho extraction error: {str(zoho_error)}"
                )

        # ============= PHASE 2: S3 FILE DOWNLOAD =============
        downloaded_s3_files = []
        if len(s3_file_keys_list) > 0:
            print("\n📥 PHASE 2: S3 FILE DOWNLOAD")
            print("-" * 70)
            logger.info(f"Downloading {len(s3_file_keys_list)} files from S3...")

            downloaded_s3_files = download_all_files_from_s3(
                file_list=s3_file_keys_list, company_name=company_name
            )
            logger.info(f"✅ Downloaded {len(downloaded_s3_files)} files from S3")

        all_saved_files = downloaded_s3_files

        # ============= PHASE 3: PARSE INPUTS =============
        db_uri_list = [uri.strip() for uri in db_uris.split(",")] if db_uris else []
        website_url_list = (
            [url.strip() for url in website_urls.split(",")] if website_urls else []
        )

        # ============= PHASE 4: DATA INGESTION PIPELINE =============
        print("\n🔄 PHASE 3: DATA INGESTION PIPELINE")
        print("-" * 70)
        logger.info(f"Running ingestion pipeline for: {company_name}")

        pipeline = DataIngestionPipeline(
            company_namespace=company_name,
            directory_path=company_directory,
            db_uris=db_uri_list,
            website_urls=website_url_list,
            proxies=None,
        )

        stats = pipeline.run()

        # ============= RESPONSE =============
        response_data = {
            "status": "success",
            "company_name": company_name,
            "company_directory": company_directory,
            "uploaded_files": [os.path.basename(f) for f in all_saved_files],
            "zoho_files": zoho_files,
            "statistics": {
                "total_documents_loaded": stats["total_documents"],
                "source_breakdown": {
                    "files": stats["file_documents"],
                    "databases": stats["database_documents"],
                    "websites": stats["website_documents"],
                    "json_files": stats["json_documents"],
                    "zoho_files": len(zoho_files),
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
                    "processing_time_seconds": round(stats["processing_time"], 2)
                },
            },
            "message": f"Knowledge base for '{company_name}' created successfully with {stats['vectors_upserted']} vectors",
        }
    
        namespace_info = save_name(namespace=company_name,folder_path="web_info",filename="web_info.json")
        print("==========================WEB INFO NAMESPACE SAVE===========================================================")
        print(namespace_info)
        print("=====================================================================================")

        print("\n" + "=" * 70)
        print("✅ INGESTION COMPLETE")
        print("-" * 70)
        print(f"   📊 Company: {company_name}")
        print(f"   📁 Zoho files: {len(zoho_files)}")
        print(f"   📁 S3 files: {len(downloaded_s3_files)}")
        print(f"   💾 Total vectors: {stats['vectors_upserted']:,}")
        print("=" * 70)

        return JSONResponse(status_code=200, content=response_data)

    except HTTPException:
        raise

    except Exception as e:
        error_msg = f"Data ingestion failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)