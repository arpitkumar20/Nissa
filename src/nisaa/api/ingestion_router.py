import os
import requests
import json
import time
import base64
from typing import List, Optional, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Request
from openai import BaseModel
from src.nisaa.services.zoho_data_downloader import ZohoCreatorBulkExporter
from tqdm import tqdm

from nisaa.services.s3_downloader import download_all_files_from_s3
from src.nisaa.helpers.logger import logger
from src.nisaa.controllers.ingestion_pipeline import DataIngestionPipeline
import base64

router = APIRouter(prefix="/data-ingestion", tags=["Data Ingestion"])
ZOHO_REGION = os.getenv("ZOHO_REGION", "IN")

ZOHO_REGION = os.getenv('ZOHO_REGION')

class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str
    company_name: str
    company_directory: str
    uploaded_files: List[str]
    zoho_files: List[str]
    statistics: dict
    message: str


class ZohoCreatorExporter:
    """Export Zoho Creator data to JSON (optimized for speed with progress bar)"""
    
    ZOHO_CONFIG = {
        'US': {
            'accounts': 'https://accounts.zoho.com',
            'creator': 'https://www.zohoapis.com'
        },
        'EU': {
            'accounts': 'https://accounts.zoho.eu',
            'creator': 'https://www.zohoapis.eu'
        },
        'IN': {
            'accounts': 'https://accounts.zoho.in',
            'creator': 'https://www.zohoapis.in'
        },
        'AU': {
            'accounts': 'https://accounts.zoho.com.au',
            'creator': 'https://www.zohoapis.com.au'
        },
        'CN': {
            'accounts': 'https://accounts.zoho.com.cn',
            'creator': 'https://www.zohoapis.com.cn'
        }
    }
    
    def __init__(self, client_id: str, client_secret: str, refresh_token: str, 
                owner_name: str, output_dir: str, 
                zoho_region: str = ZOHO_REGION):
        """Initialize Exporter"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.account_owner = owner_name
        self.zoho_region = zoho_region.upper()

        if self.zoho_region not in self.ZOHO_CONFIG:
            self.zoho_region = 'IN'
        
        self.accounts_url = self.ZOHO_CONFIG[self.zoho_region]['accounts']
        self.api_base_url = self.ZOHO_CONFIG[self.zoho_region]['creator']
        
        # Optimized for maximum speed
        self.export_config = {
            'output_dir': output_dir,
            'max_workers': 3,  # Increased workers
            'max_records': 200,
            'rate_limit_delay': 0.05,  # Reduced delay for faster requests
            'connection_timeout': 15,
            'read_timeout': 30
        }
                
        self.access_token = self._get_access_token()
        self.headers = {'Authorization': f'Zoho-oauthtoken {self.access_token}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.request_lock = Lock()
        self.last_request_time = 0
        
        self.stats = {
            "api_calls": 0,
            "reports_processed": 0,
            "total_reports": 0,
            "total_records": 0,
            "json_files": 0,
            "json_file_paths": [],
            "errors": [],
        }
        self.stats_lock = Lock()
    
    def _get_access_token(self) -> str:
        """Get access token"""
        token_url = f"{self.accounts_url}/oauth/v2/token"
        
        params = {
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'refresh_token'
        }
        
        try:
            response = requests.post(token_url, params=params, timeout=30)
            data = response.json()
            
            if 'error' in data:
                raise ValueError(f"Auth Error: {data.get('error')}")
            
            access_token = data.get('access_token')
            if not access_token:
                raise ValueError("No access token in response")
            
            return access_token
            
        except Exception as e:
            logger.error(f"Error getting Zoho token: {e}")
            raise
    
    def rate_limited_request(
        self, url: str, params: Optional[Dict] = None
    ) -> requests.Response:
        """OPTIMIZED: Rate-limited requests with call tracking"""
        with self.request_lock:
            elapsed = time.time() - self.last_request_time

            if elapsed < self.export_config["rate_limit_delay"]:
                time.sleep(self.export_config["rate_limit_delay"] - elapsed)

            response = self.session.get(
                url,
                params=params,
                timeout=(
                    self.export_config["connection_timeout"],
                    self.export_config["read_timeout"],
                ),
            )
            self.last_request_time = time.time()

            # Track API calls
            with self.stats_lock:
                self.stats["api_calls"] += 1

            return response
    
    def get_all_applications(self) -> List[Dict]:
        """Get all applications (1 API call)"""
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/applications"

        try:
            response = self.rate_limited_request(url)

            if response.status_code == 401:
                raise ValueError("Authorization error - check credentials")

            response.raise_for_status()
            applications = response.json().get("applications", [])

            return applications

        except Exception as e:
            logger.error(f"Error fetching applications: {e}")
            raise

    def get_all_reports(self, app_link_name: str) -> List[Dict]:
        """Get all reports from application (1 API call per app)"""
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/{app_link_name}/reports"

        try:
            response = self.rate_limited_request(url)
            response.raise_for_status()
            reports = response.json().get("reports", [])
            return reports

        except Exception as e:
            logger.error(f"Error fetching reports: {e}")
            raise
    
    def get_all_records_paginated(
        self, app_link_name: str, report_link_name: str
    ) -> List[Dict]:
        """Fetch all records with pagination"""
        all_records = []
        from_index = 1
        max_records = self.export_config["max_records"]

        while True:
            url = f"{self.api_base_url}/creator/v2.1/data/{self.account_owner}/{app_link_name}/report/{report_link_name}"
            params = {"from": from_index, "limit": max_records}

            try:
                response = self.rate_limited_request(url, params)
                response.raise_for_status()

                data = response.json()
                records = data.get("data", [])

                if not records:
                    break

                all_records.extend(records)

                if len(records) < max_records:
                    break

                from_index += max_records

            except Exception as e:
                logger.error(f"Error fetching records at index {from_index}: {e}")
                break

        return all_records

    def sanitize_filename(self, name: str) -> str:
        """Clean filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()
    
    def export_report_data(self, app_link_name: str, report: Dict, output_dir: Path) -> Dict:
        """Export report data to JSON (silent execution)"""
        report_name = report.get('display_name', 'Unknown')
        report_link_name = report.get('link_name', '')
        
        result = {
            'report_name': report_name,
            'success': False,
            'records_count': 0,
            'json_file': None,
            'error': None
        }
        
        try:
            records = self.get_all_records_paginated(app_link_name, report_link_name, report_name)
            
            if not records:
                result['success'] = True
                return result
            
            safe_name = self.sanitize_filename(report_name)
            json_file = output_dir / f"{safe_name}.json"
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            result['json_file'] = str(json_file)
            result['records_count'] = len(records)
            result['success'] = True
            
        except Exception as e:
            error_msg = f"Error exporting '{report_name}': {str(e)}"
            result['error'] = error_msg
        
        return result
      
    def export_all_data(self) -> Dict:
        """OPTIMIZED: Export with progress bar and API call tracking"""
        start_time = time.time()

        try:
            # Step 1: Get all applications (1 API call)
            print("\n🔄 Step 1/4: Fetching Zoho applications...")
            applications = self.get_all_applications()

            if not applications:
                raise ValueError("No applications found")

            print(f"✅ Found {len(applications)} applications")

            output_dir = Path(self.export_config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 2: Get all reports from all apps
            print("\n🔄 Step 2/4: Scanning applications for reports...")
            all_reports_with_app = []

            with tqdm(
                total=len(applications), desc="Scanning apps", unit="app"
            ) as pbar:
                for app in applications:
                    app_display_name = app.get("application_name", "Unknown")
                    app_link_name = app["link_name"]

                    try:
                        reports = self.get_all_reports(app_link_name)

                        if reports:
                            for report in reports:
                                all_reports_with_app.append(
                                    {
                                        "app_link_name": app_link_name,
                                        "app_display_name": app_display_name,
                                        "report": report,
                                    }
                                )

                        pbar.set_postfix_str(
                            f"{len(all_reports_with_app)} reports found"
                        )

                    except Exception as e:
                        self.stats["errors"].append(f"{app_display_name}: {str(e)}")

                    pbar.update(1)

            if not all_reports_with_app:
                print("⚠️ No reports found")
                return self.stats

            self.stats["total_reports"] = len(all_reports_with_app)
            print(f"✅ Found {len(all_reports_with_app)} total reports")

            # Step 3: Export reports
            print(f"\n🔄 Step 3/4: Exporting {len(all_reports_with_app)} reports...")

            with tqdm(
                total=len(all_reports_with_app), desc="Exporting reports", unit="report"
            ) as pbar:
                with ThreadPoolExecutor(
                    max_workers=self.export_config["max_workers"]
                ) as executor:
                    future_to_report = {
                        executor.submit(
                            self.export_report_data,
                            item["app_link_name"],
                            item["report"],
                            output_dir,
                        ): item
                        for item in all_reports_with_app
                    }

                    for future in as_completed(future_to_report):
                        item = future_to_report[future]

                        try:
                            result = future.result()

                            with self.stats_lock:
                                self.stats["reports_processed"] += 1

                                if result["success"] and result["records_count"] > 0:
                                    self.stats["total_records"] += result[
                                        "records_count"
                                    ]
                                    self.stats["json_files"] += 1
                                    self.stats["json_file_paths"].append(
                                        result["json_file"]
                                    )

                                if result["error"]:
                                    self.stats["errors"].append(result["error"])

                            pbar.set_postfix_str(
                                f"{self.stats['json_files']} files, {self.stats['total_records']} records"
                            )

                        except Exception as e:
                            with self.stats_lock:
                                self.stats["errors"].append(str(e))
                                self.stats["reports_processed"] += 1

                        pbar.update(1)

            execution_time = time.time() - start_time

            # Summary
            print(f"\n✅ Step 4/4: Zoho Export Complete")
            print(f"   📊 Applications: {len(applications)}")
            print(
                f"   📄 Reports: {self.stats['reports_processed']}/{self.stats['total_reports']}"
            )
            print(f"   💾 JSON Files: {self.stats['json_files']}")
            print(f"   📝 Records: {self.stats['total_records']:,}")
            print(f"   🔌 API Calls: {self.stats['api_calls']}")
            print(f"   ⏱️ Time: {execution_time:.1f}s")

            if self.stats["errors"]:
                print(f"   ⚠️ Errors: {len(self.stats['errors'])}")

            return self.stats

        except Exception as e:
            logger.error(f"Zoho export failed: {e}")
            raise
        finally:
            self.session.close()           

def decode_zoho_credentials(base64_str: str) -> Dict[str, str]:
    """Decode base64 encoded Zoho credentials"""
    try:
        decoded_bytes = base64.b64decode(base64_str)
        decoded_str = decoded_bytes.decode("utf-8")
        credentials = json.loads(decoded_str)
        print("=============================================")
        print(credentials)
        print("=============================================")
        return credentials
    except Exception as e:
        logger.error(f"Failed to decode credentials: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid credentials: {str(e)}")
    
@router.post("/ingest", response_model=IngestionResponse)
# async def run_data_ingestion(
#     company_name: str = Form(None),
#     db_uris: Optional[str] = Form(None),
#     website_urls: Optional[str] = Form(None),
#     s3_file_keys: Optional[str] = Form(None), 
#     zoho_cred_encrypted: Optional[str] = Form(None),
#     zoho_region: Optional[str] = Form('IN'),
# ):

async def run_data_ingestion(
    request: Request
):
    data = await request.json()

    company_name = data.get('company_name', None)
    db_uris = data.get('db_uris', None)
    website_urls = data.get('website_urls', None)
    s3_file_keys = data.get('s3_file_keys', [])
    zoho_cred_encrypted = data.get('zoho_cred_encrypted', None)
    zoho_region = data.get('zoho_region', 'IN')
    
    """
    OPTIMIZED: Run data ingestion pipeline with progress tracking

    Args:
        company_name: Company identifier
        db_uris: Database URIs (comma-separated)
        website_urls: Website URLs (comma-separated)
        s3_file_keys: JSON list of file names in S3
        zoho_cred_encrypted: Base64 encoded Zoho credentials
        zoho_region: Zoho region (default: IN)
    """
    company_directory = None
    zoho_files = []
    base_directory = "data"
    company_directory = os.path.join(base_directory, company_name)
    os.makedirs(company_directory, exist_ok=True)

    try:
        # Validation
        if not company_name:
            raise HTTPException(status_code=400, detail="'company_name' is required")
            
        print("\n" + "=" * 60)
        print(f"🚀 DATA INGESTION PIPELINE: {company_name.upper()}")
        print("=" * 60)

        # Parse s3_file_keys from JSON string to list
        # s3_file_keys_list = []
        # if s3_file_keys and s3_file_keys.strip():
        #     try:
        #         # Try to parse as JSON array
        #         parsed = json.loads(s3_file_keys)
        #         if isinstance(parsed, list):
        #             s3_file_keys_list = [str(key).strip() for key in parsed if key and str(key).strip()]
        #         else:
        #             raise HTTPException(
        #                 status_code=400,
        #                 detail="s3_file_keys must be a JSON array, e.g., [\"file1.pdf\", \"file2.docx\"]"
        #             )
        #     except json.JSONDecodeError as e:
        #         raise HTTPException(
        #             status_code=400,
        #             detail=f"Invalid JSON format for s3_file_keys: {str(e)}. Expected: [\"file1.pdf\", \"file2.docx\"]"
        #         )
        # Parse s3_file_keys from JSON string to list
        s3_file_keys_list = []
        if s3_file_keys:
            if isinstance(s3_file_keys, list):
                # Already a list
                s3_file_keys_list = [str(key).strip() for key in s3_file_keys if key and str(key).strip()]
            elif isinstance(s3_file_keys, str):
                # Try to parse JSON string
                try:
                    parsed = json.loads(s3_file_keys)
                    if isinstance(parsed, list):
                        s3_file_keys_list = [str(key).strip() for key in parsed if key and str(key).strip()]
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="s3_file_keys must be a JSON array, e.g., [\"file1.pdf\", \"file2.docx\"]"
                        )
                except json.JSONDecodeError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON format for s3_file_keys: {str(e)}. Expected: [\"file1.pdf\", \"file2.docx\"]"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="s3_file_keys must be a JSON array or a JSON string representing an array."
                )
    
        # STEP 1: Fetch Zoho data
        if zoho_cred_encrypted and zoho_cred_encrypted.strip():
            print("\n📥 PHASE 1: ZOHO DATA EXTRACTION")
            print("-" * 60)

            try:
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
                        status_code=400, detail="Incomplete Zoho credentials"
                    )
                
                zoho_exporter = ZohoCreatorBulkExporter(
                    client_id=zoho_client_id,
                    client_secret=zoho_client_secret,
                    refresh_token=zoho_refresh_token,
                    owner_name=zoho_owner_name,
                    output_dir=company_directory,
                    zoho_region=zoho_region or "IN",
                )

                zoho_stats = zoho_exporter.export_all_data()
                zoho_files = [
                    os.path.basename(f) for f in zoho_stats["json_file_paths"]
                ]

                print(f"\n✅ Zoho extraction complete: {len(zoho_files)} files")

            except Exception as zoho_error:
                print(f"\n❌ Zoho extraction failed: {zoho_error}")
                raise HTTPException(
                    status_code=500, detail=f"Zoho error: {str(zoho_error)}"
                )

        # # Setup directory
        # base_directory = "data"
        # company_directory = os.path.join(base_directory, company_name)
        # os.makedirs(company_directory, exist_ok=True)
        
        # logger.info(f"Company directory: {company_directory}")

        # STEP 2: Save uploaded files
        downloaded_s3_files = []
        has_s3_files = len(s3_file_keys_list) > 0        

        if has_s3_files:
            logger.info("=" * 70)
            logger.info(f"STEP 2A: Downloading {len(s3_file_keys_list)} files from S3")
            logger.info("=" * 70)
            
            downloaded_s3_files = download_all_files_from_s3(
                file_list=s3_file_keys_list, 
                company_name=company_name
            )
            logger.info(f"âœ… Downloaded {len(downloaded_s3_files)} files from S3")
            
        # Combine all file paths
        all_saved_files = downloaded_s3_files
        
        # Parse other inputs
        db_uri_list = [uri.strip() for uri in db_uris.split(",")] if db_uris else []
        website_url_list = [url.strip() for url in website_urls.split(",")] if website_urls else []
        
        # STEP 3: Run DataIngestionPipeline on all collected data
        logger.info(f"STEP 3: Running DataIngestionPipeline for: {company_name}")
        print("-" * 60)

        pipeline = DataIngestionPipeline(
            company_namespace=company_name,
            directory_path=company_directory,
            db_uris=db_uri_list,
            website_urls=website_url_list,
            proxies=None,
        )

        stats = pipeline.run()
        
        # response
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
        
        logger.info("=" * 70)
        logger.info(f"   Complete ingestion finished for {company_name}")
        logger.info(f"   Zoho files: {len(zoho_files)}")
        logger.info(f"   S3 files: {len(downloaded_s3_files)}")
        logger.info(f"   Total files: {len(all_saved_files)}")
        logger.info(f"   Total vectors: {stats['vectors_upserted']:,}")
        logger.info("=" * 70)
        
        return JSONResponse(status_code=200, content=response_data)
        
    except HTTPException as http_err:
        raise http_err
        
    except Exception as e:
        error_msg = f"Data ingestion failed for {company_name or 'Unknown Company'}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
