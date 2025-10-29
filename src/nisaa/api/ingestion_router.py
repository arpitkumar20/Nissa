import os
import requests
import json
import time
from typing import List, Optional, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
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
                 owner_name: str, app_link_name: str, output_dir: str, 
                 zoho_region: str = 'IN'):
        """Initialize Exporter"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.account_owner = owner_name
        self.app_link_name = app_link_name
        self.zoho_region = zoho_region.upper()
        
        if self.zoho_region not in self.ZOHO_CONFIG:
            self.zoho_region = 'IN'
        
        self.accounts_url = self.ZOHO_CONFIG[self.zoho_region]['accounts']
        self.api_base_url = self.ZOHO_CONFIG[self.zoho_region]['creator']
        
        # Optimized for maximum speed
        self.export_config = {
            'output_dir': output_dir,
            'max_workers': 15,  # Increased workers
            'max_records': 200,
            'rate_limit_delay': 0.05,  # Reduced delay for faster requests
            'connection_timeout': 10,
            'read_timeout': 20
        }
        
        logger.info(f"⚡ Zoho Exporter initialized (Region: {self.zoho_region}, Workers: {self.export_config['max_workers']})")
        
        self.access_token = self._get_access_token()
        self.headers = {'Authorization': f'Zoho-oauthtoken {self.access_token}'}
        
        # Session for connection pooling (faster)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.request_lock = Lock()
        self.last_request_time = 0
        
        self.stats = {
            'reports_processed': 0,
            'total_reports': 0,
            'total_records': 0,
            'json_files': 0,
            'json_file_paths': [],
            'errors': []
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
            
            logger.info("✓ Zoho access token obtained")
            return access_token
            
        except Exception as e:
            logger.error(f"✗ Error getting Zoho token: {e}")
            raise
    
    def rate_limited_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Make rate-limited request with session (faster connection pooling)"""
        with self.request_lock:
            elapsed = time.time() - self.last_request_time
            
            if elapsed < self.export_config['rate_limit_delay']:
                time.sleep(self.export_config['rate_limit_delay'] - elapsed)
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=(self.export_config['connection_timeout'], self.export_config['read_timeout'])
            )
            self.last_request_time = time.time()
            
            return response
    
    def get_application_by_name(self, app_name: str) -> Optional[Dict]:
        """Get application details"""
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/applications"
        
        try:
            response = self.rate_limited_request(url)
            
            if response.status_code == 401:
                raise ValueError("Authorization error - check OWNER_NAME and token")
            
            response.raise_for_status()
            applications = response.json().get('applications', [])
            
            if not applications:
                logger.warning("No applications found")
                return None
            
            logger.info(f"Found {len(applications)} Zoho applications")
            
            for app in applications:
                app_display = app.get('application_name', '')
                app_link = app.get('link_name', '')
                
                if app_display == app_name or app_link == app_name:
                    logger.info(f"✓ Found Zoho app: {app_display}")
                    return app
            
            available = [a.get('application_name') for a in applications]
            logger.warning(f"✗ '{app_name}' not found. Available: {available}")
            return None
            
        except Exception as e:
            logger.error(f"✗ Error fetching Zoho apps: {e}")
            raise
    
    def get_all_reports(self, app_link_name: str) -> List[Dict]:
        """Get all reports from application"""
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/{app_link_name}/reports"
        
        try:
            response = self.rate_limited_request(url)
            response.raise_for_status()
            
            reports = response.json().get('reports', [])
            logger.info(f"✓ Found {len(reports)} Zoho reports")
            
            return reports
            
        except Exception as e:
            logger.error(f"✗ Error fetching Zoho reports: {e}")
            raise
    
    def get_all_records_paginated(self, app_link_name: str, report_link_name: str, 
                                 report_name: str) -> List[Dict]:
        """Fetch all records with pagination (no verbose logging)"""
        all_records = []
        from_index = 1
        max_records = self.export_config['max_records']
        
        while True:
            url = f"{self.api_base_url}/creator/v2.1/data/{self.account_owner}/{app_link_name}/report/{report_link_name}"
            params = {'from': from_index, 'limit': max_records}
            
            try:
                response = self.rate_limited_request(url, params)
                response.raise_for_status()
                
                data = response.json()
                records = data.get('data', [])
                
                if not records:
                    break
                
                all_records.extend(records)
                
                if len(records) < max_records:
                    break
                
                from_index += max_records
                
            except Exception as e:
                logger.error(f"✗ Error fetching '{report_name}' at index {from_index}: {e}")
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
            
            # Save JSON with minimal I/O
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
    
    def _update_progress(self, completed: int, total: int, report_name: str = None):
        """Update progress bar"""
        percentage = (completed / total * 100) if total > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * completed / total) if total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        status = f"✓ {report_name}" if report_name else "Processing..."
        
        logger.info(f"Progress: [{bar}] {percentage:.1f}% ({completed}/{total}) {status}")
    
    def export_all_data(self) -> Dict:
        """Export all reports from application with progress bar"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Starting Zoho data export: {self.app_link_name}")
            
            app = self.get_application_by_name(self.app_link_name)
            if not app:
                raise ValueError(f"Zoho application '{self.app_link_name}' not found")
            
            app_link_name = app['link_name']
            
            output_dir = Path(self.export_config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            reports = self.get_all_reports(app_link_name)
            
            if not reports:
                logger.warning("No Zoho reports found")
                return self.stats
            
            self.stats['total_reports'] = len(reports)
            
            logger.info(f"📊 Exporting {len(reports)} Zoho reports (Workers: {self.export_config['max_workers']})...")
            logger.info("=" * 70)
            
            # Initial progress bar
            self._update_progress(0, len(reports))
            
            with ThreadPoolExecutor(max_workers=self.export_config['max_workers']) as executor:
                future_to_report = {
                    executor.submit(
                        self.export_report_data,
                        app_link_name,
                        report,
                        output_dir
                    ): report
                    for report in reports
                }
                
                for future in as_completed(future_to_report):
                    report = future_to_report[future]
                    
                    try:
                        result = future.result()
                        
                        with self.stats_lock:
                            self.stats['reports_processed'] += 1
                            
                            if result['success']:
                                if result['records_count'] > 0:
                                    self.stats['total_records'] += result['records_count']
                                    self.stats['json_files'] += 1
                                    self.stats['json_file_paths'].append(result['json_file'])
                                    
                                    # Update progress with report name
                                    self._update_progress(
                                        self.stats['reports_processed'], 
                                        len(reports),
                                        f"{result['report_name']} ({result['records_count']} records)"
                                    )
                                else:
                                    # Empty report
                                    self._update_progress(
                                        self.stats['reports_processed'], 
                                        len(reports),
                                        f"{result['report_name']} (empty)"
                                    )
                            else:
                                if result['error']:
                                    self.stats['errors'].append(result['error'])
                                    self._update_progress(
                                        self.stats['reports_processed'], 
                                        len(reports),
                                        f"{result['report_name']} (ERROR)"
                                    )
                    
                    except Exception as e:
                        error_msg = f"Error processing '{report.get('display_name')}': {str(e)}"
                        with self.stats_lock:
                            self.stats['errors'].append(error_msg)
                            self.stats['reports_processed'] += 1
                            self._update_progress(
                                self.stats['reports_processed'], 
                                len(reports),
                                f"{report.get('display_name')} (FAILED)"
                            )
            
            execution_time = time.time() - start_time
            
            logger.info("=" * 70)
            logger.info(f"✅ Zoho Export Complete:")
            logger.info(f"   Reports Exported: {self.stats['reports_processed']}/{len(reports)}")
            logger.info(f"   JSON Files Created: {self.stats['json_files']}")
            logger.info(f"   Total Records: {self.stats['total_records']:,}")
            logger.info(f"   Errors: {len(self.stats['errors'])}")
            logger.info(f"   Time: {execution_time:.2f}s")
            logger.info(f"   Speed: {self.stats['total_records']/execution_time:.0f} records/sec" if execution_time > 0 else "")
            logger.info("=" * 70)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"✗ Zoho export failed: {e}")
            raise
        finally:
            # Close session
            self.session.close()


@router.post("/ingest", response_model=IngestionResponse)
async def run_data_ingestion(
    company_name: str = Form(None),
    db_uris: Optional[str] = Form(None),
    website_urls: Optional[str] = Form(None),
    file_documents: Optional[List[UploadFile]] = File(None),
    # Zoho credentials
    zoho_client_id: Optional[str] = Form(None),
    zoho_client_secret: Optional[str] = Form(None),
    zoho_refresh_token: Optional[str] = Form(None),
    zoho_owner_name: Optional[str] = Form(None),
    zoho_app_link_name: Optional[str] = Form(None),
    zoho_region: Optional[str] = Form('IN'),
):
    """
    Run comprehensive data ingestion pipeline with Zoho integration
    
    Process order:
    1. Fetch Zoho JSON files (if credentials provided) - WITH PROGRESS BAR
    2. Save uploaded files
    3. Run DataIngestionPipeline on all collected data
    
    Args:
        company_name: Company identifier (used as Pinecone namespace)
        db_uris: Database connection strings (comma-separated)
        website_urls: Website URLs to scrape (comma-separated)
        file_documents: Files to upload and process
        zoho_client_id: Zoho OAuth Client ID
        zoho_client_secret: Zoho OAuth Client Secret
        zoho_refresh_token: Zoho OAuth Refresh Token
        zoho_owner_name: Zoho account owner name
        zoho_app_link_name: Zoho Creator application link name
        zoho_region: Zoho region (US/EU/IN/AU/CN, default: IN)
        
    Returns:
        Detailed statistics and status of the ingestion process
    """
    company_directory = None
    zoho_files = []
    
    try:
        # Validation
        if not company_name:
            raise HTTPException(status_code=400, detail="'company_name' is required")
        
        # Check if at least one data source is provided
        has_zoho = all([zoho_client_id, zoho_client_secret, zoho_refresh_token, 
                       zoho_owner_name, zoho_app_link_name])
        
        if not db_uris and not website_urls and not file_documents and not has_zoho:
            raise HTTPException(
                status_code=400,
                detail="At least one data source must be provided (db_uris, website_urls, file_documents, or Zoho credentials)"
            )
        
        # Setup directory
        base_directory = "data"
        company_directory = os.path.join(base_directory, company_name)
        os.makedirs(company_directory, exist_ok=True)
        
        logger.info(f"📁 Company directory: {company_directory}")
        
        # STEP 1: Fetch Zoho data first (if credentials provided)
        if has_zoho:
            try:
                logger.info("=" * 70)
                logger.info("STEP 1: Fetching Zoho Data")
                logger.info("=" * 70)
                
                zoho_exporter = ZohoCreatorExporter(
                    client_id=zoho_client_id,
                    client_secret=zoho_client_secret,
                    refresh_token=zoho_refresh_token,
                    owner_name=zoho_owner_name,
                    app_link_name=zoho_app_link_name,
                    output_dir=company_directory,
                    zoho_region=zoho_region or 'IN'
                )
                
                zoho_stats = zoho_exporter.export_all_data()
                zoho_files = [os.path.basename(f) for f in zoho_stats['json_file_paths']]
                
                if zoho_stats['errors']:
                    logger.warning(f"⚠️ Zoho export had {len(zoho_stats['errors'])} errors")
                    for error in zoho_stats['errors'][:5]:  # Show first 5 errors
                        logger.warning(f"  - {error}")
                    if len(zoho_stats['errors']) > 5:
                        logger.warning(f"  ... and {len(zoho_stats['errors']) - 5} more errors")
                
                logger.info(f"✅ Zoho data saved: {len(zoho_files)} JSON files with {zoho_stats['total_records']:,} total records")
                
            except Exception as zoho_error:
                logger.error(f"❌ Zoho export failed: {zoho_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Zoho data export failed: {str(zoho_error)}"
                )
        
        # STEP 2: Save uploaded files
        saved_files = []
        if file_documents:
            logger.info("=" * 70)
            logger.info(f"STEP 2: Saving {len(file_documents)} uploaded files")
            logger.info("=" * 70)
            
            for i, file in enumerate(file_documents, 1):
                file_path = os.path.join(company_directory, file.filename)
                
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                
                saved_files.append(file_path)
                
                # Progress indicator
                percentage = (i / len(file_documents)) * 100
                logger.info(f"   [{percentage:.0f}%] ✓ {file.filename}")
            
            logger.info(f"✅ Uploaded files saved: {len(saved_files)} files")
        
        # Parse other inputs
        db_uri_list = [uri.strip() for uri in db_uris.split(",")] if db_uris else []
        website_url_list = [url.strip() for url in website_urls.split(",")] if website_urls else []
        
        # STEP 3: Run DataIngestionPipeline on all collected data
        logger.info("=" * 70)
        logger.info(f"STEP 3: Running DataIngestionPipeline for: {company_name}")
        logger.info("=" * 70)
        
        pipeline = DataIngestionPipeline(
            company_namespace=company_name,
            directory_path=company_directory,
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
        
        logger.info("=" * 70)
        logger.info(f"✅ Complete ingestion finished for {company_name}")
        logger.info(f"   Zoho files: {len(zoho_files)}")
        logger.info(f"   Uploaded files: {len(saved_files)}")
        logger.info(f"   Total vectors: {stats['vectors_upserted']:,}")
        logger.info("=" * 70)
        
        return JSONResponse(status_code=200, content=response_data)
        
    except HTTPException as http_err:
        raise http_err
        
    except Exception as e:
        error_msg = f"Data ingestion failed for {company_name or 'Unknown Company'}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)