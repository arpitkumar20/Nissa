import os
import requests
import json
import time
import zipfile
import io
import csv
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from threading import Lock
from io import StringIO

from src.nisaa.helpers.logger import logger


class ZohoCreatorExporter:
    """
    FINAL PRODUCTION: Zoho Creator Bulk Read API Exporter

    ✅ VALIDATED API IMPLEMENTATION:
    - Endpoint 1: POST /creator/v2.1/bulk/{owner}/{app}/report/{report}/read
    - Endpoint 2: GET /creator/v2.1/bulk/{owner}/{app}/report/{report}/read/{job_id}
    - Endpoint 3: GET {download_url}

    ✅ REQUIRED OAUTH SCOPE: ZohoCreator.bulk.READ

    ✅ FEATURES:
    - Auto token refresh (every 55 minutes)
    - Smart rate limiting (~2.5 req/sec)
    - Exponential backoff retry
    - 429 rate limit handling
    - 401 auth error recovery
    - Comprehensive error logging
    - Progress tracking
    """

    ZOHO_CONFIG = {
        "US": {
            "accounts": "https://accounts.zoho.com",
            "creator": "https://www.zohoapis.com",
        },
        "EU": {
            "accounts": "https://accounts.zoho.eu",
            "creator": "https://www.zohoapis.eu",
        },
        "IN": {
            "accounts": "https://accounts.zoho.in",
            "creator": "https://www.zohoapis.in",
        },
        "AU": {
            "accounts": "https://accounts.zoho.com.au",
            "creator": "https://www.zohoapis.com.au",
        },
        "CN": {
            "accounts": "https://accounts.zoho.com.cn",
            "creator": "https://www.zohoapis.com.cn",
        },
    }

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        # owner_name: str,
        output_dir: str,
        zoho_region: str = "IN",
    ):
        """
        Initialize Zoho Creator Exporter

        Args:
            client_id: Zoho OAuth client ID
            client_secret: Zoho OAuth client secret
            refresh_token: Zoho OAuth refresh token (MUST have ZohoCreator.bulk.READ scope)
            owner_name: Zoho account owner name
            output_dir: Directory to save exported JSON files
            zoho_region: Zoho region (US, EU, IN, AU, CN)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        # self.account_owner = owner_name
        self.zoho_region = zoho_region.upper()

        # Validate region
        if self.zoho_region not in self.ZOHO_CONFIG:
            logger.warning(f"⚠️ Invalid region '{zoho_region}', defaulting to IN")
            self.zoho_region = "IN"

        self.accounts_url = self.ZOHO_CONFIG[self.zoho_region]["accounts"]
        self.api_base_url = self.ZOHO_CONFIG[self.zoho_region]["creator"]

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Token management
        self.access_token = None
        self.token_expires_at = 0
        self._refresh_access_token()

        # HTTP session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Zoho-oauthtoken {self.access_token}",
                "Content-Type": "application/json",
            }
        )

        # Rate limiting
        self.request_lock = Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.4  # 2.5 requests/second (safe)

        # Statistics tracking
        self.stats = {
            "api_calls": 0,
            "reports_processed": 0,
            "total_reports": 0,
            "total_records": 0,
            "json_files": 0,
            "json_file_paths": [],
            "errors": [],
        }

        logger.info(
            f"✅ ZohoCreatorExporter initialized for region: {self.zoho_region}"
        )

    def _refresh_access_token(self) -> None:
        """
        Refresh OAuth access token using refresh token
        Token expires in 1 hour, we refresh at 55 minutes
        """
        token_url = f'{self.accounts_url}/oauth/v2/token?refresh_token={self.refresh_token}&client_id={self.client_id}&client_secret={self.client_secret}&grant_type=refresh_token'

        try:
            response = requests.post(token_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error_msg = data.get("error")
                logger.error(f"❌ OAuth Error: {error_msg}")
                raise ValueError(f"Auth Error: {error_msg}")

            self.access_token = data.get("access_token")
            if not self.access_token:
                raise ValueError("No access token received from Zoho")

            # Token expires in 3600 seconds (1 hour), refresh at 3300 (55 min)
            self.token_expires_at = time.time() + 3300

            # Update session headers
            if hasattr(self, "session"):
                self.session.headers.update(
                    {"Authorization": f"Zoho-oauthtoken {self.access_token}"}
                )

            logger.info("✅ Access token refreshed successfully")

        except Exception as e:
            logger.error(f"❌ Token refresh failed: {e}")
            raise

    def _check_and_refresh_token(self) -> None:
        """Check if token is about to expire and refresh if needed"""
        if time.time() >= self.token_expires_at:
            logger.info("🔄 Token expired, refreshing...")
            self._refresh_access_token()

    def rate_limited_request(
        self, method: str, url: str, max_retries: int = 3, **kwargs
    ) -> requests.Response:
        """
        Make rate-limited HTTP request with retry logic

        Features:
        - Automatic token refresh
        - Rate limiting (2.5 req/sec)
        - Exponential backoff retry
        - 429 rate limit handling
        - 401 auth error recovery
        """
        with self.request_lock:
            # Check and refresh token if needed
            self._check_and_refresh_token()

            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            # Retry loop
            for attempt in range(max_retries):
                try:
                    # Make request
                    if method.upper() == "GET":
                        response = self.session.get(url, **kwargs)
                    elif method.upper() == "POST":
                        response = self.session.post(url, **kwargs)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    self.last_request_time = time.time()
                    self.stats["api_calls"] += 1

                    # Handle rate limit (429)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(
                            f"⚠️ Rate limited (429). Waiting {retry_after}s..."
                        )
                        time.sleep(retry_after)
                        continue

                    # Handle auth error (401)
                    if response.status_code == 401:
                        if attempt < max_retries - 1:
                            logger.warning("⚠️ Auth error (401). Refreshing token...")
                            self._refresh_access_token()
                            continue
                        else:
                            # Log response body for debugging
                            try:
                                error_body = response.json()
                                logger.error(f"❌ 401 Error Body: {error_body}")
                            except:
                                logger.error(f"❌ 401 Error: {response.text}")

                    response.raise_for_status()
                    return response

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"❌ Request failed after {max_retries} attempts: {e}"
                        )
                        raise

                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    time.sleep(wait_time)

            raise Exception(f"Max retries ({max_retries}) exceeded for URL: {url}")
    
    def get_owner_name(self):
        try:
            url = f"{self.api_base_url}/creator/v2.1/meta/applications"
            response = self.rate_limited_request("GET", url, timeout=30)

            if response.status_code != 200:
                print("Error fetching applications:", response.text)
                return None

            data = response.json()
            apps = data.get("applications", [])
            owner_name = next(iter(set(app.get("workspace_name") or app.get("created_by") for app in apps)), None)

            return owner_name

        except Exception as e:
            logger.error({'message': 'Error getting applications', 'error': e})
            raise

    def get_all_applications(self) -> List[Dict]:
        """
        Get all Zoho Creator applications
        API Call: 1 per account
        """
        self.account_owner = self.get_owner_name()
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/applications"

        try:
            response = self.rate_limited_request("GET", url, timeout=30)
            applications = response.json().get("applications", [])
            logger.info(f"✅ Found {len(applications)} application(s)")
            return applications

        except Exception as e:
            logger.error(f"❌ Error fetching applications: {e}")
            raise

    def get_all_reports(self, app_link_name: str) -> List[Dict]:
        """
        Get all reports from a Zoho Creator application
        API Call: 1 per application
        """
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/{app_link_name}/reports"

        try:
            response = self.rate_limited_request("GET", url, timeout=30)
            reports = response.json().get("reports", [])
            return reports

        except Exception as e:
            logger.error(f"❌ Error fetching reports for app '{app_link_name}': {e}")
            return []

    def create_bulk_read_job(
        self, app_link_name: str, report_link_name: str
    ) -> Optional[str]:
        """
        STEP 1: Create Bulk Read Job

        Endpoint: POST /creator/v2.1/bulk/{owner}/{app}/report/{report}/read
        Payload: {"query": {"max_records": 200000}}
        Response: {"details": {"id": "job_id"}}

        Returns: job_id or None
        """
        url = f"{self.api_base_url}/creator/v2.1/bulk/{self.account_owner}/{app_link_name}/report/{report_link_name}/read"

        payload = {"query": {"max_records": 200000}}  # Max allowed by Zoho

        try:
            response = self.rate_limited_request(
                "POST",
                url,
                json=payload,
                timeout=30,
            )

            data = response.json()

            # Check for OAuth scope errors
            error_code = data.get("code")
            if error_code in [2945, 2946]:  # OAuth scope errors
                error_desc = data.get("description", "OAuth scope error")
                logger.error(f"❌ OAuth Scope Error: {error_desc}")
                logger.error(
                    "⚠️ Your refresh token MUST have scope: ZohoCreator.bulk.READ"
                )
                logger.error("⚠️ Regenerate token at: https://api-console.zoho.in/")
                return None

            # Extract job ID
            job_id = data.get("details", {}).get("id")
            if not job_id:
                logger.error(f"❌ No job_id in response: {data}")
                return None

            return job_id

        except requests.exceptions.HTTPError as e:
            # Log response body for debugging
            try:
                error_body = e.response.json()
                logger.error(f"❌ Bulk job creation failed: {error_body}")
            except:
                logger.error(
                    f"❌ Bulk job creation failed: {e.response.text if hasattr(e.response, 'text') else e}"
                )
            return None

        except Exception as e:
            logger.error(f"❌ Error creating bulk job: {e}")
            return None

    def check_bulk_job_status(
        self, app_link_name: str, report_link_name: str, job_id: str
    ) -> Dict:
        """
        STEP 2: Check Bulk Read Job Status

        Endpoint: GET /creator/v2.1/bulk/{owner}/{app}/report/{report}/read/{job_id}
        Response: {
            "details": {
                "status": "IN PROGRESS" | "COMPLETED" | "FAILED",
                "result": {
                    "download_url": "/path/to/download",
                    "count": 12345
                }
            }
        }

        Returns: status info dict
        """
        url = f"{self.api_base_url}/creator/v2.1/bulk/{self.account_owner}/{app_link_name}/report/{report_link_name}/read/{job_id}"

        try:
            response = self.rate_limited_request("GET", url, timeout=30)
            data = response.json()
            details = data.get("details", {})

            status = details.get("status", "").lower()
            result = details.get("result", {})

            return {
                "status": status,
                "download_url": result.get("download_url"),
                "count": result.get("count", 0),
            }

        except Exception as e:
            logger.error(f"❌ Error checking job status for job_id '{job_id}': {e}")
            return {"status": "failed"}

    def download_bulk_result(self, download_url: str, app_link_name: str, report_link_name: str, job_id: str) -> Optional[bytes]:
        """
        STEP 3: Download Bulk Read Result (ZIP file)

        Endpoint: GET /creator/v2.1/bulk/{owner}/{app}/report/{report}/read/{job_id}/result
        Response: ZIP file containing CSV

        Returns: ZIP file bytes or None
        """
        # Construct the correct download URL
        full_url = f"{self.api_base_url}/creator/v2.1/bulk/{self.account_owner}/{app_link_name}/report/{report_link_name}/read/{job_id}/result"

        try:
            response = self.rate_limited_request("GET", full_url, timeout=90)
            return response.content

        except Exception as e:
            logger.error(f"❌ Error downloading bulk result: {e}")
            return None
        
    def parse_csv_to_json(self, csv_content: str) -> List[Dict]:
        """
        Convert CSV content to JSON (list of dicts)

        Args:
            csv_content: CSV string from Zoho

        Returns:
            List of record dictionaries
        """
        try:
            reader = csv.DictReader(StringIO(csv_content))
            records = [row for row in reader]
            return records
        except Exception as e:
            logger.error(f"❌ Error parsing CSV to JSON: {e}")
            return []

    def sanitize_filename(self, name: str) -> str:
        """
        Clean filename for safe file system usage

        Removes: < > : " / \ | ? *
        Limits: 200 characters
        """
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")
        return name.strip()[:200]

    def export_report_bulk(self, app_link_name: str, report: Dict) -> Dict:
        """
        Export single report using Bulk Read API

        Process:
        1. Create bulk read job (1 API call)
        2. Poll for completion (1-5 API calls)
        3. Download ZIP result (1 API call)
        4. Extract CSV and convert to JSON
        5. Save JSON file

        Total: ~3 API calls per report
        """
        report_name = report.get("display_name", "Unknown")
        report_link_name = report.get("link_name", "")

        result = {
            "report_name": report_name,
            "success": False,
            "records_count": 0,
            "json_file": None,
            "error": None,
        }

        try:
            logger.info(f"🔄 Exporting: {report_name}")

            # STEP 1: Create bulk read job
            job_id = self.create_bulk_read_job(app_link_name, report_link_name)

            if not job_id:
                result["error"] = "Failed to create bulk job (check OAuth scopes)"
                logger.error(f"❌ {result['error']} for report: {report_name}")
                return result

            logger.info(f"   📋 Job created: {job_id}")

            # STEP 2: Poll for completion
            max_attempts = 30  # 30 attempts × 3 seconds = 90 seconds max wait
            poll_interval = 3  # seconds

            for attempt in range(max_attempts):
                time.sleep(poll_interval)

                status_info = self.check_bulk_job_status(
                    app_link_name, report_link_name, job_id
                )

                current_status = status_info["status"]

                if current_status == "completed":
                    record_count = status_info["count"]
                    logger.info(f"   ✅ Job completed: {record_count} records")

                    # STEP 3: Download result
                    download_url = status_info["download_url"]
                    if not download_url:
                        result["error"] = "No download URL in completed job"
                        logger.error(f"❌ {result['error']}")
                        return result

                    zip_content = self.download_bulk_result(download_url, app_link_name, report_link_name, job_id)
                    
                    if not zip_content:
                        result["error"] = "Failed to download ZIP result"
                        logger.error(f"❌ {result['error']}")
                        return result

                    # Extract CSV from ZIP
                    try:
                        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                            csv_filename = z.namelist()[0]
                            csv_content = z.read(csv_filename).decode("utf-8")
                    except Exception as e:
                        result["error"] = f"Failed to extract ZIP: {e}"
                        logger.error(f"❌ {result['error']}")
                        return result

                    # Convert CSV to JSON
                    records = self.parse_csv_to_json(csv_content)

                    if records:
                        # Save JSON file
                        safe_name = self.sanitize_filename(report_name)
                        json_file = self.output_dir / f"{safe_name}.json"

                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(records, f, indent=2, ensure_ascii=False)

                        result["json_file"] = str(json_file)
                        result["records_count"] = len(records)
                        result["success"] = True
                        logger.info(
                            f"   💾 Saved: {json_file.name} ({len(records)} records)"
                        )
                    else:
                        # Empty report
                        result["success"] = True
                        logger.info(f"   ⚠️ No records in report: {report_name}")

                    break

                elif current_status == "failed":
                    result["error"] = "Bulk job failed (check Zoho logs)"
                    logger.error(f"❌ {result['error']} for report: {report_name}")
                    break

                elif current_status in ["in-progress", "in progress", "inprogress"]:
                    logger.info(
                        f"   ⏳ Job in progress... (check {attempt + 1}/{max_attempts})"
                    )
                    continue

                else:
                    logger.warning(f"   ⚠️ Unknown status: {current_status}")
                    continue

            else:
                # Timeout
                result["error"] = f"Job timeout after {max_attempts * poll_interval}s"
                logger.error(f"❌ {result['error']} for report: {report_name}")

        except Exception as e:
            result["error"] = f"Exception: {str(e)}"
            logger.error(f"❌ Export error for '{report_name}': {e}", exc_info=True)

        return result

    def export_all_data(self) -> Dict:
        """
        MAIN FUNCTION: Export all Zoho Creator data using Bulk Read API

        Process:
        1. Get all applications (1 API call)
        2. Get all reports from each app (N API calls for N apps)
        3. Export each report using bulk read (~3 API calls per report)

        Expected API calls for 25 reports across 5 apps:
        - 1 (get apps) + 5 (get reports) + 75 (bulk export) = 81 calls

        Returns:
            Statistics dictionary with export results
        """
        start_time = time.time()
        self.account_owner = self.get_owner_name()

        try:
            print("\n" + "=" * 70)
            print("🚀 ZOHO CREATOR BULK EXPORT (FINAL)")
            print("=" * 70)
            print(f"   Region: {self.zoho_region}")
            print(f"   Owner: {self.account_owner}")
            print(f"   Output: {self.output_dir}")
            print("=" * 70)

            # ============= STEP 1: GET APPLICATIONS =============
            print("\n🔄 Step 1/4: Fetching applications...")
            applications = self.get_all_applications()

            if not applications:
                raise ValueError("❌ No applications found in Zoho Creator account")

            print(f"✅ Found {len(applications)} application(s)")

            # ============= STEP 2: GET ALL REPORTS =============
            print("\n🔄 Step 2/4: Scanning for reports in all applications...")
            all_reports_with_app = []

            with tqdm(
                total=len(applications), desc="Scanning apps", unit="app"
            ) as pbar:
                for app in applications:
                    app_display_name = app.get("application_name", "Unknown")
                    app_link_name = app["link_name"]

                    try:
                        reports = self.get_all_reports(app_link_name)

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
                        error_msg = f"{app_display_name}: {str(e)}"
                        self.stats["errors"].append(error_msg)
                        logger.error(f"❌ {error_msg}")

                    pbar.update(1)

            if not all_reports_with_app:
                print("⚠️ No reports found in any application")
                return self.stats

            self.stats["total_reports"] = len(all_reports_with_app)
            print(f"✅ Found {len(all_reports_with_app)} total report(s)")

            # ============= STEP 3: EXPORT REPORTS USING BULK API =============
            print(
                f"\n🔄 Step 3/4: Exporting {len(all_reports_with_app)} report(s) "
                f"using Bulk Read API..."
            )

            with tqdm(
                total=len(all_reports_with_app), desc="Exporting reports", unit="report"
            ) as pbar:
                for item in all_reports_with_app:
                    result = self.export_report_bulk(
                        item["app_link_name"], item["report"]
                    )

                    self.stats["reports_processed"] += 1

                    if result["success"] and result["records_count"] > 0:
                        self.stats["total_records"] += result["records_count"]
                        self.stats["json_files"] += 1
                        self.stats["json_file_paths"].append(result["json_file"])

                    if result["error"]:
                        self.stats["errors"].append(
                            f"{result['report_name']}: {result['error']}"
                        )

                    pbar.set_postfix_str(
                        f"{self.stats['json_files']} files, "
                        f"{self.stats['total_records']:,} records"
                    )
                    pbar.update(1)

            execution_time = time.time() - start_time

            # ============= STEP 4: SUMMARY =============
            print(f"\n✅ Step 4/4: Export Complete!")
            print("=" * 70)
            print(f"   📊 Applications Scanned: {len(applications)}")
            print(
                f"   📄 Reports Processed: {self.stats['reports_processed']}/{self.stats['total_reports']}"
            )
            print(f"   💾 JSON Files Created: {self.stats['json_files']}")
            print(f"   📝 Total Records Exported: {self.stats['total_records']:,}")
            print(f"   🔌 Total API Calls: {self.stats['api_calls']}")
            print(f"   ⏱️  Execution Time: {execution_time:.1f}s")

            if self.stats["errors"]:
                print(f"   ⚠️  Errors Encountered: {len(self.stats['errors'])}")
                print("\n   Error Details:")
                for i, error in enumerate(
                    self.stats["errors"][:10], 1
                ):  # Show first 10
                    print(f"      {i}. {error}")
                if len(self.stats["errors"]) > 10:
                    print(f"      ... and {len(self.stats['errors']) - 10} more")

            print("=" * 70)

            return self.stats

        except Exception as e:
            logger.error(f"❌ Export failed: {e}", exc_info=True)
            raise
        finally:
            self.session.close()
            logger.info("🔒 HTTP session closed")