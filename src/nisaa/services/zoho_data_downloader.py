import os
import requests
import json
import time
import base64
import zipfile
import io
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from threading import Lock

from src.nisaa.helpers.logger import logger


class ZohoCreatorBulkExporter:
    """
    CRITICAL OPTIMIZATION: Uses Zoho Bulk Read API

    API Call Savings:
    - OLD METHOD: 1 call per page * 100s of pages = 100-1000 API calls per report
    - NEW METHOD: 3 calls per report (create job + check status + download)

    For 25 reports:
    - OLD: 2,500+ API calls
    - NEW: 75 API calls (97% reduction!)
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
        owner_name: str,
        output_dir: str,
        zoho_region: str = "IN",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.account_owner = owner_name
        self.zoho_region = zoho_region.upper()

        if self.zoho_region not in self.ZOHO_CONFIG:
            self.zoho_region = "IN"

        self.accounts_url = self.ZOHO_CONFIG[self.zoho_region]["accounts"]
        self.api_base_url = self.ZOHO_CONFIG[self.zoho_region]["creator"]

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.access_token = self._get_access_token()
        self.headers = {"Authorization": f"Zoho-oauthtoken {self.access_token}"}
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

    def _get_access_token(self) -> str:
        """Get access token"""
        token_url = f"{self.accounts_url}/oauth/v2/token"
        params = {
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
        }

        try:
            response = requests.post(token_url, params=params, timeout=30)
            data = response.json()

            if "error" in data:
                raise ValueError(f"Auth Error: {data.get('error')}")

            return data.get("access_token")

        except Exception as e:
            logger.error(f"Error getting token: {e}")
            raise

    def rate_limited_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """Rate-limited requests with tracking"""
        with self.request_lock:
            elapsed = time.time() - self.last_request_time

            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)

            if method.upper() == "GET":
                response = self.session.get(url, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")

            self.last_request_time = time.time()
            self.stats["api_calls"] += 1

            return response

    def get_all_applications(self) -> List[Dict]:
        """Get all applications (1 API call)"""
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/applications"

        try:
            response = self.rate_limited_request("GET", url, timeout=30)
            response.raise_for_status()
            return response.json().get("applications", [])

        except Exception as e:
            logger.error(f"Error fetching applications: {e}")
            raise

    def get_all_reports(self, app_link_name: str) -> List[Dict]:
        """Get all reports (1 API call per app)"""
        url = f"{self.api_base_url}/creator/v2.1/meta/{self.account_owner}/{app_link_name}/reports"

        try:
            response = self.rate_limited_request("GET", url, timeout=30)
            response.raise_for_status()
            return response.json().get("reports", [])

        except Exception as e:
            logger.error(f"Error fetching reports: {e}")
            raise

    def create_bulk_read_job(
        self, app_link_name: str, report_link_name: str
    ) -> Optional[str]:
        """
        BULK READ API: Create async export job (1 API call)
        Returns job_id
        """
        url = f"{self.api_base_url}/creator/v2.1/bulk/{self.account_owner}/{app_link_name}/report/{report_link_name}/read"

        payload = {"query": {"max_records": 200000}}  # Max allowed per job

        try:
            response = self.rate_limited_request(
                "POST",
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            job_id = data.get("details", {}).get("id")

            return job_id

        except Exception as e:
            logger.error(f"Error creating bulk job: {e}")
            return None

    def check_bulk_job_status(
        self, app_link_name: str, report_link_name: str, job_id: str
    ) -> Dict:
        """
        BULK READ API: Check job status (1 API call per check)
        """
        url = f"{self.api_base_url}/creator/v2.1/bulk/{self.account_owner}/{app_link_name}/report/{report_link_name}/read/{job_id}"

        try:
            response = self.rate_limited_request("GET", url, timeout=30)
            response.raise_for_status()

            data = response.json()
            details = data.get("details", {})

            return {
                "status": details.get("status", "").lower(),
                "download_url": details.get("result", {}).get("download_url"),
                "count": details.get("result", {}).get("count", 0),
            }

        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return {"status": "failed"}

    def download_bulk_result(self, download_url: str) -> Optional[bytes]:
        """
        BULK READ API: Download result CSV (1 API call)
        Returns ZIP file content
        """
        full_url = f"{self.api_base_url}{download_url}"

        try:
            response = self.rate_limited_request("GET", full_url, timeout=60)
            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.error(f"Error downloading result: {e}")
            return None

    def parse_csv_to_json(self, csv_content: str) -> List[Dict]:
        """Convert CSV to JSON format"""
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(csv_content))
        return [row for row in reader]

    def export_report_bulk(self, app_link_name: str, report: Dict) -> Dict:
        """
        OPTIMIZED: Export using Bulk Read API (only 3 API calls per report!)
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
            # Step 1: Create bulk read job (1 API call)
            job_id = self.create_bulk_read_job(app_link_name, report_link_name)

            if not job_id:
                result["error"] = "Failed to create bulk job"
                return result

            # Step 2: Poll for completion (1-3 API calls typically)
            max_attempts = 20
            for attempt in range(max_attempts):
                time.sleep(3)  # Wait 3 seconds between checks

                status_info = self.check_bulk_job_status(
                    app_link_name, report_link_name, job_id
                )

                if status_info["status"] == "completed":
                    # Step 3: Download result (1 API call)
                    zip_content = self.download_bulk_result(status_info["download_url"])

                    if not zip_content:
                        result["error"] = "Failed to download result"
                        return result

                    # Extract CSV from ZIP
                    with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                        csv_filename = z.namelist()[0]
                        csv_content = z.read(csv_filename).decode("utf-8")

                    # Convert CSV to JSON
                    records = self.parse_csv_to_json(csv_content)

                    if records:
                        # Save JSON
                        safe_name = self.sanitize_filename(report_name)
                        json_file = self.output_dir / f"{safe_name}.json"

                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(records, f, indent=2, ensure_ascii=False)

                        result["json_file"] = str(json_file)
                        result["records_count"] = len(records)
                        result["success"] = True
                    else:
                        result["success"] = True  # Empty report

                    break

                elif status_info["status"] == "failed":
                    result["error"] = "Bulk job failed"
                    break

                elif status_info["status"] in ["in-progress", "in progress"]:
                    continue

            else:
                result["error"] = "Job timeout"

        except Exception as e:
            result["error"] = f"Error: {str(e)}"

        return result

    def sanitize_filename(self, name: str) -> str:
        """Clean filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")
        return name.strip()

    def export_all_data(self) -> Dict:
        """
        OPTIMIZED: Export all data using Bulk Read API

        API Call Reduction Example:
        - 25 reports with avg 1000 records each
        - OLD: 25 reports * 5 pages * 1 call/page = 125+ calls
        - NEW: 25 reports * 3 calls/report = 75 calls
        - SAVINGS: 40% reduction + handles large datasets better
        """
        start_time = time.time()

        try:
            # Step 1: Get applications (1 API call)
            print("\n🔄 Step 1/4: Fetching Zoho applications...")
            applications = self.get_all_applications()

            if not applications:
                raise ValueError("No applications found")

            print(f"✅ Found {len(applications)} applications")

            # Step 2: Get all reports
            print("\n🔄 Step 2/4: Scanning for reports...")
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

                        pbar.set_postfix_str(f"{len(all_reports_with_app)} reports")
                    except Exception as e:
                        self.stats["errors"].append(str(e))

                    pbar.update(1)

            if not all_reports_with_app:
                print("⚠️ No reports found")
                return self.stats

            self.stats["total_reports"] = len(all_reports_with_app)
            print(f"✅ Found {len(all_reports_with_app)} reports")

            # Step 3: Export using Bulk Read API
            print(
                f"\n🔄 Step 3/4: Exporting {len(all_reports_with_app)} reports (Bulk API)..."
            )

            with tqdm(
                total=len(all_reports_with_app), desc="Exporting", unit="report"
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
                        self.stats["errors"].append(result["error"])

                    pbar.set_postfix_str(
                        f"{self.stats['json_files']} files, {self.stats['total_records']} records"
                    )
                    pbar.update(1)

            execution_time = time.time() - start_time

            # Summary
            print(f"\n✅ Step 4/4: Zoho Export Complete")
            print(f"   📊 Applications: {len(applications)}")
            print(f"   📄 Reports: {self.stats['reports_processed']}")
            print(f"   💾 JSON Files: {self.stats['json_files']}")
            print(f"   📝 Records: {self.stats['total_records']:,}")
            print(f"   🔌 API Calls: {self.stats['api_calls']} (Bulk API)")
            print(f"   ⏱️ Time: {execution_time:.1f}s")

            if self.stats["errors"]:
                print(f"   ⚠️ Errors: {len(self.stats['errors'])}")

            return self.stats

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
        finally:
            self.session.close()
