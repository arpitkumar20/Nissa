import os
import json
import hashlib
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class FileDeduplicator:
    """Handles file hashing and deduplication checks"""
    
    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute MD5 hash of file contents"""
        hash_func = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            raise
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, any]:
        """Get file metadata"""
        try:
            stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                "file_path": file_path,
                "file_name": path_obj.name,
                "file_type": path_obj.suffix,
                "file_size": stat.st_size,
                "file_hash": FileDeduplicator.compute_file_hash(file_path),
                "modified_time": stat.st_mtime
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise
    
    @staticmethod
    def filter_new_files(
        file_paths: List[str],
        job_manager,
        company_name: str
    ) -> tuple:
        """
        Filter out already processed files
        
        Returns:
            Tuple of (new_files, skipped_files)
        """
        logger.info(f"Checking {len(file_paths)} files for duplicates...")
        
        new_files = []
        skipped_files = []
        
        for file_path in file_paths:
            try:
                file_info = FileDeduplicator.get_file_info(file_path)
                
                existing = job_manager.is_file_processed(
                    company_name=company_name,
                    file_path=file_path,
                    file_hash=file_info["file_hash"]
                )
                
                if existing:
                    logger.info(
                        f"Skipping {file_info['file_name']} "
                        f"(already processed in job {existing['job_id']})"
                    )
                    skipped_files.append({
                        **file_info,
                        "reason": "already_processed",
                        "previous_job": existing['job_id'],
                        "previous_vectors": existing['vector_count'],
                        "processed_at": existing['processed_at']
                    })
                else:
                    new_files.append(file_info)
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(
            f"Deduplication complete: "
            f"{len(new_files)} new, {len(skipped_files)} skipped"
        )
        
        return new_files, skipped_files


class WebsiteDeduplicator:
    """Handles website content hashing and deduplication"""

    @staticmethod
    def compute_content_hash(documents: List[Document]) -> str:
        """
        Compute hash of all page contents from a website
        
        Args:
            documents: List of Document objects from website
            
        Returns:
            MD5 hash of combined content
        """
        hash_func = hashlib.md5()

        try:
            sorted_docs = sorted(documents, key=lambda d: d.metadata.get('source', ''))

            for doc in sorted_docs:
                content = doc.page_content.encode('utf-8')
                hash_func.update(content)

            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash website content: {e}")
            raise

    @staticmethod
    def filter_new_websites(
        website_urls: List[str],
        website_documents_map: Dict[str, List[Document]],
        job_manager,
        company_name: str
    ) -> tuple:
        """
        Filter out already processed websites
        
        Args:
            website_urls: List of website URLs
            website_documents_map: Dict mapping URL to its documents
            job_manager: JobManager instance
            company_name: Company namespace
            
        Returns:
            Tuple of (new_websites, skipped_websites)
            new_websites: List of dicts with url, documents, content_hash
            skipped_websites: List of dicts with url, reason, previous_job
        """
        logger.info(f"Checking {len(website_urls)} websites for duplicates...")

        new_websites = []
        skipped_websites = []

        for url in website_urls:
            documents = website_documents_map.get(url, [])

            if not documents:
                logger.warning(f"No documents found for {url}")
                continue

            try:
                content_hash = WebsiteDeduplicator.compute_content_hash(documents)

                existing = job_manager.is_website_processed(
                    company_name=company_name,
                    website_url=url,
                    content_hash=content_hash
                )

                if existing:
                    logger.info(
                        f"Skipping {url} "
                        f"(already processed in job {existing['job_id']})"
                    )
                    skipped_websites.append({
                        "url": url,
                        "content_hash": content_hash,
                        "page_count": len(documents),
                        "reason": "already_processed",
                        "previous_job": existing['job_id'],
                        "previous_vectors": existing['vector_count'],
                        "processed_at": existing['processed_at']
                    })
                else:
                    new_websites.append({
                        "url": url,
                        "documents": documents,
                        "content_hash": content_hash,
                        "page_count": len(documents)
                    })

            except Exception as e:
                logger.warning(f"Error processing {url}: {e}")
                continue

        logger.info(
            f"Website deduplication complete: "
            f"{len(new_websites)} new, {len(skipped_websites)} skipped"
        )

        return new_websites, skipped_websites


class DBDeduplicator:
    """Handles database connection hashing and table-level deduplication"""

    @staticmethod
    def compute_db_hash(db_uri: str) -> str:
        """
        Compute hash of database URI (without password for security)
        """
        try:
            from urllib.parse import urlparse, urlunparse

            parsed = urlparse(db_uri)
            netloc_without_pwd = parsed.hostname or ""
            if parsed.port:
                netloc_without_pwd += f":{parsed.port}"
            if parsed.username:
                netloc_without_pwd = f"{parsed.username}@{netloc_without_pwd}"

            sanitized = urlunparse(
                (
                    parsed.scheme,
                    netloc_without_pwd,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )

            hash_func = hashlib.md5()
            hash_func.update(sanitized.encode("utf-8"))
            return hash_func.hexdigest()

        except Exception as e:
            logger.error(f"Failed to hash database URI: {e}")
            raise

    @staticmethod
    def compute_table_hash(db_uri: str, table_name: str, row_count: int = 0) -> str:
        """
        Compute hash for a specific table including row count
        This helps detect if table data has changed

        Args:
            db_uri: Database connection string
            table_name: Name of the table
            row_count: Number of rows in table (optional, for change detection)

        Returns:
            MD5 hash of db_uri + table_name + row_count
        """
        try:
            db_hash = DBDeduplicator.compute_db_hash(db_uri)
            content = f"{db_hash}:{table_name}:{row_count}"

            hash_func = hashlib.md5()
            hash_func.update(content.encode("utf-8"))
            return hash_func.hexdigest()

        except Exception as e:
            logger.error(f"Failed to hash table {table_name}: {e}")
            raise

    @staticmethod
    def get_db_info(db_uri: str) -> Dict[str, any]:
        """Get database metadata"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(db_uri)

            return {
                "db_uri": db_uri,
                "db_type": parsed.scheme,
                "db_host": parsed.hostname,
                "db_port": parsed.port,
                "db_name": parsed.path.lstrip("/") if parsed.path else None,
                "db_hash": DBDeduplicator.compute_db_hash(db_uri),
            }
        except Exception as e:
            logger.error(f"Failed to get DB info: {e}")
            raise

    @staticmethod
    def filter_new_tables(
        db_uri: str,
        available_tables: List[Dict[str, any]],
        job_manager,
        company_name: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter out already processed tables from a database

        Args:
            db_uri: Database connection string
            available_tables: List of dicts with 'table_name' and 'row_count'
            job_manager: JobManager instance
            company_name: Company namespace

        Returns:
            Tuple of (new_tables, skipped_tables)

        Example:
            available_tables = [
                {"table_name": "users", "row_count": 1000},
                {"table_name": "orders", "row_count": 5000}
            ]
        """
        logger.info(f"Checking {len(available_tables)} tables for duplicates...")

        new_tables = []
        skipped_tables = []

        db_info = DBDeduplicator.get_db_info(db_uri)

        for table_info in available_tables:
            table_name = table_info.get("table_name")
            row_count = table_info.get("row_count", 0)

            try:
                table_hash = DBDeduplicator.compute_table_hash(
                    db_uri=db_uri, table_name=table_name, row_count=row_count
                )

                # Check if this specific table is already processed
                existing = job_manager.is_db_table_processed(
                    company_name=company_name,
                    db_hash=db_info["db_hash"],
                    table_name=table_name,
                    table_hash=table_hash,
                )

                if existing:
                    logger.info(
                        f"Skipping table {table_name} ({row_count} rows) "
                        f"(already processed in job {existing['job_id']})"
                    )
                    skipped_tables.append(
                        {
                            "table_name": table_name,
                            "row_count": row_count,
                            "table_hash": table_hash,
                            "reason": "already_processed",
                            "previous_job": existing["job_id"],
                            "previous_vectors": existing.get("vector_count", 0),
                            "processed_at": existing.get("processed_at"),
                        }
                    )
                else:
                    new_tables.append(
                        {
                            "table_name": table_name,
                            "row_count": row_count,
                            "table_hash": table_hash,
                            **db_info,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing table {table_name}: {e}")
                continue

        logger.info(
            f"Table deduplication complete: "
            f"{len(new_tables)} new, {len(skipped_tables)} skipped"
        )

        return new_tables, skipped_tables

    @staticmethod
    def filter_new_databases(
        db_uris: List[str], job_manager, company_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        LEGACY METHOD: Filter databases at URI level only

        ⚠️ WARNING: This doesn't handle partial processing!
        Use filter_new_tables() instead for table-level tracking

        Args:
            db_uris: List of database URIs
            job_manager: JobManager instance
            company_name: Company namespace

        Returns:
            Tuple of (new_databases, skipped_databases)
        """
        logger.warning(
            "Using URI-level deduplication. "
            "Consider using table-level tracking for better reliability."
        )

        logger.info(f"Checking {len(db_uris)} databases for duplicates...")

        new_databases = []
        skipped_databases = []

        for db_uri in db_uris:
            try:
                db_info = DBDeduplicator.get_db_info(db_uri)

                existing = job_manager.is_database_processed(
                    company_name=company_name, db_uri=db_uri, db_hash=db_info["db_hash"]
                )

                if existing:
                    logger.info(
                        f"Skipping {db_info['db_name']} "
                        f"(already processed in job {existing['job_id']})"
                    )
                    skipped_databases.append(
                        {
                            **db_info,
                            "reason": "already_processed",
                            "previous_job": existing["job_id"],
                            "previous_vectors": existing["vector_count"],
                            "processed_at": existing["processed_at"],
                        }
                    )
                else:
                    new_databases.append(db_info)

            except Exception as e:
                logger.warning(f"Error processing {db_uri}: {e}")
                continue

        logger.info(
            f"Database deduplication complete: "
            f"{len(new_databases)} new, {len(skipped_databases)} skipped"
        )

        return new_databases, skipped_databases


class ZohoDeduplicator:
    """
    NEW: Handles Zoho JSON content hashing and deduplication
    
    Unlike file deduplication (which hashes the file), this hashes the 
    JSON CONTENT to detect if Zoho data has actually changed.
    """
    
    @staticmethod
    def compute_content_hash(json_file_path: str) -> str:
        """
        Compute hash of JSON file CONTENT (not file bytes)
        
        This allows detecting data changes even if file metadata differs
        
        Args:
            json_file_path: Path to Zoho JSON export file
            
        Returns:
            MD5 hash of JSON content
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            json_string = json.dumps(data, sort_keys=True, ensure_ascii=False)
            
            hash_func = hashlib.md5()
            hash_func.update(json_string.encode('utf-8'))
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash Zoho JSON content: {e}")
            raise
    
    @staticmethod
    def extract_report_info(json_file_path: str) -> Dict[str, any]:
        """
        Extract report metadata from Zoho JSON file
        
        Args:
            json_file_path: Path to Zoho JSON file
            
        Returns:
            Dict with report_name, record_count, content_hash
        """
        try:
            report_name = Path(json_file_path).stem
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            record_count = len(data) if isinstance(data, list) else 1            
            content_hash = ZohoDeduplicator.compute_content_hash(json_file_path)
            
            return {
                "file_path": json_file_path,
                "report_name": report_name,
                "record_count": record_count,
                "content_hash": content_hash,
                "app_name": "Unknown"
            }
            
        except Exception as e:
            logger.error(f"Failed to extract Zoho report info: {e}")
            raise
    
    @staticmethod
    def filter_new_zoho_reports(
        zoho_json_files: List[str],
        job_manager,
        company_name: str
    ) -> tuple:
        """
        Filter out already processed Zoho reports based on CONTENT hash
        
        Args:
            zoho_json_files: List of Zoho JSON file paths
            job_manager: JobManager instance
            company_name: Company namespace
            
        Returns:
            Tuple of (new_reports, skipped_reports)
        """
        if not zoho_json_files:
            return [], []
        
        logger.info(f"Checking {len(zoho_json_files)} Zoho reports for duplicates...")
        
        new_reports = []
        skipped_reports = []
        
        for json_file in zoho_json_files:
            try:
                report_info = ZohoDeduplicator.extract_report_info(json_file)
                
                existing = job_manager.is_zoho_report_processed(
                    company_name=company_name,
                    report_name=report_info["report_name"],
                    content_hash=report_info["content_hash"]
                )
                
                if existing:
                    logger.info(
                        f"Skipping {report_info['report_name']} "
                        f"({report_info['record_count']} records - already processed in job {existing['job_id']})"
                    )
                    skipped_reports.append({
                        **report_info,
                        "reason": "already_processed",
                        "previous_job": existing['job_id'],
                        "previous_vectors": existing['vector_count'],
                        "processed_at": existing['processed_at']
                    })
                else:
                    new_reports.append(report_info)
                    logger.debug(
                        f"New report: {report_info['report_name']} "
                        f"({report_info['record_count']} records)"
                    )
                    
            except Exception as e:
                logger.warning(f"Error processing Zoho file {json_file}: {e}")
                continue
        
        logger.info(
            f"Zoho deduplication complete: "
            f"{len(new_reports)} new, {len(skipped_reports)} skipped"
        )
        
        return new_reports, skipped_reports
