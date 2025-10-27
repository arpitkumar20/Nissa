import os
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from nisaa.helpers.logger import logger
from nisaa.services.text_extract import TextPreprocessor
from src.nisaa.services.website_scrap import WebsiteIngester
from src.nisaa.services.document_loader import  DocumentLoader
from src.nisaa.services.sql_database import SQLDatabaseIngester


class DataIngestionPipeline:
    """
    Unified data ingestion pipeline supporting files, databases, and websites
    """
    def __init__(self, directory_path, company_namespace, db_uris=None, website_urls=None, proxies=None, preprocess_config=None):
        self.directory_path = directory_path
        self.company_namespace = company_namespace
        self.db_uris = db_uris or []
        self.website_urls = website_urls or []
        self.proxies = proxies
        self.preprocess_config = preprocess_config or {}
        """
        Initialize data ingestion pipeline

        Args:
            directory_path: Path to data directory (optional)
            company_namespace: Company identifier
            preprocess_config: Configuration for preprocessing
            db_uris: List of database URIs (optional)
            website_urls: List of website URLs to scrape (optional)
            proxies: Proxy configuration for web scraping (optional)
        """
        self.directory_path = directory_path
        self.company_namespace = company_namespace
        self.db_uris = db_uris or []
        self.website_urls = website_urls or []

        # Initialize loaders
        self.document_loader = None
        if directory_path:
            self.document_loader = DocumentLoader(directory_path, company_namespace)

        self.sql_ingester = None
        if db_uris:
            self.sql_ingester = SQLDatabaseIngester(company_namespace)

        self.website_ingester = None
        if website_urls:
            self.website_ingester = WebsiteIngester(company_namespace, proxies)

        self.preprocessor = TextPreprocessor(preprocess_config)

        self.stats = {
            "company": company_namespace,
            "total_documents": 0,
            "file_documents": 0,
            "database_documents": 0,
            "website_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "total_characters": 0,
            "total_urls_extracted": 0,
            "processing_time": 0,
        }

    def enrich_metadata(self, doc: Document, urls: List[Dict[str, str]]) -> Document:
        """Enrich document metadata with additional information"""
        source_type = doc.metadata.get("source_type", "file")

        if source_type == "file":
            source_path = doc.metadata.get("source", "")
            file_name = os.path.basename(source_path)
            file_extension = os.path.splitext(file_name)[1]

            doc.metadata.update(
                {
                    "company_namespace": self.company_namespace,
                    "file_name": file_name,
                    "file_type": file_extension,
                    "file_size": (
                        os.path.getsize(source_path)
                        if os.path.exists(source_path)
                        else 0
                    ),
                    "processed_at": datetime.now().isoformat(),
                    "char_count": len(doc.page_content),
                    "word_count": len(doc.page_content.split()),
                    "pipeline_version": "3.0.0",
                }
            )
        elif source_type == "website":
            doc.metadata.update(
                {
                    "processed_at": datetime.now().isoformat(),
                    "char_count": len(doc.page_content),
                    "word_count": len(doc.page_content.split()),
                    "pipeline_version": "3.0.0",
                }
            )
        else:
            doc.metadata.update(
                {
                    "company_namespace": self.company_namespace,
                    "processed_at": datetime.now().isoformat(),
                    "char_count": len(doc.page_content),
                    "word_count": len(doc.page_content.split()),
                    "pipeline_version": "3.0.0",
                }
            )

        if urls:
            doc.metadata["urls"] = urls
            doc.metadata["url_count"] = len(urls)
            doc.metadata["has_references"] = True
        else:
            doc.metadata["has_references"] = False
            doc.metadata["url_count"] = 0

        return doc

    def process_document(self, doc: Document) -> Optional[Document]:
        """Process a single document with preprocessing"""
        try:
            original_content = doc.page_content
            processed_content, urls = self.preprocessor.preprocess(original_content)

            if not processed_content or len(processed_content.strip()) < 10:
                source = (
                    doc.metadata.get("source")
                    or doc.metadata.get("table_name")
                    or doc.metadata.get("website_url", "unknown")
                )
                logger.warning(f"Document {source} has insufficient content")
                return None

            doc.page_content = processed_content
            doc = self.enrich_metadata(doc, urls)

            doc.metadata["original_char_count"] = len(original_content)
            doc.metadata["preprocessing_reduction"] = (
                round((1 - len(processed_content) / len(original_content)) * 100, 2)
                if len(original_content) > 0
                else 0
            )

            if urls:
                self.stats["total_urls_extracted"] += len(urls)

            return doc

        except Exception as e:
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("table_name")
                or doc.metadata.get("website_url", "unknown")
            )
            logger.error(f"Error processing document {source}: {str(e)}")
            return None

    def run(self) -> List[Document]:
        """Run the complete data ingestion pipeline"""
        start_time = datetime.now()

        print("\n" + "=" * 80)
        print(f"DATA INGESTION PIPELINE - {self.company_namespace.upper()}")
        print("=" * 80 + "\n")

        raw_documents = []

        if self.document_loader:
            print("LOADING FILES...")
            file_docs = self.document_loader.load_all_documents()
            raw_documents.extend(file_docs)
            self.stats["file_documents"] = len(file_docs)
            print(f"Loaded {len(file_docs)} documents from files\n")

        if self.sql_ingester and self.db_uris:
            db_docs = self.sql_ingester.ingest_multiple_databases(self.db_uris)
            raw_documents.extend(db_docs)
            self.stats["database_documents"] = len(db_docs)
            print()

        if self.website_ingester and self.website_urls:
            website_docs = self.website_ingester.ingest_multiple_websites(
                self.website_urls
            )
            raw_documents.extend(website_docs)
            self.stats["website_documents"] = len(website_docs)
            print()

        self.stats["total_documents"] = len(raw_documents)

        if raw_documents:
            print(f"PREPROCESSING {len(raw_documents)} DOCUMENTS...")
            processed_documents = []

            with tqdm(
                total=len(raw_documents),
                desc="Processing",
                unit="doc",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for doc in raw_documents:
                    processed_doc = self.process_document(doc)
                    if processed_doc:
                        processed_documents.append(processed_doc)
                        self.stats["processed_documents"] += 1
                        self.stats["total_characters"] += len(
                            processed_doc.page_content
                        )
                    else:
                        self.stats["failed_documents"] += 1
                    pbar.update(1)
        else:
            processed_documents = []
            print("⚠️  No documents loaded from any source")

        end_time = datetime.now()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()

        self.log_statistics()

        return processed_documents

    def log_statistics(self):
        """Log pipeline statistics"""
        print("\n" + "=" * 80)
        print(f"PIPELINE STATISTICS - {self.company_namespace.upper()}")
        print("=" * 80)
        print(f"Company Namespace      : {self.stats['company']}")
        print(f"Total Documents Loaded : {self.stats['total_documents']}")
        print(f"  ├─ From Files        : {self.stats['file_documents']}")
        print(f"  ├─ From Databases    : {self.stats['database_documents']}")
        print(f"  └─ From Websites     : {self.stats['website_documents']}")
        print(f"Successfully Processed : {self.stats['processed_documents']}")
        print(f"Failed/Skipped         : {self.stats['failed_documents']}")
        print(f"Total Characters       : {self.stats['total_characters']:,}")
        print(f"URLs Extracted         : {self.stats['total_urls_extracted']}")
        print(f"Processing Time        : {self.stats['processing_time']:.2f} seconds")
        print(
            f"Avg Time/Document      : {self.stats['processing_time'] / max(self.stats['total_documents'], 1):.2f}s"
        )
        print("=" * 80 + "\n")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats