from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from tqdm import tqdm

from src.nisaa.controllers.file_deduplication import WebsiteDeduplicator,DBDeduplicator
from src.nisaa.helpers.logger import logger
from src.nisaa.services.embedding_service import EmbeddingService
from src.nisaa.services.json_processor import JSONProcessor
from src.nisaa.services.text_extract import TextPreprocessor
from src.nisaa.services.vector_store_service import VectorStoreService
from src.nisaa.services.website_scrap import WebsiteIngester
from src.nisaa.services.document_loader import DocumentLoader
from src.nisaa.services.sql_database import SQLDatabaseIngester

class DataIngestionPipeline:
    """OPTIMIZED: Unified data ingestion pipeline with clean progress tracking"""

    def __init__(
        self,
        company_namespace: str,
        directory_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        db_uris: Optional[List[str]] = None,
        website_urls: Optional[List[str]] = None,
        preprocess_config: Optional[Dict[str, bool]] = None,
        proxies: Optional[dict] = None,
    ):
        self.company_namespace = company_namespace
        self.directory_path = directory_path
        self.file_paths = file_paths
        self.db_uris = db_uris or []
        self.website_urls = website_urls or []

        # Initialize components
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
        self.json_processor = JSONProcessor(
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            int(os.getenv("MAX_WORKERS", "3")),
        )

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Statistics
        self.stats = {
            "company": company_namespace,
            "total_documents": 0,
            "file_documents": 0,
            "database_documents": 0,
            "website_documents": 0,
            "json_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "json_chunks": 0,
            "total_embeddings": 0,
            "vectors_upserted": 0,
            "processing_time": 0,
        }

        self.website_info = {
            "new": [],
            "skipped": []
        }

        self.db_info = {
            "new": [],
            "skipped": []
        }

    def enrich_metadata(self, doc: Document, urls: List[Dict[str, str]]) -> Document:
        """Enrich document metadata"""
        source_type = doc.metadata.get("source_type", "file")

        if source_type == "file":
            source_path = doc.metadata.get("source", "")
            file_name = os.path.basename(source_path)
            file_extension = os.path.splitext(file_name)[1]

            doc.metadata.update(
                {
                    "file_name": file_name,
                    "file_type": file_extension,
                    "file_size": (
                        os.path.getsize(source_path)
                        if os.path.exists(source_path)
                        else 0
                    ),
                }
            )

        doc.metadata.update(
            {
                "company_namespace": self.company_namespace,
                "processed_at": datetime.now().isoformat(),
                "char_count": len(doc.page_content),
                "word_count": len(doc.page_content.split()),
                "pipeline_version": "5.0.0",
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
        """Process single document"""
        try:
            # Skip preprocessing for database documents
            if doc.metadata.get("source_type") == "database":
                doc.metadata["company_namespace"] = self.company_namespace
                doc.metadata["processed_at"] = datetime.now().isoformat()
                doc.metadata["char_count"] = len(doc.page_content)
                doc.metadata["word_count"] = len(doc.page_content.split())
                return doc

            original_content = doc.page_content
            processed_content, urls = self.preprocessor.preprocess(original_content)

            if not processed_content or len(processed_content.strip()) < 10:
                return None

            doc.page_content = processed_content
            doc = self.enrich_metadata(doc, urls)

            doc.metadata["original_char_count"] = len(original_content)
            doc.metadata["preprocessing_reduction"] = (
                round((1 - len(processed_content) / len(original_content)) * 100, 2)
                if len(original_content) > 0
                else 0
            )

            return doc

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None

    def load_and_process_standard_documents(self) -> List[Document]:
        """OPTIMIZED: Load and process with progress bars + website deduplication"""
        raw_documents = []

        print("\n📄 Loading source documents...")

        # Load files (already deduplicated)
        if self.document_loader:
            with tqdm(total=1, desc="📂 Files", leave=False) as pbar:
                try:
                    if self.file_paths:
                        docs = self.document_loader.load_specific_files(self.file_paths)
                    else:
                        docs = self.document_loader.load_all_documents(True)
                    
                    raw_documents.extend(docs)
                    self.stats["file_documents"] = len(docs)
                    pbar.set_postfix_str(f"{len(docs)} docs")
                except Exception as e:
                    logger.error(f"File loading failed: {e}")
                pbar.update(1)

        # Load databases
        # if self.sql_ingester and self.db_uris:
        #     with tqdm(total=1, desc="🗄️ Databases", leave=False) as pbar:
        #         try:
        #             docs = self.sql_ingester.ingest_multiple_databases(self.db_uris)
        #             raw_documents.extend(docs)
        #             self.stats["database_documents"] = len(docs)
        #             pbar.set_postfix_str(f"{len(docs)} docs")
        #         except Exception as e:
        #             logger.error(f"Database loading failed: {e}")
        #         pbar.update(1)

        self.stats["total_documents"] = len(raw_documents)
        print(f"✅ Loaded {len(raw_documents)} documents")

        # Process documents
        if raw_documents:
            print(f"\n📄 Processing {len(raw_documents)} documents...")
            processed_documents = []

            with tqdm(total=len(raw_documents), desc="Processing", unit="doc") as pbar:
                with ThreadPoolExecutor(
                    max_workers=int(os.getenv("MAX_WORKERS", "3"))
                ) as executor:
                    futures = {
                        executor.submit(self.process_document, doc): doc
                        for doc in raw_documents
                    }

                    for future in as_completed(futures):
                        try:
                            processed_doc = future.result()
                            if processed_doc:
                                processed_documents.append(processed_doc)
                                self.stats["processed_documents"] += 1
                            else:
                                self.stats["failed_documents"] += 1
                        except Exception as e:
                            logger.error(f"Processing error: {e}")
                            self.stats["failed_documents"] += 1

                        pbar.update(1)
                        pbar.set_postfix_str(f"{self.stats['processed_documents']} OK")

            print(f"✅ Processed {len(processed_documents)} documents")
            return processed_documents

        return []
    
    def load_and_deduplicate_websites(self, job_manager, company_name: str) -> List[Document]:
        """Load websites with deduplication"""
        if not self.website_ingester or not self.website_urls:
            return []
        
        print("\n🌐 Loading and deduplicating websites...")
        
        # Load all websites first (get documents per URL)
        website_documents_map = {}
        
        with tqdm(total=len(self.website_urls), desc="🌐 Scraping", leave=False) as pbar:
            for url in self.website_urls:
                try:
                    docs = self.website_ingester.scrape_website(url, pbar)
                    if docs:
                        website_documents_map[url] = docs
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    pbar.update(1)
        
        # Deduplicate websites
        new_websites, skipped_websites = WebsiteDeduplicator.filter_new_websites(
            website_urls=self.website_urls,
            website_documents_map=website_documents_map,
            job_manager=job_manager,
            company_name=company_name
        )
        
        # Store website info for later marking
        self.website_info = {
            "new": new_websites,
            "skipped": skipped_websites
        }
        
        # Collect all documents from new websites
        all_website_docs = []
        for website in new_websites:
            all_website_docs.extend(website['documents'])
        
        self.stats["website_documents"] = len(all_website_docs)
        logger.info(f"✅ {len(all_website_docs)} documents from {len(new_websites)} new websites")
        
        return all_website_docs
    
    def load_and_deduplicate_databases(self, job_manager, company_name: str) -> List[Document]:
        """Load databases with deduplication"""
        if not self.sql_ingester or not self.db_uris:
            return []
        
        print("\n🗄️ Loading and deduplicating databases...")
        
        # Deduplicate databases
        new_databases, skipped_databases = DBDeduplicator.filter_new_databases(
            db_uris=self.db_uris,
            job_manager=job_manager,
            company_name=company_name
        )
        
        # Store database info for later marking
        self.db_info = {
            "new": new_databases,
            "skipped": skipped_databases
        }
        
        # Process only new databases
        all_db_docs = []
        if new_databases:
            new_db_uris = [db['db_uri'] for db in new_databases]
            
            with tqdm(total=1, desc="🗄️ Databases", leave=False) as pbar:
                try:
                    docs = self.sql_ingester.ingest_multiple_databases(new_db_uris)
                    all_db_docs.extend(docs)
                    pbar.set_postfix_str(f"{len(docs)} docs")
                except Exception as e:
                    logger.error(f"Database loading failed: {e}")
                pbar.update(1)
        
        self.stats["database_documents"] = len(all_db_docs)
        logger.info(f"✅ {len(all_db_docs)} documents from {len(new_databases)} new databases")
        
        return all_db_docs

    def process_json_files(self) -> tuple:
        """
        ✅ UPDATED: Process JSON files with Zoho awareness
        
        Zoho files have already been deduplicated by content hash,
        so we only process the new ones passed to the pipeline
        """
        if not self.document_loader:
            return [], []

        # Filter JSON files from file_paths (already deduplicated)
        if self.file_paths:
            json_files = [f for f in self.file_paths if f.lower().endswith('.json')]
        else:
            json_files = self.document_loader.get_json_files()

        if not json_files:
            return [], []

        print(f"\n📊 Processing {len(json_files)} JSON files...")
        
        # ✅ Note: Zoho JSON files here have already passed deduplication
        # They are guaranteed to be NEW or CHANGED content

        all_chunks = []
        all_entities = []

        with tqdm(total=len(json_files), desc="JSON files", unit="file") as pbar:
            for json_file in json_files:
                try:
                    chunks, entities = self.json_processor.process_json_file(json_file)
                    all_chunks.extend(chunks)
                    all_entities.extend(entities)
                    self.stats["json_documents"] += 1
                    pbar.set_postfix_str(f"{len(all_chunks)} chunks")
                except Exception as e:
                    logger.error(f"JSON processing failed for {json_file}: {e}")
                pbar.update(1)

        self.stats["json_chunks"] = len(all_chunks)
        print(f"✅ Created {len(all_chunks)} JSON chunks")

        return all_chunks, all_entities

    def chunk_and_embed_documents(self, documents: List[Document]) -> tuple:
        """OPTIMIZED: Chunk and embed with progress"""
        if not documents:
            return [], [], []

        print(f"\n📄 Chunking {len(documents)} documents...")

        with tqdm(total=1, desc="Chunking", leave=False) as pbar:
            chunks = self.text_splitter.split_documents(documents)
            self.stats["total_chunks"] = len(chunks)
            pbar.update(1)

        print(f"✅ Created {len(chunks)} chunks")

        # Extract data
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings SEQUENTIALLY (updated service handles this)
        print(f"\n📄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_service.generate_for_documents(texts)

        self.stats["total_embeddings"] = len(embeddings)
        print(f"✅ Generated {len(embeddings)} embeddings")

        return texts, embeddings, metadatas

    def store_to_pinecone(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        json_chunks: List[str] = None,
        json_embeddings: List[List[float]] = None,
        json_entities: List[tuple] = None,
    ):
        """OPTIMIZED: Store with progress tracking"""
        namespace = self.company_namespace
        total_upserted = 0

        # Store standard documents
        if texts and embeddings:
            print(f"\n📄 Storing {len(texts)} document vectors...")

            with tqdm(total=1, desc="Preparing vectors", leave=False) as pbar:
                ids, vectors, processed_metadatas = (
                    self.vector_store.prepare_document_vectors(
                        texts, embeddings, metadatas, namespace
                    )
                )
                pbar.update(1)

            upserted = self.vector_store.upsert_vectors(
                ids,
                vectors,
                processed_metadatas,
                namespace,
                int(os.getenv("PINECONE_BATCH_SIZE", "100")),
            )
            total_upserted += upserted
            print(f"✅ Stored {upserted} vectors")

        # Store JSON chunks
        if json_chunks and json_embeddings:
            print(f"\n📄 Storing {len(json_chunks)} JSON vectors...")

            with tqdm(total=1, desc="Preparing JSON vectors", leave=False) as pbar:
                json_vectors = self.vector_store.prepare_json_vectors(
                    json_chunks, json_embeddings, json_entities, namespace
                )
                pbar.update(1)

            upserted = self.vector_store.upsert_json_vectors(
                json_vectors, namespace, 100
            )
            total_upserted += upserted
            print(f"✅ Stored {upserted} JSON vectors")

        self.stats["vectors_upserted"] = total_upserted

        # Verify
        print("\n📄 Verifying upload...")
        self.vector_store.verify_upsert(namespace, total_upserted)

    def run(self, job_manager=None, company_name=None) -> Dict[str, Any]:
        """Execute pipeline with file AND website deduplication"""
        start_time = datetime.now()

        # ✅ CRITICAL FIX: Initialize ALL variables at the start
        processed_documents = []
        website_docs = []  # ✅ Added this
        texts = []
        embeddings = []
        metadatas = []
        json_chunks = []
        json_embeddings = []
        json_entities = []

        print("\n" + "="*70)
        print(f"🚀 SEQUENTIAL DATA INGESTION: {self.company_namespace.upper()}")
        print("="*70)

        # =================================================================
        # PHASE 1: PROCESS STANDARD DOCUMENTS (Files, DB, Websites)
        # =================================================================
        print("\n" + "="*70)
        print("📁 PHASE 1: LOADING & PROCESSING STANDARD DOCUMENTS")
        print("="*70)

        # PHASE 1A: Website Deduplication (if applicable)
        if job_manager and company_name and self.website_urls:
            print("\n" + "="*70)
            print("🌐 PHASE 1A: WEBSITE DEDUPLICATION")
            print("="*70)
            website_docs = self.load_and_deduplicate_websites(job_manager, company_name)
        
        # PHASE 1B: Database Deduplication (if applicable)
        db_docs = []
        if job_manager and company_name and self.db_uris:
            print("\n" + "="*70)
            print("🗄️ PHASE 1B: DATABASE DEDUPLICATION")
            print("="*70)
            db_docs = self.load_and_deduplicate_databases(job_manager, company_name)
        
        # PHASE 1C: Load standard documents (files only)
        processed_documents = self.load_and_process_standard_documents()
        
        # Merge database documents with processed documents
        if db_docs:
            processed_documents.extend(db_docs)
            self.stats["total_documents"] += len(db_docs)
        
        # Merge website documents with processed documents
        if website_docs:
            processed_documents.extend(website_docs)
            self.stats["total_documents"] += len(website_docs)

        # Process and store if we have documents
        if processed_documents:
            # Chunk and embed standard documents
            texts, embeddings, metadatas = self.chunk_and_embed_documents(
                processed_documents
            )

            # Store standard documents BEFORE moving to JSON
            if texts and embeddings:
                print("\n" + "="*70)
                print("💾 STORING STANDARD DOCUMENT VECTORS")
                print("="*70)
                
                namespace = self.company_namespace
                
                print(f"\n📄 Storing {len(texts)} document vectors...")
                with tqdm(total=1, desc="Preparing vectors", leave=False) as pbar:
                    ids, vectors, processed_metadatas = (
                        self.vector_store.prepare_document_vectors(
                            texts, embeddings, metadatas, namespace
                        )
                    )
                    pbar.update(1)

                upserted = self.vector_store.upsert_vectors(
                    ids,
                    vectors,
                    processed_metadatas,
                    namespace,
                    int(os.getenv("PINECONE_BATCH_SIZE", "100")),
                )
                self.stats["vectors_upserted"] += upserted
                print(f"✅ Stored {upserted} standard document vectors")
                
                # Verify standard documents
                print(f"\n📄 Verifying standard document upload...")
                self.vector_store.verify_upsert(namespace, upserted)
        else:
            print("\n⚠️ No standard documents to process (all skipped or none provided)")

        # =================================================================
        # PHASE 2: PROCESS JSON FILES (Zoho data)
        # ONLY STARTS AFTER PHASE 1 IS COMPLETE
        # =================================================================
        print("\n" + "="*70)
        print("📊 PHASE 2: PROCESSING JSON FILES (ZOHO DATA)")
        print("="*70)
        
        json_chunks, json_entities = self.process_json_files()

        if json_chunks:
            # Embed JSON chunks
            print(f"\n📄 Generating embeddings for {len(json_chunks)} JSON chunks...")
            json_embeddings = self.embedding_service.generate_for_json_chunks(
                json_chunks
            )
            print(f"✅ Generated {len(json_embeddings)} JSON embeddings")

            # Store JSON vectors SEPARATELY
            print("\n" + "="*70)
            print("💾 STORING JSON VECTORS")
            print("="*70)
            
            namespace = self.company_namespace
            
            print(f"\n📄 Storing {len(json_chunks)} JSON vectors...")
            with tqdm(total=1, desc="Preparing JSON vectors", leave=False) as pbar:
                json_vectors = self.vector_store.prepare_json_vectors(
                    json_chunks, json_embeddings, json_entities, namespace
                )
                pbar.update(1)

            upserted = self.vector_store.upsert_json_vectors(
                json_vectors, namespace, 100
            )
            self.stats["vectors_upserted"] += upserted
            print(f"✅ Stored {upserted} JSON vectors")
            
            # Verify JSON upload
            print(f"\n📄 Verifying JSON upload...")
            self.vector_store.verify_upsert(namespace, self.stats["vectors_upserted"])
        else:
            print("\n⚠️ No JSON files to process")

        # =================================================================
        # FINAL STATISTICS
        # =================================================================
        end_time = datetime.now()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()

        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        print(f"   Total Documents: {self.stats['total_documents']}")
        print(f"   - Files: {self.stats['file_documents']}")
        print(f"   - Databases: {self.stats['database_documents']}")
        print(f"   - Websites: {self.stats['website_documents']}")
        print(f"   - JSON: {self.stats['json_documents']}")
        print(f"\n   Processed: {self.stats['processed_documents']}")
        print(f"   Failed: {self.stats['failed_documents']}")
        print(f"   Total Chunks: {self.stats['total_chunks']}")
        print(f"   Total Embeddings: {self.stats['total_embeddings'] + self.stats['json_chunks']}")
        print(f"   Vectors Stored: {self.stats['vectors_upserted']}")
        print(f"   Processing Time: {self.stats['processing_time']:.2f}s")
        print("="*70 + "\n")

        return self.stats