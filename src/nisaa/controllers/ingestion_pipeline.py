"""
PRODUCTION-READY: Ingestion Pipeline with Deterministic Ordering
Ensures consistent document order for reliable checkpoint recovery
"""
import os
import json
import time
import shutil
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from nisaa.services.file_deduplication import WebsiteDeduplicator, DBDeduplicator
from src.nisaa.config.logger import logger
from src.nisaa.services.embedding_service import EmbeddingService, compute_text_hash
from nisaa.utils.json_processor import JSONProcessor
from nisaa.utils.text_extract import TextPreprocessor
from src.nisaa.services.vector_store_service import VectorStoreService
from nisaa.utils.website_scrap import WebsiteIngester
from nisaa.utils.document_loader import DocumentLoader
from nisaa.utils.sql_database import SQLDatabaseIngester
from nisaa.services.checkpoint_manager import CheckpointManager
import threading


class DataIngestionPipeline:
    """
    FIXED: Unified data ingestion pipeline with deterministic ordering
    
    Key Features:
    - Ensures consistent document ordering across runs
    - Content-based checkpoint matching
    - Prevents embedding/text mismatches
    - Safe resume from any interruption point
    - Document-level checkpointing to skip re-scraping
    """

    def __init__(
        self,
        company_namespace: str,
        directory_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        db_uris: Optional[List[str]] = None,
        website_urls: Optional[List[str]] = None,
        preprocess_config: Optional[Dict[str, bool]] = None,
        proxies: Optional[dict] = None,
        db_pool = None,
        job_id: str = None,
    ):
        self.company_namespace = company_namespace
        self.directory_path = directory_path
        self.file_paths = file_paths
        self.db_uris = db_uris or []
        self.website_urls = website_urls or []

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

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

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
            "phases_completed": []
        }

        self.website_info = {"new": [], "skipped": []}
        self.db_info = {"new": [], "skipped": []}

        self.job_id = job_id
        self.checkpoint_manager = None
        self.cancellation_event = threading.Event()

        if db_pool and job_id:
            self.checkpoint_manager = CheckpointManager(db_pool)
            self.checkpoint_manager.company_name = company_namespace
            logger.info("✓ Checkpoint/recovery system initialized")

    def ensure_deterministic_ordering(self, documents: List[Document]) -> List[Document]:
        """
        CRITICAL: Ensure documents are ALWAYS in the same order
        This prevents checkpoint mismatch issues
        
        Sorting strategy:
        1. By source type (file, database, website)
        2. By source path/URL (alphabetical)
        3. By content hash (for stability)
        """
        def sort_key(doc):
            # Get source type (file, database, website)
            source_type = doc.metadata.get('source_type', 'unknown')
            
            # Get source identifier
            source = doc.metadata.get('source', '')
            if not source:
                source = doc.metadata.get('table_name', '')
            if not source:
                source = doc.metadata.get('website_url', '')
            
            # Compute content hash for final stability
            content_hash = compute_text_hash(doc.page_content)
            
            # Sort by: source_type, source, content_hash
            return (source_type, source, content_hash)
        
        sorted_docs = sorted(documents, key=sort_key)
        
        # Log sorting verification (first 3 docs)
        logger.info(f"✓ Sorted {len(documents)} documents for deterministic ordering")
        for i, doc in enumerate(sorted_docs[:3]):
            logger.debug(
                f"  Doc {i}: {doc.metadata.get('source_type')} - "
                f"{doc.metadata.get('source', 'N/A')[:50]}"
            )
        
        return sorted_docs

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
                "pipeline_version": "7.0.0-safe",  # Updated version
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
        """Load and process with progress bars"""
        raw_documents = []

        print("\nPhase 1: Loading source documents...")

        if self.document_loader:
            with tqdm(total=1, desc="Files", leave=False) as pbar:
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

        self.stats["total_documents"] = len(raw_documents)
        print(f"Loaded {len(raw_documents)} documents")

        if raw_documents:
            print(f"\nPhase 1A: Processing {len(raw_documents)} documents...")
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
                        if self.cancellation_event.is_set():
                            logger.warning("Document processing cancelled")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
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

            self.stats["phases_completed"].append("document_processing")
            print(f"✓ Processed {len(processed_documents)} documents")
            return processed_documents

        return []
    
    def load_and_deduplicate_websites(self, job_manager, company_name: str) -> List[Document]:
        """Load websites with deduplication"""
        if not self.website_ingester or not self.website_urls:
            return []
        
        print("\nPhase 1B: Loading and deduplicating websites...")
        
        website_documents_map = {}
        
        with tqdm(total=len(self.website_urls), desc="Scraping", leave=False) as pbar:
            for url in self.website_urls:
                try:
                    docs = self.website_ingester.scrape_website(url, pbar)
                    if docs:
                        website_documents_map[url] = docs
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    pbar.update(1)
        
        new_websites, skipped_websites = WebsiteDeduplicator.filter_new_websites(
            website_urls=self.website_urls,
            website_documents_map=website_documents_map,
            job_manager=job_manager,
            company_name=company_name
        )
        
        self.website_info = {
            "new": new_websites,
            "skipped": skipped_websites
        }
        
        all_website_docs = []
        for website in new_websites:
            all_website_docs.extend(website['documents'])
        
        self.stats["website_documents"] = len(all_website_docs)
        self.stats["phases_completed"].append("website_loading")
        logger.info(f"✓ {len(all_website_docs)} documents from {len(new_websites)} new websites")
        
        return all_website_docs
    
    def load_and_deduplicate_databases(
        self, 
        job_manager, 
        company_name: str,
        db_uri_list_with_tables: List[Dict]
    ) -> List[Document]:
        """Load databases with table-level filtering"""
        if not self.sql_ingester or not db_uri_list_with_tables:
            return []

        print("\nPhase 1C: Loading filtered database tables...")

        all_db_docs = []

        with tqdm(total=len(db_uri_list_with_tables), desc="Databases", leave=False) as pbar:
            for db_config in db_uri_list_with_tables:
                db_uri = db_config["db_uri"]
                new_tables = db_config.get("new_tables", [])
                
                try:
                    docs = self.sql_ingester.ingest_specific_tables(db_uri, new_tables)
                    all_db_docs.extend(docs)
                    pbar.set_postfix_str(f"{len(docs)} docs from {len(new_tables)} tables")
                except Exception as e:
                    logger.error(f"Database loading failed for {db_uri}: {e}")
                
                pbar.update(1)

        self.stats["database_documents"] = len(all_db_docs)
        self.stats["phases_completed"].append("database_loading")
        logger.info(f"✓ {len(all_db_docs)} documents from filtered tables")

        return all_db_docs

    def process_json_files(self) -> tuple:
        """Process JSON files with checkpointing"""
        if not self.document_loader:
            return [], []

        if self.file_paths:
            json_files = [f for f in self.file_paths if f.lower().endswith('.json')]
        else:
            json_files = self.document_loader.get_json_files()

        if not json_files:
            return [], []

        print(f"\nPhase 2B: Processing {len(json_files)} JSON files...")

        all_chunks = []
        all_entities = []

        with tqdm(total=len(json_files), desc="JSON files", unit="file") as pbar:
            for json_file in json_files:
                try:
                    chunks, entities = self.json_processor.process_json_file(
                        json_file,
                        checkpoint_manager=self.checkpoint_manager,
                        job_id=self.job_id,
                        company_name=self.company_namespace,
                        cancellation_event=self.cancellation_event
                    )                    
                    all_chunks.extend(chunks)
                    all_entities.extend(entities)
                    self.stats["json_documents"] += 1
                    pbar.set_postfix_str(f"{len(all_chunks)} chunks")
                except Exception as e:
                    logger.error(f"JSON processing failed for {json_file}: {e}")
                pbar.update(1)

        self.stats["json_chunks"] = len(all_chunks)
        self.stats["phases_completed"].append("json_processing")
        print(f"✓ Created {len(all_chunks)} JSON chunks")

        return all_chunks, all_entities

    def store_to_pinecone(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        json_chunks: List[str] = None,
        json_embeddings: List[List[float]] = None,
        json_entities: List[tuple] = None,
    ):
        """Store to Pinecone with verification"""
        namespace = self.company_namespace
        total_upserted = 0

        if texts and embeddings:
            print(f"\nPhase 4: Storing {len(texts)} document vectors...")

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
                checkpoint_manager=self.checkpoint_manager,
                job_id=self.job_id,
                cancellation_event=self.cancellation_event
            )
            total_upserted += upserted
            print(f"✓ Stored {upserted} vectors")

        if json_chunks and json_embeddings:
            print(f"\nPhase 4B: Storing {len(json_chunks)} JSON vectors...")

            with tqdm(total=1, desc="Preparing JSON vectors", leave=False) as pbar:
                json_vectors = self.vector_store.prepare_json_vectors(
                    json_chunks, json_embeddings, json_entities, namespace
                )
                pbar.update(1)

            upserted = self.vector_store.upsert_json_vectors(
                json_vectors, 
                namespace, 
                100,
                checkpoint_manager=self.checkpoint_manager,
                job_id=self.job_id
            )
            total_upserted += upserted
            print(f"✓ Stored {upserted} JSON vectors")

        self.stats["vectors_upserted"] = total_upserted
        self.stats["phases_completed"].append("vector_storage")

        print("\nVerifying upload...")
        verified = self.vector_store.verify_upsert(namespace, total_upserted)
        
        if not verified:
            raise ValueError(
                f"Vector count mismatch after upload. "
                f"Expected {total_upserted}, but Pinecone returned different count."
            )

    def run(self, job_manager=None, company_name=None) -> Dict[str, Any]:
        
        try:
            start_time = datetime.now()
            
            print(f"\n{'='*60}")
            print(f"INGESTION PIPELINE: {self.company_namespace.upper()}")
            print(f"{'='*60}")
            
            # Check if resuming
            checkpoint_exists = False
            expected_doc_count = None
            current_phase = "loading"  # FIX: Track which phase we should be in
            
            if self.checkpoint_manager and self.job_id:
                checkpoints = self.checkpoint_manager.get_all_checkpoints(self.job_id)
                
                if checkpoints:
                    checkpoint_exists = True
                    
                    # CRITICAL FIX: Check phases in REVERSE ORDER (latest first)
                    # Priority: upsert > embedding > json_conversion > loading
                    
                    if 'upserting_vectors' in checkpoints:
                        current_phase = "upsert_vectors"
                        logger.info(
                            f"✓ Resuming from VECTOR UPSERT checkpoint "
                            f"(batch {checkpoints['upserting_vectors'].get('last_batch_index', 0) + 1}/"
                            f"{checkpoints['upserting_vectors'].get('total_batches', '?')})"
                        )
                        
                    elif 'upserting_json_vectors' in checkpoints:
                        current_phase = "upsert_json"
                        logger.info(
                            f"✓ Resuming from JSON UPSERT checkpoint "
                            f"(batch {checkpoints['upserting_json_vectors'].get('last_batch_index', 0) + 1}/"
                            f"{checkpoints['upserting_json_vectors'].get('total_batches', '?')})"
                        )
                        
                    elif 'embedding_documents' in checkpoints:
                        current_phase = "embedding_documents"
                        checkpoint_data = checkpoints['embedding_documents']
                        logger.info(
                            f"✓ Resuming from EMBEDDING checkpoint "
                            f"({checkpoint_data.get('embeddings_count', 0)} embeddings completed)"
                        )
                        
                    elif 'embedding_json' in checkpoints:
                        current_phase = "embedding_json"
                        logger.info("✓ Resuming from JSON EMBEDDING checkpoint")
                        
                    elif 'json_conversion' in checkpoints:
                        current_phase = "json_conversion"
                        checkpoint_data = checkpoints['json_conversion']
                        logger.info(
                            f"✓ Resuming from JSON CONVERSION checkpoint "
                            f"({checkpoint_data.get('processed', 0)} entities processed)"
                        )
                    else:
                        # Unknown checkpoint state - start fresh
                        logger.warning("⚠️ Unknown checkpoint state, starting from beginning")
                        current_phase = "loading"
            
            logger.info(f"Pipeline state: checkpoint_exists={checkpoint_exists}, current_phase={current_phase}")
        
            # =================================================================
            # PHASE 1: ALWAYS Load Documents (needed for correct chunking)
            # =================================================================
            processed_documents = []
            website_docs = []
            db_docs = []
            loaded_from_cache = False

            # SKIP PHASE 1 if resuming from embedding/upsert phase
            if current_phase not in ["embedding_documents", "embedding_json", "upsert_vectors", "upsert_json"]:
                # NEW: Try loading from document checkpoint (skips scraping on resume)
                if self.checkpoint_manager and self.job_id:
                    doc_checkpoint_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_docs.pkl"
                    if doc_checkpoint_path.exists():
                        try:
                            with open(doc_checkpoint_path, 'rb') as f:
                                processed_documents = pickle.load(f)
                            logger.info(f"✓ Resumed: Loaded {len(processed_documents)} documents from disk checkpoint")
                            loaded_from_cache = True
                            self.stats["total_documents"] = len(processed_documents)
                        except Exception as e:
                            logger.warning(f"Failed to load document checkpoint: {e}")
                
                if not loaded_from_cache:
                    if job_manager and company_name and self.website_urls:
                        print("\n[PHASE 1A] Loading and deduplicating websites...")
                        website_docs = self.load_and_deduplicate_websites(job_manager, company_name)
                    
                    if job_manager and company_name and self.db_uris:
                        print("\n[PHASE 1B] Loading and deduplicating database tables...")
                        db_docs = self.load_and_deduplicate_databases(job_manager, company_name, self.db_uris)
                    
                    print("\n[PHASE 1C] Loading and processing standard documents...")
                    processed_documents = self.load_and_process_standard_documents()
                    
                    # Combine all documents
                    if db_docs:
                        processed_documents.extend(db_docs)
                        self.stats["total_documents"] += len(db_docs)
                    
                    if website_docs:
                        processed_documents.extend(website_docs)
                        self.stats["total_documents"] += len(website_docs)
                    
                    if not processed_documents:
                        logger.warning("No documents loaded!")
                        return self.stats
                    
                    # CRITICAL: Ensure deterministic ordering
                    logger.info(f"Ensuring deterministic ordering for {len(processed_documents)} documents...")
                    processed_documents = self.ensure_deterministic_ordering(processed_documents)

                    # NEW: Save document checkpoint to prevent re-scraping on failures
                    if self.checkpoint_manager and self.job_id and processed_documents:
                        try:
                            doc_checkpoint_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_docs.pkl"
                            with open(doc_checkpoint_path, 'wb') as f:
                                pickle.dump(processed_documents, f)
                            logger.info(f"✓ Checkpoint: Saved {len(processed_documents)} documents to disk")
                        except Exception as e:
                            logger.warning(f"Failed to save document checkpoint: {e}")
            else:
                logger.info(f"SKIPPING Phase 1 (document loading) - resuming from {current_phase}")
                # Load documents from cache for chunking
                if self.checkpoint_manager and self.job_id:
                    doc_checkpoint_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_docs.pkl"
                    if doc_checkpoint_path.exists():
                        try:
                            with open(doc_checkpoint_path, 'rb') as f:
                                processed_documents = pickle.load(f)
                            logger.info(f"✓ Loaded cached documents: {len(processed_documents)} docs")
                        except Exception as e:
                            logger.error(f"Failed to load cached documents: {e}")
                            raise

            # =================================================================
            # PHASE 2: Chunking ALL Documents
            # =================================================================
            print(f"\nPhase 2: Chunking {len(processed_documents)} documents...")
            
            with tqdm(total=1, desc="Chunking", leave=False) as pbar:
                chunks = self.text_splitter.split_documents(processed_documents)
                self.stats["total_chunks"] = len(chunks)
                pbar.update(1)
            
            print(f"✓ Created {len(chunks)} chunks")
            
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Log chunk verification info
            logger.info(f"Chunk count: {len(chunks)}")
            if texts:
                logger.info(f"First 3 chunk hashes: {[compute_text_hash(t)[:8] for t in texts[:3]]}")

            # =================================================================
            # PHASE 3: Generate Embeddings (with SAFE content-based matching)
            # =================================================================
            embeddings = []
            if current_phase not in ["upsert_vectors", "upsert_json"]:
                print(f"\nPhase 3: Generating embeddings for {len(texts)} chunks...")
                
                embeddings = self.embedding_service.generate_for_documents(
                    texts,
                    checkpoint_manager=self.checkpoint_manager,
                    job_id=self.job_id,
                    cancellation_event=self.cancellation_event
                )
                
                # Verify counts match
                if len(embeddings) != len(texts):
                    logger.error(
                        f"CRITICAL: Embedding count mismatch! "
                        f"Embeddings: {len(embeddings)}, Texts: {len(texts)}"
                    )
                    raise ValueError("Embedding/text count mismatch - pipeline cannot continue safely")
                
                self.stats["total_embeddings"] = len(embeddings)
                self.stats["phases_completed"].append("embedding")
                print(f"✓ Generated {len(embeddings)} embeddings")
                
                # CRITICAL: Save texts and embeddings to disk for upsert resume
                if self.checkpoint_manager and self.job_id and embeddings:
                    try:
                        vectors_cache_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_vectors_cache.pkl"
                        with open(vectors_cache_path, 'wb') as f:
                            pickle.dump({
                                'texts': texts,
                                'embeddings': embeddings,
                                'metadatas': metadatas,
                                'count': len(texts)
                            }, f)
                        logger.info(f"✓ Vector cache saved: {len(texts)} texts + {len(embeddings)} embeddings")
                    except Exception as e:
                        logger.warning(f"Failed to save vector cache: {e}")
            else:
                logger.info(f"SKIPPING Phase 3 (embeddings) - resuming from {current_phase}")
                # Retrieve embeddings from checkpoint
                if self.checkpoint_manager and self.job_id:
                    checkpoint_data = self.checkpoint_manager.load_checkpoint(self.job_id, "embedding_documents")
                    if checkpoint_data:
                        embeddings_count = checkpoint_data.get("embeddings_count", 0)
                        logger.info(f"✓ Using cached embeddings: {embeddings_count} embeddings from previous run")
                        self.stats["total_embeddings"] = embeddings_count
            
            # =================================================================
            # PHASE 4: JSON Processing
            # =================================================================
            json_chunks = []
            json_embeddings = []
            json_entities = []
            
            if current_phase not in ["embedding_json", "upsert_vectors", "upsert_json"]:
                if "json_processing" not in self.stats.get("phases_completed", []):
                    print("\n[PHASE 2B] Processing JSON files...")
                    json_chunks, json_entities = self.process_json_files()
                    
                    if json_chunks:
                        print(f"Generating embeddings for {len(json_chunks)} JSON chunks...")
                        json_embeddings = self.embedding_service.generate_for_json_chunks(
                            json_chunks,
                            checkpoint_manager=self.checkpoint_manager,
                            job_id=self.job_id,
                            cancellation_event=self.cancellation_event
                        )
                        
                        if not json_embeddings:
                            raise ValueError("JSON embedding phase failed")
                        
                        logger.info(f"JSON embedding complete: {len(json_embeddings)} embeddings")
                        
                        # CRITICAL: Save JSON vectors cache for upsert resume
                        if self.checkpoint_manager and self.job_id and json_embeddings:
                            try:
                                json_cache_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_json_vectors_cache.pkl"
                                with open(json_cache_path, 'wb') as f:
                                    pickle.dump({
                                        'json_chunks': json_chunks,
                                        'json_embeddings': json_embeddings,
                                        'json_entities': json_entities,
                                        'count': len(json_chunks)
                                    }, f)
                                logger.info(f"✓ JSON vector cache saved: {len(json_chunks)} chunks + {len(json_embeddings)} embeddings")
                            except Exception as e:
                                logger.warning(f"Failed to save JSON vector cache: {e}")
            else:
                logger.info(f"SKIPPING Phase 4 (JSON processing) - resuming from {current_phase}")
            
            # =================================================================
            # PHASE 5: Vector Storage
            # FIX: This should ALWAYS run (never skip, even on resume)
            # =================================================================
            print("\n[PHASE 3] Storing to Pinecone...")
            
            # CRITICAL: If resuming from upsert, load cached vectors
            if current_phase in ["upsert_vectors", "upsert_json"]:
                logger.info(f"Resuming from {current_phase} - loading cached vectors...")
                
                # Load document vectors cache (ALWAYS needed)
                vectors_cache_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_vectors_cache.pkl"
                if vectors_cache_path.exists():
                    try:
                        with open(vectors_cache_path, 'rb') as f:
                            cache = pickle.load(f)
                        texts = cache.get('texts', [])
                        embeddings = cache.get('embeddings', [])
                        metadatas = cache.get('metadatas', [])
                        logger.info(f"✓ Loaded {len(texts)} texts and {len(embeddings)} embeddings from cache")
                    except Exception as e:
                        logger.error(f"Failed to load vector cache: {e}")
                        raise ValueError(f"Cannot resume upsert without vector cache: {e}")
                else:
                    logger.error(f"Vector cache not found at {vectors_cache_path}")
                    raise ValueError(
                        f"Cannot resume upsert phase without cached vectors. "
                        f"Cache should have been saved during embedding phase."
                    )
                
                # Load JSON vectors cache if resuming from JSON upsert
                if current_phase == "upsert_json":
                    json_cache_path = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_json_vectors_cache.pkl"
                    if json_cache_path.exists():
                        try:
                            with open(json_cache_path, 'rb') as f:
                                json_cache = pickle.load(f)
                            json_chunks = json_cache.get('json_chunks', [])
                            json_embeddings = json_cache.get('json_embeddings', [])
                            json_entities = json_cache.get('json_entities', [])
                            logger.info(f"✓ Loaded {len(json_chunks)} JSON chunks from cache")
                        except Exception as e:
                            logger.error(f"Failed to load JSON vector cache: {e}")
                            json_chunks, json_embeddings, json_entities = [], [], []
            
            self.store_to_pinecone(
                texts, embeddings, metadatas,
                json_chunks, json_embeddings, json_entities
            )
            
            # Calculate final stats
            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()
            
            # Print summary
            print(f"\n{'='*60}")
            print("FINAL STATISTICS:")
            print(f"{'='*60}")
            print(f"Total documents: {self.stats['total_documents']}")
            print(f"  - Files: {self.stats['file_documents']}")
            print(f"  - Databases: {self.stats['database_documents']}")
            print(f"  - Websites: {self.stats['website_documents']}")
            print(f"  - JSON: {self.stats['json_documents']}")
            print(f"\nProcessed: {self.stats['processed_documents']}")
            print(f"Failed: {self.stats['failed_documents']}")
            print(f"Total chunks: {self.stats['total_chunks']}")
            print(f"Total embeddings: {self.stats['total_embeddings']}")
            print(f"Vectors stored: {self.stats['vectors_upserted']}")
            print(f"Processing time: {self.stats['processing_time']:.2f}s")
            print(f"Phases completed: {', '.join(self.stats['phases_completed'])}")
            print(f"{'='*60}\n")
            
            # Cleanup checkpoints on success
            # Only clear checkpoints/caches when ALL expected vectors were upserted.
            if self.checkpoint_manager and self.job_id:
                try:
                    expected_vectors = 0
                    # Prefer stats but fallback to available variables
                    expected_vectors = self.stats.get('total_embeddings', 0)
                    # include json embeddings if present
                    expected_vectors += self.stats.get('json_chunks', 0)

                    actual_upserted = self.stats.get('vectors_upserted', 0)

                    # If we upserted everything (or more), clear checkpoints
                    if expected_vectors > 0 and actual_upserted >= expected_vectors:
                        self.checkpoint_manager.clear_checkpoint(self.job_id)
                        # Also clear the document cache and vector caches
                        doc_cache = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_docs.pkl"
                        if doc_cache.exists():
                            doc_cache.unlink()

                        vectors_cache = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_vectors_cache.pkl"
                        if vectors_cache.exists():
                            vectors_cache.unlink()

                        json_cache = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_json_vectors_cache.pkl"
                        if json_cache.exists():
                            json_cache.unlink()

                        logger.info("All checkpoints and caches cleared after successful completion")
                    else:
                        # Partial upsert or no expected vectors: preserve checkpoints for resume
                        logger.info(
                            f"Partial or no upsert detected (expected={expected_vectors}, upserted={actual_upserted}). "
                            f"Preserving checkpoints for resume."
                        )
                except Exception as e:
                    logger.error(f"Error during checkpoint cleanup: {e}")
            
            # Cleanup ingestion directory
            try:
                if self.directory_path and os.path.exists(self.directory_path):
                    shutil.rmtree(self.directory_path)
                    logger.info(f"Cleaned up ingestion directory: {self.directory_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup ingestion directory: {e}")
            
            # Mark interrupted if cancellation event was set during run
            if self.cancellation_event and self.cancellation_event.is_set():
                self.stats['interrupted'] = True
            else:
                self.stats['interrupted'] = False

            return self.stats
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            if self.checkpoint_manager and self.job_id:
                logger.info(
                    f"Checkpoints saved for job {self.job_id}. "
                    f"Resume by rerunning with same company name"
                )
            
            raise