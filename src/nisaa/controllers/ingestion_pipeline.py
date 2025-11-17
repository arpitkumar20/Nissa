from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from tqdm import tqdm

from nisaa.services.file_deduplication import WebsiteDeduplicator,DBDeduplicator
from src.nisaa.config.logger import logger
from src.nisaa.services.embedding_service import EmbeddingService
from nisaa.utils.json_processor import JSONProcessor
from nisaa.utils.text_extract import TextPreprocessor
from src.nisaa.services.vector_store_service import VectorStoreService
from nisaa.utils.website_scrap import WebsiteIngester
from nisaa.utils.document_loader import DocumentLoader
from nisaa.utils.sql_database import SQLDatabaseIngester

from nisaa.services.checkpoint_manager import CheckpointManager, ProcessedItemTracker
import threading    



class DataIngestionPipeline:
    """Unified data ingestion pipeline with clean progress tracking"""

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
        }

        self.website_info = {
            "new": [],
            "skipped": []
        }

        self.db_info = {
            "new": [],
            "skipped": []
        }

        self.job_id = job_id
        self.checkpoint_manager = None
        self.cancellation_event = threading.Event() 

        if db_pool and job_id:
            self.checkpoint_manager = CheckpointManager(db_pool)
            self.checkpoint_manager.company_name = company_namespace
            logger.info("Checkpoint/recovery system initialized")

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
        """Load and process with progress bars and website deduplication"""
        raw_documents = []

        print("\nLoading source documents...")

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
            print(f"\nProcessing {len(raw_documents)} documents...")
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

            print(f"Processed {len(processed_documents)} documents")
            return processed_documents

        return []
    
    def load_and_deduplicate_websites(self, job_manager, company_name: str) -> List[Document]:
        """Load websites with deduplication"""
        if not self.website_ingester or not self.website_urls:
            return []
        
        print("\nLoading and deduplicating websites...")
        
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
        logger.info(f"{len(all_website_docs)} documents from {len(new_websites)} new websites")
        
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

        print("\nLoading filtered database tables...")

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
        logger.info(f"{len(all_db_docs)} documents from filtered tables")

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

        print(f"\nProcessing {len(json_files)} JSON files...")

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
        print(f"Created {len(all_chunks)} JSON chunks")

        return all_chunks, all_entities


    def chunk_and_embed_documents(self, documents: List[Document]) -> tuple:
        """Chunk and embed with progress and checkpointing"""
        if not documents:
            return [], [], []

        print(f"\nChunking {len(documents)} documents...")

        with tqdm(total=1, desc="Chunking", leave=False) as pbar:
            chunks = self.text_splitter.split_documents(documents)
            self.stats["total_chunks"] = len(chunks)
            pbar.update(1)

        print(f"Created {len(chunks)} chunks")

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        print(f"\nGenerating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_service.generate_for_documents(
            texts,
            checkpoint_manager=self.checkpoint_manager,
            job_id=self.job_id,
            cancellation_event=self.cancellation_event  
        )

        self.stats["total_embeddings"] = len(embeddings)
        print(f"Generated {len(embeddings)} embeddings")

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
        """Store with progress tracking and checkpointing"""
        namespace = self.company_namespace
        total_upserted = 0

        if texts and embeddings:
            print(f"\nStoring {len(texts)} document vectors...")

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
            print(f"Stored {upserted} vectors")

        if json_chunks and json_embeddings:
            print(f"\nStoring {len(json_chunks)} JSON vectors...")

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
            print(f"Stored {upserted} JSON vectors")

        self.stats["vectors_upserted"] = total_upserted

        print("\nVerifying upload...")
        self.vector_store.verify_upsert(namespace, total_upserted)


    def run(self, job_manager=None, company_name=None) -> Dict[str, Any]:
        """Execute pipeline with checkpoint resume support"""
        
        try:
            start_time = datetime.now()
            
            resume_data = None
            if self.checkpoint_manager and self.job_id:
                checkpoint = self.checkpoint_manager.load_checkpoint(self.job_id, 'pipeline_state')
                if checkpoint:
                    logger.info(f"Found pipeline checkpoint - attempting to restore state")
                    
                    data_file = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_pipeline_state_data.json"
                    if data_file.exists():
                        try:
                            import pickle
                            with open(data_file, 'rb') as f:
                                resume_data = pickle.load(f)
                            
                            logger.info(
                                f"Restored pipeline state: "
                                f"{len(resume_data.get('texts', []))} texts, "
                                f"{len(resume_data.get('embeddings', []))} embeddings, "
                                f"{len(resume_data.get('json_chunks', []))} JSON chunks"
                            )
                        except Exception as e:
                            logger.error(f"Failed to load pipeline state: {e}")
                            resume_data = None
            
            if resume_data:
                logger.info("RESUMING PIPELINE FROM CHECKPOINT")
                
                texts = resume_data.get('texts', [])
                embeddings = resume_data.get('embeddings', [])
                metadatas = resume_data.get('metadatas', [])
                json_chunks = resume_data.get('json_chunks', [])
                json_embeddings = resume_data.get('json_embeddings', [])
                json_entities = resume_data.get('json_entities', [])
                
                self.stats = resume_data.get('stats', self.stats)
                
                logger.info(
                    f"Resume summary - Documents: {self.stats['total_documents']}, "
                    f"Chunks: {self.stats['total_chunks']}, "
                    f"Embeddings: {len(embeddings)}, "
                    f"Remaining: {len(texts) - len(embeddings)}, "
                    f"JSON chunks: {len(json_chunks)}"
                )
                
                if len(texts) > len(embeddings):
                    logger.info("RESUMING: CONTINUE EMBEDDING")
                    
                    remaining_texts = texts[len(embeddings):]
                    logger.info(f"Generating embeddings for {len(remaining_texts)} remaining texts...")
                    
                    remaining_embeddings = self.embedding_service.generate_for_documents(
                        texts=remaining_texts,
                        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
                        checkpoint_manager=self.checkpoint_manager,
                        job_id=self.job_id,
                        cancellation_event=self.cancellation_event
                    )
                    
                    embeddings.extend(remaining_embeddings)
                    logger.info(f"Total embeddings: {len(embeddings)}/{len(texts)}")
                    
                    self.stats['total_embeddings'] = len(embeddings)
                
                logger.info("RESUMING: VECTOR STORAGE")
                
                total_upserted = 0
                
                if texts and embeddings:
                    print(f"\nStoring {len(texts)} document vectors...")
                    namespace = self.company_namespace
                    
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
                    print(f"Stored {upserted} vectors")
                
                if json_chunks and json_embeddings:
                    print(f"\nStoring {len(json_chunks)} JSON vectors...")
                    namespace = self.company_namespace
                    
                    with tqdm(total=1, desc="Preparing JSON vectors", leave=False) as pbar:
                        json_vectors = self.vector_store.prepare_json_vectors(
                            json_chunks, json_embeddings, json_entities, namespace
                        )
                        pbar.update(1)

                    upserted = self.vector_store.upsert_json_vectors(
                        json_vectors, namespace, 100,
                        checkpoint_manager=self.checkpoint_manager,
                        job_id=self.job_id
                    )
                    total_upserted += upserted
                    print(f"Stored {upserted} JSON vectors")
                
                self.stats["vectors_upserted"] = total_upserted
                
                print("\nVerifying upload...")
                self.vector_store.verify_upsert(self.company_namespace, total_upserted)
                
                end_time = datetime.now()
                self.stats["processing_time"] = (end_time - start_time).total_seconds()
                
                print(f"\nFinal statistics (resumed):")
                print(f"Total documents: {self.stats['total_documents']}")
                print(f"Vectors stored: {self.stats['vectors_upserted']}")
                print(f"Processing time: {self.stats['processing_time']:.2f}s")
                
                if self.checkpoint_manager and self.job_id:
                    self.checkpoint_manager.clear_checkpoint(self.job_id)
                    
                    data_file = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_pipeline_state_data.json"
                    if data_file.exists():
                        data_file.unlink()
                    
                    logger.info(f"Cleared all checkpoints for job {self.job_id}")
                
                return self.stats
            
            processed_documents = []
            website_docs = []
            texts = []
            embeddings = []
            metadatas = []
            json_chunks = []
            json_embeddings = []
            json_entities = []

            print(f"\nSequential data ingestion: {self.company_namespace.upper()}")
            print("Phase 1: Loading and processing standard documents")

            if job_manager and company_name and self.website_urls:
                print("\nPhase 1A: Website deduplication")
                website_docs = self.load_and_deduplicate_websites(job_manager, company_name)
            
            db_docs = []
            if job_manager and company_name and self.db_uris:
                print("\nPhase 1B: Database table-level processing")
                db_docs = self.load_and_deduplicate_databases(job_manager, company_name, self.db_uris)
            
            processed_documents = self.load_and_process_standard_documents()
            
            if db_docs:
                processed_documents.extend(db_docs)
                self.stats["total_documents"] += len(db_docs)
            
            if website_docs:
                processed_documents.extend(website_docs)
                self.stats["total_documents"] += len(website_docs)

            if processed_documents:
                texts, embeddings, metadatas = self.chunk_and_embed_documents(
                    processed_documents
                )
                
                if self.checkpoint_manager and self.job_id and texts and embeddings:
                    logger.info("Saving pipeline state checkpoint...")
                    
                    self.checkpoint_manager.save_checkpoint(
                        job_id=self.job_id,
                        company_name=self.company_namespace,
                        phase='pipeline_state',
                        checkpoint_data={
                            'total_documents': self.stats['total_documents'],
                            'total_chunks': self.stats['total_chunks'],
                            'total_embeddings': len(embeddings),
                            'has_json': False,
                            'timestamp': time.time()
                        }
                    )
                    
                    import pickle
                    data_file = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_pipeline_state_data.json"
                    with open(data_file, 'wb') as f:
                        pickle.dump({
                            'texts': texts,
                            'embeddings': embeddings,
                            'metadatas': metadatas,
                            'json_chunks': [],
                            'json_embeddings': [],
                            'json_entities': [],
                            'stats': self.stats
                        }, f)
                    
                    logger.info(f"Pipeline state saved ({len(texts)} texts, {len(embeddings)} embeddings)")

                if texts and embeddings:
                    print("\nStoring standard document vectors")
                    
                    namespace = self.company_namespace
                    
                    print(f"\nStoring {len(texts)} document vectors...")
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
                    self.stats["vectors_upserted"] += upserted
                    print(f"Stored {upserted} standard document vectors")

                    self.vector_store.verify_upsert(namespace, upserted)
            else:
                print("\nNo standard documents to process")

            print("\nPhase 2: Processing JSON files")
            
            json_chunks, json_entities = self.process_json_files()

            if json_chunks:
                print(f"\nGenerating embeddings for {len(json_chunks)} JSON chunks...")
                json_embeddings = self.embedding_service.generate_for_json_chunks(
                    json_chunks,
                    checkpoint_manager=self.checkpoint_manager,
                    job_id=self.job_id,
                    cancellation_event=self.cancellation_event
                )
                print(f"Generated {len(json_embeddings)} JSON embeddings")
                
                if self.checkpoint_manager and self.job_id:
                    logger.info("Updating pipeline state with JSON data...")
                    
                    self.checkpoint_manager.save_checkpoint(
                        job_id=self.job_id,
                        company_name=self.company_namespace,
                        phase='pipeline_state',
                        checkpoint_data={
                            'total_documents': self.stats['total_documents'],
                            'total_chunks': self.stats['total_chunks'],
                            'total_embeddings': len(embeddings) if embeddings else 0,
                            'json_chunks': len(json_chunks),
                            'has_json': True,
                            'timestamp': time.time()
                        }
                    )
                    
                    import pickle
                    data_file = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_pipeline_state_data.json"
                    with open(data_file, 'wb') as f:
                        pickle.dump({
                            'texts': texts,
                            'embeddings': embeddings,
                            'metadatas': metadatas,
                            'json_chunks': json_chunks,
                            'json_embeddings': json_embeddings,
                            'json_entities': json_entities,
                            'stats': self.stats
                        }, f)
                    
                    logger.info(f"Pipeline state updated with JSON data")
                
                print("\nStoring JSON vectors")
                
                namespace = self.company_namespace
                
                print(f"\nStoring {len(json_chunks)} JSON vectors...")
                with tqdm(total=1, desc="Preparing JSON vectors", leave=False) as pbar:
                    json_vectors = self.vector_store.prepare_json_vectors(
                        json_chunks, json_embeddings, json_entities, namespace
                    )
                    pbar.update(1)

                upserted = self.vector_store.upsert_json_vectors(
                    json_vectors, namespace, 100,
                    checkpoint_manager=self.checkpoint_manager,
                    job_id=self.job_id
                )
                self.stats["vectors_upserted"] += upserted
                print(f"Stored {upserted} JSON vectors")
                
                print(f"\nVerifying JSON upload...")
                self.vector_store.verify_upsert(namespace, self.stats["vectors_upserted"])
            else:
                print("\nNo JSON files to process")

            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()

            print("\nFinal statistics:")
            print(f"Total documents: {self.stats['total_documents']}")
            print(f"- Files: {self.stats['file_documents']}")
            print(f"- Databases: {self.stats['database_documents']}")
            print(f"- Websites: {self.stats['website_documents']}")
            print(f"- JSON: {self.stats['json_documents']}")
            print(f"\nProcessed: {self.stats['processed_documents']}")
            print(f"Failed: {self.stats['failed_documents']}")
            print(f"Total chunks: {self.stats['total_chunks']}")
            print(f"Total embeddings: {self.stats['total_embeddings'] + self.stats['json_chunks']}")
            print(f"Vectors stored: {self.stats['vectors_upserted']}")
            print(f"Processing time: {self.stats['processing_time']:.2f}s")

            if self.checkpoint_manager and self.job_id:
                if self.stats.get('vectors_upserted', 0) > 0:
                    self.checkpoint_manager.clear_checkpoint(self.job_id)
                    
                    data_file = self.checkpoint_manager.checkpoint_dir / f"{self.job_id}_pipeline_state_data.json"
                    if data_file.exists():
                        data_file.unlink()
                    
                    logger.info(f"Cleared checkpoints after successful completion")
                else:
                    logger.info(f"Checkpoints preserved for resume (job incomplete)")
                
            try:
                import shutil
                if self.directory_path and os.path.exists(self.directory_path):
                    shutil.rmtree(self.directory_path)
                    logger.info(f"Cleaned up ingestion directory: {self.directory_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup ingestion directory: {e}")

            return self.stats
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            if self.checkpoint_manager and self.job_id:
                logger.info(
                    f"Checkpoints saved for job {self.job_id}. "
                    f"Resume by rerunning with same data"
                )
            
            raise