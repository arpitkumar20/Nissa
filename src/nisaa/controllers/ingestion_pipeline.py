from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from src.nisaa.helpers.logger import logger
from src.nisaa.services.embedding_service import EmbeddingService
from src.nisaa.services.json_processor import JSONProcessor
from src.nisaa.services.text_extract import TextPreprocessor
from src.nisaa.services.vector_store_service import VectorStoreService
from src.nisaa.services.website_scrap import WebsiteIngester
from src.nisaa.services.document_loader import  DocumentLoader
from src.nisaa.services.sql_database import SQLDatabaseIngester


class DataIngestionPipeline:
    """
    Unified data ingestion pipeline supporting files, databases, websites, and JSON
    with parallel processing and vector storage
    """
    
    def __init__(
        self,
        company_namespace: str,
        directory_path: Optional[str] = None,
        db_uris: Optional[List[str]] = None,
        website_urls: Optional[List[str]] = None,
        preprocess_config: Optional[Dict[str, bool]] = None,
        proxies: Optional[dict] = None,
    ):
        """
        Initialize data ingestion pipeline
        
        Args:
            company_namespace: Company identifier (used as Pinecone namespace)
            directory_path: Path to data directory (optional)
            db_uris: List of database URIs (optional)
            website_urls: List of website URLs (optional)
            preprocess_config: Text preprocessing configuration
            proxies: Proxy configuration for web scraping
        """
        self.company_namespace = company_namespace
        self.directory_path = directory_path
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
            os.getenv('OPENAI_API_KEY'),
            os.getenv('OPENAI_MODEL'),
            int(os.getenv('MAX_WORKERS', '10'))
        )
        
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
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
    
    def enrich_metadata(self, doc: Document, urls: List[Dict[str, str]]) -> Document:
        """Enrich document metadata with additional information"""
        source_type = doc.metadata.get("source_type", "file")
        
        if source_type == "file":
            source_path = doc.metadata.get("source", "")
            file_name = os.path.basename(source_path)
            file_extension = os.path.splitext(file_name)[1]
            
            doc.metadata.update({
                "file_name": file_name,
                "file_type": file_extension,
                "file_size": (
                    os.path.getsize(source_path)
                    if os.path.exists(source_path)
                    else 0
                ),
            })
        
        doc.metadata.update({
            "company_namespace": self.company_namespace,
            "processed_at": datetime.now().isoformat(),
            "char_count": len(doc.page_content),
            "word_count": len(doc.page_content.split()),
            "pipeline_version": "4.0.0",
        })
        
        # URL metadata
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
            # Skip database documents from preprocessing
            if doc.metadata.get("source_type") == "database":
                doc.metadata["company_namespace"] = self.company_namespace
                doc.metadata["processed_at"] = datetime.now().isoformat()
                doc.metadata["char_count"] = len(doc.page_content)
                doc.metadata["word_count"] = len(doc.page_content.split())
                return doc
            
            original_content = doc.page_content
            processed_content, urls = self.preprocessor.preprocess(original_content)
            
            if not processed_content or len(processed_content.strip()) < 10:
                logger.warning(f"Document has insufficient content after preprocessing")
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
        """Load and process files, databases, and websites in parallel"""
        raw_documents = []
        
        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            # Load files
            if self.document_loader:
                logger.info("📁 Loading files (excluding JSON)...")
                futures[executor.submit(self.document_loader.load_all_documents, True)] = "files"
            
            # Load databases
            if self.sql_ingester and self.db_uris:
                logger.info("🗄️ Loading databases...")
                futures[executor.submit(self.sql_ingester.ingest_multiple_databases, self.db_uris)] = "databases"
            
            # Load websites
            if self.website_ingester and self.website_urls:
                logger.info("🌐 Loading websites...")
                futures[executor.submit(self.website_ingester.ingest_multiple_websites, self.website_urls)] = "websites"
            
            # Collect results
            for future in as_completed(futures):
                source = futures[future]
                try:
                    docs = future.result()
                    raw_documents.extend(docs)
                    
                    if source == "files":
                        self.stats["file_documents"] = len(docs)
                    elif source == "databases":
                        self.stats["database_documents"] = len(docs)
                    elif source == "websites":
                        self.stats["website_documents"] = len(docs)
                    
                    logger.info(f"✅ {source.capitalize()}: {len(docs)} documents loaded")
                except Exception as e:
                    logger.error(f"❌ Failed to load {source}: {e}")
        
        self.stats["total_documents"] = len(raw_documents)
        
        # Process documents in parallel
        if raw_documents:
            logger.info(f"🔄 Processing {len(raw_documents)} documents...")
            processed_documents = []
            
            with ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', '10'))) as executor:
                futures = {executor.submit(self.process_document, doc): doc for doc in raw_documents}
                
                for future in as_completed(futures):
                    try:
                        processed_doc = future.result()
                        if processed_doc:
                            processed_documents.append(processed_doc)
                            self.stats["processed_documents"] += 1
                        else:
                            self.stats["failed_documents"] += 1
                    except Exception as e:
                        logger.error(f"Document processing failed: {e}")
                        self.stats["failed_documents"] += 1
            
            logger.info(f"✅ Processed {len(processed_documents)} documents")
            return processed_documents
        
        return []
    
    def process_json_files(self) -> tuple:
        """Process JSON files separately and return chunks with entities"""
        if not self.document_loader:
            return [], []
        
        json_files = self.document_loader.get_json_files()
        
        if not json_files:
            logger.info("ℹ️ No JSON files found")
            return [], []
        
        logger.info(f"📄 Processing {len(json_files)} JSON files...")
        
        all_chunks = []
        all_entities = []
        
        for json_file in json_files:
            try:
                chunks, entities = self.json_processor.process_json_file(json_file)
                all_chunks.extend(chunks)
                all_entities.extend(entities)
                self.stats["json_documents"] += 1
            except Exception as e:
                logger.error(f"Failed to process JSON file {json_file}: {e}")
        
        self.stats["json_chunks"] = len(all_chunks)
        logger.info(f"✅ Processed {len(json_files)} JSON files into {len(all_chunks)} chunks")
        
        return all_chunks, all_entities
    
    def chunk_and_embed_documents(self, documents: List[Document]) -> tuple:
        """Chunk documents and generate embeddings"""
        if not documents:
            return [], [], []
        
        logger.info(f"✂️ Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        self.stats["total_chunks"] = len(chunks)
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Extract texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_for_documents(
            texts,
            int(os.getenv('EMBEDDING_BATCH_SIZE', '100')),
            int(os.getenv('MAX_WORKERS', '10'))
        )
        
        self.stats["total_embeddings"] = len(embeddings)
        
        return texts, embeddings, metadatas
    
    def store_to_pinecone(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        json_chunks: List[str] = None,
        json_embeddings: List[List[float]] = None,
        json_entities: List[tuple] = None
    ):
        """Store all vectors to Pinecone"""
        namespace = self.company_namespace
        total_upserted = 0
        
        # Store standard documents
        if texts and embeddings:
            logger.info(f"💾 Storing {len(texts)} document vectors to Pinecone...")
            ids, vectors, processed_metadatas = self.vector_store.prepare_document_vectors(
                texts, embeddings, metadatas, namespace
            )
            
            upserted = self.vector_store.upsert_vectors(
                ids, vectors, processed_metadatas, namespace, int(os.getenv('PINECONE_BATCH_SIZE', '100'))
            )
            total_upserted += upserted
        
        # Store JSON chunks
        if json_chunks and json_embeddings:
            logger.info(f"💾 Storing {len(json_chunks)} JSON vectors to Pinecone...")
            json_vectors = self.vector_store.prepare_json_vectors(
                json_chunks, json_embeddings, json_entities, namespace
            )
            
            upserted = self.vector_store.upsert_json_vectors(
                json_vectors, namespace, 100
            )
            total_upserted += upserted
        
        self.stats["vectors_upserted"] = total_upserted
        
        # Verify
        self.vector_store.verify_upsert(namespace, total_upserted)
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete data ingestion pipeline"""
        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info(f"🚀 DATA INGESTION PIPELINE - {self.company_namespace.upper()}")
        logger.info("=" * 80)
        
        # Phase 1: Load and process standard documents
        logger.info("\n📦 PHASE 1: Loading Standard Documents (Files, Databases, Websites)")
        processed_documents = self.load_and_process_standard_documents()
        
        # Phase 2: Process JSON files
        logger.info("\n📄 PHASE 2: Processing JSON Files")
        json_chunks, json_entities = self.process_json_files()
        
        # Phase 3: Chunk and embed standard documents
        logger.info("\n✂️ PHASE 3: Chunking and Embedding Standard Documents")
        texts, embeddings, metadatas = self.chunk_and_embed_documents(processed_documents)
        
        # Phase 4: Embed JSON chunks
        json_embeddings = []
        if json_chunks:
            logger.info("\n🔢 PHASE 4: Embedding JSON Chunks")
            json_embeddings = self.embedding_service.generate_for_json_chunks(json_chunks)
        
        # Phase 5: Store to Pinecone
        logger.info("\n💾 PHASE 5: Storing to Pinecone Vector Database")
        self.store_to_pinecone(
            texts, embeddings, metadatas,
            json_chunks, json_embeddings, json_entities
        )
        
        # Calculate processing time
        end_time = datetime.now()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()
        
        # Log statistics
        self.log_statistics()
        
        return self.stats
    
    def log_statistics(self):
        """Log pipeline statistics"""
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 PIPELINE STATISTICS - {self.company_namespace.upper()}")
        logger.info("=" * 80)
        logger.info(f"Company Namespace      : {self.stats['company']}")
        logger.info(f"Total Documents Loaded : {self.stats['total_documents']}")
        logger.info(f"  ├─ From Files        : {self.stats['file_documents']}")
        logger.info(f"  ├─ From Databases    : {self.stats['database_documents']}")
        logger.info(f"  ├─ From Websites     : {self.stats['website_documents']}")
        logger.info(f"  └─ JSON Files        : {self.stats['json_documents']}")
        logger.info(f"Successfully Processed : {self.stats['processed_documents']}")
        logger.info(f"Failed/Skipped         : {self.stats['failed_documents']}")
        logger.info(f"Total Chunks Created   : {self.stats['total_chunks']}")
        logger.info(f"JSON Chunks Created    : {self.stats['json_chunks']}")
        logger.info(f"Total Embeddings       : {self.stats['total_embeddings'] + self.stats['json_chunks']}")
        logger.info(f"Vectors Upserted       : {self.stats['vectors_upserted']}")
        logger.info(f"Processing Time        : {self.stats['processing_time']:.2f} seconds")
        logger.info("=" * 80)
        logger.info(f"✅ Knowledge base for '{self.company_namespace}' is ready!")
        logger.info("=" * 80 + "\n")