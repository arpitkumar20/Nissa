from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from tqdm import tqdm

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
        db_uris: Optional[List[str]] = None,
        website_urls: Optional[List[str]] = None,
        preprocess_config: Optional[Dict[str, bool]] = None,
        proxies: Optional[dict] = None,
    ):
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
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            int(os.getenv("MAX_WORKERS", "10")),
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
        """OPTIMIZED: Load and process with progress bars"""
        raw_documents = []

        print("\n🔄 Loading source documents...")

        # Load files
        if self.document_loader:
            with tqdm(total=1, desc="📁 Files", leave=False) as pbar:
                try:
                    docs = self.document_loader.load_all_documents(True)
                    raw_documents.extend(docs)
                    self.stats["file_documents"] = len(docs)
                    pbar.set_postfix_str(f"{len(docs)} docs")
                except Exception as e:
                    logger.error(f"File loading failed: {e}")
                pbar.update(1)

        # Load databases
        if self.sql_ingester and self.db_uris:
            with tqdm(total=1, desc="🗄️ Databases", leave=False) as pbar:
                try:
                    docs = self.sql_ingester.ingest_multiple_databases(self.db_uris)
                    raw_documents.extend(docs)
                    self.stats["database_documents"] = len(docs)
                    pbar.set_postfix_str(f"{len(docs)} docs")
                except Exception as e:
                    logger.error(f"Database loading failed: {e}")
                pbar.update(1)

        # Load websites
        if self.website_ingester and self.website_urls:
            with tqdm(total=1, desc="🌐 Websites", leave=False) as pbar:
                try:
                    docs = self.website_ingester.ingest_multiple_websites(
                        self.website_urls
                    )
                    raw_documents.extend(docs)
                    self.stats["website_documents"] = len(docs)
                    pbar.set_postfix_str(f"{len(docs)} docs")
                except Exception as e:
                    logger.error(f"Website loading failed: {e}")
                pbar.update(1)

        self.stats["total_documents"] = len(raw_documents)
        print(f"✅ Loaded {len(raw_documents)} documents")

        # Process documents
        if raw_documents:
            print(f"\n🔄 Processing {len(raw_documents)} documents...")
            processed_documents = []

            with tqdm(total=len(raw_documents), desc="Processing", unit="doc") as pbar:
                with ThreadPoolExecutor(
                    max_workers=int(os.getenv("MAX_WORKERS", "10"))
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

    def process_json_files(self) -> tuple:
        """OPTIMIZED: Process JSON files with progress"""
        if not self.document_loader:
            return [], []

        json_files = self.document_loader.get_json_files()

        if not json_files:
            return [], []

        print(f"\n🔄 Processing {len(json_files)} JSON files...")

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

        print(f"\n🔄 Chunking {len(documents)} documents...")

        with tqdm(total=1, desc="Chunking", leave=False) as pbar:
            chunks = self.text_splitter.split_documents(documents)
            self.stats["total_chunks"] = len(chunks)
            pbar.update(1)

        print(f"✅ Created {len(chunks)} chunks")

        # Extract data
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings
        print(f"\n🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_service.generate_for_documents(
            texts,
            int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            int(os.getenv("MAX_WORKERS", "10")),
        )

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
            print(f"\n🔄 Storing {len(texts)} document vectors...")

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
            print(f"\n🔄 Storing {len(json_chunks)} JSON vectors...")

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
        print("\n🔄 Verifying upload...")
        self.vector_store.verify_upsert(namespace, total_upserted)

    def run(self) -> Dict[str, Any]:
        """OPTIMIZED: Execute pipeline with clean progress"""
        start_time = datetime.now()

        # Phase 1: Load standard documents
        processed_documents = self.load_and_process_standard_documents()

        # Phase 2: Process JSON files
        json_chunks, json_entities = self.process_json_files()

        # Phase 3: Chunk and embed standard documents
        texts, embeddings, metadatas = self.chunk_and_embed_documents(
            processed_documents
        )

        # Phase 4: Embed JSON chunks
        json_embeddings = []
        if json_chunks:
            print(f"\n🔄 Generating embeddings for {len(json_chunks)} JSON chunks...")
            json_embeddings = self.embedding_service.generate_for_json_chunks(
                json_chunks
            )
            print(f"✅ Generated {len(json_embeddings)} JSON embeddings")

        # Phase 5: Store to Pinecone
        self.store_to_pinecone(
            texts, embeddings, metadatas, json_chunks, json_embeddings, json_entities
        )

        # Calculate time
        end_time = datetime.now()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()

        return self.stats
