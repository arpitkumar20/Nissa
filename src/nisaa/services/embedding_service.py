"""
Embedding generation service with parallel processing
"""
import os
from typing import List
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.nisaa.helpers.logger import logger

class EmbeddingService:
    """Handles embedding generation for both regular documents and JSON chunks"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('EMBEDDING_MODEL')
        self.langchain_embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
            show_progress_bar=False
        )
        self.openai_client = OpenAI(api_key=self.api_key)
    
    def generate_for_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        max_workers: int = None
    ) -> List[List[float]]:
        """
        Generate embeddings for document texts using parallel processing
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            max_workers: Number of parallel workers
            
        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
        max_workers = max_workers or int(os.getenv('MAX_WORKERS', '10'))
        
        logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
        logger.info(f"   Using model: {self.model}")
        logger.info(f"   Batch size: {batch_size}, Workers: {max_workers}")
        
        all_embeddings = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        def process_batch(batch_data):
            batch_idx, batch = batch_data
            try:
                vectors = self.langchain_embeddings.embed_documents(batch)
                logger.info(f"   ✓ Batch {batch_idx + 1}/{len(batches)} completed")
                return batch_idx, vectors
            except Exception as e:
                logger.error(f"   ✗ Batch {batch_idx + 1} failed: {e}")
                raise
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_batch, (idx, batch)): idx
                for idx, batch in enumerate(batches)
            }
            
            results = [None] * len(batches)
            for future in as_completed(futures):
                batch_idx, vectors = future.result()
                results[batch_idx] = vectors
        
        for result in results:
            all_embeddings.extend(result)
        
        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def generate_for_json_chunks(
        self,
        chunks: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for JSON chunks using OpenAI API directly
        
        Args:
            chunks: List of text chunks
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"🔢 Generating embeddings for {len(chunks)} JSON chunks...")
        
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"   ✓ Generated {len(embeddings)}/{len(chunks)} embeddings")
                
            except Exception as e:
                logger.error(f"✗ Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                embeddings.extend([[0.0] * 1536] * len(batch))
        
        logger.info(f"✅ Generated {len(embeddings)} JSON embeddings")
        return embeddings