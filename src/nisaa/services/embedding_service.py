"""
Embedding generation service with parallel processing and rate limit handling
"""
import os
import time
from typing import List
from openai import OpenAI, RateLimitError
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
    
    def _retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        """
        Retry function with exponential backoff for rate limit errors
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for delay on each retry
        """
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                    raise
                
                # Extract wait time from error message if available
                error_msg = str(e)
                wait_time = delay
                
                if "Please try again in" in error_msg:
                    try:
                        # Try to extract wait time from error message
                        import re
                        match = re.search(r'try again in (\d+)ms', error_msg)
                        if match:
                            wait_time = max(float(match.group(1)) / 1000, delay)
                    except:
                        pass
                
                logger.warning(
                    f"Rate limit hit. Retry {attempt + 1}/{max_retries} "
                    f"after {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                delay *= backoff_factor
            except Exception as e:
                logger.error(f"Non-retryable error: {e}")
                raise
    
    def generate_for_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        max_workers: int = None
    ) -> List[List[float]]:
        """
        Generate embeddings for document texts using parallel processing
        with rate limit handling
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (default: 25)
            max_workers: Number of parallel workers (default: 3)
            
        Returns:
            List of embedding vectors
        """
        # CRITICAL: Reduce these values to stay under rate limits
        batch_size = batch_size or int(os.getenv('EMBEDDING_BATCH_SIZE', '25'))  # Reduced from 50
        max_workers = max_workers or int(os.getenv('MAX_WORKERS', '3'))  # Reduced from 5
        
        logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
        logger.info(f"   Using model: {self.model}")
        logger.info(f"   Batch size: {batch_size}, Workers: {max_workers}")
        
        all_embeddings = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        def process_batch(batch_data):
            batch_idx, batch = batch_data
            
            def embed_batch():
                vectors = self.langchain_embeddings.embed_documents(batch)
                logger.info(f"   ✓ Batch {batch_idx + 1}/{len(batches)} completed")
                return vectors
            
            try:
                # Retry with exponential backoff on rate limit errors
                vectors = self._retry_with_exponential_backoff(embed_batch)
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
        batch_size: int = 50  # Keep smaller batches for JSON
    ) -> List[List[float]]:
        """
        Generate embeddings for JSON chunks using OpenAI API directly
        with rate limit handling
        
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
            
            def embed_json_batch():
                response = self.openai_client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                return [item.embedding for item in response.data]
            
            try:
                # Retry with exponential backoff
                batch_embeddings = self._retry_with_exponential_backoff(embed_json_batch)
                embeddings.extend(batch_embeddings)
                
                logger.info(f"   ✓ Generated {len(embeddings)}/{len(chunks)} embeddings")
                
            except Exception as e:
                logger.error(f"✗ Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                embeddings.extend([[0.0] * 1536] * len(batch))
        
        logger.info(f"✅ Generated {len(embeddings)} JSON embeddings")
        return embeddings