"""
Embedding generation service with SEQUENTIAL processing and rate limit handling
"""
import os
import time
from typing import List
from openai import OpenAI, RateLimitError
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
            show_progress_bar=False,
            max_retries=0
        )
        self.openai_client = OpenAI(
            api_key=self.api_key,
            max_retries=0
        )
        
        self.min_batch_delay = float(os.getenv('MIN_BATCH_DELAY', '2.0'))
        self.last_request_time = 0
    
    def _rate_limit_delay(self):
        """Enforce minimum delay between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_batch_delay:
            sleep_time = self.min_batch_delay - time_since_last
            logger.info(f"   ⏳ Rate limit delay: {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 5,
        initial_delay: float = 5.0,
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
                
                error_msg = str(e)
                wait_time = delay
                
                if "Please try again in" in error_msg:
                    try:
                        import re
                        match = re.search(r'try again in (\d+\.?\d*)m?s', error_msg)
                        if match:
                            extracted_time = float(match.group(1))

                            if 'ms' in error_msg:
                                extracted_time = extracted_time / 1000
                            wait_time = max(extracted_time, delay)
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
        Generate embeddings for document texts using SEQUENTIAL processing
        with rate limit handling
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (default: 20)
            max_workers: DEPRECATED - Sequential processing only
            
        Returns:
            List of embedding vectors
        """

        batch_size = batch_size or int(os.getenv('EMBEDDING_BATCH_SIZE', '20'))

        logger.info(f"Generating embeddings for {len(texts)} texts -- Using model: {self.model} -- Batch size: {batch_size} (SEQUENTIAL processing)")
        
        all_embeddings = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        for batch_idx, batch in enumerate(batches):
            self._rate_limit_delay()
            
            def embed_batch():
                vectors = self.langchain_embeddings.embed_documents(batch)
                return vectors
            
            try:
                vectors = self._retry_with_exponential_backoff(embed_batch)
                all_embeddings.extend(vectors)
                
                logger.info(
                    f"Batch {batch_idx + 1}/{len(batches)} completed "
                    f"({len(all_embeddings)}/{len(texts)} embeddings)"
                )
                
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                raise
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def generate_for_json_chunks(
        self,
        chunks: List[str],
        batch_size: int = 20
    ) -> List[List[float]]:
        """
        Generate embeddings for JSON chunks using OpenAI API directly
        with SEQUENTIAL processing and rate limit handling
        
        Args:
            chunks: List of text chunks
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(chunks)} JSON chunks...")
        
        embeddings = []
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            self._rate_limit_delay()
            
            def embed_json_batch():
                response = self.openai_client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                return [item.embedding for item in response.data]
            
            try:
                batch_embeddings = self._retry_with_exponential_backoff(embed_json_batch)
                embeddings.extend(batch_embeddings)
                
                logger.info(
                    f"Batch {batch_num}/{num_batches} completed "
                    f"({len(embeddings)}/{len(chunks)} embeddings)"
                )
                
            except Exception as e:
                logger.error(f"✗ Error generating embeddings for batch {batch_num}: {e}")
                embeddings.extend([[0.0] * 1536] * len(batch))
        
        logger.info(f"Generated {len(embeddings)} JSON embeddings")
        return embeddings