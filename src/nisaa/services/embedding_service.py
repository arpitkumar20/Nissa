"""
Embedding generation service with ROBUST checkpoint recovery
Key fixes:
1. Proper index alignment when resuming
2. Better error handling for connection failures
3. Checkpoint saves even on connection errors
4. Graceful cancellation handling
"""
import os
import time
import hashlib
from typing import List, Optional
from openai import OpenAI, RateLimitError, APIConnectionError
from langchain_openai import OpenAIEmbeddings
from src.nisaa.config.logger import logger
import json


class EmbeddingService:
    """Handles embedding generation with checkpoint/recovery support"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('EMBEDDING_MODEL')
        self.langchain_embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
            show_progress_bar=False,
            max_retries=2  
        )
        self.openai_client = OpenAI(
            api_key=self.api_key,
            max_retries=2,
            timeout=60.0  
        )
        
        self.min_batch_delay = float(os.getenv('MIN_BATCH_DELAY', '2.0'))
        self.last_request_time = 0
    
    def _rate_limit_delay(self):
        """Enforce minimum delay between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_batch_delay:
            sleep_time = self.min_batch_delay - time_since_last
            logger.info(f"Rate limit delay: {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 3,  # Reduced from 5
        initial_delay: float = 5.0,
        backoff_factor: float = 2.0
    ):
        """Retry function with exponential backoff for rate limit errors"""
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except (RateLimitError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                    raise
                
                error_msg = str(e)
                wait_time = delay
                
                if isinstance(e, APIConnectionError):
                    logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}")
                    wait_time = delay * 2  
                elif "Please try again in" in error_msg:
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
                    f"API error. Retry {attempt + 1}/{max_retries} "
                    f"after {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                delay *= backoff_factor
            except Exception as e:
                logger.error(f"Non-retryable error: {e}")
                raise

    def _save_checkpoint_safe(
        self,
        checkpoint_manager,
        job_id: str,
        company_name: str,
        phase: str,
        checkpoint_data: dict,
        embeddings: List[List[float]]
    ):
        """
        Safely save checkpoint with multiple fallback methods
        CRITICAL: This must never fail completely
        """
        try:
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}.json"
            data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
            
            checkpoint_data['timestamp'] = time.time()
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            with open(data_file, 'w') as f:
                json.dump({'embeddings': embeddings}, f)
            
            logger.info(
                f"✓ Checkpoint saved to file: batch {checkpoint_data.get('last_batch_index', 0) + 1}, "
                f"{checkpoint_data.get('embeddings_count', 0)} embeddings"
            )
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to save checkpoint to file: {e}")
 
        try:
            checkpoint_manager.save_checkpoint(
                job_id=job_id,
                company_name=company_name,
                phase=phase,
                checkpoint_data=checkpoint_data
            )
        except Exception as e:
            logger.warning(f"DB checkpoint save failed (file checkpoint is safe): {e}")

    def generate_for_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        checkpoint_manager = None,
        job_id: str = None,
        cancellation_event = None
    ) -> List[List[float]]:
        """Generate embeddings with ROBUST checkpoint recovery"""
        batch_size = batch_size or int(os.getenv('EMBEDDING_BATCH_SIZE', '10'))
        phase = 'embedding_documents'
        
        all_embeddings = []
        start_index = 0
        
        if checkpoint_manager and job_id:
            checkpoint = checkpoint_manager.load_checkpoint(job_id, phase)
            if checkpoint:
                data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
                if data_file.exists():
                    try:
                        with open(data_file, 'r') as f:
                            saved_data = json.load(f)
                            all_embeddings = saved_data.get('embeddings', [])
                        
                        start_index = len(all_embeddings)
                        
                        logger.info(
                            f"Resuming: {len(all_embeddings)} embeddings restored, "
                            f"{len(texts) - start_index} texts remaining"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint data: {e}")
                        all_embeddings = []
                        start_index = 0
        
        texts_remaining = texts[start_index:]  
        
        if not texts_remaining:
            logger.info("✓ All texts already embedded, returning cached embeddings")
            return all_embeddings
        
        logger.info(
            f"Generating embeddings for {len(texts_remaining)}/{len(texts)} texts -- "
            f"Model: {self.model} -- Batch size: {batch_size}"
        )
        
        batches = [texts_remaining[i:i + batch_size] 
                for i in range(0, len(texts_remaining), batch_size)]
        total_batches = len(batches)
        
        for batch_idx, batch in enumerate(batches):
            if cancellation_event and cancellation_event.is_set():
                logger.warning(
                    f"Embedding cancelled at batch {batch_idx + 1}/{total_batches}. "
                    f"Returning {len(all_embeddings)} embeddings."
                )
                return all_embeddings
            
            self._rate_limit_delay()
            
            def embed_batch():
                return self.langchain_embeddings.embed_documents(batch)
            
            try:
                vectors = self._retry_with_exponential_backoff(embed_batch)
                all_embeddings.extend(vectors)
                
                if checkpoint_manager and job_id:
                    checkpoint_data = {
                        'embeddings_count': len(all_embeddings),  
                        'last_batch_index': batch_idx,
                        'total_batches': total_batches,
                        'texts_processed': start_index + (batch_idx + 1) * batch_size
                    }
                    
                    self._save_checkpoint_safe(
                        checkpoint_manager,
                        job_id,
                        checkpoint_manager.company_name,
                        phase,
                        checkpoint_data,
                        all_embeddings
                    )
                
                logger.info(
                    f"✓ Batch {batch_idx + 1}/{total_batches} completed "
                    f"({len(all_embeddings)}/{len(texts)} total embeddings)"
                )
                
            except Exception as e:
                logger.error(f" Batch {batch_idx + 1} failed: {e}")
                
                if checkpoint_manager and job_id:
                    checkpoint_data = {
                        'embeddings_count': len(all_embeddings),
                        'last_batch_index': batch_idx - 1,
                        'total_batches': total_batches,
                        'error': str(e)[:500]
                    }
                    
                    self._save_checkpoint_safe(
                        checkpoint_manager,
                        job_id,
                        checkpoint_manager.company_name,
                        phase,
                        checkpoint_data,
                        all_embeddings
                    )
                    
                    logger.info(f"✓ Checkpoint saved at {len(all_embeddings)} embeddings")
                
                raise
        
        if checkpoint_manager and job_id and len(all_embeddings) == len(texts):
            try:
                checkpoint_manager.clear_checkpoint(job_id, phase)
                data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
                if data_file.exists():
                    data_file.unlink()
                logger.info("✓ Checkpoint cleared after successful completion")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")
        
        logger.info(f"Generated {len(all_embeddings)} total embeddings")
        return all_embeddings

    def generate_for_json_chunks(
        self,
        chunks: List[str],
        batch_size: int = 20,
        checkpoint_manager = None,
        job_id: str = None,
        cancellation_event = None
    ) -> List[List[float]]:
        """
        Generate embeddings for JSON chunks with robust checkpoint recovery
        """
        phase = 'embedding_json'
        
        embeddings_already_done = 0
        embeddings = []
        global_batch_index = 0
        
        if checkpoint_manager and job_id:
            checkpoint = checkpoint_manager.load_checkpoint(job_id, phase)
            if checkpoint:
                embeddings_already_done = checkpoint.get('embeddings_count', 0)
                global_batch_index = checkpoint.get('last_batch_index', -1) + 1
                
                data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
                if data_file.exists():
                    try:
                        with open(data_file, 'r') as f:
                            saved_data = json.load(f)
                            embeddings = saved_data.get('embeddings', [])
                        
                        logger.info(
                            f"Resuming JSON embedding: "
                            f"{len(embeddings)} embeddings restored"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load JSON checkpoint: {e}")
                        embeddings = []
                        embeddings_already_done = 0
                        global_batch_index = 0
        
        chunks_remaining = chunks[embeddings_already_done:]
        
        if not chunks_remaining:
            logger.info("All JSON chunks already embedded")
            return embeddings
        
        logger.info(f"Generating embeddings for {len(chunks_remaining)}/{len(chunks)} JSON chunks...")
        
        num_batches = (len(chunks_remaining) + batch_size - 1) // batch_size
        
        for local_batch_idx in range(0, len(chunks_remaining), batch_size):
            if cancellation_event and cancellation_event.is_set():
                logger.warning(
                    f"JSON embedding cancelled. "
                    f"Returning {len(embeddings)} embeddings."
                )
                return embeddings
            
            batch = chunks_remaining[local_batch_idx:local_batch_idx + batch_size]
            batch_num = local_batch_idx // batch_size + 1
            
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
                
                if checkpoint_manager and job_id:
                    current_global_batch = global_batch_index + batch_num - 1
                    
                    checkpoint_data = {
                        'last_batch_index': current_global_batch,
                        'total_batches': (len(chunks) + batch_size - 1) // batch_size,
                        'embeddings_count': len(embeddings)
                    }
                    
                    self._save_checkpoint_safe(
                        checkpoint_manager,
                        job_id,
                        checkpoint_manager.company_name,
                        phase,
                        checkpoint_data,
                        embeddings
                    )
                
                logger.info(
                    f"✓ JSON Batch {batch_num}/{num_batches} completed "
                    f"({len(embeddings)}/{len(chunks)} total embeddings)"
                )
                
            except Exception as e:
                logger.error(f"JSON Batch {batch_num} error: {e}")
                
                if checkpoint_manager and job_id:
                    current_global_batch = global_batch_index + batch_num - 2
                    
                    checkpoint_data = {
                        'last_batch_index': max(current_global_batch, -1),
                        'total_batches': (len(chunks) + batch_size - 1) // batch_size,
                        'embeddings_count': len(embeddings),
                        'error': str(e)[:500]
                    }
                    
                    self._save_checkpoint_safe(
                        checkpoint_manager,
                        job_id,
                        checkpoint_manager.company_name,
                        phase,
                        checkpoint_data,
                        embeddings
                    )
                
                raise
        
        if checkpoint_manager and job_id and len(embeddings) == len(chunks):
            try:
                checkpoint_manager.clear_checkpoint(job_id, phase)
                data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
                if data_file.exists():
                    data_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clear JSON checkpoint: {e}")
        
        logger.info(f"✓ Generated {len(embeddings)} JSON embeddings")
        return embeddings