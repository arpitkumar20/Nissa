"""
PRODUCTION-READY: Embedding Service with Content-Based Checkpoint Matching
Ensures embeddings are ALWAYS matched to correct text content
"""
import os
import time
import hashlib
import json
from typing import List, Optional, Tuple, Dict
from openai import OpenAI, RateLimitError, APIConnectionError
from langchain_openai import OpenAIEmbeddings
from src.nisaa.config.logger import logger


def compute_text_hash(text: str) -> str:
    """
    Compute stable SHA256 hash of text content
    This ensures same text always gets same hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def compute_data_hash(data: List) -> str:
    """Compute SHA256 hash of data list for integrity checking"""
    try:
        content = json.dumps(
            [(str(i), len(str(item)[:100])) for i, item in enumerate(data[:100])],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()
    except:
        return "unknown"


class EmbeddingService:
    """
    FIXED: Embedding generation with SAFE content-based checkpoint recovery
    
    Key Features:
    - Content hashing for reliable text→embedding mapping
    - Reuses embeddings only when text content matches exactly
    - Generates new embeddings for changed/new content
    - Prevents data corruption from non-deterministic document ordering
    """
    
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
        max_retries: int = 3,
        initial_delay: float = 5.0,
        backoff_factor: float = 2.0
    ):
        """Retry function with exponential backoff"""
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

    def _save_checkpoint_with_hashes(
        self,
        checkpoint_manager,
        job_id: str,
        company_name: str,
        phase: str,
        checkpoint_data: dict,
        embeddings: List[List[float]],
        texts: List[str]
    ):
        """
        Save checkpoint with text content hashes for reliable matching
        """
        try:
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}.json"
            data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
            
            # Compute hashes for all texts
            text_hashes = [compute_text_hash(text) for text in texts]
            
            # Add integrity information
            checkpoint_data.update({
                'timestamp': time.time(),
                'embedding_count': len(embeddings),
                'text_count': len(texts),
                'unique_hashes': len(set(text_hashes)),
                'batch_size': checkpoint_data.get('batch_size', 10),
                'company_name': company_name  
            })
            
            # Save metadata
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"✓ Checkpoint metadata saved: {checkpoint_file.name}")
            
            # Save data with hashes for content-based matching
            with open(data_file, 'w') as f:
                json.dump({
                    'embeddings': embeddings,
                    'texts': texts,
                    'text_hashes': text_hashes,  # CRITICAL: Save hashes
                    'metadata': {
                        'count': len(embeddings),
                        'unique_hashes': len(set(text_hashes)),
                        'company_name': company_name 
                    }
                }, f)
            
            logger.info(
                f"✓ Checkpoint data saved: {len(embeddings)} embeddings with content hashes"
            )
            
        except Exception as e:
            logger.critical(f"FILE CHECKPOINT FAILED: {e}")
            raise
 
        # Try DB save (non-critical)
        try:
            checkpoint_manager.save_checkpoint(
                job_id=job_id,
                company_name=company_name,
                phase=phase,
                checkpoint_data=checkpoint_data
            )
        except Exception as e:
            logger.warning(f"DB checkpoint save failed (file is safe): {e}")

    def _load_checkpoint_with_verification(
        self,
        checkpoint_manager,
        job_id: str,
        phase: str
    ) -> Tuple[Dict[str, List], bool]:
        """
        Load checkpoint and return hash→embedding mapping
        Returns: (hash_to_embedding_map, is_valid)
        """
        try:
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}.json"
            data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
            
            if not checkpoint_file.exists() or not data_file.exists():
                logger.info("No checkpoint files found")
                return {}, False
            
            # Load checkpoint metadata
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Load checkpoint data
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            embeddings = data.get('embeddings', [])
            texts = data.get('texts', [])
            text_hashes = data.get('text_hashes', [])
            
            # Verify integrity
            if len(embeddings) != len(texts) or len(embeddings) != len(text_hashes):
                logger.error(
                    f"Checkpoint corrupted: mismatched lengths "
                    f"(emb:{len(embeddings)}, txt:{len(texts)}, hash:{len(text_hashes)})"
                )
                return {}, False
            
            if len(embeddings) != checkpoint.get('embedding_count', 0):
                logger.error(
                    f"Checkpoint metadata mismatch: {len(embeddings)} vs "
                    f"{checkpoint.get('embedding_count', 0)}"
                )
                return {}, False
            
            # Build hash → embedding mapping
            hash_to_embedding = {}
            for i, (text_hash, embedding, text) in enumerate(zip(text_hashes, embeddings, texts)):
                hash_to_embedding[text_hash] = {
                    'embedding': embedding,
                    'text_preview': text[:100],
                    'original_index': i
                }
            
            logger.info(
                f"✓ Checkpoint verified: {len(embeddings)} embeddings, "
                f"{len(hash_to_embedding)} unique content hashes"
            )
            
            # Log first few hashes for debugging
            sample_hashes = list(hash_to_embedding.keys())[:3]
            logger.debug(f"Sample checkpoint hashes: {sample_hashes}")
            
            return hash_to_embedding, True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return {}, False

    def generate_for_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        checkpoint_manager = None,
        job_id: str = None,
        cancellation_event = None
    ) -> List[List[float]]:
        """
        FIXED: Generate embeddings with SAFE content-based checkpoint recovery
        
        Key Safety Features:
        1. Matches embeddings to text by CONTENT HASH (not position)
        2. Reuses embeddings only when content matches exactly
        3. Generates new embeddings for changed/new content
        4. Works correctly even if document order changes
        """
        batch_size = batch_size or int(os.getenv('EMBEDDING_BATCH_SIZE', '10'))
        phase = 'embedding_documents'
        
        # Step 1: Compute hashes for all current texts
        text_hashes = [compute_text_hash(text) for text in texts]
        unique_hashes = set(text_hashes)
        
        logger.info(
            f"Processing {len(texts)} texts with {len(unique_hashes)} unique content hashes"
        )
        
        # Log first few hashes for debugging
        logger.debug(f"Sample current hashes: {list(unique_hashes)[:3]}")
        
        # Step 2: Load checkpoint (hash → embedding mapping)
        hash_to_embedding = {}
        
        if checkpoint_manager and job_id:
            hash_to_embedding, is_valid = self._load_checkpoint_with_verification(
                checkpoint_manager, job_id, phase
            )
            
            if is_valid and hash_to_embedding:
                logger.info(f"Found checkpoint with {len(hash_to_embedding)} embeddings")
        
        # Step 3: Match current texts to checkpoint embeddings by content hash
        result_embeddings = [None] * len(texts)
        texts_to_generate = []
        indices_to_generate = []
        
        matched_count = 0
        new_content_count = 0
        
        for i, (text, text_hash) in enumerate(zip(texts, text_hashes)):
            if text_hash in hash_to_embedding:
                # EXACT content match found - reuse embedding
                result_embeddings[i] = hash_to_embedding[text_hash]['embedding']
                matched_count += 1
                
                # Verify it's really the same content
                if i < 3:  # Log first few matches for verification
                    checkpoint_preview = hash_to_embedding[text_hash]['text_preview']
                    current_preview = text[:100]
                    logger.debug(
                        f"Match {i}: hash {text_hash[:8]} "
                        f"checkpoint: '{checkpoint_preview[:50]}...' "
                        f"current: '{current_preview[:50]}...'"
                    )
            else:
                # New or changed content - need to generate
                texts_to_generate.append(text)
                indices_to_generate.append(i)
                new_content_count += 1
        
        logger.info(
            f"Content matching: {matched_count} reused from checkpoint, "
            f"{new_content_count} new/changed texts need generation"
        )
        
        # Step 4: Generate embeddings for new/changed content
        if texts_to_generate:
            logger.info(
                f"Generating embeddings for {len(texts_to_generate)}/{len(texts)} texts -- "
                f"Model: {self.model} -- Batch size: {batch_size}"
            )
            
            num_remaining = len(texts_to_generate)
            total_batches = (num_remaining + batch_size - 1) // batch_size
            
            logger.info(f"Total batches to process: {total_batches}")
            
            generated_embeddings = []
            
            for batch_idx in range(total_batches):
                if cancellation_event and cancellation_event.is_set():
                    logger.warning(
                        f"Embedding cancelled at batch {batch_idx + 1}/{total_batches}. "
                        f"Returning {matched_count + len(generated_embeddings)} embeddings."
                    )
                    break
                
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_remaining)
                batch = texts_to_generate[batch_start:batch_end]
                
                logger.debug(f"Batch {batch_idx}: indices {batch_start}-{batch_end}, size {len(batch)}")
                
                self._rate_limit_delay()
                
                def embed_batch():
                    return self.langchain_embeddings.embed_documents(batch)
                
                try:
                    vectors = self._retry_with_exponential_backoff(embed_batch)
                    generated_embeddings.extend(vectors)
                    
                    # Insert newly generated embeddings into result
                    for j, emb in enumerate(vectors):
                        idx = indices_to_generate[batch_start + j]
                        result_embeddings[idx] = emb
                    
                    # Save checkpoint with ALL current progress
                    if checkpoint_manager and job_id:
                        # Collect all completed embeddings and texts
                        completed_embeddings = []
                        completed_texts = []
                        
                        for i, emb in enumerate(result_embeddings):
                            if emb is not None:
                                completed_embeddings.append(emb)
                                completed_texts.append(texts[i])
                        
                        checkpoint_data = {
                            'last_batch_index': batch_idx,
                            'total_batches': total_batches,
                            'embeddings_count': len(completed_embeddings),
                            'matched_from_checkpoint': matched_count,
                            'newly_generated': len(generated_embeddings),
                            'batch_size': batch_size
                        }
                        
                        self._save_checkpoint_with_hashes(
                            checkpoint_manager,
                            job_id,
                            checkpoint_manager.company_name,
                            phase,
                            checkpoint_data,
                            completed_embeddings,
                            completed_texts
                        )
                    
                    logger.info(
                        f"✓ Batch {batch_idx + 1}/{total_batches} completed "
                        f"({len(generated_embeddings)}/{len(texts_to_generate)} generated, "
                        f"{matched_count + len(generated_embeddings)}/{len(texts)} total)"
                    )
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")
                    
                    # Save progress before failing
                    if checkpoint_manager and job_id:
                        completed_embeddings = []
                        completed_texts = []
                        
                        for i, emb in enumerate(result_embeddings):
                            if emb is not None:
                                completed_embeddings.append(emb)
                                completed_texts.append(texts[i])
                        
                        checkpoint_data = {
                            'last_batch_index': batch_idx - 1,
                            'total_batches': total_batches,
                            'embeddings_count': len(completed_embeddings),
                            'error': str(e)[:500]
                        }
                        
                        self._save_checkpoint_with_hashes(
                            checkpoint_manager,
                            job_id,
                            checkpoint_manager.company_name,
                            phase,
                            checkpoint_data,
                            completed_embeddings,
                            completed_texts
                        )
                    
                    raise
        
        # Step 5: Verify all embeddings are present
        none_count = sum(1 for e in result_embeddings if e is None)
        if none_count > 0:
            logger.error(f"CRITICAL: {none_count} embeddings still missing!")
            raise ValueError(f"{none_count} texts have no embeddings")
        
        logger.info(
            f"✓ Complete: {len(result_embeddings)} total embeddings "
            f"({matched_count} reused from checkpoint, {len(texts_to_generate)} generated)"
        )
        
        # Cleanup checkpoint on success
        if checkpoint_manager and job_id and len(result_embeddings) == len(texts):
            try:
                checkpoint_manager.clear_checkpoint(job_id, phase)
                checkpoint_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}.json"
                data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_{phase}_data.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                if data_file.exists():
                    data_file.unlink()
                logger.info("✓ Checkpoint cleared after successful completion")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")
        
        return result_embeddings

    def generate_for_json_chunks(
        self,
        chunks: List[str],
        batch_size: int = 20,
        checkpoint_manager = None,
        job_id: str = None,
        cancellation_event = None
    ) -> List[List[float]]:
        """
        Generate embeddings for JSON chunks with content-based checkpoint recovery
        """
        phase = 'embedding_json'
        batch_size = batch_size or int(os.getenv('EMBEDDING_BATCH_SIZE', '20'))
        
        # Use the same content-based matching approach
        chunk_hashes = [compute_text_hash(chunk) for chunk in chunks]
        unique_hashes = set(chunk_hashes)
        
        logger.info(
            f"Processing {len(chunks)} JSON chunks with {len(unique_hashes)} unique hashes"
        )
        
        # Load checkpoint
        hash_to_embedding = {}
        
        if checkpoint_manager and job_id:
            hash_to_embedding, is_valid = self._load_checkpoint_with_verification(
                checkpoint_manager, job_id, phase
            )
        
        # Match and generate
        result_embeddings = [None] * len(chunks)
        chunks_to_generate = []
        indices_to_generate = []
        
        matched_count = 0
        
        for i, (chunk, chunk_hash) in enumerate(zip(chunks, chunk_hashes)):
            if chunk_hash in hash_to_embedding:
                result_embeddings[i] = hash_to_embedding[chunk_hash]['embedding']
                matched_count += 1
            else:
                chunks_to_generate.append(chunk)
                indices_to_generate.append(i)
        
        logger.info(
            f"JSON matching: {matched_count} reused, {len(chunks_to_generate)} need generation"
        )
        
        # Generate remaining
        if chunks_to_generate:
            num_remaining = len(chunks_to_generate)
            total_batches = (num_remaining + batch_size - 1) // batch_size
            
            generated_embeddings = []
            
            for batch_idx in range(total_batches):
                if cancellation_event and cancellation_event.is_set():
                    logger.warning(f"JSON embedding cancelled at batch {batch_idx + 1}/{total_batches}")
                    break
                
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_remaining)
                batch = chunks_to_generate[batch_start:batch_end]
                
                self._rate_limit_delay()
                
                def embed_json_batch():
                    response = self.openai_client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    return [item.embedding for item in response.data]
                
                try:
                    batch_embeddings = self._retry_with_exponential_backoff(embed_json_batch)
                    generated_embeddings.extend(batch_embeddings)
                    
                    # Insert into result
                    for j, emb in enumerate(batch_embeddings):
                        idx = indices_to_generate[batch_start + j]
                        result_embeddings[idx] = emb
                    
                    # Save checkpoint
                    if checkpoint_manager and job_id:
                        completed_embeddings = [e for e in result_embeddings if e is not None]
                        completed_chunks = [chunks[i] for i, e in enumerate(result_embeddings) if e is not None]
                        
                        checkpoint_data = {
                            'last_batch_index': batch_idx,
                            'total_batches': total_batches,
                            'embeddings_count': len(completed_embeddings),
                            'batch_size': batch_size
                        }
                        
                        self._save_checkpoint_with_hashes(
                            checkpoint_manager,
                            job_id,
                            checkpoint_manager.company_name,
                            phase,
                            checkpoint_data,
                            completed_embeddings,
                            completed_chunks
                        )
                    
                    logger.info(
                        f"✓ JSON Batch {batch_idx + 1}/{total_batches} completed "
                        f"({len(generated_embeddings)}/{len(chunks_to_generate)} generated)"
                    )
                    
                except Exception as e:
                    logger.error(f"JSON Batch {batch_idx + 1} error: {e}")
                    raise
        
        # Verify
        none_count = sum(1 for e in result_embeddings if e is None)
        if none_count > 0:
            logger.error(f"CRITICAL: {none_count} JSON embeddings missing!")
            raise ValueError(f"{none_count} chunks have no embeddings")
        
        logger.info(
            f"✓ JSON Complete: {len(result_embeddings)} embeddings "
            f"({matched_count} reused, {len(chunks_to_generate)} generated)"
        )
        
        # Cleanup
        if checkpoint_manager and job_id and len(result_embeddings) == len(chunks):
            try:
                checkpoint_manager.clear_checkpoint(job_id, phase)
            except Exception as e:
                logger.warning(f"Failed to clear JSON checkpoint: {e}")
        
        return result_embeddings