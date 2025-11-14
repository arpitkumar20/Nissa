"""
Vector store service for Pinecone operations with checkpoint/recovery support
"""
import os
import time
import uuid
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from pinecone import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from src.nisaa.config.logger import logger
from nisaa.utils.json_processor import extract_all_ids


class VectorStoreService:
    """Handles all Pinecone vector store operations with checkpointing"""
    
    def __init__(self, api_key: str = None, index_name: str = None):
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX')
        self.pc = LangchainPinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)
        logger.info(f"📌 Connected to Pinecone index: {self.index_name}")
    
    def prepare_document_vectors(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        namespace: str
    ) -> Tuple[List[str], List[List[float]], List[Dict]]:
        """Prepare document vectors for upsert"""
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        processed_metadatas = []
        for i, metadata in enumerate(metadatas):
            meta_copy = metadata.copy()
            meta_copy['text'] = texts[i][:40000]
            
            final_metadata = {}
            for key, value in meta_copy.items():
                if isinstance(value, (str, int, float, bool)):
                    final_metadata[key] = value
                elif isinstance(value, list):
                    final_metadata[key] = [str(item) for item in value if item is not None][:100]
                else:
                    final_metadata[key] = str(value)[:1000]
            
            processed_metadatas.append(final_metadata)
        
        logger.info(f"Prepared {len(ids)} document vectors for namespace '{namespace}'")
        return ids, embeddings, processed_metadatas
    
    def prepare_json_vectors(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        entities: List[Tuple[str, Dict]],
        namespace: str
    ) -> List[Dict]:
        """Prepare JSON chunk vectors with entity metadata"""
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = hashlib.md5(chunk.encode()).hexdigest()[:16]
            
            entity_id, entity_data = entities[i] if i < len(entities) else (None, {})
            all_ids = extract_all_ids(entity_data)
            
            metadata = {
                "text": chunk[:40000],
                "chunk_index": i,
                "full_text_length": len(chunk),
                "entity_id": entity_id if entity_id else "unknown",
                "source_type": "json"
            }
            
            if 'record_id' in all_ids:
                metadata["record_id"] = all_ids['record_id']
            if 'unique_id' in all_ids:
                metadata["unique_id"] = all_ids['unique_id']
            if 'full_name' in all_ids:
                metadata["full_name"] = all_ids['full_name']
            if 'email' in all_ids:
                metadata["email"] = all_ids['email']
            if 'phone' in all_ids:
                metadata["phone"] = all_ids['phone']
            
            has_nested_lists = any(key.count('[') > 1 for key in entity_data.keys())
            if has_nested_lists:
                metadata["has_nested_arrays"] = True
            
            lines = chunk.split('\n')
            for line in lines:
                if 'Key identifiers:' in line:
                    metadata["identifiers"] = line.split('Key identifiers:')[1].strip()[:1000]
                    break
            
            vectors.append({
                "id": f"{namespace}_{vector_id}_{i}",
                "values": embedding,
                "metadata": metadata
            })
        
        logger.info(f"Prepared {len(vectors)} JSON vectors for namespace '{namespace}'")
        return vectors
    
    def _compute_batch_hash(self, batch_ids: List[str]) -> str:
        """Compute hash for a batch to track processed batches"""
        combined = "".join(sorted(batch_ids))
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    """
    Replace upsert_vectors and upsert_json_vectors in vector_store_service.py
    SIMPLIFIED: Only batch index tracking
    """

    def upsert_vectors(
    self,
    ids: List[str],
    vectors: List[List[float]],
    metadatas: List[Dict],
    namespace: str,
    batch_size: int = None,
    checkpoint_manager = None,
    job_id: str = None,
    cancellation_event = None  

    ) -> int:
        """Upsert vectors with proper checkpoint recovery"""
        batch_size = batch_size or int(os.getenv('PINECONE_BATCH_SIZE', '100'))
        phase = 'upserting_vectors'
        
        start_batch_idx = 0
        total_upserted = 0
        
        # Load checkpoint
        if checkpoint_manager and job_id:
            checkpoint = checkpoint_manager.load_checkpoint(job_id, phase)
            if checkpoint:
                start_batch_idx = checkpoint.get('last_batch_index', -1) + 1
                total_upserted = checkpoint.get('total_upserted', 0)
                logger.info(
                    f"🔄 Resuming upsert from batch {start_batch_idx} "
                    f"({total_upserted} vectors already upserted)"
                )
        
        upsert_batches = (len(vectors) - 1) // batch_size + 1
        
        logger.info(f"Upserting {len(vectors)} vectors to namespace '{namespace}'...")
        
        for i in range(start_batch_idx * batch_size, len(vectors), batch_size):

            if cancellation_event and cancellation_event.is_set():
                logger.warning(
                    f"⚠️ Upsert cancelled at batch {i // batch_size + 1}/{upsert_batches}. "
                    f"Progress saved in checkpoint."
                )
                return total_upserted 
            
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            vectors_to_upsert = list(zip(batch_ids, batch_vectors, batch_metadatas))
            
            try:
                upsert_response = self.index.upsert(
                    vectors=vectors_to_upsert,
                    namespace=namespace
                )
                
                upserted_count = upsert_response.get('upserted_count', 0)
                total_upserted += upserted_count
                
                # Save checkpoint
                if checkpoint_manager and job_id:
                    checkpoint_manager.save_checkpoint(
                        job_id=job_id,
                        company_name=checkpoint_manager.company_name,
                        phase=phase,
                        checkpoint_data={
                            'last_batch_index': batch_num - 1,
                            'total_batches': upsert_batches,
                            'total_upserted': total_upserted,
                            'namespace': namespace,
                            'timestamp': time.time()
                        }
                    )
                
                logger.info(f"✅ Batch {batch_num}/{upsert_batches}: {upserted_count} vectors")
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                
                if checkpoint_manager and job_id:
                    checkpoint_manager.save_checkpoint(
                        job_id=job_id,
                        company_name=checkpoint_manager.company_name,
                        phase=phase,
                        checkpoint_data={
                            'last_batch_index': batch_num - 2,
                            'total_batches': upsert_batches,
                            'total_upserted': total_upserted,
                            'namespace': namespace,
                            'error': str(e),
                            'timestamp': time.time()
                        }
                    )
                raise
        
        # Clear checkpoint on success
        if checkpoint_manager and job_id:
            checkpoint_manager.clear_checkpoint(job_id, phase)
        
        logger.info(f"✅ Total upserted: {total_upserted}/{len(vectors)}")
        return total_upserted


    def upsert_json_vectors(
        self,
        vectors: List[Dict],
        namespace: str,
        batch_size: int = 100,
        checkpoint_manager = None,
        item_tracker = None,
        job_id: str = None
    ) -> int:
        """
        Upsert JSON vectors to Pinecone with checkpoint/recovery support
        SIMPLIFIED: Only batch tracking
        """
        phase = 'upserting_json_vectors'
        
        # Check for existing checkpoint
        start_batch_idx = 0
        total_upserted = 0
        
        if checkpoint_manager and job_id:
            checkpoint = checkpoint_manager.load_checkpoint(job_id, phase)
            if checkpoint:
                start_batch_idx = checkpoint.get('last_batch_index', -1) + 1
                total_upserted = checkpoint.get('total_upserted', 0)
                logger.info(
                    f"🔄 Resuming JSON upsert from batch {start_batch_idx} "
                    f"({total_upserted} vectors already upserted)"
                )
        
        upsert_batches = (len(vectors) - 1) // batch_size + 1
        
        logger.info(f"Upserting {len(vectors)} JSON vectors to namespace '{namespace}'...")
        
        for i in range(start_batch_idx * batch_size, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                
                # Save checkpoint
                if checkpoint_manager and job_id:
                    checkpoint_manager.save_checkpoint(
                        job_id=job_id,
                        company_name=checkpoint_manager.company_name,
                        phase=phase,
                        checkpoint_data={
                            'last_batch_index': batch_num - 1,
                            'total_batches': upsert_batches,
                            'total_upserted': total_upserted,
                            'namespace': namespace,
                            'timestamp': time.time()
                        }
                    )
                
                logger.info(f"✓ JSON Batch {batch_num}/{upsert_batches}: {len(batch)} vectors")
                
            except Exception as e:
                logger.error(f"❌ JSON Batch {batch_num} failed: {e}")
                
                # Save checkpoint on error
                if checkpoint_manager and job_id:
                    checkpoint_manager.save_checkpoint(
                        job_id=job_id,
                        company_name=checkpoint_manager.company_name,
                        phase=phase,
                        checkpoint_data={
                            'last_batch_index': batch_num - 2,
                            'total_batches': upsert_batches,
                            'total_upserted': total_upserted,
                            'namespace': namespace,
                            'error': str(e),
                            'timestamp': time.time()
                        }
                    )
                raise
        
        # Clear checkpoint on success
        if checkpoint_manager and job_id:
            checkpoint_manager.clear_checkpoint(job_id, phase)
        
        logger.info(f"✅ Total JSON vectors upserted: {total_upserted}")
        return total_upserted
    def verify_upsert(self, namespace: str, expected_count: int):
        """Verify vectors were successfully uploaded"""
        time.sleep(2)
        try:
            stats = self.index.describe_index_stats()
            namespace_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
            
            logger.info(f"Verification - Vectors in '{namespace}': {namespace_count}")
            
            if namespace_count < expected_count:
                logger.warning(f"Expected {expected_count}, found {namespace_count}")
            else:
                logger.info(f"✅ All {expected_count} vectors stored correctly")
        except Exception as e:
            logger.warning(f"Could not verify: {e}")
    
    def create_langchain_vectorstore(
        self,
        namespace: str,
        embeddings: OpenAIEmbeddings
    ) -> LangchainPinecone:
        """Create LangChain PineconeVectorStore wrapper"""
        return LangchainPinecone(
            index=self.index,
            embedding=embeddings,
            namespace=namespace,
            text_key='text'
        )