"""
Vector store service for Pinecone operations
"""
import os
import time
import uuid
import hashlib
from typing import List, Dict, Tuple, Any
from pinecone import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from src.nisaa.helpers.logger import logger
from src.nisaa.services.json_processor import extract_all_ids


class VectorStoreService:
    """Handles all Pinecone vector store operations"""
    
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
        """
        Prepare document vectors for upsert
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            namespace: Pinecone namespace
            
        Returns:
            Tuple of (ids, vectors, processed_metadatas)
        """
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        processed_metadatas = []
        for i, metadata in enumerate(metadatas):
            meta_copy = metadata.copy()
            meta_copy['text'] = texts[i][:40000]  # Pinecone metadata limit
            
            # Convert all values to Pinecone-compatible types
            final_metadata = {}
            for key, value in meta_copy.items():
                if isinstance(value, (str, int, float, bool)):
                    final_metadata[key] = value
                elif isinstance(value, list):
                    final_metadata[key] = [str(item) for item in value if item is not None][:100]
                else:
                    final_metadata[key] = str(value)[:1000]
            
            processed_metadatas.append(final_metadata)
        
        logger.info(f"📦 Prepared {len(ids)} document vectors for namespace '{namespace}'")
        return ids, embeddings, processed_metadatas
    
    def prepare_json_vectors(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        entities: List[Tuple[str, Dict]],
        namespace: str
    ) -> List[Dict]:
        """
        Prepare JSON chunk vectors with entity metadata
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            entities: List of (entity_id, entity_data) tuples
            namespace: Pinecone namespace
            
        Returns:
            List of vector dictionaries ready for upsert
        """
        
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
            
            # Add ID metadata
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
            
            # Check for nested lists
            has_nested_lists = any(key.count('[') > 1 for key in entity_data.keys())
            if has_nested_lists:
                metadata["has_nested_arrays"] = True
            
            # Extract identifiers from chunk
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
        
        logger.info(f"📦 Prepared {len(vectors)} JSON vectors for namespace '{namespace}'")
        return vectors
    
    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
        namespace: str,
        batch_size: int = None
    ) -> int:
        """
        Upsert vectors to Pinecone in batches
        
        Args:
            ids: List of vector IDs
            vectors: List of embedding vectors
            metadatas: List of metadata dicts
            namespace: Pinecone namespace
            batch_size: Batch size for upserts
            
        Returns:
            Total number of vectors upserted
        """
        batch_size = batch_size or int(os.getenv('PINECONE_BATCH_SIZE', '100'))
        total_upserted = 0
        upsert_batches = (len(vectors) - 1) // batch_size + 1
        
        logger.info(f"📤 Upserting {len(vectors)} vectors to namespace '{namespace}'...")
        
        for i in range(0, len(vectors), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            vectors_to_upsert = list(zip(batch_ids, batch_vectors, batch_metadatas))
            batch_num = i // batch_size + 1
            
            try:
                upsert_response = self.index.upsert(
                    vectors=vectors_to_upsert,
                    namespace=namespace
                )
                
                upserted_count = upsert_response.get('upserted_count', 0)
                total_upserted += upserted_count
                
                logger.info(f"   ✓ Batch {batch_num}/{upsert_batches}: {upserted_count} vectors")
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                logger.error(f"   ✗ Batch {batch_num} failed: {e}")
                raise
        
        logger.info(f"✅ Total upserted: {total_upserted}/{len(vectors)}")
        return total_upserted
    
    def upsert_json_vectors(
        self,
        vectors: List[Dict],
        namespace: str,
        batch_size: int = 100
    ) -> int:
        """
        Upsert JSON vectors to Pinecone
        
        Args:
            vectors: List of vector dictionaries
            namespace: Pinecone namespace
            batch_size: Batch size for upserts
            
        Returns:
            Total number of vectors upserted
        """
        total_upserted = 0
        upsert_batches = (len(vectors) - 1) // batch_size + 1
        
        logger.info(f"📤 Upserting {len(vectors)} JSON vectors to namespace '{namespace}'...")
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.info(f"   ✓ Batch {batch_num}/{upsert_batches}: {len(batch)} vectors")
            except Exception as e:
                logger.error(f"   ✗ Batch {batch_num} failed: {e}")
                raise
        
        logger.info(f"✅ Total JSON vectors upserted: {total_upserted}")
        return total_upserted
    
    def verify_upsert(self, namespace: str, expected_count: int):
        """Verify vectors were successfully uploaded"""
        time.sleep(2)
        try:
            stats = self.index.describe_index_stats()
            namespace_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
            
            logger.info(f"📊 Verification - Vectors in '{namespace}': {namespace_count}")
            
            if namespace_count < expected_count:
                logger.warning(f"⚠️ Expected {expected_count}, found {namespace_count}")
            else:
                logger.info(f"✅ All {expected_count} vectors stored correctly")
        except Exception as e:
            logger.warning(f"Could not verify: {e}")
    
    def create_langchain_vectorstore(
        self,
        namespace: str,
        embeddings: OpenAIEmbeddings
    ) -> LangchainPinecone:
        """
        Create LangChain PineconeVectorStore wrapper
        
        Args:
            namespace: Pinecone namespace
            embeddings: OpenAI embeddings instance
            
        Returns:
            LangChain Pinecone vectorstore
        """
        return LangchainPinecone(
            index=self.index,
            embedding=embeddings,
            namespace=namespace,
            text_key='text'
        )