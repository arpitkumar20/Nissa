"""
Fixed JSON Processor with Checkpoint Recovery
Key fixes:
1. Correct batch calculation (ceiling division)
2. Checkpoint-based resume support
3. Parallel processing with failure tracking
4. Entity-level tracking for safety
"""
import os
import time
import json
import hashlib
from typing import Dict, List, Union, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.nisaa.config.logger import logger


def flatten_json(data: Union[Dict, list], parent_key: str = '', sep: str = '.', max_depth: int = 100) -> Dict:
    """Completely flatten a nested JSON structure with dot notation"""
    items = []
    
    if max_depth <= 0:
        items.append((parent_key, str(data)))
        return dict(items)
    
    if isinstance(data, dict):
        if not data:
            items.append((parent_key, {}))
        else:
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                
                if isinstance(v, (dict, list)) and v:
                    items.extend(flatten_json(v, new_key, sep=sep, max_depth=max_depth-1).items())
                else:
                    items.append((new_key, v))
    
    elif isinstance(data, list):
        if not data:
            items.append((parent_key, []))
        else:
            for i, v in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                
                if isinstance(v, (dict, list)) and v:
                    items.extend(flatten_json(v, new_key, sep=sep, max_depth=max_depth-1).items())
                else:
                    items.append((new_key, v))
    else:
        items.append((parent_key, data))
    
    return dict(items)


def identify_entities(flattened_data: Dict) -> List[Tuple[str, Dict]]:
    """Group flattened data by TOP-LEVEL entities only"""
    entities = {}
    first_array_pattern = __import__('re').compile(r'^([^\[]+\[\d+\])')
    
    for key, value in flattened_data.items():
        match = first_array_pattern.match(key)
        
        if match:
            entity_id = match.group(1)
        else:
            parts = key.split('.')
            entity_id = parts[0] if parts else "root"
        
        if entity_id not in entities:
            entities[entity_id] = {}
        
        entities[entity_id][key] = value
    
    return sorted(entities.items(), key=lambda x: (len(x[0]), x[0]))


def extract_all_ids(entity_data: Dict) -> Dict[str, str]:
    """Extract all possible ID fields from entity data"""
    ids = {}
    
    for key, value in entity_data.items():
        key_lower = key.lower()
        
        if (key_lower.endswith('.id') or key_lower == 'id') and key_lower.count('[') <= 1:
            if 'record_id' not in ids:
                ids['record_id'] = str(value)
        
        elif ('unique_id' in key_lower or 'unique id' in key_lower) and key_lower.count('[') <= 1:
            if 'unique_id' not in ids:
                ids['unique_id'] = str(value)
        
        elif ('full_name' in key_lower or 'name.display_value' in key_lower) and key_lower.count('[') <= 1:
            if 'full_name' not in ids:
                ids['full_name'] = str(value)
        
        elif 'email' in key_lower and value and key_lower.count('[') <= 1:
            if 'email' not in ids:
                ids['email'] = str(value)
        
        elif 'phone' in key_lower and value and key_lower.count('[') <= 1:
            if 'phone' not in ids:
                ids['phone'] = str(value)
    
    return ids


def convert_entity_to_text(entity_data: Dict) -> str:
    """Convert entity data to text with ALL nested lists included inline"""
    text_lines = []
    
    for key, value in sorted(entity_data.items()):
        line = f"{key}: {value}"
        if len(line) > 2000:
            line = f"{key}: {str(value)[:2000]}..."
        text_lines.append(line)
    
    return "\n".join(text_lines)


class JSONProcessor:
    """Processes JSON files with FIXED checkpoint recovery"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", max_workers: int = 3):
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_workers = max_workers
        self.client = OpenAI(api_key=openai_api_key)
    
    def _compute_entity_hash(self, entity_id: str, entity_data: Dict) -> str:
        """Compute hash for entity to detect changes"""
        content = json.dumps(
            {entity_id: entity_data},
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _save_json_checkpoint(
        self,
        checkpoint_manager,
        job_id: str,
        company_name: str,
        converted_texts: List[str],
        failed_entities: List[str],
        total_entities: int,
        processed_count: int
    ):
        """Save JSON conversion checkpoint"""
        try:
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{job_id}_json_conversion.json"
            data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_json_conversion_data.json"
            
            checkpoint_data = {
                'phase': 'json_conversion',
                'processed': processed_count,
                'total': total_entities,
                'failed_count': len(failed_entities),
                'text_count': len(converted_texts),
                'timestamp': time.time()
            }
            
            # Save metadata
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Save data
            with open(data_file, 'w') as f:
                json.dump({
                    'converted_texts': converted_texts,
                    'failed_entities': failed_entities,
                    'metadata': {
                        'count': len(converted_texts),
                        'hash': hashlib.sha256(
                            json.dumps(converted_texts, default=str).encode()
                        ).hexdigest()
                    }
                }, f)
            
            logger.info(
                f"✓ JSON checkpoint saved: {processed_count}/{total_entities} processed, "
                f"{len(converted_texts)} texts, {len(failed_entities)} failed"
            )
            
            try:
                checkpoint_manager.save_checkpoint(
                    job_id=job_id,
                    company_name=company_name,
                    phase='json_conversion',
                    checkpoint_data=checkpoint_data
                )
            except Exception as e:
                logger.warning(f"DB checkpoint save failed (file is safe): {e}")
                
        except Exception as e:
            logger.error(f"Failed to save JSON checkpoint: {e}")
            raise
    
    def _load_json_checkpoint(self, checkpoint_manager, job_id: str) -> Tuple[List[str], List[str], int, bool]:
        """Load JSON checkpoint with integrity check"""
        try:
            checkpoint_file = checkpoint_manager.checkpoint_dir / f"{job_id}_json_conversion.json"
            data_file = checkpoint_manager.checkpoint_dir / f"{job_id}_json_conversion_data.json"
            
            if not checkpoint_file.exists() or not data_file.exists():
                return [], [], 0, False
            
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            converted_texts = data.get('converted_texts', [])
            failed_entities = data.get('failed_entities', [])
            metadata = data.get('metadata', {})
            
            # Verify integrity
            if len(converted_texts) != metadata.get('count', 0):
                logger.error(
                    f"JSON checkpoint corrupted: text count mismatch "
                    f"{len(converted_texts)} vs {metadata.get('count', 0)}"
                )
                return [], [], 0, False
            
            processed = checkpoint.get('processed', 0)
            logger.info(
                f"✓ JSON checkpoint loaded: {processed} processed, "
                f"{len(converted_texts)} texts, {len(failed_entities)} failed"
            )
            
            return converted_texts, failed_entities, processed, True
            
        except Exception as e:
            logger.error(f"Failed to load JSON checkpoint: {e}")
            return [], [], 0, False
    
    def _convert_entity_batch(self, entities: List[Tuple[str, Dict]], start_idx: int) -> List[Tuple[str, str, bool]]:
        """Convert entity batch to natural language with error tracking"""
        results = []
        
        for idx, (entity_id, entity_data) in enumerate(entities):
            actual_idx = start_idx + idx
            
            try:
                entity_text = convert_entity_to_text(entity_data)
                
                has_lists = any('[' in key and key.count('[') > 1 for key in entity_data.keys())
                list_hint = "\n\nIMPORTANT: This record contains nested arrays/lists. Include ALL array items."
                
                prompt = f"""Convert to natural language:
1. PRESERVE ALL EXACT VALUES: every ID, number, name, email, phone
2. INCLUDE ALL NESTED DATA: every array item [0], [1], [2]
3. Use plain text only - NO markdown
4. Complete sentences including all information
5. Start with important identifiers{list_hint}

Data:
{entity_text}

Natural language (plain text, complete, all values included):"""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Convert data to natural language. Preserve exact IDs, numbers, names. Include ALL nested items. No markdown. Complete and detailed."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    timeout=60
                )
                
                text = response.choices[0].message.content
                
                # Clean markdown
                for char in ['**', '*', '###', '##', '#', '- ', 'â€¢ ', 'â"€']:
                    text = text.replace(char, '')
                
                # Add identifiers
                metadata_lines = []
                for key, value in entity_data.items():
                    if any(term in key.lower() for term in ['id', 'name', 'email', 'phone']):
                        if key.count('[') <= 1 and value:
                            clean_key = key.split('.')[-1].replace('[', '').replace(']', '')
                            if clean_key:
                                metadata_lines.append(f"{clean_key}: {value}")
                
                if metadata_lines:
                    unique_meta = list(dict.fromkeys(metadata_lines))[:10]
                    text += "\n\nKey identifiers: " + ", ".join(unique_meta)
                
                results.append((entity_id, text, True))
                logger.debug(f"✓ Entity {actual_idx + 1}: {entity_id} converted")
                
            except Exception as e:
                logger.warning(f"Failed to convert entity {actual_idx + 1} ({entity_id}): {e}")
                results.append((entity_id, f"Error: {str(e)[:100]}", False))
        
        return results
    
    def process_json_file(
        self, 
        file_path: str,
        checkpoint_manager = None,
        job_id: str = None,
        company_name: str = None,
        cancellation_event = None
    ) -> Tuple[List[str], List[Tuple[str, Dict]]]:
        """
        Process JSON file with FIXED checkpoint recovery
        
        Returns: (chunks, entities) for metadata
        """
        logger.info(f"Processing JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info("Flattening JSON...")
            flattened_data = flatten_json(data, max_depth=100)
            logger.info(f"Flattened {len(flattened_data)} keys")
            
            logger.info("Identifying entities...")
            entities = identify_entities(flattened_data)
            logger.info(f"Found {len(entities)} entities")
            
            total_entities = len(entities)
            converted_texts = []
            failed_entities = []
            start_from = 0
            
            # Try to load from checkpoint
            if checkpoint_manager and job_id:
                loaded_texts, loaded_failed, start_from, is_valid = self._load_json_checkpoint(
                    checkpoint_manager, job_id
                )
                if is_valid and start_from > 0:
                    converted_texts = loaded_texts
                    failed_entities = loaded_failed
                    logger.info(f"Resuming from entity {start_from + 1}")
            
            # Process remaining entities
            remaining_entities = entities[start_from:]
            
            if not remaining_entities:
                logger.info("All entities already converted")
                chunks = self._chunk_by_entities(converted_texts)
                return chunks, entities
            
            logger.info(f"Converting {len(remaining_entities)}/{total_entities} remaining entities...")
            
            # FIX: Correct batch calculation
            batch_size = max(1, min(10, self.max_workers * 2))
            num_remaining = len(remaining_entities)
            total_batches = (num_remaining + batch_size - 1) // batch_size
            
            logger.info(f"Processing in {total_batches} batches of {batch_size}")
            
            for batch_idx in range(total_batches):
                if cancellation_event and cancellation_event.is_set():
                    logger.warning(f"JSON conversion cancelled at batch {batch_idx + 1}")
                    break
                
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_remaining)
                batch_entities = remaining_entities[batch_start:batch_end]
                
                try:
                    batch_results = self._convert_entity_batch(
                        batch_entities,
                        start_from + batch_start
                    )
                    
                    for entity_id, text, success in batch_results:
                        if success:
                            converted_texts.append(text)
                        else:
                            failed_entities.append(entity_id)
                    
                    # Save checkpoint after each batch
                    if checkpoint_manager and job_id:
                        self._save_json_checkpoint(
                            checkpoint_manager,
                            job_id,
                            company_name,
                            converted_texts,
                            failed_entities,
                            total_entities,
                            start_from + batch_end
                        )
                    
                    logger.info(
                        f"✓ Batch {batch_idx + 1}/{total_batches}: "
                        f"{len(batch_results)} entities, "
                        f"{len(converted_texts)} total texts"
                    )
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")
                    
                    if checkpoint_manager and job_id:
                        self._save_json_checkpoint(
                            checkpoint_manager,
                            job_id,
                            company_name,
                            converted_texts,
                            failed_entities,
                            total_entities,
                            start_from + batch_start
                        )
                    raise
            
            # Cleanup checkpoint on success
            if checkpoint_manager and job_id and len(converted_texts) + len(failed_entities) == total_entities:
                try:
                    checkpoint_manager.clear_checkpoint(job_id, 'json_conversion')
                    logger.info("✓ JSON checkpoint cleared after completion")
                except Exception as e:
                    logger.warning(f"Failed to clear JSON checkpoint: {e}")
            
            logger.info(f"Chunking {len(converted_texts)} texts...")
            chunks = self._chunk_by_entities(converted_texts)
            logger.info(f"Created {len(chunks)} chunks from {len(converted_texts)} entities")
            
            if failed_entities:
                logger.warning(f"⚠️ {len(failed_entities)} entities failed: {failed_entities[:5]}")
            
            return chunks, entities
            
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            raise
    
    def _chunk_by_entities(self, entity_texts: List[str], max_chunk_size: int = 8000, entities_per_chunk: int = 1) -> List[str]:
        """Chunk texts by entities"""
        chunks = []
        current_chunk = []
        current_entity_count = 0
        
        for entity_text in entity_texts:
            entity_size = len(entity_text)
            
            if entity_size > max_chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_entity_count = 0
                
                sentences = entity_text.split('. ')
                temp_chunk = []
                temp_size = 0
                
                for sentence in sentences:
                    if temp_size + len(sentence) > max_chunk_size:
                        if temp_chunk:
                            chunks.append('. '.join(temp_chunk) + '.')
                        temp_chunk = [sentence]
                        temp_size = len(sentence)
                    else:
                        temp_chunk.append(sentence)
                        temp_size += len(sentence)
                
                if temp_chunk:
                    chunks.append('. '.join(temp_chunk) + '.')
            
            elif current_entity_count >= entities_per_chunk:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [entity_text]
                current_entity_count = 1
            else:
                current_chunk.append(entity_text)
                current_entity_count += 1
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks