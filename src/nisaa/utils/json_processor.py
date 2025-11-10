"""
JSON processor with nested list handling - EXISTING LOGIC INTACT
"""
import json
import re
import time
import hashlib
from pathlib import Path
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
    first_array_pattern = re.compile(r'^([^\[]+\[\d+\])')
    
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
    """Extract all possible ID fields from entity data for metadata"""
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


def convert_to_natural_language_batch(entities: List[Tuple[str, Dict]], api_key: str, 
                                      model: str = "gpt-4o-mini", max_workers: int = 3) -> List[str]:
    """Convert entities to natural language with ALL nested data included"""
    client = OpenAI(api_key=api_key)
    
    def process_entity(entity_tuple):
        entity_id, entity_data = entity_tuple
        entity_text = convert_entity_to_text(entity_data)
        
        has_lists = any('[' in key and key.count('[') > 1 for key in entity_data.keys())
        
        list_hint = ""
        if has_lists:
            list_hint = "\n\nIMPORTANT: This record contains nested arrays/lists (shown with [0], [1], [2] notation). Include ALL array items in your natural language output - they are part of this single record."
        
        prompt = f"""Convert this data to natural language with these CRITICAL requirements:

1. PRESERVE ALL EXACT VALUES: Include every ID, number, name, email, phone, code, URL exactly as shown
2. INCLUDE ALL NESTED DATA: When you see keys with [0], [1], [2] etc., these are items in arrays/lists that belong to this record. Include ALL of them in your description.
3. Use plain text only - NO markdown, bullets, asterisks, or formatting
4. Write flowing sentences that include all information
5. Start with the most important identifiers (ID, name, code)
6. Be complete - include every field without omitting anything{list_hint}

Data:
{entity_text}

Natural language (plain text, all exact values and nested list items included):"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You convert data to plain natural language. Always preserve exact IDs, numbers, names, codes. Include ALL nested array items as they belong to the same record. Never use markdown or formatting. Write clear, complete sentences."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                text = response.choices[0].message.content
                text = text.replace('**', '').replace('*', '').replace('###', '').replace('##', '').replace('#', '')
                text = text.replace('- ', '').replace('• ', '').replace('─', '')
                
                metadata_lines = []
                for key, value in entity_data.items():
                    if any(term in key.lower() for term in ['id', 'name', 'email', 'phone', 'code', 'aadhaar', 'number']):
                        if key.count('[') <= 1:
                            clean_key = key.split('.')[-1].replace('[', '').replace(']', '')
                            if clean_key and value:
                                metadata_lines.append(f"{clean_key}: {value}")
                
                if metadata_lines:
                    seen = set()
                    unique_metadata = []
                    for item in metadata_lines:
                        if item not in seen:
                            seen.add(item)
                            unique_metadata.append(item)
                    
                    text += "\n\nKey identifiers: " + ", ".join(unique_metadata[:10])
                
                return (entity_id, text)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return (entity_id, f"Error processing {entity_id}: {str(e)}")
    
    logger.info(f"Processing {len(entities)} JSON entities in parallel (max {max_workers} workers)...")
    
    results = [None] * len(entities)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entity, entity): idx 
                  for idx, entity in enumerate(entities)}
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            entity_id, text = future.result()
            results[idx] = text
            completed += 1
            
            if completed % 10 == 0 or completed == len(entities):
                logger.info(f"Completed {completed}/{len(entities)} entities")
    
    return results


def chunk_by_entities(entity_texts: List[str], max_chunk_size: int = 8000, 
                     entities_per_chunk: int = 1) -> List[str]:
    """Chunk texts by entities. Default is 1 entity per chunk for best retrieval"""
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


class JSONProcessor:
    """Processes JSON files with nested list handling"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", max_workers: int = 3):
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_workers = max_workers
    
    def process_json_file(self, file_path: str) -> Tuple[List[str], List[Tuple[str, Dict]]]:
        """
        Process a single JSON file
        
        Returns:
            Tuple of (chunks, entities) for metadata extraction
        """
        logger.info(f"Processing JSON file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Flattening JSON...")
        flattened_data = flatten_json(data, max_depth=100)
        logger.info(f"Flattened {len(flattened_data)} keys")
        
        logger.info(f"Identifying entities...")
        entities = identify_entities(flattened_data)
        logger.info(f"Found {len(entities)} entities")
        
        logger.info(f"Converting to natural language...")
        entity_texts = convert_to_natural_language_batch(
            entities, self.openai_api_key, self.model, self.max_workers
        )
        logger.info(f"Converted all entities")
        
        logger.info(f"Chunking data...")
        chunks = chunk_by_entities(entity_texts, max_chunk_size=8000, entities_per_chunk=1)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks, entities