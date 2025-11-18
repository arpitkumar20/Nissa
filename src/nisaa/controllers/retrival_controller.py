import os
import re
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "nisaa-knowledge")

# Optimized defaults
TOP_K = int(os.getenv("TOP_K", "10"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("rapidfuzz not available - fuzzy matching disabled")


class PineconeFuzzyCache:
    """
    Cache for Pinecone metadata to enable fast fuzzy matching
    Loads once per namespace and reuses for the session
    """
    
    def __init__(self):
        self._cache = {}
        self._initialized = {}
        self._processed_cache = {}  # Pre-processed for fuzzy matching
    
    def get_or_load(self, namespace: str, index, max_samples: int = 500) -> Dict[str, List[str]]:
        """
        Get cached metadata or load from Pinecone
        
        Returns:
            Dict with keys: 'doctors', 'specialties', 'departments', 'full_names', 'phones'
        """
        if namespace in self._cache:
            return self._cache[namespace]
        
        metadata_cache = {
            'doctors': set(),
            'specialties': set(),
            'departments': set(),
            'full_names': set(),
            'phones': set(),
            'record_ids': set(),
            'unique_ids': set()
        }
        
        try:
            # Sample vectors to extract metadata
            stats = index.describe_index_stats()
            namespace_stats = stats.get('namespaces', {}).get(namespace, {})
            vector_count = namespace_stats.get('vector_count', 0)
            
            if vector_count == 0:
                logger.warning(f"No vectors found in namespace: {namespace}")
                return {k: [] for k in metadata_cache.keys()}
            
            # Query with dummy vector to get samples
            sample_size = min(max_samples, vector_count)
            results = index.query(
                vector=[0.0] * 1536,
                top_k=sample_size,
                namespace=namespace,
                include_metadata=True
            )
            
            # Extract unique values from metadata - optimized with set operations
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                
                # Use get with default to avoid key checks
                doctor_name = metadata.get('doctor_name') or metadata.get('name')
                if doctor_name:
                    metadata_cache['doctors'].add(str(doctor_name).strip())
                
                if specialty := metadata.get('specialty'):
                    metadata_cache['specialties'].add(str(specialty).strip())
                
                if department := metadata.get('department'):
                    metadata_cache['departments'].add(str(department).strip())
                
                if full_name := metadata.get('full_name'):
                    metadata_cache['full_names'].add(str(full_name).strip())
                
                if phone := metadata.get('phone'):
                    metadata_cache['phones'].add(str(phone).strip())
                
                if record_id := metadata.get('record_id'):
                    metadata_cache['record_ids'].add(str(record_id).strip())
                
                if unique_id := metadata.get('unique_id'):
                    metadata_cache['unique_ids'].add(str(unique_id).strip())
            
            # Convert sets to sorted lists once
            final_cache = {k: sorted(v) for k, v in metadata_cache.items()}
            
            self._cache[namespace] = final_cache
            self._initialized[namespace] = True
            
        except Exception as e:
            logger.error(f"Failed to load metadata cache: {e}")
            return {k: [] for k in metadata_cache.keys()}
        
        return self._cache[namespace]


# Global cache instance
_pinecone_cache = PineconeFuzzyCache()


class HybridRAGQueryEngine:
    """
    ENHANCED: Optimized RAG Engine with Pinecone fuzzy matching
    """

    def __init__(
        self,
        namespace: str,
        pinecone_index_name: str = PINECONE_INDEX_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ):
        if not namespace:
            raise ValueError("Namespace is required and cannot be None or empty")

        self.namespace = namespace
        self.pinecone_index_name = pinecone_index_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Initialize embeddings with caching
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_API_KEY,
            show_progress_bar=False,
            chunk_size=1000,
        )

        # Initialize Pinecone
        self.pc = PineconeClient(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(pinecone_index_name)

        # Initialize vectorstore
        self.vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings,
            namespace=namespace,
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY,
        )

        # Load metadata cache for fuzzy matching
        self.metadata_cache = _pinecone_cache.get_or_load(
            namespace=namespace,
            index=self.index
        )

        # Compile regex patterns once
        self._compile_patterns()

        self._word_cache = {}

    def _compile_patterns(self):
        """Pre-compile all regex patterns for better performance"""
        self.patterns = {
            "record_id": re.compile(r"\b\d{15,}\b"),
            "phone_formatted": re.compile(
                r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\.][(]?[0-9]{1,4}[)]?[-\s\.][0-9]{3,4}[-\s\.][0-9]{3,4}"
            ),
            "phone_plain": re.compile(r"(?:\+91)?[6-9]\d{9}\b"),
            "unique_id_hyphen": re.compile(r"\b[A-Z]{2,}-\d+-\d+\b", re.IGNORECASE),
            "unique_id_plain": re.compile(r"\b[A-Z]{2,}\d{5,}\b", re.IGNORECASE),
        }

        self._clean_pattern = re.compile(r'[^\w\s]')

    def fuzzy_correct_query(self, query: str, threshold: int = 80) -> Tuple[str, List[Dict]]:
        """
        Apply fuzzy matching to correct typos in query against Pinecone metadata
        
        Args:
            query: User's input query (potentially with typos)
            threshold: Minimum fuzzy match score (0-100)
        
        Returns:
            Tuple of (corrected_query, list_of_corrections)
        """
        if not FUZZY_AVAILABLE:
            return query, []
        
        corrections = []
        corrected_query = query
        words = query.split()
        
        corrected_positions = set()
        
        if self.metadata_cache['doctors']:
            for i in range(len(words)):
                if i in corrected_positions:
                    continue
                    
                for j in range(min(i + 4, len(words)), i, -1):  # Up to 4-word phrases
                    if any(k in corrected_positions for k in range(i, j)):
                        continue
                        
                    phrase = ' '.join(words[i:j])
                    
                    result = process.extractOne(
                        phrase,
                        self.metadata_cache['doctors'],
                        scorer=fuzz.WRatio,
                        score_cutoff=threshold - 5
                    )
                    
                    if result:
                        matched_name, score, _ = result
                        if score >= threshold - 5:
                            corrected_query = corrected_query.replace(phrase, matched_name, 1)
                            corrections.append({
                                'original': phrase,
                                'corrected': matched_name,
                                'type': 'doctor_name',
                                'confidence': score,
                                'source': 'pinecone_metadata'
                            })
                            logger.info(f"Fuzzy matched doctor: '{phrase}' -> '{matched_name}' (score: {score})")
                            corrected_positions.update(range(i, j))
                            break
        
        for idx, word in enumerate(words):
            if idx in corrected_positions:
                continue
                
            clean_word = self._clean_pattern.sub('', word)
            if len(clean_word) < 3:
                continue
            
            # Try specialty matching first (usually more specific)
            if self.metadata_cache['specialties']:
                result = process.extractOne(
                    clean_word,
                    self.metadata_cache['specialties'],
                    scorer=fuzz.WRatio,
                    score_cutoff=threshold
                )
                
                if result:
                    matched_specialty, score, _ = result
                    corrected_query = re.sub(
                        r'\b' + re.escape(clean_word) + r'\b',
                        matched_specialty,
                        corrected_query,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    corrections.append({
                        'original': clean_word,
                        'corrected': matched_specialty,
                        'type': 'specialty',
                        'confidence': score,
                        'source': 'pinecone_metadata'
                    })
                    logger.info(f"Fuzzy matched specialty: '{clean_word}' -> '{matched_specialty}' (score: {score})")
                    continue
            
            # Try department matching
            if self.metadata_cache['departments']:
                result = process.extractOne(
                    clean_word,
                    self.metadata_cache['departments'],
                    scorer=fuzz.WRatio,
                    score_cutoff=threshold
                )
                
                if result:
                    matched_dept, score, _ = result
                    corrected_query = re.sub(
                        r'\b' + re.escape(clean_word) + r'\b',
                        matched_dept,
                        corrected_query,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    corrections.append({
                        'original': clean_word,
                        'corrected': matched_dept,
                        'type': 'department',
                        'confidence': score,
                        'source': 'pinecone_metadata'
                    })
                    logger.info(f"Fuzzy matched department: '{clean_word}' -> '{matched_dept}' (score: {score})")
        
        if corrections:
            logger.info(f"Query correction: '{query}' -> '{corrected_query}' ({len(corrections)} changes)")
        
        return corrected_query, corrections

    def fuzzy_match_metadata_field(
        self, 
        field_name: str, 
        query_value: str,
        top_k: int = 5,
        threshold: int = 75
    ) -> List[Dict]:
        """
        Fuzzy match against specific metadata field from Pinecone cache
        
        Args:
            field_name: 'doctor_name', 'specialty', 'phone', 'full_name', etc.
            query_value: Value to match (with potential typos)
            top_k: Number of matches to return
            threshold: Minimum score (0-100)
        
        Returns:
            List of matches with scores
        """
        if not FUZZY_AVAILABLE:
            return []
        
        # Map field names to cache keys
        field_mapping = {
            'doctor_name': 'doctors',
            'specialty': 'specialties',
            'department': 'departments',
            'full_name': 'full_names',
            'phone': 'phones',
            'record_id': 'record_ids',
            'unique_id': 'unique_ids'
        }
        
        cache_key = field_mapping.get(field_name, field_name)
        candidates = self.metadata_cache.get(cache_key, [])
        
        if not candidates:
            logger.warning(f"No cached values for field: {field_name}")
            return []
        
        try:
            # Use process.extract to get multiple matches
            matches = process.extract(
                query_value,
                candidates,
                scorer=fuzz.WRatio,
                limit=top_k,
                score_cutoff=threshold
            )
            
            results = [
                {
                    'value': matched_value,
                    'score': score,
                    'type': field_name,
                    'original_query': query_value
                }
                for matched_value, score, _ in matches
            ]
            
            if results:
                logger.info(
                    f"Fuzzy matched {field_name}: '{query_value}' -> "
                    f"{len(results)} matches (best: {results[0]['value']}, score: {results[0]['score']})"
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy matching failed for {field_name}: {e}")
            return []

    def detect_id_in_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced ID detection with fuzzy fallback
        """
        # Check for record ID
        if match := self.patterns["record_id"].search(query):
            record_id = match.group()
            
            # Try fuzzy match if exact pattern found
            if FUZZY_AVAILABLE and self.metadata_cache['record_ids']:
                fuzzy_matches = self.fuzzy_match_metadata_field(
                    'record_id', record_id, top_k=1, threshold=90
                )
                if fuzzy_matches:
                    logger.info(f"Using fuzzy matched record_id: {fuzzy_matches[0]['value']}")
                    record_id = fuzzy_matches[0]['value']
            
            return {
                "is_id_query": True,
                "id_type": "record_id",
                "id_value": record_id,
                "original_query": query,
            }

        # Check for phone numbers
        if match := self.patterns["phone_formatted"].search(query):
            phone = match.group().strip()
            phone_digits = sum(c.isdigit() for c in phone)  # Faster than regex
            
            if 10 <= phone_digits <= 14:
                # Try fuzzy match
                if FUZZY_AVAILABLE and self.metadata_cache['phones']:
                    fuzzy_matches = self.fuzzy_match_metadata_field(
                        'phone', phone, top_k=1, threshold=85
                    )
                    if fuzzy_matches:
                        logger.info(f"Using fuzzy matched phone: {fuzzy_matches[0]['value']}")
                        phone = fuzzy_matches[0]['value']
                
                return {
                    "is_id_query": True,
                    "id_type": "phone",
                    "id_value": phone,
                    "original_query": query,
                }

        if match := self.patterns["phone_plain"].search(query):
            phone = match.group()
            
            # Fuzzy fallback
            if FUZZY_AVAILABLE and self.metadata_cache['phones']:
                fuzzy_matches = self.fuzzy_match_metadata_field(
                    'phone', phone, top_k=1, threshold=85
                )
                if fuzzy_matches:
                    phone = fuzzy_matches[0]['value']
            
            return {
                "is_id_query": True,
                "id_type": "phone",
                "id_value": phone,
                "original_query": query,
            }

        # Check for unique IDs
        if match := self.patterns["unique_id_hyphen"].search(query):
            unique_id = match.group()
            
            if FUZZY_AVAILABLE and self.metadata_cache['unique_ids']:
                fuzzy_matches = self.fuzzy_match_metadata_field(
                    'unique_id', unique_id, top_k=1, threshold=85
                )
                if fuzzy_matches:
                    unique_id = fuzzy_matches[0]['value']
            
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": unique_id,
                "original_query": query,
            }

        logger.info("No ID/Phone detected - using semantic search")
        return {
            "is_id_query": False,
            "id_type": None,
            "id_value": None,
            "original_query": query,
        }

    def _normalize_phone_number(self, phone: str) -> str:
        """
        Normalize phone number to include country code if missing.
        Indian numbers: 10 digits starting with 6-9 → add +91

        Args:
            phone: Raw phone number string

        Returns:
            Normalized phone number with country code
        """
        cleaned = re.sub(r'[^\d+]', '', phone)

        if re.match(r'^[6-9]\d{9}$', cleaned):
            return f"+91{cleaned}"

        return cleaned

    def retrieve_by_id(
        self, id_type: str, id_value: str, top_k: int = 10
    ) -> List[Document]:
        """
        Enhanced exact ID matching with fuzzy fallback
        """
        logger.info(f"Exact ID Search: {id_type} = {id_value}")
        # Normalize phone numbers before searching
        if id_type == "phone":
            id_value = self._normalize_phone_number(id_value)
            logger.info(f"Normalized phone number: {id_value}")

        logger.info(f"Exact ID Search: {id_type} = {id_value}")
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(
                f"Find record with {id_type} {id_value}"
            )

            # Query with exact filter
            results = self.index.query(
                vector=query_embedding,
                filter={id_type: {"$eq": id_value}},
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )

            if results.matches:
                logger.info(f"Found {len(results.matches)} exact matches")
                # List comprehension is faster than append loop
                return [
                    Document(
                        page_content=match.metadata.get("text", ""),
                        metadata=match.metadata,
                    )
                    for match in results.matches
                ]

            # Fuzzy fallback - try similar IDs
            logger.warning(f"No exact matches, trying fuzzy ID search")
            
            if FUZZY_AVAILABLE:
                fuzzy_matches = self.fuzzy_match_metadata_field(
                    field_name=id_type,
                    query_value=id_value,
                    top_k=3,
                    threshold=80
                )
                
                if fuzzy_matches:
                    logger.info(f"Found {len(fuzzy_matches)} fuzzy ID matches")
                    all_docs = []
                    
                    for fuzzy_match in fuzzy_matches:
                        corrected_id = fuzzy_match['value']
                        
                        # Query with corrected ID
                        results = self.index.query(
                            vector=query_embedding,
                            filter={id_type: {"$eq": corrected_id}},
                            top_k=3,
                            namespace=self.namespace,
                            include_metadata=True,
                        )
                        
                        for match in results.matches:
                            # Create metadata dict once
                            metadata = {
                                **match.metadata,
                                'fuzzy_match': True,
                                'fuzzy_confidence': fuzzy_match['score'],
                                'original_query_value': id_value,
                                'matched_value': corrected_id
                            }
                            
                            all_docs.append(Document(
                                page_content=metadata.get("text", ""),
                                metadata=metadata
                            ))
                    
                    if all_docs:
                        return all_docs[:top_k]
            
            # Final fallback: semantic search
            return self.retrieve_semantic(
                f"information about {id_type} {id_value}",
                top_k=top_k
            )

        except Exception as e:
            logger.error(f"Error in ID search: {str(e)}")
            return self.retrieve_semantic(f"{id_type} {id_value}", top_k)

    def retrieve_semantic(
        self, query: str, top_k: int = None, score_threshold: float = None
    ) -> List[Document]:
        """
        ENHANCED: Semantic search with automatic query correction
        """
        if top_k is None:
            top_k = self.top_k
        if score_threshold is None:
            score_threshold = self.similarity_threshold

        # Apply fuzzy correction before semantic search
        corrected_query, corrections = self.fuzzy_correct_query(query)
        
        if corrections:
            logger.info(f"Using corrected query for semantic search: '{corrected_query}'")
            query = corrected_query

        try:
            # Use MMR for diversity
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": top_k * 3,
                    "lambda_mult": 0.7,
                },
            )

            docs = retriever.invoke(query)

            # Add correction info to metadata
            if corrections:
                for doc in docs:
                    doc.metadata['query_corrected'] = True
                    doc.metadata['original_query'] = query
                    doc.metadata['corrections'] = corrections

            # Filter by relevance score if available
            if docs and hasattr(docs[0], "metadata") and "score" in docs[0].metadata:
                docs = [
                    d for d in docs if d.metadata.get("score", 1.0) >= score_threshold
                ]

            logger.info(f"Found {len(docs)} semantic matches")
            return docs

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def hybrid_retrieve(
        self, query: str, top_k: int = None
    ) -> Tuple[List[Document], str]:
        """
        ENHANCED: Hybrid retrieval with fuzzy correction
        """
        if top_k is None:
            top_k = self.top_k

        # First, try to correct the query
        corrected_query, corrections = self.fuzzy_correct_query(query)
        
        # Use corrected query for ID detection
        id_info = self.detect_id_in_query(corrected_query)

        if id_info["is_id_query"]:
            docs = self.retrieve_by_id(id_info["id_type"], id_info["id_value"], top_k)

            # If exact match returns too few results, supplement with semantic
            if len(docs) < 3:
                logger.info("Supplementing ID search with semantic search")
                semantic_docs = self.retrieve_semantic(corrected_query, top_k // 2)
                
                # Merge and deduplicate efficiently
                seen_ids = {doc.metadata.get("id") for doc in docs if "id" in doc.metadata}
                
                for doc in semantic_docs:
                    doc_id = doc.metadata.get("id")
                    if doc_id not in seen_ids:
                        docs.append(doc)
                        seen_ids.add(doc_id)
                        if len(docs) >= top_k:
                            break

            return docs, f"{id_info['id_type']}_match"
        else:
            docs = self.retrieve_semantic(corrected_query, top_k)
            return docs, "semantic"
