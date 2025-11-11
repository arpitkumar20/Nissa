import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
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
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.55")
)


class HybridRAGQueryEngine:
    """
    Optimized RAG Engine with improved retrieval accuracy and speed
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
            chunk_size=1000,  # Batch embeddings
        )

        # Initialize Pinecone
        self.pc = PineconeClient(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(pinecone_index_name)

        # Initialize vectorstore with optimized settings
        self.vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings,
            namespace=namespace,
        )

        # Initialize LLM with lower temperature for accuracy
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY,
        )

        # Compile regex patterns once
        self._compile_patterns()

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
        self.keywords = {
            "phone": ["phone", "mobile", "contact", "number"],
            "id": ["id", "unique id", "record id", "identifier", "reference", "ref"],
        }

    def detect_id_in_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced ID detection with better pattern matching
        """
        query_lower = query.lower()

        # Check for record ID
        if match := self.patterns["record_id"].search(query):
            logger.info(f"Detected RECORD_ID: {match.group()}")
            return {
                "is_id_query": True,
                "id_type": "record_id",
                "id_value": match.group(),
                "original_query": query,
            }

        # Check for formatted phone
        if match := self.patterns["phone_formatted"].search(query):
            cleaned_phone = match.group().strip()
            phone_digits = len(re.sub(r"\D", "", cleaned_phone))
            if 10 <= phone_digits <= 14:
                logger.info(f"Detected PHONE (formatted): {cleaned_phone}")
                return {
                    "is_id_query": True,
                    "id_type": "phone",
                    "id_value": cleaned_phone,
                    "original_query": query,
                }

        # Check for plain phone
        if match := self.patterns["phone_plain"].search(query):
            phone_number = match.group()
            logger.info(f"Detected PHONE (plain): {phone_number}")
            return {
                "is_id_query": True,
                "id_type": "phone",
                "id_value": phone_number,
                "original_query": query,
            }

        # Check for unique ID with hyphens
        if match := self.patterns["unique_id_hyphen"].search(query):
            logger.info(f"Detected UNIQUE_ID (hyphenated): {match.group()}")
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": match.group(),
                "original_query": query,
            }

        # Check for unique ID without hyphens
        if match := self.patterns["unique_id_plain"].search(query):
            logger.info(f"Detected UNIQUE_ID (plain): {match.group()}")
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": match.group(),
                "original_query": query,
            }

        # Context-based detection for phone
        if any(kw in query_lower for kw in self.keywords["phone"]):
            words = query.split()
            for word in words:
                cleaned = word.strip(',.?!:;()[]{}"\'"')
                if cleaned.isdigit() and 10 <= len(cleaned) <= 14:
                    logger.info(f"Detected PHONE from context: {cleaned}")
                    return {
                        "is_id_query": True,
                        "id_type": "phone",
                        "id_value": cleaned,
                        "original_query": query,
                    }

        # Context-based detection for ID
        if any(kw in query_lower for kw in self.keywords["id"]):
            words = query.split()
            for word in words:
                cleaned = word.strip(',.?!:;()[]{}"\'"')
                if cleaned.isdigit() and len(cleaned) >= 15:
                    logger.info(f"Detected RECORD_ID from context: {cleaned}")
                    return {
                        "is_id_query": True,
                        "id_type": "record_id",
                        "id_value": cleaned,
                        "original_query": query,
                    }

        logger.info("No ID/Phone detected - using semantic search")
        return {
            "is_id_query": False,
            "id_type": None,
            "id_value": None,
            "original_query": query,
        }

    def retrieve_by_id(
        self, id_type: str, id_value: str, top_k: int = 10
    ) -> List[Document]:
        """
        Optimized exact ID matching with better fallback
        """
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
                docs = [
                    Document(
                        page_content=match.metadata.get("text", ""),
                        metadata=match.metadata,
                    )
                    for match in results.matches
                ]
                return docs

            # Fallback: Partial match on text content
            logger.warning(f"No exact matches, trying partial search")
            return self.retrieve_semantic(
                f"information about {id_type} {id_value}",
                top_k=top_k * 2,  # Cast wider net
            )

        except Exception as e:
            logger.error(f"Error in ID search: {str(e)}")
            return self.retrieve_semantic(f"{id_type} {id_value}", top_k)

    def retrieve_semantic(
        self, query: str, top_k: int = None, score_threshold: float = None
    ) -> List[Document]:
        """
        Optimized semantic search with MMR for diversity
        """
        if top_k is None:
            top_k = self.top_k
        if score_threshold is None:
            score_threshold = self.similarity_threshold

        try:
            # Use MMR (Maximal Marginal Relevance) for diversity
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Changed from "similarity"
                search_kwargs={
                    "k": top_k,
                    "fetch_k": top_k * 3,  # Fetch more candidates
                    "lambda_mult": 0.7,  # Balance relevance vs diversity
                },
            )

            docs = retriever.invoke(query)

            # Filter by relevance score if available
            if hasattr(docs[0], "metadata") and "score" in docs[0].metadata:
                docs = [
                    d for d in docs if d.metadata.get("score", 1.0) >= score_threshold
                ]

            logger.info(f"Found {len(docs)} semantic matches")
            return docs

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

    def hybrid_retrieve(
        self, query: str, top_k: int = None
    ) -> Tuple[List[Document], str]:
        """
        Optimized hybrid retrieval with better fallback logic
        """
        if top_k is None:
            top_k = self.top_k

        id_info = self.detect_id_in_query(query)

        if id_info["is_id_query"]:
            docs = self.retrieve_by_id(id_info["id_type"], id_info["id_value"], top_k)

            # If exact match returns too few results, supplement with semantic
            if len(docs) < 3:
                logger.info("Supplementing with semantic search")
                semantic_docs = self.retrieve_semantic(query, top_k // 2)
                # Merge and deduplicate
                seen_ids = {
                    doc.metadata.get("id") for doc in docs if "id" in doc.metadata
                }
                for doc in semantic_docs:
                    if doc.metadata.get("id") not in seen_ids:
                        docs.append(doc)
                        if len(docs) >= top_k:
                            break

            return docs, f"{id_info['id_type']}_match"
        else:
            docs = self.retrieve_semantic(query, top_k)
            return docs, "semantic"

    def rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """
        Optional: Rerank documents for better relevance
        """
        if not docs:
            return docs

        # Simple keyword-based boosting
        query_terms = set(query.lower().split())

        def relevance_score(doc):
            content_lower = doc.page_content.lower()
            # Count query term matches
            matches = sum(1 for term in query_terms if term in content_lower)
            # Boost if metadata matches
            metadata_boost = 0
            for key, value in doc.metadata.items():
                if key in ["full_name", "phone", "record_id", "unique_id"]:
                    if str(value).lower() in query.lower():
                        metadata_boost += 2
            return matches + metadata_boost

        # Sort by relevance
        docs_with_scores = [(doc, relevance_score(doc)) for doc in docs]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in docs_with_scores]
