import json
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

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "nisaa-knowledge")
NAMESPACE = os.getenv("NAMESPACE", "IHI-MEDICAL-CAMP")

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))


# ============================================================================
# HYBRID SEARCH RAG QUERY ENGINE
# ============================================================================


class HybridRAGQueryEngine:
    """
    Enhanced RAG Engine with Hybrid Search for LangGraph
    Thread-safe and optimized for production use
    """

    def __init__(
        self,
        pinecone_index_name: str = PINECONE_INDEX_NAME,
        namespace: str = NAMESPACE,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """Initialize Hybrid RAG Query Engine"""
        self.pinecone_index_name = pinecone_index_name
        self.namespace = namespace
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Initialize embeddings
        logger.info(f"Initializing embeddings: {embedding_model}")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model, openai_api_key=OPENAI_API_KEY
        )

        # Initialize Pinecone client
        logger.info(f"Connecting to Pinecone: {pinecone_index_name}")
        self.pc = PineconeClient(api_key=PINECONE_API_KEY)
        
        # Initialize vector store using langchain-pinecone
        logger.info(f"Initializing vector store: {namespace}")
        self.vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings,
            namespace=namespace,
        )
        
        # Get the index for direct queries
        self.index = self.pc.Index(pinecone_index_name)

        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model}")
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY,
        )

        logger.info("✅ Hybrid RAG Engine initialized")

    # ========================================================================
    # ID DETECTION AND EXTRACTION
    # ========================================================================

    def detect_id_in_query(self, query: str) -> Dict[str, Any]:
        """
        Detect if query contains an ID or phone number

        Returns:
            {
                'is_id_query': bool,
                'id_type': str ('record_id', 'unique_id', 'phone', or None),
                'id_value': str,
                'original_query': str
            }
        """
        query_lower = query.lower()

        # Pattern 1: Long numeric IDs (15+ digits - record IDs)
        long_id_pattern = r"\b\d{15,}\b"
        long_id_match = re.search(long_id_pattern, query)
        if long_id_match:
            logger.info(f"🔍 Detected RECORD_ID: {long_id_match.group()}")
            return {
                "is_id_query": True,
                "id_type": "record_id",
                "id_value": long_id_match.group(),
                "original_query": query,
            }

        # Pattern 2: Formatted phone numbers
        formatted_phone_pattern = r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\.][(]?[0-9]{1,4}[)]?[-\s\.][0-9]{3,4}[-\s\.][0-9]{3,4}"
        formatted_phone_match = re.search(formatted_phone_pattern, query)
        if formatted_phone_match:
            cleaned_phone = formatted_phone_match.group().strip()
            phone_digits = len(re.sub(r"\D", "", cleaned_phone))
            if 10 <= phone_digits <= 14:
                logger.info(f"📱 Detected PHONE (formatted): {cleaned_phone}")
                return {
                    "is_id_query": True,
                    "id_type": "phone",
                    "id_value": cleaned_phone,
                    "original_query": query,
                }

        # Pattern 3: Plain phone numbers (10-14 digits)
        plain_phone_pattern = r"(?:\+91)?[6-9]\d{9}\b"
        plain_phone_match = re.search(plain_phone_pattern, query)
        if plain_phone_match:
            phone_number = plain_phone_match.group()
            logger.info(f"📱 Detected PHONE (plain): {phone_number}")
            return {
                "is_id_query": True,
                "id_type": "phone",
                "id_value": phone_number,
                "original_query": query,
            }

        # Pattern 4: Unique IDs WITH hyphens (IHI-5-130)
        unique_id_hyphenated_pattern = r"\b[A-Z]{2,}-\d+-\d+\b"
        unique_id_hyphenated_match = re.search(
            unique_id_hyphenated_pattern, query, re.IGNORECASE
        )
        if unique_id_hyphenated_match:
            logger.info(
                f"🔍 Detected UNIQUE_ID (hyphenated): {unique_id_hyphenated_match.group()}"
            )
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": unique_id_hyphenated_match.group(),
                "original_query": query,
            }

        # Pattern 5: Unique IDs WITHOUT hyphens (IHI2302145)
        unique_id_no_hyphen_pattern = r"\b[A-Z]{2,}\d{5,}\b"
        unique_id_no_hyphen_match = re.search(
            unique_id_no_hyphen_pattern, query, re.IGNORECASE
        )
        if unique_id_no_hyphen_match:
            logger.info(
                f"🔍 Detected UNIQUE_ID (no hyphens): {unique_id_no_hyphen_match.group()}"
            )
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": unique_id_no_hyphen_match.group(),
                "original_query": query,
            }

        # Pattern 6: Keywords indicating ID/phone queries
        phone_keywords = ["phone", "mobile", "contact", "number"]
        id_keywords = ["id", "unique id", "record id", "identifier"]

        # Check for phone keywords
        if any(kw in query_lower for kw in phone_keywords):
            words = query.split()
            for word in words:
                cleaned = word.strip(',.?!:;()[]{}"\'"')
                if cleaned.isdigit() and 10 <= len(cleaned) <= 14:
                    logger.info(f"📱 Detected PHONE from context: {cleaned}")
                    return {
                        "is_id_query": True,
                        "id_type": "phone",
                        "id_value": cleaned,
                        "original_query": query,
                    }

        # Check for ID keywords
        if any(kw in query_lower for kw in id_keywords):
            words = query.split()
            for word in words:
                cleaned = word.strip(',.?!:;()[]{}"\'"')
                if cleaned.isdigit() and len(cleaned) >= 15:
                    logger.info(f"🔍 Detected RECORD_ID from context: {cleaned}")
                    return {
                        "is_id_query": True,
                        "id_type": "record_id",
                        "id_value": cleaned,
                        "original_query": query,
                    }

        logger.info("🔎 No ID/Phone - using semantic search")
        return {
            "is_id_query": False,
            "id_type": None,
            "id_value": None,
            "original_query": query,
        }
 
 
    # ========================================================================
    # HYBRID RETRIEVAL
    # ========================================================================

    def retrieve_by_id(
        self, id_type: str, id_value: str, top_k: int = 5
    ) -> List[Document]:
        """
        Retrieve documents using EXACT ID matching via Pinecone metadata filter

        Args:
            id_type: 'record_id', 'unique_id', or 'phone'
            id_value: The ID value to search for
            top_k: Number of results to return

        Returns:
            List of Document objects
        """
        logger.info(f"🎯 Exact ID Search: {id_type} = {id_value}")

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(
                f"Find record with {id_type} {id_value}"
            )

            # Query Pinecone with metadata filter
            results = self.index.query(
                vector=query_embedding,
                filter={id_type: {"$eq": id_value}},
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )

            if results.matches:
                logger.info(f"✅ Found {len(results.matches)} exact matches")

                docs = []
                for match in results.matches:
                    doc = Document(
                        page_content=match.metadata.get("text", ""),
                        metadata=match.metadata,
                    )
                    docs.append(doc)

                return docs
            else:
                logger.warning(f"⚠️ No exact matches for {id_type}={id_value}")
                logger.info("🔄 Falling back to semantic search")
                return self.retrieve_semantic(
                    f"Find information about {id_type} {id_value}", top_k
                )

        except Exception as e:
            logger.error(f"❌ Error in ID search: {str(e)}")
            return self.retrieve_semantic(f"{id_type} {id_value}", top_k)

    def retrieve_semantic(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using SEMANTIC search

        Args:
            query: Natural language query
            top_k: Number of results

        Returns:
            List of Document objects
        """
        logger.info(f"🔎 Semantic Search: {query[:100]}")

        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )

            docs = retriever.invoke(query)
            logger.info(f"✅ Found {len(docs)} semantic matches")
            return docs

        except Exception as e:
            logger.error(f"❌ Error in semantic search: {str(e)}")
            return []

    def hybrid_retrieve(
        self, query: str, top_k: int = None
    ) -> Tuple[List[Document], str]:
        """
        MAIN RETRIEVAL: Auto-selects between ID and semantic search

        Args:
            query: User's question
            top_k: Number of results

        Returns:
            (documents, search_type) tuple
        """
        if top_k is None:
            top_k = self.top_k

        # Detect ID/phone
        id_info = self.detect_id_in_query(query)

        if id_info["is_id_query"]:
            # Use exact ID matching
            docs = self.retrieve_by_id(id_info["id_type"], id_info["id_value"], top_k)
            return docs, f"{id_info['id_type']}_match"
        else:
            # Use semantic search
            docs = self.retrieve_semantic(query, top_k)
            return docs, "semantic"