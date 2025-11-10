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
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "nisaa-knowledge")

TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))


class HybridRAGQueryEngine:
    """
    Enhanced RAG Engine with Hybrid Search for LangGraph
    
    FIXED: Thread-safe and designed for per-request instantiation
    No longer uses singleton pattern or file reading
    """

    def __init__(
        self,
        namespace: str,
        pinecone_index_name: str = PINECONE_INDEX_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Initialize Hybrid RAG Query Engine
        
        CRITICAL FIX: Namespace is now REQUIRED parameter
        
        Args:
            namespace: Company-specific namespace (MUST be provided)
            pinecone_index_name: Pinecone index name
            embedding_model: OpenAI embedding model
            llm_model: OpenAI LLM model
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            temperature: LLM temperature
            max_tokens: Max tokens in response
            
        Raises:
            ValueError: If namespace is None or empty
        """
        
        if not namespace:
            raise ValueError(
                "Namespace is REQUIRED and cannot be None or empty. "
                "Pass company_namespace from GraphState."
            )

        self.namespace = namespace
        self.pinecone_index_name = pinecone_index_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model, 
            openai_api_key=OPENAI_API_KEY
        )

        self.pc = PineconeClient(api_key=PINECONE_API_KEY)
        
        self.vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=self.embeddings,
            namespace=namespace,
        )
        
        self.index = self.pc.Index(pinecone_index_name)

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY,
        )

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

        unique_id_hyphenated_pattern = r"\b[A-Z]{2,}-\d+-\d+\b"
        unique_id_hyphenated_match = re.search(
            unique_id_hyphenated_pattern, query, re.IGNORECASE
        )
        if unique_id_hyphenated_match:
            logger.info(
                f"Detected UNIQUE_ID (hyphenated): {unique_id_hyphenated_match.group()}"
            )
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": unique_id_hyphenated_match.group(),
                "original_query": query,
            }

        unique_id_no_hyphen_pattern = r"\b[A-Z]{2,}\d{5,}\b"
        unique_id_no_hyphen_match = re.search(
            unique_id_no_hyphen_pattern, query, re.IGNORECASE
        )
        if unique_id_no_hyphen_match:
            logger.info(
                f"Detected UNIQUE_ID (no hyphens): {unique_id_no_hyphen_match.group()}"
            )
            return {
                "is_id_query": True,
                "id_type": "unique_id",
                "id_value": unique_id_no_hyphen_match.group(),
                "original_query": query,
            }

        phone_keywords = ["phone", "mobile", "contact", "number"]
        id_keywords = ["id", "unique id", "record id", "identifier"]

        if any(kw in query_lower for kw in phone_keywords):
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

        if any(kw in query_lower for kw in id_keywords):
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
        logger.info(f"Exact ID Search: {id_type} = {id_value} in namespace '{self.namespace}'")

        try:
            query_embedding = self.embeddings.embed_query(
                f"Find record with {id_type} {id_value}"
            )

            results = self.index.query(
                vector=query_embedding,
                filter={id_type: {"$eq": id_value}},
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )

            if results.matches:
                logger.info(f"Found {len(results.matches)} exact matches")

                docs = []
                for match in results.matches:
                    doc = Document(
                        page_content=match.metadata.get("text", ""),
                        metadata=match.metadata,
                    )
                    docs.append(doc)

                return docs
            else:
                logger.warning(f"No exact matches for {id_type}={id_value}")
                return self.retrieve_semantic(
                    f"Find information about {id_type} {id_value}", top_k
                )

        except Exception as e:
            logger.error(f"Error in ID search: {str(e)}")
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
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
            )

            docs = retriever.invoke(query)
            logger.info(f"Found {len(docs)} semantic matches")
            return docs

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
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

        id_info = self.detect_id_in_query(query)

        if id_info["is_id_query"]:
            docs = self.retrieve_by_id(id_info["id_type"], id_info["id_value"], top_k)
            return docs, f"{id_info['id_type']}_match"
        else:
            docs = self.retrieve_semantic(query, top_k)
            return docs, "semantic"