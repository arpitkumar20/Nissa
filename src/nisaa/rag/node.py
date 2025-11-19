import logging
from langchain_core.messages import HumanMessage, SystemMessage
from src.nisaa.controllers.retrival_controller import HybridRAGQueryEngine
from src.nisaa.rag.state import GraphState, log_state

logger = logging.getLogger(__name__)


def get_rag_engine(namespace: str) -> HybridRAGQueryEngine:
    """
    FIX: Create a NEW RAG engine instance for each request

    This eliminates:
    - Race conditions between concurrent requests
    - Namespace mixing between different companies
    - Singleton-related bugs

    Args:
        namespace: Company-specific namespace from state

    Returns:
        Fresh HybridRAGQueryEngine instance
    """
    if not namespace:
        raise ValueError("Namespace is required for RAG engine")

    return HybridRAGQueryEngine(
        namespace=namespace,
        top_k=15,  # CHANGED: Increased from 5 to 15
        similarity_threshold=0.65,
        temperature=0.7,
        max_tokens=2000,
        use_reranker=True,  # NEW: Enable reranker for better accuracy
    )


def detect_id_or_phone(state: GraphState) -> GraphState:
    """
    MOVED TO FIRST: Detect ID/phone before anything else
    This is cheap and determines our search strategy
    """
    try:
        log_state(state, "DETECT_ID_PHONE")

        query = state["user_query"]
        namespace = state["company_namespace"]

        rag_engine = get_rag_engine(namespace)
        id_info = rag_engine.detect_id_in_query(query)

        return {
            **state,
            "is_id_query": id_info["is_id_query"],
            "id_type": id_info["id_type"],
            "id_value": id_info["id_value"],
        }

    except Exception as e:
        logger.error(f"Error detecting ID/phone: {e}")
        return {**state, "is_id_query": False, "id_type": None, "id_value": None}


def generate_embedding(state: GraphState) -> GraphState:
    """
    Generate embedding for the user query
    Uses OpenAI text-embedding-3-small
    """
    try:
        log_state(state, "GENERATE_EMBEDDING")

        query = state["user_query"]
        namespace = state["company_namespace"]

        rag_engine = get_rag_engine(namespace)
        embedding = rag_engine.embeddings.embed_query(query)

        logger.info(f"Generated embedding of dimension {len(embedding)}")

        return {**state, "query_embedding": embedding}

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return {**state, "query_embedding": None}


def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve documents using hybrid search with reranking
    - Exact match for ID/phone queries
    - Semantic search for natural language
    - Reranks results for better accuracy
    """
    try:
        log_state(state, "RETRIEVE_DOCUMENTS")

        namespace = state["company_namespace"]
        rag_engine = get_rag_engine(namespace)
        query = state["user_query"]

        # top_k is now 15 with reranking enabled
        docs, search_type = rag_engine.hybrid_retrieve(query, top_k=15)

        doc_list = []
        for doc in docs:
            doc_dict = {"content": doc.page_content, "metadata": doc.metadata}
            doc_list.append(doc_dict)

        logger.info(f"Retrieved {len(docs)} documents using {search_type} (reranked)")

        return {
            **state,
            "retrieved_documents": doc_list,
            "search_type": search_type,
            "num_documents": len(docs),
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return {
            **state,
            "retrieved_documents": [],
            "search_type": "error",
            "num_documents": 0,
            "error": str(e),
        }


def format_context(state: GraphState) -> GraphState:
    """
    Format retrieved documents into context string for LLM
    Includes metadata, IDs, and complete data
    """
    try:
        log_state(state, "FORMAT_CONTEXT")

        docs = state["retrieved_documents"]

        if not docs:
            return {
                **state,
                "formatted_context": "No relevant information found in the knowledge base.",
            }

        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc["content"]
            metadata = doc["metadata"]

            formatted_text = f"[Document {i}]\n"

            if metadata.get("record_id"):
                formatted_text += f"Record ID: {metadata['record_id']}\n"
            if metadata.get("unique_id"):
                formatted_text += f"Unique ID: {metadata['unique_id']}\n"
            if metadata.get("full_name"):
                formatted_text += f"Name: {metadata['full_name']}\n"
            if metadata.get("phone"):
                formatted_text += f"Phone: {metadata['phone']}\n"

            formatted_text += f"\nContent:\n{content}\n"

            if metadata.get("complete_data"):
                formatted_text += f"\nComplete Data:\n{metadata['complete_data']}\n"

            formatted_parts.append(formatted_text)

        context = "\n\n" + ("=" * 80 + "\n\n").join(formatted_parts)

        logger.info(f"Formatted context: {len(context)} characters")

        return {**state, "formatted_context": context}

    except Exception as e:
        logger.error(f"Error formatting context: {e}")
        return {
            **state,
            "formatted_context": "Error formatting context.",
            "error": str(e),
        }


def generate_response(state: GraphState) -> GraphState:
    """
    IMPROVED: Generate response WITHOUT history first
    Faster and sufficient for most queries
    """
    try:
        log_state(state, "GENERATE_RESPONSE")

        namespace = state["company_namespace"]
        rag_engine = get_rag_engine(namespace)

        system_message = """You are a helpful AI assistant with access to a knowledge base. 

    Context from Knowledge Base:
    {context}

    Instructions:
    - Answer ONLY based on the provided context
    - If information is complete and clear, provide a confident, complete answer
    - If information is ambiguous or incomplete, clearly state: "I need to check our conversation history for more context"
    - When asked about IDs, names, or phone numbers, extract them directly from context
    - Be comprehensive and cite specific field names when relevant
    - Maintain professional tone"""

        context = state["formatted_context"]
        query = state["user_query"]

        llm_messages = [
            SystemMessage(content=system_message.format(context=context)),
            HumanMessage(content=query),
        ]

        response = rag_engine.llm.invoke(llm_messages)
        answer = response.content

        logger.info(f"Generated initial response: {len(answer)} characters")

        return {**state, "model_response": answer}

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_msg = "I apologize, but I encountered an error processing your request. Please try again."
        return {
            **state,
            "model_response": error_msg,
            "error": str(e),
        }
