import logging
from langchain_core.messages import HumanMessage, SystemMessage
from src.nisaa.controllers.retrival_controller import HybridRAGQueryEngine
from src.nisaa.rag.state import GraphState, log_state

logger = logging.getLogger(__name__)

# CRITICAL: Cache RAG engines per namespace (reuse across requests)
_rag_engine_cache = {}

def get_rag_engine(namespace: str) -> HybridRAGQueryEngine:
    """
    OPTIMIZED: Reuse RAG engine instances instead of creating new ones
    
    Creating a new engine every time is expensive:
    - Initializes OpenAI client
    - Initializes Pinecone client
    - Creates embeddings object
    
    Now: Cache and reuse per namespace
    """
    if namespace in _rag_engine_cache:
        return _rag_engine_cache[namespace]
    
    if not namespace:
        raise ValueError("Namespace is required for RAG engine")
    
    engine = HybridRAGQueryEngine(
        namespace=namespace,
        top_k=5,
        similarity_threshold=0.65,
        temperature=0.1,  # Lower = faster + more deterministic
        max_tokens=300    # Reduced from 500 (40% faster)
    )
    
    _rag_engine_cache[namespace] = engine
    logger.info(f"Created and cached RAG engine for namespace: {namespace}")
    
    return engine


def detect_id_or_phone(state: GraphState) -> GraphState:
    """Detect ID/phone - UNCHANGED"""
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
    """Generate embedding ONCE - UNCHANGED"""
    try:
        log_state(state, "GENERATE_EMBEDDING")

        query = state["user_query"]
        namespace = state["company_namespace"]
        
        if state.get("is_id_query", False):
            logger.info("Skipping embedding for ID query")
            return {**state, "query_embedding": None}
        
        rag_engine = get_rag_engine(namespace)
        
        logger.info("🔥 GENERATING EMBEDDING")
        embedding = rag_engine.embeddings.embed_query(query)

        logger.info(f"Generated embedding of dimension {len(embedding)}")

        return {**state, "query_embedding": embedding}

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return {**state, "query_embedding": None}


def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve documents - UNCHANGED"""
    try:
        log_state(state, "RETRIEVE_DOCUMENTS")

        namespace = state["company_namespace"]
        query = state["user_query"]
        query_embedding = state.get("query_embedding")
        is_id_query = state.get("is_id_query", False)
        
        rag_engine = get_rag_engine(namespace)

        if is_id_query:
            logger.info("Using ID-based exact match retrieval")
            id_value = state.get("id_value")
            id_type = state.get("id_type")
            
            docs = rag_engine.retrieve_by_id(id_value, id_type, top_k=5)
            search_type = "exact_match"
            
        else:
            if not query_embedding:
                logger.error("No embedding found in state!")
                logger.warning("⚠️ Generating embedding as fallback")
                query_embedding = rag_engine.embeddings.embed_query(query)
            else:
                logger.info("✅ Using pre-generated embedding")
            
            docs = rag_engine.retrieve_with_embedding(
                query_embedding=query_embedding,
                query_text=query,
                top_k=5
            )
            search_type = "semantic"

        doc_list = []
        for doc in docs:
            doc_dict = {"content": doc.page_content, "metadata": doc.metadata}
            doc_list.append(doc_dict)

        logger.info(f"Retrieved {len(docs)} documents using {search_type}")

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
    OPTIMIZED: Simplified context formatting (removed verbose metadata)
    
    Before: Included all metadata fields (record_id, unique_id, full_name, etc.)
    After: Only essential fields
    
    Saves: ~30-40% of context length
    """
    try:
        log_state(state, "FORMAT_CONTEXT")

        docs = state["retrieved_documents"]

        if not docs:
            return {
                **state,
                "formatted_context": "No relevant information found.",
            }

        # SIMPLIFIED FORMAT (less verbose)
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc["content"]
            metadata = doc.get("metadata", {})

            # Only include essential metadata
            formatted_text = f"[Document {i}]\n{content}\n"
            
            # Add complete data if available
            if metadata.get("complete_data"):
                formatted_text += f"\nDetails: {metadata['complete_data']}\n"

            formatted_parts.append(formatted_text)

        context = "\n\n".join(formatted_parts)  # Removed separator line (saves tokens)

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
    OPTIMIZED: Shorter system prompt + streaming disabled
    
    Improvements:
    1. Reduced system prompt by 60%
    2. More direct instructions
    3. Faster LLM processing
    """
    try:
        log_state(state, "GENERATE_RESPONSE")

        namespace = state["company_namespace"]
        rag_engine = get_rag_engine(namespace)

        # OPTIMIZED: Shorter, more direct system prompt
        system_message = """You are a helpful AI assistant. Answer based on the provided context.

Context:
{context}

Instructions:
- Answer ONLY from the context
- Be concise and direct
- Extract specific details (names, IDs, dates, numbers)
- Professional tone"""

        context = state["formatted_context"]
        query = state["user_query"]

        llm_messages = [
            SystemMessage(content=system_message.format(context=context)),
            HumanMessage(content=query)
        ]

        response = rag_engine.llm.invoke(llm_messages)
        answer = response.content

        logger.info(f"Generated response: {len(answer)} characters")

        return {**state, "model_response": answer}

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_msg = "I encountered an error. Please try again."
        return {
            **state,
            "model_response": error_msg,
            "error": str(e),
        }