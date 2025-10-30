"""
LangGraph Node Implementations for Agentic RAG Pipeline
Each node performs a specific step in the conversation workflow
"""

import time
import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.nisaa.controllers.retrival_controller import HybridRAGQueryEngine
from src.nisaa.graphs.state import GraphState, log_state
from src.nisaa.helpers.postgres_store import PostgresMemoryStore

logger = logging.getLogger(__name__)

# Initialize RAG engine (singleton)
_rag_engine = None


def get_rag_engine() -> HybridRAGQueryEngine:
    """Get or create the RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        logger.info("Initializing Hybrid RAG Engine...")
        _rag_engine = HybridRAGQueryEngine(
            top_k=5, similarity_threshold=0.65, temperature=0.7, max_tokens=2000
        )
        logger.info("RAG Engine initialized successfully")
    return _rag_engine


# ============================================================================
# NODE 1: Load Chat History
# ============================================================================


def load_chat_history(state: GraphState) -> GraphState:
    """
    Load recent conversation history from PostgreSQL
    Adds historical messages to state for contextualization
    """
    try:
        log_state(state, "LOAD_CHAT_HISTORY")
        thread_id = state["thread_id"]

        # Initialize memory store
        memory = PostgresMemoryStore(thread_id)

        # Get last 10 messages (5 conversation turns)
        history_messages = memory.get_langchain_messages(n=10)

        logger.info(f"Loaded {len(history_messages)} messages from history")

        # Add user's current query as HumanMessage
        current_message = HumanMessage(content=state["user_query"])

        # Combine history + current message
        all_messages = history_messages + [current_message]

        return {**state, "messages": all_messages}

    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        # Continue with just the current query
        return {**state, "messages": [HumanMessage(content=state["user_query"])]}


# ============================================================================
# NODE 2: Contextualize Query
# ============================================================================


def contextualize_query(state: GraphState) -> GraphState:
    """
    Reformulate query based on chat history to make it standalone
    If no history, use original query
    """
    try:
        log_state(state, "CONTEXTUALIZE_QUERY")

        messages = state["messages"]
        user_query = state["user_query"]

        # If no history (only current message), use original query
        if len(messages) <= 1:
            logger.info("No history - using original query")
            return {**state, "contextualized_query": user_query}

        # Build contextualization prompt
        rag_engine = get_rag_engine()

        context_prompt = """Given the conversation history, reformulate the last user question to be a standalone question that can be understood without the conversation context.

Conversation History:
"""
        # Add history (excluding last message which is current query)
        for msg in messages[:-1]:
            if isinstance(msg, HumanMessage):
                context_prompt += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context_prompt += f"Assistant: {msg.content}\n"

        context_prompt += f"\nCurrent Question: {user_query}\n\nStandalone Question:"

        # Use LLM to reformulate
        standalone_question = rag_engine.llm.invoke(context_prompt).content

        logger.info(f"Contextualized: '{user_query}' -> '{standalone_question}'")

        return {**state, "contextualized_query": standalone_question}

    except Exception as e:
        logger.error(f"Error contextualizing query: {e}")
        # Fallback to original query
        return {**state, "contextualized_query": state["user_query"]}


# ============================================================================
# NODE 3: Detect ID/Phone
# ============================================================================


def detect_id_or_phone(state: GraphState) -> GraphState:
    """
    Detect if query contains ID or phone number
    Uses regex patterns from HybridRAGQueryEngine
    """
    try:
        log_state(state, "DETECT_ID_PHONE")

        query = state["contextualized_query"]
        rag_engine = get_rag_engine()

        # Use RAG engine's ID detection
        id_info = rag_engine.detect_id_in_query(query)

        logger.info(f"ID Detection: {id_info}")

        return {
            **state,
            "is_id_query": id_info["is_id_query"],
            "id_type": id_info["id_type"],
            "id_value": id_info["id_value"],
        }

    except Exception as e:
        logger.error(f"Error detecting ID/phone: {e}")
        return {**state, "is_id_query": False, "id_type": None, "id_value": None}


# ============================================================================
# NODE 4: Generate Embedding
# ============================================================================


def generate_embedding(state: GraphState) -> GraphState:
    """
    Generate embedding for the contextualized query
    Uses OpenAI text-embedding-3-small
    """
    try:
        log_state(state, "GENERATE_EMBEDDING")

        query = state["contextualized_query"]
        rag_engine = get_rag_engine()

        # Generate embedding
        embedding = rag_engine.embeddings.embed_query(query)

        logger.info(f"Generated embedding of dimension {len(embedding)}")

        return {**state, "query_embedding": embedding}

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return {**state, "query_embedding": None}


# ============================================================================
# NODE 5: Retrieve Documents (HYBRID)
# ============================================================================


def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve documents using hybrid search
    - Exact match for ID/phone queries
    - Semantic search for natural language
    """
    try:
        log_state(state, "RETRIEVE_DOCUMENTS")

        rag_engine = get_rag_engine()
        query = state["contextualized_query"]

        # Use hybrid retrieval from RAG engine
        docs, search_type = rag_engine.hybrid_retrieve(query, top_k=5)

        # Convert docs to serializable format
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

 
# ============================================================================
# NODE 6: Format Context
# ============================================================================


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

            # Build document section
            formatted_text = f"[Document {i}]\n"

            # Add metadata
            if metadata.get("record_id"):
                formatted_text += f"Record ID: {metadata['record_id']}\n"
            if metadata.get("unique_id"):
                formatted_text += f"Unique ID: {metadata['unique_id']}\n"
            if metadata.get("full_name"):
                formatted_text += f"Name: {metadata['full_name']}\n"
            if metadata.get("phone"):
                formatted_text += f"Phone: {metadata['phone']}\n"

            formatted_text += f"\nContent:\n{content}\n"

            # Add complete structured data if available
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


# ============================================================================
# NODE 7: Generate Response
# ============================================================================


def generate_response(state: GraphState) -> GraphState:
    """
    Generate final AI response using LLM with RAG context
    Includes conversation history and retrieved documents
    """
    try:
        log_state(state, "GENERATE_RESPONSE")

        rag_engine = get_rag_engine()

        # Build prompt with context and history
        system_message = """You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately and completely.

Context from Knowledge Base:
{context}

Instructions:
- Use ALL relevant information from the context
- When asked about IDs, names, or phone numbers, extract them directly
- Be comprehensive and cite specific field names when relevant
- If information is not in the context, clearly state that
- Maintain conversational tone based on chat history"""

        context = state["formatted_context"]
        query = state["contextualized_query"]
        messages = state["messages"]

        # Build conversation for LLM
        llm_messages = [SystemMessage(content=system_message.format(context=context))]

        # Add history (excluding last message which is current query)
        for msg in messages[:-1]:
            llm_messages.append(msg)

        # Add current query
        llm_messages.append(HumanMessage(content=query))

        # Generate response
        response = rag_engine.llm.invoke(llm_messages)
        answer = response.content

        logger.info(f"Generated response: {len(answer)} characters")

        # Update messages with AI response
        updated_messages = messages + [AIMessage(content=answer)]

        return {**state, "model_response": answer, "messages": updated_messages}

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_msg = "I apologize, but I encountered an error processing your request. Please try again."
        return {
            **state,
            "model_response": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)],
            "error": str(e),
        }


# ============================================================================
# NODE 8: Save to Memory
# ============================================================================


def save_to_memory(state: GraphState) -> GraphState:
    """
    Save conversation turn to PostgreSQL
    Stores contextualized query (user) and AI response
    """
    try:
        log_state(state, "SAVE_TO_MEMORY")

        thread_id = state["thread_id"]
        contextualized_query = state["contextualized_query"]
        model_response = state["model_response"]

        # Initialize memory store
        memory = PostgresMemoryStore(thread_id)

        # Save user message (contextualized version)
        memory.put("user", contextualized_query)

        # Save assistant response
        memory.put("assistant", model_response)

        logger.info(f"Saved conversation turn to PostgreSQL for thread {thread_id}")

        return state

    except Exception as e:
        logger.error(f"Error saving to memory: {e}")
        # Don't fail the whole pipeline if memory save fails
        return {**state, "error": f"Memory save error: {str(e)}"}
 