import time
import logging
from langgraph.graph import StateGraph, START, END
from src.nisaa.rag.node import (
    detect_id_or_phone,
    generate_embedding,
    retrieve_documents,
    format_context,
    generate_response
)

from src.nisaa.rag.state import GraphState, create_initial_state

logger = logging.getLogger(__name__)


def create_rag_graph():
    """
    Create the RAG conversation graph with nodes and edges
    """
    builder = StateGraph(GraphState)
    logger.info("Building RAG graph with NEW FLOW...")

    builder.add_node("detect_id", detect_id_or_phone)
    builder.add_node("embed", generate_embedding)
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node("format", format_context)
    builder.add_node("generate", generate_response)
    
    # Add edges
    builder.add_edge(START, "detect_id")
    builder.add_edge("detect_id", "embed")
    builder.add_edge("embed", "retrieve")
    builder.add_edge("retrieve", "format")
    builder.add_edge("format", "generate")
    builder.add_edge("generate", END)

    graph = builder.compile()
    return graph

def execute_rag_pipeline(
    user_query: str, 
    user_phone_number: str, 
    company_namespace: str, 
) -> dict:
    """
    Execute the RAG pipeline for a user query
    
    FIXED: Now properly passes company_namespace through state

    Args:
        user_query: User's input message
        user_phone_number: User's phone number (thread ID)
        company_namespace: Company-specific namespace for vector store
        
    Returns:
        Final state dictionary with response
    """
    start_time = time.time()

    try:
        initial_state = create_initial_state(
            user_query, 
            user_phone_number, 
            company_namespace
        )

        config = {"configurable": {"thread_id": str(user_phone_number)}}
        graph = create_rag_graph()

        final_state = graph.invoke(initial_state, config)

        processing_time = time.time() - start_time
        final_state["processing_time"] = processing_time

        used_history = final_state.get("needs_history", False)

        logger.info(f"Pipeline completed in {processing_time:.2f}s - History used: {used_history} - Search type: {final_state.get('search_type', 'unknown')} - Documents: {final_state.get('num_documents', 0)}")

        return final_state

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        processing_time = time.time() - start_time

        return {
            "model_response": "I apologize, but I encountered an error. Please try again.",
            "error": str(e),
            "processing_time": processing_time,
            "thread_id": str(user_phone_number),
            "company_namespace": company_namespace
        }