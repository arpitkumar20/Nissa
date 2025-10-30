import time
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from src.nisaa.graphs.node import contextualize_query, detect_id_or_phone, format_context, generate_embedding, generate_response, load_chat_history, retrieve_documents, save_to_memory

from src.nisaa.graphs.state import GraphState, create_initial_state
from src.nisaa.helpers.db import DB_URI

logger = logging.getLogger(__name__)

# ============================================================================
# Graph Builder
# ============================================================================


def create_rag_graph():
    """
    Create and compile the RAG graph workflow

    Returns:
        Compiled StateGraph ready for invocation
    """

    # Initialize graph builder
    builder = StateGraph(GraphState)

    # Add nodes
    logger.info("Building RAG graph...")

    builder.add_node("load_history", load_chat_history)
    builder.add_node("contextualize", contextualize_query)
    builder.add_node("detect_id", detect_id_or_phone)
    builder.add_node("embed", generate_embedding)
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node("format", format_context)
    builder.add_node("generate", generate_response)
    builder.add_node("save_memory", save_to_memory)

    # Define edges (linear flow)
    builder.add_edge(START, "load_history")
    builder.add_edge("load_history", "contextualize")
    builder.add_edge("contextualize", "detect_id")
    builder.add_edge("detect_id", "embed")
    builder.add_edge("embed", "retrieve")
    builder.add_edge("retrieve", "format")
    builder.add_edge("format", "generate")
    builder.add_edge("generate", "save_memory")
    builder.add_edge("save_memory", END)

    # Setup PostgreSQL checkpointer for persistence
    try:
        checkpointer = PostgresSaver.from_conn_string(DB_URI)
        checkpointer.setup()
        logger.info("PostgreSQL checkpointer initialized")
    except Exception as e:
        logger.warning(f"Checkpointer setup failed: {e}. Using memory checkpointer.")
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    # Compile graph
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("✅ RAG Graph compiled successfully")

    return graph


# ============================================================================
# Graph Execution Wrapper
# ============================================================================


def execute_rag_pipeline(
    user_query: str, user_phone_number: str, stream: bool = False
) -> dict:
    """
    Execute the RAG pipeline for a user query

    Args:
        user_query: User's input message
        user_phone_number: User's phone number (thread ID)
        stream: Whether to stream responses (future enhancement)

    Returns:
        Final state dictionary with response
    """
    start_time = time.time()

    try:
        # Create initial state
        initial_state = create_initial_state(user_query, user_phone_number)

        # Configuration with thread ID
        config = {"configurable": {"thread_id": str(user_phone_number)}}

        # Create and invoke graph
        graph = create_rag_graph()

        logger.info(f"Executing RAG pipeline for thread: {user_phone_number}")

        # Invoke graph
        final_state = graph.invoke(initial_state, config)

        # Calculate processing time
        processing_time = time.time() - start_time
        final_state["processing_time"] = processing_time

        logger.info(f"✅ Pipeline completed in {processing_time:.2f}s")

        return final_state

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}", exc_info=True)
        processing_time = time.time() - start_time

        return {
            "model_response": "I apologize, but I encountered an error. Please try again.",
            "error": str(e),
            "processing_time": processing_time,
            "thread_id": str(user_phone_number),
        }


# ============================================================================
# Streaming Support (Future Enhancement)
# ============================================================================


async def stream_rag_pipeline(user_query: str, user_phone_number: str):
    """
    Stream responses from the RAG pipeline
    Useful for real-time UI updates
    """
    initial_state = create_initial_state(user_query, user_phone_number)

    config = {"configurable": {"thread_id": str(user_phone_number)}}

    graph = create_rag_graph()

    async for event in graph.astream(initial_state, config):
        for node_name, node_state in event.items():
            if node_name == "generate":
                # Stream the response
                yield {
                    "type": "response",
                    "content": node_state.get("model_response", ""),
                }
            else:
                # Stream progress updates
                yield {"type": "progress", "node": node_name, "status": "completed"}