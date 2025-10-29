from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import logging

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    State schema for the RAG conversation graph

    Uses add_messages reducer to properly handle message history
    while maintaining additional metadata fields
    """

    # Core conversation state (with message reducer)
    messages: Annotated[List[AnyMessage], add_messages]

    # User identification
    user_phone_number: str
    thread_id: str

    # Query processing
    user_query: str  # Original query from user
    contextualized_query: str  # Reformulated standalone query

    # ID/Phone detection
    is_id_query: bool
    id_type: Optional[str]  # 'record_id', 'unique_id', 'phone', or None
    id_value: Optional[str]

    # Embedding & retrieval
    query_embedding: Optional[List[float]]
    retrieved_documents: List[Dict[str, Any]]
    search_type: str  # 'exact_match', 'semantic', 'hybrid'
    num_documents: int

    # Context formatting
    formatted_context: str

    # LLM response
    model_response: str

    # Metadata
    processing_time: float
    error: Optional[str]


def create_initial_state(user_query: str, user_phone_number: str) -> GraphState:
    """
    Create initial state for a new conversation turn

    Args:
        user_query: User's input message
        user_phone_number: User's phone number (thread ID)

    Returns:
        Initialized GraphState
    """
    return GraphState(
        messages=[],
        user_phone_number=user_phone_number,
        thread_id=str(user_phone_number),
        user_query=user_query,
        contextualized_query="",
        is_id_query=False,
        id_type=None,
        id_value=None,
        query_embedding=None,
        retrieved_documents=[],
        search_type="semantic",
        num_documents=0,
        formatted_context="",
        model_response="",
        processing_time=0.0,
        error=None,
    )


def log_state(state: GraphState, node_name: str):
    """Debug utility to log state at each node"""
    logger.info(f"[{node_name}] State snapshot:")
    logger.info(f"  - Thread: {state.get('thread_id')}")
    logger.info(f"  - Query: {state.get('user_query', '')[:100]}")
    logger.info(f"  - Messages: {len(state.get('messages', []))}")
    logger.info(f"  - ID Query: {state.get('is_id_query')}")
    logger.info(f"  - Documents: {state.get('num_documents')}")
    logger.info(f"  - Search Type: {state.get('search_type')}")

 