"""
Graph State Definition - FIXED VERSION

✅ Key Changes:
1. Added needs_history field for uncertainty tracking
2. Added history_messages field for conditional history loading
3. Proper initialization of all fields
4. Better typing and documentation
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import logging

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    State schema for the RAG conversation graph
    
    ✅ UPDATED: Added fields for new uncertainty-based retry flow

    Uses add_messages reducer to properly handle message history
    while maintaining additional metadata fields
    """

    # Core conversation state (with message reducer)
    messages: Annotated[List[AnyMessage], add_messages]

    # User identification
    user_phone_number: str
    thread_id: str
    company_namespace: str  # ✅ CRITICAL: Used for vector store namespace

    # Query processing
    user_query: str  # Original query from user
    contextualized_query: str  # Reformulated standalone query (deprecated in new flow)

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

    # ✅ NEW: Uncertainty detection & retry
    needs_history: bool  # True if response was uncertain
    history_messages: List[AnyMessage]  # Loaded only if needed

    # Metadata
    processing_time: float
    error: Optional[str]


def create_initial_state(
    user_query: str, 
    user_phone_number: str, 
    company_namespace: str
) -> GraphState:
    """
    Create initial state for a new conversation turn
    
    ✅ FIXED: Properly initializes all fields including new ones

    Args:
        user_query: User's input message
        user_phone_number: User's phone number (thread ID)
        company_namespace: Company-specific namespace for vector store

    Returns:
        Initialized GraphState with all required fields
    """
    if not company_namespace:
        raise ValueError("company_namespace is required in initial state")
    
    logger.info(f"Creating initial state for namespace: {company_namespace}")
    
    return GraphState(
        # Core conversation
        messages=[],
        
        # User identification
        user_phone_number=user_phone_number,
        thread_id=str(user_phone_number),
        company_namespace=company_namespace,
        
        # Query processing
        user_query=user_query,
        contextualized_query="",  # Deprecated in new flow
        
        # ID/Phone detection
        is_id_query=False,
        id_type=None,
        id_value=None,
        
        # Embedding & retrieval
        query_embedding=None,
        retrieved_documents=[],
        search_type="semantic",
        num_documents=0,
        
        # Context formatting
        formatted_context="",
        
        # LLM response
        model_response="",
        
        # ✅ NEW: Uncertainty & retry
        needs_history=False,
        history_messages=[],
        
        # Metadata
        processing_time=0.0,
        error=None,
    )


def log_state(state: GraphState, node_name: str):
    """
    Debug utility to log state at each node
    
    ✅ UPDATED: Includes new fields in logging
    """
    logger.info(f"[{node_name}] State snapshot:")
    logger.info(f"  - Thread: {state.get('thread_id')}")
    logger.info(f"  - Namespace: {state.get('company_namespace')}")
    logger.info(f"  - Query: {state.get('user_query', '')[:100]}")
    logger.info(f"  - Messages: {len(state.get('messages', []))}")
    logger.info(f"  - ID Query: {state.get('is_id_query')}")
    logger.info(f"  - Documents: {state.get('num_documents')}")
    logger.info(f"  - Search Type: {state.get('search_type')}")
    logger.info(f"  - Needs History: {state.get('needs_history', False)}")