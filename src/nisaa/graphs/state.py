"""
Graph State Definition - FIXED VERSION

Key Changes:
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
    
    UPDATED: Added fields for new uncertainty-based retry flow

    Uses add_messages reducer to properly handle message history
    while maintaining additional metadata fields
    """

    messages: Annotated[List[AnyMessage], add_messages]

    user_phone_number: str
    thread_id: str
    company_namespace: str

    user_query: str
    contextualized_query: str

    is_id_query: bool
    id_type: Optional[str]
    id_value: Optional[str]

    query_embedding: Optional[List[float]]
    retrieved_documents: List[Dict[str, Any]]
    search_type: str
    num_documents: int

    formatted_context: str
    model_response: str

    needs_history: bool
    history_messages: List[AnyMessage]

    processing_time: float
    error: Optional[str]


def create_initial_state(
    user_query: str, 
    user_phone_number: str, 
    company_namespace: str
) -> GraphState:
    """
    Create initial state for a new conversation turn
    
    FIXED: Properly initializes all fields including new ones

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
        messages=[],
        
        user_phone_number=user_phone_number,
        thread_id=str(user_phone_number),
        company_namespace=company_namespace,
        
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
        
        needs_history=False,
        history_messages=[],
        
        processing_time=0.0,
        error=None,
    )


def log_state(state: GraphState, node_name: str):
    """
    Debug utility to log state at each node
    
    UPDATED: Includes new fields in logging
    """

    logger.info(f"[{node_name}] State snapshot:  - Thread: {state.get('thread_id')}  - Namespace: {state.get('company_namespace')} - Query: {state.get('user_query', '')[:100]}  - Messages: {len(state.get('messages', []))}  - ID Query: {state.get('is_id_query')} - Documents: {state.get('num_documents')}  - Documents: {state.get('num_documents')}  - Search Type: {state.get('search_type')}  - Needs History: {state.get('needs_history', False)}")