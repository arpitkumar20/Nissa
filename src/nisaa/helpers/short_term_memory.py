import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from src.nisaa.helpers.postgres_store import PostgresMemoryStore

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
DB_URI = os.getenv("DB_URI")
if not DB_URI and OPENAI_MODEL:
    raise ValueError("DB_URI and OPENAI_MODEL environment variable is not set!")

def get_user_info(inputs: dict):
    """
    Retrieve basic user information and return a personalized greeting.

    Args:
        inputs (dict): A dictionary containing user details such as 'name'.

    Returns:
        dict: A response dictionary with a personalized message.
    """
    return {"info": f"Hello {inputs.get('name', 'User')}! I’m your assistant."}


def create_chat_agent():
    """
    Create a chat agent with either short-term or long-term memory.

    Args:
        short_term_memory (bool): If True, use in-memory storage for session memory;
                                  if False, use persistent PostgreSQL memory.

    Returns:
        AgentExecutor: A fully initialized chat agent instance.
    """
    
    agent = create_agent(
        OPENAI_MODEL,
        [get_user_info],
        checkpointer=InMemorySaver(),
    )
    return agent

def build_chat_history(thread_id):
    store = PostgresMemoryStore(thread_id)
    last_messages = store.get_last_n_messages(n=20)

    chat_history = []
    for msg in last_messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history
