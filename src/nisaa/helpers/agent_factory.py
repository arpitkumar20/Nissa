# import os
# from langchain.agents import create_agent

# from src.nisaa.helpers.memory_store import checkpointer

# OPENAI_MODEL=os.getenv('OPENAI_MODEL')

# def get_user_info(inputs: dict):
#     """
#     Returns a greeting message for the user based on the input dictionary.
#     Expects 'name' key in inputs, defaults to 'User'.
#     """
#     return {"info": f"Hello {inputs.get('name', 'User')}! I’m your assistant."}

# def create_chat_agent():
#     agent = create_agent(
#         OPENAI_MODEL,
#         [get_user_info],
#         checkpointer=checkpointer,
#     )
#     return agent


import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from src.nisaa.helpers.memory_store import checkpointer

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

def get_user_info(inputs: dict):
    """
    Retrieve basic user information and return a personalized greeting.

    Args:
        inputs (dict): A dictionary containing user details such as 'name'.

    Returns:
        dict: A response dictionary with a personalized message.
    """
    return {"info": f"Hello {inputs.get('name', 'User')}! I’m your assistant."}


def create_chat_agent(short_term_memory=True):
    """
    Create a chat agent with either short-term or long-term memory.

    Args:
        short_term_memory (bool): If True, use in-memory storage for session memory;
                                  if False, use persistent PostgreSQL memory.

    Returns:
        AgentExecutor: A fully initialized chat agent instance.
    """
    if short_term_memory:
        # Use in-memory (temporary) memory for short-term sessions
        agent = create_agent(
            OPENAI_MODEL,
            [get_user_info],
            checkpointer=InMemorySaver(),  # Short-term memory
        )
    else:
        # Use PostgreSQL or file-based persistent memory
        agent = create_agent(
            OPENAI_MODEL,
            [get_user_info],
            checkpointer=checkpointer,     # Long-term memory
        )

    return agent
