from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nisaa.helpers.short_term_memory import build_chat_history
from src.nisaa.helpers.short_term_memory import build_chat_history

ROLE = """You are an intelligent assistant for a social media platform with access to platform documentation and conversation history."""

GOAL = """
Objectives:
1. Answer questions using the provided context and chat history
2. Check content against moderation guidelines when needed
3. Provide accurate, concise responses (1-3 sentences)
4. Use conversation history to maintain context

Rules:
- Only use information from the provided context
- Never make up information
- Keep responses short and factual
- Make responses natural and human-readable
"""

FACTS_DUMP = """

{CONTEXT}

Current Question:
{MESSAGE}
"""

NEW_TEMPLATE = """
SYSTEM:
{ROLE}

{GOAL}

{FACTS_DUMP}
"""

def create_chat_prompt(thread_id, user_input, context=""):
    chat_history = build_chat_history(thread_id)

    facts = FACTS_DUMP.format(CONTEXT=context, MESSAGE=user_input)
    prompt_text = NEW_TEMPLATE.format(
        ROLE=ROLE,
        GOAL=GOAL,
        FACTS_DUMP=facts
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=prompt_text)
    ])

    return prompt, chat_history