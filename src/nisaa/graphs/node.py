import os
from openai import OpenAI
from typing import Optional
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from src.nisaa.helpers.logger import logger
from src.nisaa.graphs.state import ChatBotState
from src.nisaa.prompt.chat_bot import create_chat_prompt
from nisaa.helpers.top_k_fetch import query_pinecone_topk
from src.nisaa.helpers.short_term_memory import create_chat_agent

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL=os.getenv('OPENAI_MODEL')
LLM_TEMPERATURE=os.getenv('LLM_TEMPERATURE')
PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE')
PINECONE_TOP_K = os.getenv('PINECONE_TOP_K')

client = OpenAI(api_key=OPENAI_API_KEY)
agent = create_chat_agent()

def embed_query(state: ChatBotState) -> dict:
    try:
        user_query = state.get("user_query")
        if not user_query:
            raise HTTPException(status_code=400, detail="Missing user_query in state")

        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=user_query
        )
        embedding = response.data[0].embedding

        state["embedding"] = embedding
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

def top_k_finding(
    state: ChatBotState,
    top_k: int = int(PINECONE_TOP_K),
    namespace: Optional[str] = PINECONE_NAMESPACE,
    similarity_threshold: float = 0.3) -> dict:

    try:
        embedding = state.get("embedding")
        if not embedding:
            raise HTTPException(status_code=400, detail="Missing embedding in state")

        results = query_pinecone_topk(
            query_vector=embedding,
            top_k=top_k,
            namespace=namespace,
            similarity_threshold=similarity_threshold
        )

        state["neighbors"] = results
        logger.info(f"Top-K search returned {len(results)} results")
        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone top-K search failed: {str(e)}")

def prompt_builder_node(state: dict) -> dict:
    try:
        logger.info("Building prompt from context and user message...")

        context = "\n".join(
            neighbor.get('source_text', '').strip()
            for neighbor in state.get('neighbors', [])
            if neighbor.get('source_text')
        )

        user_message = state.get('user_query')
        phone = state.get('user_phone_number', 'unknown_thread')

        if not user_message:
            raise ValueError("User message is missing from state.")

        prompt, chat_history = create_chat_prompt(phone, user_message, context)

        state["prompt"] = prompt
        state["history"] = chat_history

        logger.info(f"Prompt built successfully")
        return state

    except Exception as e:
        logger.error(f"Prompt building failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prompt building failed: {str(e)}")
    

def invoke_model_node(state: dict) -> dict:
    try:
        prompt = state.get("prompt")
        history = state.get("history", [])

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt not found in state")

        if hasattr(prompt, "format_messages"):
            prompt = prompt.format_messages(history=history)
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        raw_response = llm.invoke(prompt)
        state["model_response"] = raw_response.content.strip()
        logger.info("Model invoked successfully.")
        return state

    except Exception as e:
        logger.error(f"Chat response failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat response failed: {str(e)}")