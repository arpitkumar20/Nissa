import os
from typing import Optional
from fastapi import HTTPException
from openai import OpenAI
from src.nisaa.graphs.state import ChatBotState
from src.nisaa.services.top_k_fetch import query_pinecone_topk

from dotenv import load_dotenv
load_dotenv()

from src.nisaa.helpers.logger import logger

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    top_k: int = 5,
    namespace: Optional[str] = None,
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

        state["pinecone_results"] = results
        logger.info(f"Top-K search returned {len(results)} results")
        print(">>>>>>>>state>>>>>>>>>>>",state)
        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone top-K search failed: {str(e)}")
