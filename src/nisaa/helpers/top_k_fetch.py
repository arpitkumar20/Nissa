import os
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import json
from src.nisaa.helpers.logger import logger
from src.nisaa.services.pinecone_client import pinecone_index

def clean_text(text: str) -> str:
    return text.strip()

def query_pinecone_topk(
    query_vector: List[float],
    top_k: int = 5,
    namespace: Optional[str] = None,
    similarity_threshold: float = 0.3
) -> List[Dict]:

    print("================================top k namespace for query ============================================================")
    namespaces_file="web_info/web_info.json"
    if not os.path.exists(namespaces_file):
        raise FileNotFoundError(f"JSON file not found: {namespaces_file}")

    with open(namespaces_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        namespace = data.get("namespace")
    print(namespace)
    print("============================================================================================")

    logger.info(f"Querying Pinecone index with top_k={top_k}, namespace={namespace}")

    query_vec_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    query_vec_np = normalize(query_vec_np, axis=1)

    response = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        include_values=True
    )

    matches = response.get("matches", [])
    results = []

    for match in matches:
        match_values = match.get("values")
        if not match_values:
            continue

        match_vec_np = np.array(match_values, dtype=np.float32).reshape(1, -1)
        match_vec_np = normalize(match_vec_np, axis=1)

        similarity = float(cosine_similarity(query_vec_np, match_vec_np)[0][0])
        if similarity < similarity_threshold:
            continue

        metadata = match.get("metadata", {})
        cleaned_metadata = {k: clean_text(str(v)) for k, v in metadata.items()}

        results.append({
            "id": clean_text(match.get("id", "")),
            "score": similarity,
            **cleaned_metadata
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Found {len(results)} matches above threshold {similarity_threshold}")
    return results