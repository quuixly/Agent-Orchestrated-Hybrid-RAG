import os
import requests

RAG_URL = os.getenv("RAG_URL", "http://localhost:8000")


def list_collections():
    resp = requests.get(f"{RAG_URL}/collections")
    resp.raise_for_status()

    return resp.json()

def search(collection_name: str, query: str, limit: int = 5, rrf_reranker_k_param: int = 60):
    resp = requests.get(
        f"{RAG_URL}/collections/{collection_name}/search",
        params={
            "query": query,
            "limit": limit,
            "rrf_reranker_k_param": rrf_reranker_k_param
        }
    )
    resp.raise_for_status()

    return resp.json()