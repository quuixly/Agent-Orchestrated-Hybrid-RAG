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


def get_neighbor_chunk(collection_name: str, doc_id: int, current_seq: int, direction: str = "next"):
    url = f"{RAG_URL}/collections/{collection_name}/documents/{doc_id}/chunks/{current_seq}/neighbor"

    resp = requests.get(
        url,
        params={"direction": direction}
    )

    if resp.status_code == 404:
        return None

    resp.raise_for_status()

    return resp.json()