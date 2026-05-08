from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import os
import uvicorn

from src.hybrid_rag import setup_hybrid_rag


RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))

app = FastAPI()
hybrid_rag = setup_hybrid_rag()

class DocumentPayload(BaseModel):
    documents: List[str]


@app.get("/collections")
def list_collections():
    """Returns a list of all active document collections."""

    return {
        "collections":
            hybrid_rag.list_collections()
    }


@app.post("/collections/{collection_name}/documents")
def add_documents(collection_name: str, payload: DocumentPayload):
    """Processes and inserts a list of documents into a specific collection."""

    try:
        success = hybrid_rag.process_documents(collection_name, payload.documents)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to insert documents into the database.")

        return {
            "message": f"Successfully added {len(payload.documents)} documents to '{collection_name}'."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_name}/search")
def search(collection_name: str, query: str, limit: int = 5, rrf_reranker_k_param: int = 60):
    """Searches a collection for documents matching the query."""

    try:
        results = hybrid_rag.search(collection_name, query, limit, rrf_reranker_k_param)

        return {
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_name}/documents/{doc_id}/chunks/{current_seq}/neighbor")
def get_neighbor_chunk(
    collection_name: str,
    doc_id: int,
    current_seq: int,
    direction: str
):
    """
    Retrieves the neighboring chunk (next or previous) for a given document and sequence number.
    Used by the Agent to 'turn the page' in a specific medical document.
    """
    if direction not in ["next", "prev", "previous"]:
         raise HTTPException(status_code=400, detail="Direction must be 'next' or 'prev'.")

    try:
        normalized_direction = "prev" if direction == "previous" else direction

        result = hybrid_rag.read_neighbor_chunk(
            collection_name=collection_name,
            doc_id=doc_id,
            current_seq=current_seq,
            direction=normalized_direction
        )

        if not result:
            raise HTTPException(status_code=404, detail="Neighbor chunk not found (end or beginning of document).")

        return {
            "chunk": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
def drop_collection(collection_name: str):
    """Deletes an entire collection."""

    try:
        success = hybrid_rag.drop_collection(collection_name)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to drop collection. It might not exist.")

        return {
            "message": f"Collection '{collection_name}' dropped successfully."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT)