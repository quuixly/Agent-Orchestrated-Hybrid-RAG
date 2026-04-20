from typing import List

from embedding_model import EmbeddingModel
from knowledge_database import KnowledgeDatabase


class HybridRAG:
    def __init__(self, knowledge_database: KnowledgeDatabase):
        self.__knowledge_database = knowledge_database

    def process_documents(self, collection_name: str, documents: List[str]) -> bool:
        return self.__knowledge_database.insert(collection_name, documents)

    def drop_collection(self, collection_name: str) -> bool:
        return self.__knowledge_database.drop_collection(collection_name)

    def list_collections(self):
        return self.__knowledge_database.list_collections()

    def search(self, collection_name: str, query: str, limit: int = 5, rrf_reranker_k_param: int = 60) -> List[str]:
        results = self.__knowledge_database.search(collection_name, query, limit, rrf_reranker_k_param)

        return results


def setup_hybrid_rag(embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = EmbeddingModel(embedding_model_name)
    knowledge_database = KnowledgeDatabase(embedding_model)

    return HybridRAG(knowledge_database)
