from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingModel:
    def __init__(self, embedding_model_name: str) -> None:
        self.embedding_model_name = embedding_model_name
        self.__model = SentenceTransformer(self.embedding_model_name)
        self.vector_dimension = self.__model.get_embedding_dimension()

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        if not documents:
            return []

        embeddings = self.__model.encode(documents,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True
                                         )

        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        if not query:
            return []

        embedding = self.__model.encode(query,
                                        convert_to_numpy=True,
                                        normalize_embeddings=True
                                        )

        return embedding.tolist()