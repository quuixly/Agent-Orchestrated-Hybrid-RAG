import logging
from typing import List
from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker

from .embedding_model import EmbeddingModel


class KnowledgeDatabase:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.__client = MilvusClient(uri="http://localhost:19530")
        self.__embedding_model = embedding_model

    def create_collection(self, collection_name: str) -> bool:
        if self.__client.has_collection(collection_name):
            logging.info(f"Collection {collection_name} already exists")
            return False

        schema = self.__create_schema()
        index_params = self.__create_index_params()

        # Index will be created automatically
        try:
            self.__client.create_collection(collection_name = collection_name,
                                            schema = schema,
                                            index_params=index_params)
            return True
        except Exception as e:
            return False

    def list_collections(self) -> List[str]:
        return self.__client.list_collections()

    def __create_schema(self):
        schema = self.__client.create_schema()

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="dense",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.__embedding_model.vector_dimension
        )
        schema.add_field(
            field_name="sparse",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # Automatically generate BM25 sparse vectors from text during insertion
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        return schema

    def __create_index_params(self):
        index_params = self.__client.prepare_index_params()

        index_params.add_index(
            field_name="dense",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        return index_params

    def drop_collection(self, collection_name: str) -> bool:
        try:
            if self.__client.has_collection(collection_name):
                self.__client.drop_collection(collection_name)
                return True

            return False
        except Exception as e:
            return False

    def insert(self, collection_name: str, texts: List[str]) -> None:
        if not self.__client.has_collection(collection_name):
            self.create_collection(collection_name)

        dense_vectors = self.__embedding_model.encode_documents(texts)
        data = [
            {
                "text": text,
                "dense": vector
            }
            for text, vector in zip(texts, dense_vectors)
        ]

        self.__client.insert(
            collection_name=collection_name,
            data=data
        )
        self.__client.flush(collection_name)

    def delete(self, collection_name: str, indices: List[int]) -> bool:
        try:
            self.__client.delete(collection_name=collection_name, pks=indices)
            return True
        except Exception as e:
            return False

    def search(self, collection_name: str, query: str, limit: int, rrf_reranker_k_param: int = 60) -> List[str]:
        if not self.__client.has_collection(collection_name):
            return []

        query_embedding = self.__embedding_model.encode_query(query)

        search_params_dense = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=limit
        )

        search_params_sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit
        )

        try:
            result = self.__client.hybrid_search(
                collection_name=collection_name,
                reqs=[search_params_dense, search_params_sparse],
                ranker=RRFRanker(k = rrf_reranker_k_param),
                limit=limit,
                output_fields=["text"]
            )

            if not result:
                return []

            return [hit['entity']['text'] for hit in result[0]]

        except Exception as e:
            logging.error(f"Hybrid search failed: {e}")
            return []