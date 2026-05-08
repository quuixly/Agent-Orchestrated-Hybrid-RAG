from typing import List, Dict, Any
import re
import uuid

from .embedding_model import EmbeddingModel
from .knowledge_database import KnowledgeDatabase


class HybridRAG:
    def __init__(self, knowledge_database: KnowledgeDatabase) -> None:
        self.__knowledge_database = knowledge_database

    def process_documents(self, collection_name: str, documents: List[str]) -> bool:
        if not documents:
            return False

        all_processed_chunks = []

        for idx, document_text in enumerate(documents):
            doc_id = int(uuid.uuid4().int & (1 << 63) - 1)

            document_chunks = self.__chunk_by_sentences_with_context(
                document_text=document_text,
                doc_id=doc_id,
                min_chars=600,
                context_sentences=3
            )

            all_processed_chunks.extend(document_chunks)

        if not all_processed_chunks:
            return False

        return self.__knowledge_database.insert(collection_name, all_processed_chunks)

    def drop_collection(self, collection_name: str) -> bool:
        return self.__knowledge_database.drop_collection(collection_name)

    def list_collections(self):
        return self.__knowledge_database.list_collections()

    def search(self, collection_name: str, query: str, limit: int = 5, rrf_reranker_k_param: int = 60) -> List[Dict[str, Any]]:
        results = self.__knowledge_database.search(collection_name, query, limit, rrf_reranker_k_param)

        return results

    def read_neighbor_chunk(self, collection_name: str, doc_id: int, current_seq: int, direction: str = "next") -> dict:
        return self.__knowledge_database.read_neighbor_chunk(collection_name, doc_id, current_seq, direction)


    def __split_into_sentences(self, text: str) -> List[str]:
        clean_text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?]) +(?=[A-ZĄĆĘŁŃÓŚŹŻ])', clean_text)

        return [s.strip() for s in sentences if s.strip()]

    def __chunk_by_sentences_with_context(self, document_text: str, doc_id: int, min_chars: int = 600, context_sentences: int = 3) -> List[Dict[str, Any]]:
        sentences = self.__split_into_sentences(document_text)
        processed_chunks = []
        current_seq_num = 0

        i = 0
        while i < len(sentences):
            current_block = []
            current_length = 0
            start_idx = i

            while i < len(sentences) and current_length < min_chars:
                current_block.append(sentences[i])
                current_length += len(sentences[i]) + 1
                i += 1

            end_idx = i

            current_text = " ".join(current_block).strip()
            if not current_text:
                continue

            prev_text = " ".join(sentences[max(0, start_idx - context_sentences) : start_idx])
            next_text = " ".join(sentences[end_idx : end_idx + context_sentences])

            formatted_text = ""
            if prev_text:
                formatted_text += f"<previous_chunk>\n{prev_text}\n</previous_chunk>\n\n"

            formatted_text += f"<current_chunk>\n{current_text}\n</current_chunk>\n\n"

            if next_text:
                formatted_text += f"<next_chunk>\n{next_text}\n</next_chunk>"

            processed_chunks.append({
                "text": formatted_text.strip(),
                "doc_id": doc_id,
                "seq_num": current_seq_num
            })

            current_seq_num += 1

        return processed_chunks


def setup_hybrid_rag(embedding_model_name: str = "sdadas/mmlw-retrieval-roberta-large"):
    embedding_model = EmbeddingModel(embedding_model_name)
    knowledge_database = KnowledgeDatabase(embedding_model)

    return HybridRAG(knowledge_database)