from typing import List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embedding_model import EmbeddingModel
from .knowledge_database import KnowledgeDatabase
from .llm_client import BielikChatModel


doc2query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
Jesteś ekspertem w dziedzinie przetwarzania języka naturalnego (NLP), systemów wyszukiwania informacji (Information Retrieval) oraz techniki doc2query. 
Twoim zadaniem jest analiza tekstu, podział go na semantyczne fragmenty (chunki) oraz wygenerowanie dla każdego z nich trafnych pytań.

Otrzymasz dwa bloki tekstu:
<previous_window> - Zawiera tekst poprzedzający. Używaj go GŁÓWNIE jako kontekstu (np. do zrozumienia, kogo dotyczą zaimki takie jak "on/ona", do rozszyfrowania skrótów lub pojęcia ogólnego tematu).
<current_window> - Zawiera główny tekst, który musisz przetworzyć.

INSTRUKCJE:
1. Kontekstualizacja: Przeczytaj <previous_window>, aby zrozumieć kontekst wydarzeń w <current_window>.
2. Obsługa uciętych fragmentów (BARDZO WAŻNE): 
   - Ucięty POCZĄTEK: Jeśli tekst na samym początku <current_window> jest ucięty (zaczyna się w połowie zdania), odszukaj jego pierwszą część na końcu <previous_window>, połącz je w jedną, logiczną całość i uwzględnij w bieżącej analizie.
   - Ucięty KONIEC: Jeśli tekst na samym końcu <current_window> jest ucięty (urwany w połowie zdania), CAŁKOWICIE GO ZIGNORUJ. Nie twórz dla niego żadnego chunku ani pytań. Zostanie on przetworzony dopiero w następnej iteracji.
3. Podział semantyczny (Chunking): Podziel przygotowany tekst (pełne zdania z current_window + ewentualnie sklejone zdanie początkowe) na logiczne "chunki semantyczne". Jeden chunk to spójny fragment tekstu (zazwyczaj 1-4 zdania), który skupia się na jednej, konkretnej myśli.
4. Generowanie pytań (doc2query): Dla każdego wyodrębnionego chunku wygeneruj od 2 do 4 różnych pytań, na które ten konkretny fragment udziela idealnej i wyczerpującej odpowiedzi. Pytania muszą być w języku polskim.

OGRANICZENIA:
- Twórz chunki TYLKO z kompletnych zdań. Jeśli na końcu <current_window> znajduje się niedokończone zdanie, bezwzględnie pomiń je w generowanym formacie JSON.
- Pytania muszą być bardzo precyzyjne. Jeśli chunk mówi "W 1990 roku wydał swoją pierwszą książkę", użyj wiedzy z <previous_window>, aby wiedzieć, kim jest autor, i wygeneruj pytanie "W którym roku [Imię i Nazwisko autora] wydał pierwszą książkę?" zamiast "Kiedy wydał książkę?".
- Odpowiedź MUSI być zwrócona ściśle w formacie JSON. Nie dodawaj żadnego tekstu pobocznego, formatowania markdown (takiego jak ```json) ani wyjaśnień. Zwróć sam, czysty obiekt JSON.

FORMAT WYJŚCIOWY JSON:
{{
  "results": [
    {{
      "chunk": "tekst pierwszego semantycznego fragmentu...",
      "queries": [
        "Pierwsze pytanie, na które ten chunk odpowiada?",
        "Drugie pytanie, na które ten chunk odpowiada?"
      ]
    }},
    {{
      "chunk": "tekst drugiego semantycznego fragmentu...",
      "queries": [ ... ]
    }}
  ]
}}
"""),
    ("human", "<previous_window>\n{previous_window}\n</previous_window>\n<current_window>\n{current_window}\n</current_window>")
])


class HybridRAG:
    def __init__(self, knowledge_database: KnowledgeDatabase, llm_client: BaseChatModel) -> None:
        self.__knowledge_database = knowledge_database
        self.__llm_client = llm_client
        self.__json_output_parser = JsonOutputParser()

    def process_documents(self, collection_name: str, documents: List[str]) -> bool:
        chunks = self.__sliding_window(documents, 777)

        return self.__knowledge_database.insert(collection_name, chunks)

    def drop_collection(self, collection_name: str) -> bool:
        return self.__knowledge_database.drop_collection(collection_name)

    def list_collections(self):
        return self.__knowledge_database.list_collections()

    def search(self, collection_name: str, query: str, limit: int = 5, rrf_reranker_k_param: int = 60) -> List[str]:
        results = self.__knowledge_database.search(collection_name, query, limit, rrf_reranker_k_param)

        return results

    def __sliding_window(self, documents: List[str], window_size: int) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=window_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        results = []

        for document in documents:
            windows = splitter.split_text(document)
            previous_window = ""

            for current_window in windows:
                message = doc2query_prompt_template.format_messages(
                    previous_window=previous_window,
                    current_window=current_window
                )

                try:
                    response = self.__llm_client.invoke(message)
                    raw_text = response.text

                    parsed_json = self.__json_output_parser.parse(raw_text)
                    ready_chunks = self.__format_results(parsed_json)
                    results.extend(ready_chunks)

                except Exception as e:
                    pass

                previous_window = current_window

        return results

    def __format_results(self, parsed_json: Dict[str, Any]) -> List[str]:
        formatted_chunks = []
        results_list = parsed_json.get("results", [])

        for item in results_list:
            queries_joined = " ".join(item.get("queries", []))
            chunk_text = item.get("chunk", "")

            formatted_string = f"<queries>{queries_joined}</queries>\n{chunk_text}"
            formatted_chunks.append(formatted_string)

        return formatted_chunks


def setup_hybrid_rag(embedding_model_name: str = "sdadas/mmlw-retrieval-roberta-large"):
    embedding_model = EmbeddingModel(embedding_model_name)
    knowledge_database = KnowledgeDatabase(embedding_model)
    llm_client = BielikChatModel(temperature=0)

    return HybridRAG(knowledge_database, llm_client)