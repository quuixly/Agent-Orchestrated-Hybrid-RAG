from mcp.server.fastmcp import FastMCP
from pydantic import Field
import os
import logging
import time

from tools.hybrid_rag import list_collections, search, get_neighbor_chunk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("mcp-server")

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", 8000))

mcp = FastMCP("Environment", host=MCP_HOST, port=MCP_PORT)


def get_initial_collections(max_retries=150, delay=5):
    attempt = 0

    while attempt < max_retries:
        try:
            attempt += 1
            logger.info(f"Connecting to RAG (attempt {attempt}/{max_retries})...")

            response_data = list_collections()

            if isinstance(response_data, dict) and "collections" in response_data:
                cols = response_data["collections"]
            else:
                cols = response_data

            logger.info(f"Connected! Collections found: {cols}")
            return cols
        except Exception as e:
            logger.warning(f"RAG service not ready: {e}")
            time.sleep(delay)

    logger.error("Could not fetch collections. Starting with fallback description.")
    return []


@mcp.tool(
    name="search_ground_truth_database",
    description="Przeszukuje konkretną kolekcję bazy danych wiedzy (ground truth) za pomocą zapytania"
)
def search_ground_truth_database(
        collection_name: str = Field(
            description=(
                    f"Nazwa kolekcji do przeszukania. Dostępne kolekcje: "
                    f"[{', '.join(get_initial_collections())}]. "
            )
        ),
        query: str = Field(description="Treść zapytania do wyszukania."),
) -> list:
    return search(collection_name, query, 2)


@mcp.tool(
    name="read_neighboring_chunk",
    description=(
        "Pobiera sąsiedni fragment tekstu (następną lub poprzednią 'stronę') z konkretnego dokumentu. "
        "Użyj tego narzędzia, gdy fragment zwrócony przez 'search_ground_truth_database' wydaje się ucięty "
        "lub gdy potrzebujesz doczytać więcej kontekstu z dokumentu. "
        "Jeśli narzędzie zwróci None, oznacza to, że osiągnięto fizyczny koniec lub początek dokumentu."
    )
)
def read_neighboring_chunk(
        collection_name: str = Field(
            description=(
                    "TNazwa kolekcji, w której znajduje się dokument. Dostępne kolekcje: "
                    f"[{', '.join(get_initial_collections())}]. "
            )
        ),
        doc_id: int = Field(
            description="Unikalny identyfikator dokumentu (doc_id). Musisz wyodrębnić tę liczbę całkowitą z wyników poprzedniego wyszukiwania."
        ),
        current_seq: int = Field(
            description="Numer sekwencji (seq_num) fragmentu, który aktualnie przeglądasz. Musisz wyodrębnić tę liczbę całkowitą z wyników wyszukiwania."
        ),
        direction: str = Field(
            default="next",
            description="Kierunek czytania. Musi być ściśle wartością 'next', aby czytać do przodu, lub 'prev', aby czytać do tyłu."
        )
) -> dict:
    return get_neighbor_chunk(collection_name, doc_id, current_seq, direction)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")