from mcp.server.fastmcp import FastMCP
from pydantic import Field
import os

from tools.hybrid_rag import list_collections, search, get_neighbor_chunk


MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", 8000))

mcp = FastMCP("Environment", host=MCP_HOST, port=MCP_PORT)


@mcp.tool(
    name="list_ground_truth_database_collections",
    description="List collections of ground truth databases",
)
def list_ground_truth_database_collections():
    return list_collections()


@mcp.tool(
    name="search_ground_truth_database",
    description="Search a specific ground truth database collection using a query"
)
def search_ground_truth_database(
        collection_name: str = Field(
            description=(
                    "The name of the collection to search in. "
                    "IMPORTANT: If you are not sure which collections exist, you MUST call "
                    "list_ground_truth_database_collections first to get the valid names."
            )
        ),
        query: str = Field(description="The search query string."),
        limit: int = Field(description="Maximum number of results to return."),
) -> list:
    return search(collection_name, query, limit)


@mcp.tool(
    name="read_neighboring_chunk",
    description=(
        "Retrieves the adjacent text chunk (next or previous 'page') from a specific document. "
        "Use this tool when a chunk returned by 'search_ground_truth_database' seems cut off, "
        "or when you need to read further into the document to get more context. "
        "If this tool returns None, it means you have reached the physical end or beginning of the document."
    )
)
def read_neighboring_chunk(
        collection_name: str = Field(
            description="The name of the collection where the document resides."
        ),
        doc_id: int = Field(
            description="The unique document identifier (doc_id). You must extract this exact integer from the results of your previous search."
        ),
        current_seq: int = Field(
            description="The sequence number (seq_num) of the chunk you are currently looking at. You must extract this integer from your search results."
        ),
        direction: str = Field(
            default="next",
            description="The direction to read. Must be strictly 'next' to read forward, or 'prev' to read backward."
        )
) -> dict:
    return get_neighbor_chunk(collection_name, doc_id, current_seq, direction)


if __name__ == "__main__":
    mcp.run()