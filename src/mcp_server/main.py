from mcp.server.fastmcp import FastMCP
from pydantic import Field
from tools.hybrid_rag import list_collections, search


mcp = FastMCP("Environment")


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
        rrf_reranker_k_param: int = Field(
            default=60,
            description=(
                "Smoothing parameter (K) for Reciprocal Rank Fusion (RRF). "
                "Controls the balance between vector and text search rankings. "
                "This parameter balances two methods: "
                "1. Semantic Search (Dense Vectors/Embeddings) - finds context and meaning. "
                "2. Keyword Search (BM25/Full-text) - finds exact word matches. "
                "Default is 60 (balanced). "
                "Lower values (e.g., 10) favor top results from a single method; "
                "higher values (e.g., 100) favor consensus between methods."
                "Range: 1 (very aggressive, favors top 1 result) to 500 (very conservative, favors consensus). "
                "Optimal value is usually 60."
            )
        )
) -> list:
    return search(collection_name, query, limit, rrf_reranker_k_param)

if __name__ == "__main__":
    mcp.run()