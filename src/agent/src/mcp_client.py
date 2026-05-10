import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPClient:
    def __init__(self, server_configs):
        self.client = MultiServerMCPClient(server_configs)
        self.tools = []

    async def fetch_tools(self):
        self.tools = await self.client.get_tools()

        return self.tools

    async def execute_tool(self, name, args):
        tool = next((t for t in self.tools if t.name == name), None)
        if not tool:
            raise ValueError(f"Tool {name} does not exist.")
        return await tool.ainvoke(args)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    async def main():
        configs = {
            "Environment": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }

        mcp_client = MCPClient(configs)
        print(await mcp_client.fetch_tools())

    asyncio.run(main())