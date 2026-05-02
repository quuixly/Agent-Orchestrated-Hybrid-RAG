import asyncio
from typing import List, Any, Dict
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPClient:
    def __init__(self, server_configs: Dict[str, Dict[str, str]]):
        self.client = MultiServerMCPClient(server_configs)
        self.tools: List[Any] = []

    async def fetch_tools(self):
        self.tools = await self.client.get_tools()

        return self.tools

    async def execute_tool(self, name: str, args: Dict):
        tool = next((t for t in self.tools if t.name == name), None)

        if not tool:
            raise ValueError(f"Tool {name} does not exist.")

        return await tool.ainvoke(args)


if __name__ == "__main__":
    async def main():
        configs = {
            "weather_service": {
                "transport": "http",
                "url": "http://localhost:8000/mcp"
            }
        }

        mcp_client = MCPClient(configs)
        print(await mcp_client.fetch_tools())

    asyncio.run(main())