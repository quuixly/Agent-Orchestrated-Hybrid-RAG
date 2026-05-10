from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel
from transformers import AutoTokenizer
from dotenv import load_dotenv
import re
import json
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
import logging

from .prompts import find_user_context_prompt, think_prompt, respond_prompt, compress_thoughts_prompt


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medical_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_tag(text: str, tag_name: str) -> str:
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    return match.group(1).strip() if match else text.strip()

def parse_observation(raw_text: str) -> str:
    try:
        parsed = json.loads(raw_text)

        if "results" in parsed:
            parts = []
            for res in parsed["results"]:
                chunk = res.get("text", "")
                meta = f" [doc_id={res.get('doc_id')}, seq={res.get('seq_num')}]"
                parts.append(chunk + meta)
            return "\n---\n".join(parts)

        if "chunk" in parsed:
            res = parsed["chunk"]
            chunk = res.get("text", "")
            meta = f" [doc_id={res.get('doc_id')}, seq={res.get('seq_num')}]"
            return chunk + meta

        return raw_text

    except json.JSONDecodeError:
        return raw_text


class AgentState(BaseModel):
    user_question: str
    user_question_context: Optional[str] = None
    thoughts: Optional[str] = None
    final_response: Optional[str] = None
    step: int = 0
    used_queries: Annotated[List[str], lambda a, b: a + b] = []
    messages: Annotated[List[BaseMessage], lambda a, b: a + b] = []


class Agent:
    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("find_user_question_context", self.find_user_question_context)
        graph.add_node("think", self.think)
        graph.add_node("tool_node", self.tool_node)
        graph.add_node("compress", self.compress)
        graph.add_node("respond", self.respond)

        graph.add_edge(START, "find_user_question_context")
        graph.add_edge("find_user_question_context", "think")

        graph.add_conditional_edges(
            "think",
            self.router_think,
            {
                "continue": "tool_node",
                "end": "respond",
            }
        )

        graph.add_conditional_edges(
            "tool_node",
            self.router_tool,
            {
                "compress_thoughts": "compress",
                "continue": "think",
            }
        )

        graph.add_edge("compress", "think")
        graph.add_edge("respond", END)

        self.__workflow = graph.compile()

    async def find_user_question_context(self, state: AgentState, config: RunnableConfig):
        logger.info("--- NODE: FIND USER CONTEXT ---")
        llm = config["configurable"]["llm"]
        prompt = find_user_context_prompt.invoke({"user_message": state.user_question})

        response = await llm.ainvoke(prompt.to_messages())

        context = str(extract_tag(response.text, "analiza"))
        logger.info(f"Extracted User Context: {context}")

        return {
            "messages": [response],
            "user_question_context": context,
        }

    async def think(self, state: AgentState, config: RunnableConfig):
        max_steps = config["configurable"]["max_steps"]
        logger.info(f"--- NODE: THINK (step {state.step}/{max_steps}) ---")
        llm = config["configurable"]["llm"]
        mcp_client = config["configurable"]["mcp_client"]
        tools = await mcp_client.fetch_tools()
        llm_with_tools = llm.bind_tools(tools)

        used_queries_str = "\n".join(f"- {q}" for q in state.used_queries) if state.used_queries else "brak"

        prompt = think_prompt.invoke({
            "context": state.user_question_context,
            "thoughts": state.thoughts,
            "used_queries": used_queries_str,
            "steps_left": max_steps - state.step,
        })

        response = await llm_with_tools.ainvoke(prompt.to_messages())

        result = extract_tag(response.text, "mysli")
        logger.info(f"New thoughts: {result}")

        new_queries = []
        tool_info = ""
        if response.tool_calls:
            for call in response.tool_calls:
                query = call["args"].get("query", call["args"].get("query_text", ""))
                logger.info(f"Tool requested: {call['name']} args: {call['args']}")
                tool_info += f"\n[ACTION]: {call['name']}({call['args']})"
                if query:
                    new_queries.append(query)

        new_thoughts = (state.thoughts + "\n" if state.thoughts else "") + str(result) + tool_info

        return {
            "messages": [response],
            "thoughts": new_thoughts,
            "step": state.step + 1,
            "used_queries": new_queries,
        }

    async def tool_node(self, state: AgentState, config: RunnableConfig):
        logger.info("--- NODE: TOOL EXECUTION ---")
        mcp_client = config["configurable"]["mcp_client"]

        last_message = state.messages[-1]
        new_messages = []
        readable_thoughts = ""

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            logger.info(f"Executing tool: {tool_name} args: {tool_args}")
            try:
                observation = await mcp_client.execute_tool(tool_name, tool_args)
                if isinstance(observation, list) and len(observation) > 0:
                    raw_text = observation[0].get("text", "")
                else:
                    raw_text = str(observation)

                final_observation = parse_observation(raw_text)

            except Exception as e:
                logger.error(f"Tool error {tool_name}: {e}")
                final_observation = f"Error: {str(e)}"

            logger.info(f"Observation ({len(final_observation)} chars): {final_observation[:150]}...")

            new_messages.append(ToolMessage(
                content=final_observation,
                tool_call_id=tool_id,
                name=tool_name,
            ))
            readable_thoughts += f"\n[Obs/{tool_name}]: {final_observation}"

        return {
            "messages": new_messages,
            "thoughts": (state.thoughts or "") + readable_thoughts,
        }

    async def respond(self, state: AgentState, config: RunnableConfig):
        logger.info("--- NODE: RESPOND ---")
        llm = config["configurable"]["llm"]
        prompt = respond_prompt.invoke({
            "user_question": state.user_question,
            "context": state.user_question_context,
            "thoughts": state.thoughts,
        })

        response = await llm.ainvoke(prompt.to_messages())
        final_response = str(extract_tag(response.text, "odpowiedz"))
        logger.info(f"Final response: {final_response[:200]}...")

        return {
            "messages": [response],
            "final_response": final_response,
        }

    async def router_think(self, state: AgentState, config: RunnableConfig):
        logger.info("--- ROUTER: PO MYŚLENIU ---")
        max_steps = config["configurable"]["max_steps"]

        if state.step >= max_steps:
            logger.info(f"Decision: END (max steps {max_steps} reached)")
            return "end"

        last_msg = state.messages[-1] if state.messages else None
        if not last_msg or not getattr(last_msg, "tool_calls", None):
            logger.info("Decision: END (no tool calls or final answer reached)")
            return "end"

        logger.info("Decision: TOOL_NODE (executing tool)")
        return "continue"

    async def router_tool(self, state: AgentState, config: RunnableConfig):
        logger.info("--- ROUTER: PO NARZĘDZIU (SPRAWDZANIE LIMITU) ---")

        tokenizer = config["configurable"]["tokenizer"]
        max_context_len = config["configurable"]["max_context_len"]

        tokens = tokenizer.encode(state.thoughts or "")
        logger.info(f"Obecne tokeny historii: {len(tokens)}")

        if len(tokens) > max_context_len:
            logger.info(f"Decision: COMPRESS ({len(tokens)} > {max_context_len})")
            return "compress_thoughts"

        logger.info("Decision: THINK (context size OK)")
        return "continue"

    async def compress(self, state: AgentState, config: RunnableConfig):
        logger.info("--- NODE: COMPRESS ---")
        llm = config["configurable"]["llm"]
        prompt = compress_thoughts_prompt.invoke({
            "context": state.user_question_context,
            "thoughts": state.thoughts,
        })

        response = await llm.ainvoke(prompt.to_messages())
        result = extract_tag(response.text, "kompresja")
        logger.info("Compression finished.")

        return {
            "messages": [response],
            "thoughts": str(result),
        }

    async def __call__(self, *args, **kwargs):
        state = args[0] if args else kwargs.get("state", {})
        user_question = state.get("user_question", "<No question provided>")
        logger.info(f"Starting Agent for question: {user_question}")
        return await self.__workflow.ainvoke(*args, **kwargs)


if __name__ == "__main__":
    from bielik_llm import BielikChatModel
    from mcp_client import MCPClient
    import asyncio


    async def main():
        bielik = BielikChatModel()
        model_name = "speakleash/Bielik-11B-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        configs = {
            "medical_knowledge_base": {
                "transport": "http",
                "url": "http://127.0.0.1:8002/mcp",
            }
        }
        mcp_client = MCPClient(configs)

        conf = {
            "configurable": {
                "llm": bielik,
                "mcp_client": mcp_client,
                "tokenizer": tokenizer,
                "max_context_len": 3000,
                "max_steps": 10
            }
        }

        a = Agent()
        logger.info("--- EXECUTING TEST RUN ---")
        result = await a({"user_question": "Co to cukrzyca?"}, config=conf)
        print(result["final_response"])

    asyncio.run(main())